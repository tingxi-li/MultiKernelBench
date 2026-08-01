#!/usr/bin/env python3
"""Create deterministic preregistration or completed-campaign evidence bundles.

This utility was added after ``protocol_freeze_receipt.json`` was written.  It
is evidence packaging code only: it is deliberately absent from the frozen
protocol file map and is never consulted by launch validation or execution.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:
    from .analyze import analyze, load_outcomes
    from .campaign import CAMPAIGN_ID, canonical_json, sha256_file, validate_manifest_rows
    from .controller import EventJournal, TERMINAL_EVENT_TYPES
    from .freeze_protocol import documents as freeze_documents
    from .validate_launch import validate_launch_state
except ImportError:  # direct script execution
    from analyze import analyze, load_outcomes  # type: ignore
    from campaign import CAMPAIGN_ID, canonical_json, sha256_file, validate_manifest_rows  # type: ignore
    from controller import EventJournal, TERMINAL_EVENT_TYPES  # type: ignore
    from freeze_protocol import documents as freeze_documents  # type: ignore
    from validate_launch import validate_launch_state  # type: ignore


BASE = Path(__file__).resolve().parent
MANIFEST_MEMBER = "EVIDENCE_MANIFEST.json"
SHA256_FIELDS = (
    "locks/protocol_freeze_receipt.json",
    "locks/prompt_contract_lock.json",
    "gates/FREEZE_RECEIPT.json",
    "manifests/summary.json",
    "locks/model_resolution_lock.json",
    "locks/gpu_assignment_lock.json",
    "locks/gate_bindings.json",
    "locks/remote_preregistration_lock.json",
    "locks/reference_latency_lock.json",
)
REQUIRED_LAUNCH_CHECKS = {
    "trajectory_manifests",
    "prompt_contract_lock",
    "protocol_freeze_receipt",
    "gate_preregistration_receipt",
    "immutable_model_resolution",
    "gpu_uuid_binding",
    "robust_hidden_gate_bindings",
    "remote_preregistration",
    "provider_credentials",
    "provider_sdks",
    "frozen_reference_latencies",
}
EXCLUDED_TOP_LEVEL = {
    "analysis_outputs",
    "evidence",
    "outcomes",
    "results",
    "trajectories",
}
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".torch_ext",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
EXCLUDED_SUFFIXES = {
    ".a",
    ".cubin",
    ".ninja",
    ".o",
    ".obj",
    ".ptx",
    ".pyc",
    ".pyo",
    ".so",
}
ACTIVE_LOCK_NAMES = {"active.lock", "launcher.lock", "run.lock"}
SENSITIVE_KEYS = {"access_token", "api_key", "authorization", "password", "secret"}


class EvidenceError(RuntimeError):
    """Evidence inputs are incomplete, unsafe, or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected a JSON object: {path}")
    return value


def _stable_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative_to_base(path: Path, base: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise EvidenceError(f"evidence input must remain inside {base}: {path}") from exc


def _excluded(relative: Path, *, support_tree: bool) -> bool:
    if not relative.parts:
        return True
    if support_tree and relative.parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return True
    name = relative.name
    lowered = name.lower()
    if name in ACTIVE_LOCK_NAMES or lowered.endswith(".lock") or name.startswith(".tmp"):
        return True
    if any(
        ".partial" in part.lower() or part.lower().endswith((".tmp", ".temp"))
        for part in relative.parts
    ) or lowered.endswith("~"):
        return True
    if relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return False


def _walk_files(root: Path, base: Path, *, support_tree: bool) -> list[Path]:
    if not root.is_dir():
        raise EvidenceError(f"evidence tree is not a directory: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = _relative_to_base(path, base)
        if _excluded(relative, support_tree=support_tree):
            continue
        if path.is_symlink():
            raise EvidenceError(f"symlink is not allowed in evidence: {path}")
        if path.is_file():
            files.append(path.resolve())
    return sorted(files, key=lambda item: str(_relative_to_base(item, base)))


def _check_sensitive_keys(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                raise EvidenceError(f"sensitive key rejected at {location}.{key}")
            _check_sensitive_keys(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_sensitive_keys(item, f"{location}[{index}]")


def _scan_structured_file(path: Path) -> None:
    if path.suffix == ".json":
        _check_sensitive_keys(json.loads(path.read_text(encoding="utf-8")), str(path))
    elif path.suffix == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise EvidenceError(f"blank/partial JSONL line at {path}:{line_number}")
            _check_sensitive_keys(json.loads(line), f"{path}:{line_number}")


def verify_frozen_protocol(base: Path) -> dict[str, str]:
    """Verify all historical receipts byte-for-byte without rewriting them."""
    expected = freeze_documents(base)
    mismatches = [
        str(path.relative_to(base))
        for path, payload in expected.items()
        if not path.is_file() or path.read_bytes() != payload
    ]
    if mismatches:
        raise EvidenceError(f"frozen protocol receipt mismatch: {mismatches}")
    receipt = _read_json(base / "locks/protocol_freeze_receipt.json")
    if "capture_evidence.py" in receipt.get("files", {}):
        raise EvidenceError("post-freeze evidence utility unexpectedly entered protocol lock")
    return {
        str(path.relative_to(base)): sha256_file(path)
        for path in sorted(expected, key=lambda item: str(item.relative_to(base)))
    }


def _manifest_rows(base: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    core = [json.loads(line) for line in (base / "manifests/core_192.jsonl").read_text().splitlines()]
    extension = [json.loads(line) for line in (base / "manifests/prompt_extension_128.jsonl").read_text().splitlines()]
    validate_manifest_rows(core, extension)
    rows = core + extension
    return rows, {row["trajectory_id"]: row for row in rows}


def preregistration_state(base: Path) -> dict[str, Any]:
    freeze_hashes = verify_frozen_protocol(base)
    rows, _by_id = _manifest_rows(base)
    gpu_lock = _read_json(base / "locks/gpu_assignment_lock.json")
    if gpu_lock.get("state") == "resolved":
        verify_gpu_binding_receipt(base)
    readiness = validate_launch_state(
        base,
        environ={},
        check_gpu_runtime=False,
        check_provider_sdks=False,
    )
    failed = [
        {"name": row["name"], "detail": row["detail"]}
        for row in readiness["checks"]
        if row["passed"] is not True
    ]
    return {
        "artifact_state": {
            "gate_bindings": _read_json(base / "locks/gate_bindings.json")["state"],
            "gpu_assignment": _read_json(base / "locks/gpu_assignment_lock.json")["state"],
            "model_resolution": _read_json(base / "locks/model_resolution_lock.json")["state"],
            "reference_latencies": _read_json(base / "locks/reference_latency_lock.json")["state"],
            "remote_preregistration": _read_json(base / "locks/remote_preregistration_lock.json")["state"],
        },
        "campaign_results_complete": False,
        "capture_state": "preregistration_only",
        "execution_claimed": False,
        "frozen_receipt_sha256": freeze_hashes,
        "gpu_trajectories_claimed": False,
        "launch_permitted_claimed": False,
        "performance_results_claimed": False,
        "provider_calls_claimed": False,
        "trajectory_manifest_count": len(rows),
        "unresolved_launch_checks": failed,
    }


def verify_gpu_binding_receipt(base: Path) -> dict[int, str]:
    lock_path = base / "locks/gpu_assignment_lock.json"
    lock = _read_json(lock_path)
    if lock.get("state") != "resolved":
        raise EvidenceError("GPU assignment remains unresolved")
    relative_receipt = lock.get("binding_receipt")
    if not isinstance(relative_receipt, str) or not relative_receipt:
        raise EvidenceError("resolved GPU lock has no binding receipt")
    receipt_path = (lock_path.parent / relative_receipt).resolve()
    _relative_to_base(receipt_path, base)
    if sha256_file(receipt_path) != lock.get("binding_receipt_sha256"):
        raise EvidenceError("GPU binding receipt hash mismatch")
    receipt = _read_json(receipt_path)
    if (
        receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("record_type") != "convergence_v2_gpu_assignment_binding"
        or receipt.get("state") != "resolved_identity_runtime_recheck_pending"
        or receipt.get("runtime_recheck_required") is not True
    ):
        raise EvidenceError("GPU binding receipt has an invalid identity/state")
    lock_slots = {int(row["slot"]): row.get("gpu_uuid") for row in lock.get("slots", [])}
    receipt_slots = {int(row["slot"]): row.get("gpu_uuid") for row in receipt.get("slots", [])}
    if lock_slots != receipt_slots or set(lock_slots) != {0, 1, 2, 3} or len(set(lock_slots.values())) != 4:
        raise EvidenceError("GPU lock and binding receipt slot maps differ")
    if any(row.get("name") != "NVIDIA RTX 6000 Ada Generation" for row in receipt.get("slots", [])):
        raise EvidenceError("GPU binding receipt contains a non-RTX-6000-Ada device")
    for evidence in receipt.get("evidence", []):
        source = (base / evidence["path"]).resolve()
        if not source.is_file() or sha256_file(source) != evidence.get("sha256"):
            raise EvidenceError(f"GPU binding source evidence changed: {evidence.get('path')}")
    return lock_slots  # type: ignore[return-value]


def _required_launch_hashes(base: Path) -> dict[str, str]:
    return {name: sha256_file(base / name) for name in SHA256_FIELDS}


def _validate_launch_receipt(path: Path, base: Path) -> dict[str, Any]:
    receipt = _read_json(path)
    if (
        receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("ready") is not True
        or receipt.get("action") != "launch_permitted"
    ):
        raise EvidenceError("launch receipt does not attest launch_permitted")
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        raise EvidenceError("launch receipt has no checks list")
    by_name = {row.get("name"): row for row in checks if isinstance(row, dict)}
    if not REQUIRED_LAUNCH_CHECKS.issubset(by_name):
        raise EvidenceError("launch receipt omits required readiness checks")
    failed = [name for name in REQUIRED_LAUNCH_CHECKS if by_name[name].get("passed") is not True]
    if failed:
        raise EvidenceError(f"launch receipt contains failed checks: {sorted(failed)}")
    if receipt.get("artifact_sha256") != _required_launch_hashes(base):
        raise EvidenceError("launch receipt is not hash-bound to current frozen/mutable locks")
    return receipt


def _event_path_map(root: Path, expected_ids: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    candidates = sorted(root.rglob("*.jsonl"))
    for path in candidates:
        trajectory_id = path.stem if path.stem in expected_ids else (
            path.parent.name if path.name == "events.jsonl" and path.parent.name in expected_ids else None
        )
        if trajectory_id is None:
            raise EvidenceError(f"unexpected JSONL in trajectory root: {path}")
        if trajectory_id in result:
            raise EvidenceError(f"duplicate event journal for {trajectory_id}")
        result[trajectory_id] = path.resolve()
    if set(result) != expected_ids:
        missing = sorted(expected_ids - set(result))
        raise EvidenceError(f"event-journal census mismatch: observed={len(result)}, missing={missing[:5]}")
    return result


def _resolved_models(base: Path) -> dict[str, str]:
    lock = _read_json(base / "locks/model_resolution_lock.json")
    if lock.get("state") != "resolved":
        raise EvidenceError("model resolution remains unresolved")
    result = {}
    for row in lock.get("resolutions", []):
        if row.get("provider_attested_immutable") is not True or not row.get("immutable_revision"):
            raise EvidenceError("model lock contains an unattested/empty revision")
        result[f"{row['provider']}:{row['requested_alias']}"] = row["immutable_revision"]
    return result


def _resolved_gpus(base: Path) -> dict[int, str]:
    return verify_gpu_binding_receipt(base)


def _validate_journals(
    root: Path,
    by_id: dict[str, dict[str, Any]],
    base: Path,
) -> tuple[dict[str, Any], list[Path]]:
    paths = _event_path_map(root, set(by_id))
    revisions = _resolved_models(base)
    gpu_by_slot = _resolved_gpus(base)
    terminal_counts: Counter[str] = Counter()
    provider_events = 0
    for trajectory_id, path in paths.items():
        events = EventJournal(path, CAMPAIGN_ID, trajectory_id).read()
        if not events or events[-1]["event_type"] not in TERMINAL_EVENT_TYPES:
            raise EvidenceError(f"trajectory is not terminal: {trajectory_id}")
        started = [event for event in events if event["event_type"] == "trajectory_started"]
        if len(started) != 1:
            raise EvidenceError(f"trajectory must have exactly one start event: {trajectory_id}")
        manifest = by_id[trajectory_id]
        expected_gpu = gpu_by_slot[manifest["gpu_slot"]]
        payload = started[0]["payload"]
        if payload.get("gpu_slot") != manifest["gpu_slot"] or payload.get("gpu_uuid") != expected_gpu:
            raise EvidenceError(f"trajectory GPU binding mismatch: {trajectory_id}")
        usage = [event for event in events if event["event_type"] == "provider_usage"]
        if not usage:
            raise EvidenceError(f"trajectory has no retained provider-usage event: {trajectory_id}")
        expected_revision = revisions[manifest["model_key"]]
        if any(event["payload"].get("resolved_model_revision") != expected_revision for event in usage):
            raise EvidenceError(f"trajectory provider revision mismatch: {trajectory_id}")
        provider_events += len(usage)
        terminal_counts[events[-1]["event_type"]] += 1
    return (
        {
            "event_journals_verified": len(paths),
            "provider_usage_events_verified": provider_events,
            "terminal_event_counts": dict(sorted(terminal_counts.items())),
        },
        list(paths.values()),
    )


def _validate_outcomes_analysis(
    outcomes_path: Path,
    analysis_path: Path,
    expected_ids: set[str],
) -> dict[str, Any]:
    outcomes = load_outcomes(outcomes_path)
    observed_ids = {row.trajectory_id for row in outcomes}
    if len(outcomes) != 320 or observed_ids != expected_ids:
        raise EvidenceError("survival outcome census is not exactly the frozen 320 trajectories")
    analysis_value = _read_json(analysis_path)
    tau = analysis_value.get("tau_s")
    if not isinstance(tau, (int, float)):
        raise EvidenceError("analysis has no numeric tau_s")
    expected = analyze(outcomes, float(tau))
    expected_payload = _stable_json(expected)
    if analysis_path.read_bytes() != expected_payload:
        raise EvidenceError("analysis bytes do not exactly reproduce from outcomes")
    return {
        "analysis_sha256": sha256_file(analysis_path),
        "outcomes_sha256": sha256_file(outcomes_path),
        "survival_outcomes_verified": len(outcomes),
        "tau_s": float(tau),
    }


def completed_state(
    base: Path,
    *,
    launch_receipt: Path,
    trajectory_root: Path,
    outcomes: Path,
    analysis_path: Path,
) -> tuple[dict[str, Any], list[Path]]:
    verify_frozen_protocol(base)
    rows, by_id = _manifest_rows(base)
    _validate_launch_receipt(launch_receipt, base)
    # Recheck all durable content locks without depending on current credentials,
    # provider SDK installation, or live GPU availability. Runtime readiness is
    # attested by the hash-bound launch receipt above.
    durable = validate_launch_state(
        base,
        environ={"OPENAI_API_KEY": "present-at-validation", "ANTHROPIC_API_KEY": "present-at-validation"},
        check_gpu_runtime=False,
        check_provider_sdks=False,
    )
    durable_failures = [row for row in durable["checks"] if row["passed"] is not True]
    if durable_failures:
        raise EvidenceError(f"durable launch locks no longer validate: {durable_failures}")
    journal_summary, journal_files = _validate_journals(trajectory_root, by_id, base)
    analysis_summary = _validate_outcomes_analysis(outcomes, analysis_path, set(by_id))
    state = {
        "analysis": analysis_summary,
        "campaign_results_complete": True,
        "capture_state": "completed_campaign",
        "execution_claimed": True,
        "gpu_trajectories_claimed": True,
        "journals": journal_summary,
        "launch_permitted_claimed": True,
        "performance_results_claimed": True,
        "provider_calls_claimed": True,
        "trajectory_manifest_count": len(rows),
        "unresolved_launch_checks": [],
    }
    return state, journal_files


def _entry(path: Path, base: Path) -> dict[str, Any]:
    relative = _relative_to_base(path, base)
    return {
        "path": str(relative),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _tar_member(handle: tarfile.TarFile, name: str, payload: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    handle.addfile(info, io.BytesIO(payload))


def write_bundle(
    *,
    base: Path,
    output_prefix: Path,
    mode: str,
    state: dict[str, Any],
    explicit_files: Iterable[Path] = (),
) -> tuple[Path, Path, dict[str, Any]]:
    support = _walk_files(base, base, support_tree=True)
    files = {path.resolve() for path in support}
    for path in explicit_files:
        resolved = path.resolve()
        relative = _relative_to_base(resolved, base)
        if _excluded(relative, support_tree=False):
            continue
        if resolved.is_symlink() or not resolved.is_file():
            raise EvidenceError(f"invalid explicit evidence file: {resolved}")
        files.add(resolved)
    ordered = sorted(files, key=lambda item: str(_relative_to_base(item, base)))
    for path in ordered:
        _scan_structured_file(path)
    entries = [_entry(path, base) for path in ordered]
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "capture_mode": mode,
        "claim_state": state,
        "files": entries,
        "post_freeze_evidence_utility": {
            "included_in_protocol_freeze_receipt": False,
            "launch_input": False,
            "path": "capture_evidence.py",
            "sha256": sha256_file(base / "capture_evidence.py"),
        },
        "schema_version": 1,
    }
    manifest_payload = _stable_json(manifest)
    archive = output_prefix.with_suffix(".tar.gz")
    index = output_prefix.with_suffix(".index.json")
    if archive.exists() or index.exists():
        raise FileExistsError("refusing to overwrite evidence archive/index")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as tar:
                    _tar_member(tar, MANIFEST_MEMBER, manifest_payload)
                    for path, entry in zip(ordered, entries):
                        _tar_member(tar, entry["path"], path.read_bytes())
        os.replace(temporary, archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    index_value = {
        "archive_sha256": sha256_file(archive),
        "archive_size": archive.stat().st_size,
        "campaign_id": CAMPAIGN_ID,
        "capture_mode": mode,
        "evidence_manifest_sha256": _sha256_bytes(manifest_payload),
        "file_count": len(entries),
        "schema_version": 1,
        "state_sha256": _sha256_bytes(canonical_json(state)),
    }
    index_temporary = index.with_name(f".{index.name}.tmp.{os.getpid()}")
    try:
        index_temporary.write_bytes(_stable_json(index_value))
        os.replace(index_temporary, index)
    finally:
        if index_temporary.exists():
            index_temporary.unlink()
    return archive, index, index_value


def _safe_archive_name(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and str(path) == value


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def verify_evidence(index_path: Path) -> dict[str, Any]:
    """Verify an immutable bundle solely from its canonical index and members."""
    index_path = index_path.resolve()
    index = _read_json(index_path)
    if index_path.read_bytes() != _stable_json(index):
        raise EvidenceError("evidence index is not canonical stable JSON")
    mode = index.get("capture_mode")
    if (
        index.get("schema_version") != 1
        or index.get("campaign_id") != CAMPAIGN_ID
        or mode not in {"prereg", "complete"}
        or not _valid_sha256(index.get("archive_sha256"))
        or not _valid_sha256(index.get("evidence_manifest_sha256"))
        or not _valid_sha256(index.get("state_sha256"))
        or isinstance(index.get("archive_size"), bool)
        or not isinstance(index.get("archive_size"), int)
        or index["archive_size"] < 0
        or isinstance(index.get("file_count"), bool)
        or not isinstance(index.get("file_count"), int)
        or index["file_count"] < 0
    ):
        raise EvidenceError("evidence index identity or fields are invalid")
    suffix = ".index.json"
    if not index_path.name.endswith(suffix) or index_path.name == suffix:
        raise EvidenceError("evidence index filename must end in .index.json")
    archive = index_path.with_name(index_path.name[: -len(suffix)] + ".tar.gz")
    try:
        archive_payload = archive.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read evidence archive {archive}: {exc}") from exc
    if (
        len(archive_payload) != index["archive_size"]
        or _sha256_bytes(archive_payload) != index["archive_sha256"]
    ):
        raise EvidenceError("evidence archive bytes differ from index")
    if (
        len(archive_payload) < 10
        or archive_payload[:2] != b"\x1f\x8b"
        or archive_payload[2] != 8
        or archive_payload[3] != 0
        or archive_payload[4:8] != b"\x00\x00\x00\x00"
    ):
        raise EvidenceError("evidence gzip header is not normalized")
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:gz") as bundle:
            members = bundle.getmembers()
            if any(not member.isfile() for member in members):
                raise EvidenceError("evidence archive contains a non-file member")
            names = [member.name for member in members]
            if len(set(names)) != len(names) or not names or names[0] != MANIFEST_MEMBER:
                raise EvidenceError("evidence archive has duplicate members or no leading manifest")
            if any(not _safe_archive_name(name) for name in names):
                raise EvidenceError("evidence archive contains an unsafe member name")
            if any(
                member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mode not in {0o644, 0o755}
                for member in members
            ):
                raise EvidenceError("evidence archive metadata is not normalized")
            manifest_handle = bundle.extractfile(members[0])
            if manifest_handle is None:
                raise EvidenceError("cannot read embedded evidence manifest")
            manifest_payload = manifest_handle.read()
            try:
                manifest = json.loads(manifest_payload)
            except json.JSONDecodeError as exc:
                raise EvidenceError("embedded evidence manifest is invalid JSON") from exc
            if not isinstance(manifest, dict) or manifest_payload != _stable_json(manifest):
                raise EvidenceError("embedded evidence manifest is not canonical stable JSON")
            if _sha256_bytes(manifest_payload) != index["evidence_manifest_sha256"]:
                raise EvidenceError("embedded evidence manifest hash differs from index")
            if (
                manifest.get("schema_version") != 1
                or manifest.get("campaign_id") != CAMPAIGN_ID
                or manifest.get("capture_mode") != mode
            ):
                raise EvidenceError("embedded evidence manifest identity differs")
            state = manifest.get("claim_state")
            if not isinstance(state, dict) or _sha256_bytes(canonical_json(state)) != index["state_sha256"]:
                raise EvidenceError("embedded claim state hash differs from index")
            entries = manifest.get("files")
            if not isinstance(entries, list) or len(entries) != index["file_count"]:
                raise EvidenceError("evidence file count differs from index")
            expected_names: list[str] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise EvidenceError("evidence manifest contains a non-object file entry")
                name = entry.get("path")
                size = entry.get("size")
                if (
                    not _safe_archive_name(name)
                    or name == MANIFEST_MEMBER
                    or not _valid_sha256(entry.get("sha256"))
                    or isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                    or _excluded(Path(name), support_tree=False)
                ):
                    raise EvidenceError("evidence manifest contains an unsafe/malformed file entry")
                expected_names.append(name)
            if expected_names != sorted(expected_names) or names != [MANIFEST_MEMBER, *expected_names]:
                raise EvidenceError("evidence archive order/membership differs from manifest")
            for member, entry in zip(members[1:], entries):
                handle = bundle.extractfile(member)
                if handle is None:
                    raise EvidenceError(f"cannot read evidence member {member.name}")
                payload = handle.read()
                if len(payload) != entry["size"] or _sha256_bytes(payload) != entry["sha256"]:
                    raise EvidenceError(f"evidence member differs: {member.name}")
    except (OSError, tarfile.TarError) as exc:
        raise EvidenceError(f"cannot read evidence archive {archive}: {exc}") from exc
    claim_keys = (
        "campaign_results_complete",
        "execution_claimed",
        "gpu_trajectories_claimed",
        "launch_permitted_claimed",
        "performance_results_claimed",
        "provider_calls_claimed",
    )
    expected_claim = mode == "complete"
    if any(state.get(key) is not expected_claim for key in claim_keys):
        raise EvidenceError("evidence claim flags do not match capture mode")
    expected_capture_state = "completed_campaign" if expected_claim else "preregistration_only"
    if state.get("capture_state") != expected_capture_state:
        raise EvidenceError("evidence capture state does not match capture mode")
    utility = manifest.get("post_freeze_evidence_utility")
    by_name = {entry["path"]: entry for entry in entries}
    if (
        not isinstance(utility, dict)
        or utility.get("included_in_protocol_freeze_receipt") is not False
        or utility.get("launch_input") is not False
        or utility.get("path") != "capture_evidence.py"
        or utility.get("sha256") != by_name.get("capture_evidence.py", {}).get("sha256")
    ):
        raise EvidenceError("post-freeze evidence utility binding differs")
    return {
        "archive_sha256": index["archive_sha256"],
        "capture_mode": mode,
        "file_count": len(entries),
        "index": str(index_path),
        "ok": True,
    }


def capture(
    *,
    base: Path,
    mode: str,
    output_prefix: Path,
    launch_receipt: Path | None = None,
    trajectory_root: Path | None = None,
    outcomes: Path | None = None,
    analysis_path: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    base = base.resolve()
    if mode == "prereg":
        if any(value is not None for value in (launch_receipt, trajectory_root, outcomes, analysis_path)):
            raise EvidenceError("prereg capture does not accept execution artifacts")
        state = preregistration_state(base)
        return write_bundle(base=base, output_prefix=output_prefix, mode=mode, state=state)
    if mode != "complete":
        raise EvidenceError(f"unsupported capture mode: {mode}")
    if any(value is None for value in (launch_receipt, trajectory_root, outcomes, analysis_path)):
        raise EvidenceError("complete capture requires launch receipt, trajectory root, outcomes, and analysis")
    assert launch_receipt is not None and trajectory_root is not None and outcomes is not None and analysis_path is not None
    explicit_roots = [launch_receipt.resolve(), outcomes.resolve(), analysis_path.resolve()]
    state, journals = completed_state(
        base,
        launch_receipt=launch_receipt.resolve(),
        trajectory_root=trajectory_root.resolve(),
        outcomes=outcomes.resolve(),
        analysis_path=analysis_path.resolve(),
    )
    # Include the complete per-trajectory evidence tree (sources, measurement
    # receipts, event journals, and terminal records), while applying the same
    # cache/build/partial exclusions. The validated journals are a strict
    # subset; retaining ``journals`` explicitly documents that invariant.
    trajectory_files = _walk_files(trajectory_root.resolve(), base, support_tree=False)
    explicit_roots.extend(trajectory_files)
    if not set(journals).issubset(set(trajectory_files)):
        raise EvidenceError("validated journals escaped the retained trajectory tree")
    return write_bundle(
        base=base,
        output_prefix=output_prefix,
        mode=mode,
        state=state,
        explicit_files=explicit_roots,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prereg", "complete", "verify"))
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--trajectory-root", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--analysis", dest="analysis_path", type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        if args.index is None:
            parser.error("verify requires --index")
        if args.output_prefix is not None or any(
            value is not None for value in (args.launch_receipt, args.trajectory_root, args.outcomes, args.analysis_path)
        ):
            parser.error("verify accepts only --index (and optional --base)")
        print(json.dumps(verify_evidence(args.index), sort_keys=True))
        return 0
    if args.output_prefix is None:
        parser.error(f"{args.mode} requires --output-prefix")
    if args.index is not None:
        parser.error("--index is only valid for verify")
    archive, index, value = capture(
        base=args.base,
        mode=args.mode,
        output_prefix=args.output_prefix.resolve(),
        launch_receipt=args.launch_receipt,
        trajectory_root=args.trajectory_root,
        outcomes=args.outcomes,
        analysis_path=args.analysis_path,
    )
    print(json.dumps({"archive": str(archive), "index": str(index), **value}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
