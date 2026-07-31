"""Shared identities, hashing, and fail-closed evidence checks for v1r1."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO_ROOT = HERE.parents[3]
RESULT_TAG = "crossed_v1r1"
RECOVERY_ID = "fused-epilogue-crossed-v1r1-reporting-fix"
CAMPAIGN_ID = "fused-epilogue-crossed-v1"
PARENT_GIT_COMMIT = "4ddfc88d5712959e12161da75ce991f8c9a20248"
PARENT_LAUNCH_LOCK_SHA256 = "fe9a0123cc1e3aa64f5d0026e09d89b918838174c7964c85c4580b0962767e8a"
PARENT_SOURCE_BUNDLE_SHA256 = "f8d7745416c7254d10cfdd144d80d91409f745d521d0debb5f5f8cc87c9de0b6"

LOCK_PATH = HERE / "recovery_lock.json"
INCIDENT_PATH = HERE / "incident_receipt.json"
PARENT_RESULT_ROOT = PARENT / "results/crossed_v1"
RESULT_ROOT = PARENT / f"results/{RESULT_TAG}"
REMOTE_RECEIPT_PATH = RESULT_ROOT / "remote_verification_receipt.json"

EXPECTED_PARENT_ARTIFACTS = {
    "audit/receipts/shard00.json": {
        "sha256": "f60a5dd3ebd5e570ef92e92517c0357212de3f31f80aa1c8cd84c9c4a6665591",
        "size": 3662,
    },
    "audit/receipts/shard01.json": {
        "sha256": "d837a685f7b50f06a6de93bde4875981d752f53cf0f082f10f87da3ae8c71eac",
        "size": 3674,
    },
    "audit/receipts/shard02.json": {
        "sha256": "e73363dbaba0349da428835e54f6e74eae778835966d4a06eaf32b711063ab09",
        "size": 3684,
    },
    "audit/receipts/shard03.json": {
        "sha256": "0412ee951517342230d5e471c16e4e1b5566e0de88276ea8a5ee642bbc1fcf99",
        "size": 3680,
    },
    "remote_push_receipt.json": {
        "sha256": "e812a89f4c9ab09b89644caf5fb565cc7c8ef13e03c2b7b6d9bd4dd3abd82df3",
        "size": 773,
    },
}

BINDING_KEYS = (
    "recovery_id",
    "parent_launch_lock_sha256",
    "parent_source_bundle_sha256",
    "recovery_lock_sha256",
    "recovery_source_bundle_sha256",
    "recovery_git_commit",
    "result_tag",
)


class RecoveryError(RuntimeError):
    """A recovery provenance or append-only contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("utf-8") + b"\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read JSON {path}: {exc}") from exc


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically create JSON without ever replacing a retained artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    require(not temporary.exists(), f"unreconciled temporary artifact: {temporary}")
    temporary.write_bytes(stable_json_bytes(value))
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def exclusive_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically create finite canonical JSONL without replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    require(not temporary.exists(), f"unreconciled temporary artifact: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise RecoveryError(f"blank JSONL row {path}:{line_number}")
                value = json.loads(line)
                require(isinstance(value, dict), f"non-object JSONL row {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _load_parent_modules():
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))
    import core as parent_core  # type: ignore

    return parent_core


def parent_inventory() -> dict[str, dict[str, Any]]:
    require(PARENT_RESULT_ROOT.is_dir(), "preserved crossed_v1 result root is missing")
    require(not PARENT_RESULT_ROOT.is_symlink(), "preserved crossed_v1 result root is symlinked")
    observed: dict[str, dict[str, Any]] = {}
    for path in sorted(PARENT_RESULT_ROOT.rglob("*")):
        require(not path.is_symlink(), f"symlink in preserved crossed_v1 inventory: {path}")
        if path.is_file():
            relative = path.relative_to(PARENT_RESULT_ROOT).as_posix()
            observed[relative] = {"sha256": file_sha256(path), "size": path.stat().st_size}
    return observed


def verify_parent() -> dict[str, Any]:
    parent_core = _load_parent_modules()
    campaign, cells, launch_lock = parent_core.load_contract()
    require(campaign["campaign_id"] == CAMPAIGN_ID, "parent campaign identity changed")
    require(file_sha256(parent_core.LOCK_PATH) == PARENT_LAUNCH_LOCK_SHA256, "parent launch lock changed")
    require(launch_lock["source_bundle_sha256"] == PARENT_SOURCE_BUNDLE_SHA256, "parent source bundle changed")
    observed = parent_inventory()
    require(observed == EXPECTED_PARENT_ARTIFACTS, f"preserved crossed_v1 inventory changed: {observed}")

    remote = read_json(PARENT_RESULT_ROOT / "remote_push_receipt.json")
    require(
        remote.get("commit") == PARENT_GIT_COMMIT
        and remote.get("launch_lock_sha256") == PARENT_LAUNCH_LOCK_SHA256
        and remote.get("source_bundle_sha256") == PARENT_SOURCE_BUNDLE_SHA256
        and remote.get("verified_remote_commit") == PARENT_GIT_COMMIT
        and remote.get("verified_remote_ref") is True,
        "preserved parent remote receipt is inconsistent",
    )
    all_ids: list[str] = []
    for shard_index in range(4):
        receipt = read_json(
            PARENT_RESULT_ROOT / f"audit/receipts/shard{shard_index:02d}.json"
        )
        contract = receipt.get("contract", {})
        expected_ids = [
            cell["cell_id"] for cell in cells if cell["cell_index"] % 4 == shard_index
        ]
        require(
            receipt.get("record_type") == "fused_crossed_audit_receipt"
            and contract.get("assigned_cell_ids") == expected_ids
            and contract.get("git_commit") == PARENT_GIT_COMMIT
            and contract.get("launch_lock_sha256") == PARENT_LAUNCH_LOCK_SHA256
            and contract.get("source_bundle_sha256") == PARENT_SOURCE_BUNDLE_SHA256
            and contract.get("physical_gpu") == shard_index
            and contract.get("shard_count") == 4
            and contract.get("shard_index") == shard_index
            and contract.get("tag") == "crossed_v1",
            f"preserved parent shard receipt {shard_index} changed contract",
        )
        all_ids.extend(expected_ids)
    require(len(all_ids) == 228 and len(set(all_ids)) == 228, "parent shard census is not exactly 228 cells")
    return {
        "artifact_count": 5,
        "artifact_sha256": {key: value["sha256"] for key, value in observed.items()},
        "assigned_cells": 228,
        "gate_jsonl": 0,
        "partials": 0,
        "sealed_terminal_outcomes": 0,
        "shard_receipts": 4,
        "shard_statuses": 0,
    }


def verify_incident_receipt() -> dict[str, Any]:
    receipt = read_json(INCIDENT_PATH)
    require(
        receipt.get("schema_version") == 1
        and receipt.get("record_type") == "fused_crossed_v1_failed_launch_incident_receipt"
        and receipt.get("campaign_id") == CAMPAIGN_ID
        and receipt.get("recovery_id") == RECOVERY_ID
        and receipt.get("parent_git_commit") == PARENT_GIT_COMMIT
        and receipt.get("parent_launch_lock_sha256") == PARENT_LAUNCH_LOCK_SHA256
        and receipt.get("parent_source_bundle_sha256") == PARENT_SOURCE_BUNDLE_SHA256
        and receipt.get("preserved_artifacts") == EXPECTED_PARENT_ARTIFACTS
        and receipt.get("preserved_artifact_count") == 5
        and receipt.get("process_exit_census")
        == {"exit_code_1": 4, "zero_division_error": 4}
        and receipt.get("sealed_outcome_census")
        == {
            "audit_cell_records": 0,
            "gate_jsonl": 0,
            "sealed_terminal_outcomes": 0,
            "shard_statuses": 0,
        },
        "incident receipt is stale or inconsistent",
    )
    verify_parent()
    return receipt


def load_recovery_lock() -> dict[str, Any]:
    lock = read_json(LOCK_PATH)
    require(
        lock.get("schema_version") == 1
        and lock.get("record_type") == "fused_crossed_v1r1_recovery_lock"
        and lock.get("recovery_id") == RECOVERY_ID
        and lock.get("campaign_id") == CAMPAIGN_ID
        and lock.get("result_tag") == RESULT_TAG,
        "unexpected recovery-lock identity",
    )
    return lock


def binding_for_commit(commit: str) -> dict[str, str]:
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "invalid recovery commit")
    lock = load_recovery_lock()
    return {
        "recovery_id": RECOVERY_ID,
        "parent_launch_lock_sha256": PARENT_LAUNCH_LOCK_SHA256,
        "parent_source_bundle_sha256": PARENT_SOURCE_BUNDLE_SHA256,
        "recovery_lock_sha256": file_sha256(LOCK_PATH),
        "recovery_source_bundle_sha256": lock["source_bundle_sha256"],
        "recovery_git_commit": commit,
        "result_tag": RESULT_TAG,
    }


def add_binding(value: dict[str, Any], binding: dict[str, str]) -> dict[str, Any]:
    conflicts = [key for key, expected in binding.items() if key in value and value[key] != expected]
    require(not conflicts, f"artifact has conflicting recovery binding: {conflicts}")
    return {**value, **binding}


def validate_binding(value: dict[str, Any], binding: dict[str, str], label: str) -> None:
    mismatches = [key for key in BINDING_KEYS if value.get(key) != binding.get(key)]
    require(not mismatches, f"{label} has foreign or missing recovery binding: {mismatches}")


def _expected_gate_coverage() -> set[tuple[str, int, str]]:
    adapter = read_json(REPO_ROOT / "ako_runs/controlled_followup/fused_grid/robust_adapter_manifest.json")
    return {
        (case_id, seed_index, gate_id)
        for case_id in adapter["robust_gate"]["case_ids"]
        for seed_index in range(64)
        for gate_id in ("conformance_mixed", "semantic_mixed")
    }


def validate_retained_tree(binding: dict[str, str]) -> dict[str, int]:
    """Validate every retained v1r1 JSON object and LAUNCH_FAILED evidence."""
    if not RESULT_ROOT.exists():
        return {"json": 0, "jsonl": 0, "jsonl_rows": 0}
    require(RESULT_ROOT.is_dir() and not RESULT_ROOT.is_symlink(), "unsafe v1r1 result root")
    json_count = jsonl_count = jsonl_rows = 0
    gate_paths: set[Path] = set()
    record_paths: set[Path] = set()
    for path in sorted(RESULT_ROOT.rglob("*")):
        require(not path.is_symlink(), f"symlinked retained recovery artifact: {path}")
        if not path.is_file():
            continue
        require(".partial." not in path.name, f"unreconciled partial artifact: {path}")
        if path.suffix == ".json":
            value = read_json(path)
            require(isinstance(value, dict), f"retained JSON is not an object: {path}")
            validate_binding(value, binding, str(path))
            json_count += 1
            if path.parent.name == "records":
                record_paths.add(path.resolve())
        elif path.suffix == ".jsonl":
            rows = read_jsonl(path)
            require(rows, f"retained JSONL is empty: {path}")
            for index, row in enumerate(rows):
                validate_binding(row, binding, f"{path}:{index + 1}")
            jsonl_count += 1
            jsonl_rows += len(rows)
            if path.parent.name == "gate":
                gate_paths.add(path.resolve())
        elif path.name.endswith(".lock"):
            continue
        else:
            raise RecoveryError(f"unexpected retained recovery artifact type: {path}")

    parent_core = _load_parent_modules()
    cells = {cell["cell_id"]: cell for cell in read_json(parent_core.CELLS_PATH)}
    referenced_gates: set[Path] = set()
    expected_coverage = _expected_gate_coverage()
    for record_path in sorted(record_paths):
        record = read_json(record_path)
        cell = record.get("cell", {})
        cell_id = cell.get("cell_id")
        require(cell_id in cells and cell == cells[cell_id], f"foreign retained cell record: {record_path}")
        expected_record_path = (
            RESULT_ROOT / "audit/records" / parent_core.cell_filename(cell_id)
        ).resolve()
        require(record_path == expected_record_path, f"misplaced retained cell record: {record_path}")
        outcome = record.get("terminal_outcome")
        expected_gate_path = (
            RESULT_ROOT
            / "audit/gate"
            / (parent_core.cell_filename(cell_id).removesuffix(".json") + ".jsonl")
        ).resolve()
        if outcome in {"LAUNCH_FAILED", "GATE_FAILED", "GATE_PASSED"}:
            gate_relative = record.get("gate_jsonl_path")
            require(isinstance(gate_relative, str), f"gate path missing for {cell_id}")
            gate_path = (REPO_ROOT / gate_relative).resolve()
            require(gate_path == expected_gate_path, f"foreign gate path for {cell_id}")
            require(gate_path.is_file(), f"retained gate evidence missing for {cell_id}")
            require(file_sha256(gate_path) == record.get("gate_jsonl_sha256"), f"retained gate evidence changed for {cell_id}")
            rows = read_jsonl(gate_path)
            require(len(rows) == 512, f"retained gate row count differs for {cell_id}")
            coverage = {(row.get("case_id"), row.get("seed_index"), row.get("gate_id")) for row in rows}
            require(coverage == expected_coverage, f"retained gate coverage differs for {cell_id}")
            for row in rows:
                validate_binding(row, binding, f"gate row for {cell_id}")
                require(
                    row.get("crossed_campaign_id") == CAMPAIGN_ID
                    and row.get("crossed_cell_id") == cell_id
                    and row.get("crossed_cell_sha256") == parent_core.canonical_sha256(cell)
                    and row.get("crossed_launch_lock_sha256") == PARENT_LAUNCH_LOCK_SHA256
                    and row.get("crossed_source_bundle_sha256") == PARENT_SOURCE_BUNDLE_SHA256,
                    f"retained gate row has foreign parent binding for {cell_id}",
                )
            summary = record.get("gate_summary", {})
            require(
                summary.get("complete") is True
                and summary.get("expected_records") == 512
                and summary.get("observed_records") == 512,
                f"retained gate summary is incomplete for {cell_id}",
            )
            if outcome == "LAUNCH_FAILED":
                require(any(row.get("ok") is not True for row in rows), f"LAUNCH_FAILED has no execution failure for {cell_id}")
                require(summary.get("full_gate_pass") is False, f"LAUNCH_FAILED summary passes for {cell_id}")
            if outcome == "GATE_PASSED":
                require(summary.get("full_gate_pass") is True, f"GATE_PASSED summary fails for {cell_id}")
            if outcome == "GATE_FAILED":
                require(summary.get("full_gate_pass") is False, f"GATE_FAILED summary passes for {cell_id}")
            referenced_gates.add(gate_path)
        else:
            require(not expected_gate_path.exists(), f"orphan gate evidence for {cell_id}")
    require(gate_paths == referenced_gates, "orphan or unreferenced retained gate JSONL")
    return {"json": json_count, "jsonl": jsonl_count, "jsonl_rows": jsonl_rows}
