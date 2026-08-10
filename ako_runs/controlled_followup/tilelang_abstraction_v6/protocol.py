#!/usr/bin/env python3
"""Freeze, re-hash, and project the TileLang abstraction-v6 campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAMPAIGN_PATH = HERE / "campaign.json"
MATERIALS_PATH = HERE / "materials.json"
DEFAULT_LOCK_PATH = HERE / "campaign_lock.json"
RESULTS_ROOT = HERE / "results"
CAMPAIGN_ID = "tilelang-abstraction-v6-ada"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GPU0_UUID = "GPU-45af34ad-0c74-74d0-ef3a-652090d837ae"
CUDA_ROOT = Path("/usr/local/cuda-13.1")
NCU_DISPATCHER = CUDA_ROOT / "bin" / "ncu"
NCU_WRAPPER = Path("/opt/nvidia/nsight-compute/2026.2.1/ncu")
NCU_FINAL_ELF = Path(
    "/opt/nvidia/nsight-compute/2026.2.1/target/linux-desktop-glibc_2_11_3-x64/ncu"
)


def initialize_compiler_environment() -> None:
    """Set the frozen compiler root before any code path can import TileLang."""
    os.environ["CUDA_HOME"] = str(CUDA_ROOT)
    os.environ["CUDA_PATH"] = str(CUDA_ROOT)
    cuda_bin = str(CUDA_ROOT / "bin")
    parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    os.environ["PATH"] = os.pathsep.join([cuda_bin, *[part for part in parts if part != cuda_bin]])


initialize_compiler_environment()

TOOLCHAIN_LOCAL_MODULES = {
    "analyze": "ako_runs/controlled_followup/tilelang_abstraction_v6/analyze.py",
    "artifacts": "ako_runs/controlled_followup/tilelang_abstraction_v6/artifacts.py",
    "campaign_runner": "ako_runs/controlled_followup/tilelang_abstraction_v6/campaign_runner.py",
    "protocol": "ako_runs/controlled_followup/tilelang_abstraction_v6/protocol.py",
    "phase1_common": "ako_runs/phase1_matmul/common.py",
    "phase2_common": "ako_runs/phase2_fused_sdpa/common2.py",
    "phase2_variants": "ako_runs/phase2_fused_sdpa/variants2/__init__.py",
    "fused_abstraction": "ako_runs/phase2_fused_sdpa/variants2/fused_tilelang_abstraction.py",
    "fused_gate_recovery_audit": "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/audit.py",
    "fused_gate_recovery_common": "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/common.py",
    "fused_gate_recovery_init": "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/__init__.py",
    "fused_gate_recovery_validate": "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/validate.py",
    "phase2_runner": "ako_runs/phase2_fused_sdpa/runner2.py",
    "phase2_ncu": "ako_runs/phase2_fused_sdpa/ncu_collect2.py",
}

SOURCE_PATHS = (
    "ako_runs/controlled_followup/tilelang_abstraction_v6/.gitignore",
    "ako_runs/controlled_followup/tilelang_abstraction_v6/__init__.py",
    "ako_runs/controlled_followup/tilelang_abstraction_v6/campaign.json",
    "ako_runs/controlled_followup/tilelang_abstraction_v6/materials.json",
    "ako_runs/controlled_followup/tilelang_abstraction_v6/protocol.py",
    "ako_runs/controlled_followup/tilelang_abstraction_v6/artifacts.py",
    "ako_runs/controlled_followup/tilelang_abstraction_v6/campaign_runner.py",
    "ako_runs/controlled_followup/tilelang_abstraction_v6/analyze.py",
    "ako_runs/controlled_followup/tilelang_abstraction_v6/README.md",
    "ako_runs/controlled_followup/tilelang_abstraction_v6/test_protocol.py",
)


class ProtocolError(RuntimeError):
    """Raised when a frozen campaign binding or stage contract differs."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def stable_write(path: Path, value: object) -> None:
    """Create one canonical JSON file and never overwrite retained evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite retained artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def repo_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ProtocolError(f"material path must be repository-relative: {relative!r}")
    path = (REPO / relative).resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError as exc:
        raise ProtocolError(f"material path escapes repository: {relative!r}") from exc
    return path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError as exc:
        raise ProtocolError(f"path is outside repository: {path}") from exc


def result_root(tag: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag or ""):
        raise ProtocolError(f"unsafe result tag: {tag!r}")
    path = (RESULTS_ROOT / tag).resolve()
    path.relative_to(RESULTS_ROOT.resolve())
    return path


def artifact_identity(files: list[dict[str, Any]]) -> str:
    """Canonical identity of the complete admitted TileLang cache census."""
    if not isinstance(files, list) or not files:
        raise ProtocolError("admitted cache file census is missing")
    return canonical_sha256(files)


def validate_artifact_receipt(
    receipt: dict[str, Any], *, expected_root: Path | None = None
) -> str:
    if expected_root is None:
        raise ProtocolError("admitted cache validation requires its artifact root")
    from .artifacts import validate_cache_receipt

    return validate_cache_receipt(receipt, expected_root)


def validate_campaign(campaign: dict[str, Any]) -> None:
    if campaign.get("schema_version") != 1 or campaign.get("campaign_id") != CAMPAIGN_ID:
        raise ProtocolError("unexpected campaign identity/schema")
    if campaign.get("state") != "draft_not_frozen":
        raise ProtocolError("campaign source must remain draft_not_frozen; the lock records freezing")
    hardware = campaign.get("hardware", {})
    if hardware != {
        "compute_capability": "8.9",
        "product_name": "NVIDIA RTX 6000 Ada Generation",
        "timing_gpu": 0,
    }:
        raise ProtocolError("Ada hardware contract changed")
    inference = campaign.get("inference", {})
    if inference.get("trials") != 100 or (inference.get("primary_trial_start"), inference.get("primary_trial_stop")) != (60, 100):
        raise ProtocolError("settled-tail estimator changed")
    if inference.get("confirmation_blocks", 0) < 15:
        raise ProtocolError("at least 15 independent process blocks are required")
    if inference.get("distributions") != ["positive", "withheld_signed"]:
        raise ProtocolError("distribution order/content changed")
    if not 0 <= inference.get("delta_hw_log_ratio", -1) <= inference.get("equivalence_log_ratio", -1):
        raise ProtocolError("invalid direction/equivalence thresholds")
    if campaign.get("predecessor") != {
        "campaign_id": "tilelang-abstraction-v5-ada",
        "controlling": False,
        "reason": "correctness complete but NCU runtime side effects and a nested toolchain probe prevented admission completion",
    }:
        raise ProtocolError("noncontrolling v5 predecessor binding changed")
    pairs = campaign.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 1:
        raise ProtocolError("campaign must contain exactly the fused F1/F4c pair")
    ids, families = set(), set()
    for pair in pairs:
        pair_id, family = pair.get("pair_id"), pair.get("family")
        if not isinstance(pair_id, str) or pair_id in ids:
            raise ProtocolError("pair_id values must be unique strings")
        if pair_id != "fused_softmax_f1_f4c" or family != "fused_softmax" or family in families:
            raise ProtocolError("fused F1/F4c denominator changed")
        ids.add(pair_id)
        families.add(family)
        if pair.get("high", {}).get("level") != "TL-H" or pair.get("low", {}).get("level") != "TL-M":
            raise ProtocolError(f"{pair_id}: expected TL-H/TL-M arms")
        if pair["high"].get("variant") == pair["low"].get("variant"):
            raise ProtocolError(f"{pair_id}: treatments must use distinct variants")
        match = pair.get("match", {})
        required = {"algorithm", "dtype", "instruction_family", "logical_work", "pipeline_depth", "profile_equal_fields", "threads", "tile"}
        if set(match) != required or not match["profile_equal_fields"]:
            raise ProtocolError(f"{pair_id}: incomplete matching contract")
        gate = pair.get("gate", {})
        if gate.get("available") is True:
            if not gate.get("gate_ids") or not gate.get("manifest_material_id") or not gate.get("spec_material_id"):
                raise ProtocolError(f"{pair_id}: current gate binding is incomplete")
        elif gate.get("available") is False:
            if gate.get("gate_ids") or gate.get("manifest_material_id") is not None or gate.get("spec_material_id") is not None or not gate.get("reason"):
                raise ProtocolError(f"{pair_id}: unavailable gate must fail closed explicitly")
        else:
            raise ProtocolError(f"{pair_id}: gate availability is not boolean")
    if families != {"fused_softmax"}:
        raise ProtocolError("family denominator changed")
    scope = campaign.get("scope_policy", {})
    if scope.get("current_claim_scope") != "local_pair_only" or scope.get("legacy_results_controlling") is not False:
        raise ProtocolError("claim scope or legacy-result policy changed")


def validate_materials(materials: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Path]:
    if materials.get("schema_version") != 1 or materials.get("campaign_id") != CAMPAIGN_ID:
        raise ProtocolError("unexpected material-registry identity/schema")
    entries = materials.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ProtocolError("material registry has no entries")
    resolved: dict[str, Path] = {}
    for material_id, binding in entries.items():
        if not isinstance(material_id, str) or not isinstance(binding, dict):
            raise ProtocolError("malformed material entry")
        path, expected = repo_path(binding.get("path")), binding.get("sha256")
        if not path.is_file() or not HEX64.fullmatch(str(expected)):
            raise ProtocolError(f"{material_id}: missing material or invalid digest")
        actual = file_sha256(path)
        if actual != expected:
            raise ProtocolError(f"{material_id}: material changed: {actual} != {expected}")
        if not binding.get("role"):
            raise ProtocolError(f"{material_id}: role is required")
        resolved[material_id] = path
    family_bindings = materials.get("family_bindings")
    if not isinstance(family_bindings, dict) or set(family_bindings) != {"fused_softmax"}:
        raise ProtocolError("material family bindings differ")
    for pair in campaign["pairs"]:
        family = pair["family"]
        ids = family_bindings[family]
        if not isinstance(ids, list) or not ids or any(material_id not in resolved for material_id in ids):
            raise ProtocolError(f"{family}: material closure is incomplete")
        gate = pair["gate"]
        if gate["available"]:
            for field in ("manifest_material_id", "spec_material_id"):
                if gate[field] not in resolved or gate[field] not in ids:
                    raise ProtocolError(f"{family}: gate material is outside its closure")
            manifest = read_json(resolved[gate["manifest_material_id"]])
            spec = read_json(resolved[gate["spec_material_id"]])
            if spec.get("manifest_sha256") != canonical_sha256(manifest):
                raise ProtocolError(f"{family}: current gate does not bind its manifest")
            expected_keys = {f"fused_softmax/{gate_id}" for gate_id in gate["gate_ids"]}
            if not expected_keys <= set(spec.get("gates", {})):
                raise ProtocolError(f"{family}: gate IDs are absent from current spec")
            if manifest.get("operations", {}).get("fused_softmax", {}).get("shape") != pair.get("shape"):
                raise ProtocolError(f"{family}: pair shape differs from the current gate manifest")
            for gate_key in expected_keys:
                contract = spec["gates"][gate_key].get("contract", {})
                if any(contract.get(key) != value for key, value in pair.get("input_contract", {}).items()):
                    raise ProtocolError(f"{family}: pair input contract differs from {gate_key}")
    predecessor_campaign = read_json(resolved["v5_campaign_source"])
    predecessor_pair = next(
        (row for row in predecessor_campaign.get("pairs", []) if row.get("pair_id") == "fused_softmax_f1_f4c"),
        None,
    )
    if campaign["pairs"] != [predecessor_pair]:
        raise ProtocolError("F1/F4c design differs from the v5 predecessor")
    for field in ("hardware", "inference", "scope_policy"):
        if campaign[field] != predecessor_campaign.get(field):
            raise ProtocolError(f"{field} differs from the v5 predecessor")

    incident = read_json(resolved["v4_artifact_identity_incident"])
    closure = incident.get("artifact_closure", {})
    closure_root = repo_path(str(closure.get("root", "")))
    rows = []
    if closure_root.is_dir():
        for path in sorted(item for item in closure_root.rglob("*") if item.is_file()):
            rows.append({
                "path": path.relative_to(closure_root).as_posix(),
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
            })
    if (
        incident.get("campaign_id") != "tilelang-abstraction-v4-ada"
        or incident.get("classification") != "NON_CONTROLLING_RAW_ARTIFACT_IDENTITY_INCIDENT"
        or incident.get("result_state") != "correctness_complete_profile_identity_failed_noncontrolling"
        or incident.get("sealed_census") != {
            "correctness_gate_expected_records": 1024,
            "correctness_gate_failed_records": 0,
            "correctness_gate_observed_records": 1024,
            "expected_timing_records_if_admitted": 120,
            "projected_timing_manifest_rows": 0,
            "retained_timing_records": 0,
        }
        or incident.get("policy") != {
            "continuation_authorized": False,
            "performance_timing_authorized": False,
            "post_hoc_overlay_authorized": False,
            "reuse_authorized": False,
            "successor_requires_deterministic_preimport_toolchain_initialization": True,
            "successor_requires_exact_admitted_executable_loading": True,
            "successor_requires_fresh_artifacts": True,
            "successor_requires_new_lock": True,
        }
        or closure.get("file_count") != len(rows)
        or closure.get("total_bytes") != sum(row["size"] for row in rows)
        or closure.get("sha256") != canonical_sha256(rows)
    ):
        raise ProtocolError("v4 artifact-identity incident or result closure is inconsistent")
    parent = incident.get("parent_bindings", {})
    expected_parent = {
        "campaign_lock_sha256": file_sha256(resolved["v4_campaign_lock"]),
        "campaign_sha256": file_sha256(resolved["v4_campaign_source"]),
        "run_status_sha256": file_sha256(resolved["v4_admission_run_status"]),
        "summary_sha256": file_sha256(resolved["v4_admission_summary"]),
        "timing_manifest_sha256": file_sha256(resolved["v4_timing_manifest"]),
    }
    if any(parent.get(key) != value for key, value in expected_parent.items()):
        raise ProtocolError("v4 incident parent bindings changed")
    result_commit = str(parent.get("result_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40,64}", result_commit):
        raise ProtocolError("v4 incident result commit is malformed")
    try:
        _git("cat-file", "-e", f"{result_commit}^{{commit}}")
        _git("merge-base", "--is-ancestor", result_commit, "HEAD")
        committed = set(_git(
            "ls-tree", "-r", "--name-only", result_commit, "--", closure["root"]
        ).splitlines())
        expected_committed = {
            f"{closure['root']}/{row['path']}" for row in rows
        }
        if committed != expected_committed:
            raise ProtocolError("v4 result commit file census differs")
        for relative in committed:
            if _git("rev-parse", f"{result_commit}:{relative}") != _git("hash-object", "--", relative):
                raise ProtocolError(f"v4 result commit blob differs: {relative}")
    except ProtocolError as exc:
        raise ProtocolError(f"v4 incident result commit is not retained: {exc}") from exc
    v4_summary = read_json(resolved["v4_admission_summary"])
    v4_manifest = read_json(resolved["v4_timing_manifest"])
    if (
        v4_summary.get("complete") is not True
        or v4_summary.get("timing_eligible_pair_ids") != []
        or v4_manifest.get("rows") != []
        or v4_manifest.get("eligible_pair_ids") != []
    ):
        raise ProtocolError("v4 predecessor was not retained as noncontrolling")

    v5_incident = read_json(resolved["v5_admission_runtime_incident"])
    v5_closure = v5_incident.get("artifact_closure", {})
    v5_root = repo_path(str(v5_closure.get("root", "")))
    v5_rows = []
    if v5_root.is_dir():
        for path in sorted(item for item in v5_root.rglob("*") if item.is_file()):
            v5_rows.append({
                "path": path.relative_to(v5_root).as_posix(),
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
            })
    if (
        v5_incident.get("campaign_id") != "tilelang-abstraction-v5-ada"
        or v5_incident.get("classification") != "NON_CONTROLLING_NCU_RUNTIME_ADMISSION_INCIDENT"
        or v5_incident.get("result_state") != "correctness_complete_profile_runtime_failed_noncontrolling"
        or v5_incident.get("sealed_census") != {
            "admission_run_status_retained": False,
            "admission_summary_retained": False,
            "correctness_gate_expected_records": 1024,
            "correctness_gate_failed_records": 0,
            "correctness_gate_observed_records": 1024,
            "expected_timing_records_if_admitted": 120,
            "profile_attempted_records": 2,
            "profile_successful_records": 1,
            "retained_timing_records": 0,
        }
        or v5_incident.get("policy") != {
            "continuation_authorized": False,
            "performance_timing_authorized": False,
            "post_hoc_overlay_authorized": False,
            "reuse_authorized": False,
            "successor_requires_external_ncu_temporary_directory": True,
            "successor_requires_fresh_artifacts": True,
            "successor_requires_new_lock": True,
            "successor_requires_parent_only_live_toolchain_preflight": True,
        }
        or v5_incident.get("profile_census") != {
            "high": {
                "artifact_identity_match": True,
                "ncu_returncode": 0,
                "profile_ok": True,
                "unexpected_artifact_files": [
                    "runtime_tmp/nvidia/nsight_compute/lock",
                    "runtime_tmp/nvidia/nsight_compute/lock.45af34ad-0c74-74d0-ef3a-652090d837ae",
                ],
            },
            "low": {
                "artifact_identity_match": False,
                "load_receipt_written": False,
                "ncu_returncode": 1,
                "profile_ok": False,
                "terminal_error": "nested ncu --version timed out after 30 seconds",
            },
        }
        or v5_closure.get("file_count") != len(v5_rows)
        or v5_closure.get("total_bytes") != sum(row["size"] for row in v5_rows)
        or v5_closure.get("sha256") != canonical_sha256(v5_rows)
    ):
        raise ProtocolError("v5 NCU-runtime incident or result closure is inconsistent")

    admission_root = v5_root / "admission"
    v5_lock = read_json(resolved["v5_campaign_lock"])
    v5_materials = read_json(resolved["v5_materials"])
    v5_parent = v5_incident.get("parent_bindings", {})
    expected_v5_parent = {
        "campaign_lock_sha256": file_sha256(resolved["v5_campaign_lock"]),
        "campaign_sha256": file_sha256(resolved["v5_campaign_source"]),
        "high_admission_sha256": file_sha256(admission_root / "fused_softmax_f1_f4c__high.json"),
        "high_profile_sha256": file_sha256(admission_root / "fused_softmax_f1_f4c__high.profile.json"),
        "launch_receipt_sha256": file_sha256(admission_root / "launch_receipt.json"),
        "low_admission_sha256": file_sha256(admission_root / "fused_softmax_f1_f4c__low.json"),
        "low_profile_sha256": file_sha256(admission_root / "fused_softmax_f1_f4c__low.profile.json"),
        "materials_sha256": file_sha256(resolved["v5_materials"]),
        "result_commit": v5_parent.get("result_commit"),
        "source_commit": v5_lock.get("git_commit"),
    }
    if v5_parent != expected_v5_parent:
        raise ProtocolError("v5 incident parent bindings changed")
    source_commit, result_commit = v5_parent["source_commit"], v5_parent["result_commit"]
    launch = read_json(admission_root / "launch_receipt.json")
    lock_commit = launch.get("remote_authorization", {}).get("head_commit")
    if not all(re.fullmatch(r"[0-9a-f]{40,64}", str(value)) for value in (source_commit, lock_commit, result_commit)):
        raise ProtocolError("v5 incident commit binding is malformed")

    dependency_hashes = {
        binding["path"]: binding["sha256"] for binding in v5_materials.get("entries", {}).values()
    }
    source_hashes = v5_lock.get("source_sha256", {})
    if (
        v5_lock.get("campaign_id") != "tilelang-abstraction-v5-ada"
        or v5_materials.get("campaign_id") != "tilelang-abstraction-v5-ada"
        or v5_lock.get("campaign_sha256") != file_sha256(resolved["v5_campaign_source"])
        or v5_lock.get("materials_sha256") != file_sha256(resolved["v5_materials"])
        or v5_lock.get("dependency_sha256") != dependency_hashes
        or v5_lock.get("dependency_bundle_sha256") != canonical_sha256(dependency_hashes)
        or not isinstance(source_hashes, dict)
        or v5_lock.get("source_bundle_sha256") != canonical_sha256(source_hashes)
        or v5_lock.get("git_commit") != v5_lock.get("upstream_commit")
    ):
        raise ProtocolError("v5 frozen source/dependency lock is inconsistent")
    try:
        _git("cat-file", "-e", f"{source_commit}^{{commit}}")
        _git("cat-file", "-e", f"{lock_commit}^{{commit}}")
        _git("cat-file", "-e", f"{result_commit}^{{commit}}")
        _git("merge-base", "--is-ancestor", source_commit, lock_commit)
        _git("merge-base", "--is-ancestor", lock_commit, result_commit)
        _git("merge-base", "--is-ancestor", result_commit, "HEAD")
        lock_relative = repo_relative(resolved["v5_campaign_lock"])
        lock_blob = _git("rev-parse", f"{lock_commit}:{lock_relative}")
        if lock_blob != _git("hash-object", "--", lock_relative):
            raise ProtocolError("v5 lock-commit blob differs")
        for relative, expected in source_hashes.items():
            path = repo_path(relative)
            if file_sha256(path) != expected:
                raise ProtocolError(f"v5 frozen source changed: {relative}")
            if _git("rev-parse", f"{source_commit}:{relative}") != _git("hash-object", "--", relative):
                raise ProtocolError(f"v5 source-commit blob differs: {relative}")
        committed = set(_git(
            "ls-tree", "-r", "--name-only", result_commit, "--", v5_closure["root"]
        ).splitlines())
        expected_committed = {
            f"{v5_closure['root']}/{row['path']}" for row in v5_rows
        }
        if committed != expected_committed:
            raise ProtocolError("v5 result commit file census differs")
        for relative in committed:
            if _git("rev-parse", f"{result_commit}:{relative}") != _git("hash-object", "--", relative):
                raise ProtocolError(f"v5 result-commit blob differs: {relative}")
    except ProtocolError as exc:
        raise ProtocolError(f"v5 committed predecessor closure is not retained: {exc}") from exc

    high = read_json(admission_root / "fused_softmax_f1_f4c__high.json")
    low = read_json(admission_root / "fused_softmax_f1_f4c__low.json")
    high_profile = read_json(admission_root / "fused_softmax_f1_f4c__high.profile.json")
    low_profile = read_json(admission_root / "fused_softmax_f1_f4c__low.profile.json")
    forbidden = {
        "admission/run_status.json", "admission/summary.json", "timing_manifest.json"
    }
    retained_paths = {row["path"] for row in v5_rows}
    remote = launch.get("remote_authorization", {})
    extra_locks = {
        "admission/fused_softmax_f1_f4c__high_artifact/runtime_tmp/nvidia/nsight_compute/lock",
        "admission/fused_softmax_f1_f4c__high_artifact/runtime_tmp/nvidia/nsight_compute/lock.45af34ad-0c74-74d0-ef3a-652090d837ae",
    }
    size_by_path = {row["path"]: row["size"] for row in v5_rows}
    if (
        any(
            receipt.get("terminal_outcome") != "GATE_PASSED"
            or receipt.get("gate_summary", {}).get("observed_records") != 512
            or receipt.get("gate_summary", {}).get("failed_records") != 0
            for receipt in (high, low)
        )
        or high_profile.get("ok") is not True
        or high_profile.get("artifact_identity_match") is not True
        or low_profile.get("ok") is not False
        or low_profile.get("returncode") != 1
        or low_profile.get("artifact_identity_match") is not False
        or "ncu'" not in low_profile.get("stderr_tail", "")
        or "timed out after 30 seconds" not in low_profile.get("stderr_tail", "")
        or v5_incident.get("profile_census", {}).get("high", {}).get("profile_ok") is not True
        or v5_incident.get("profile_census", {}).get("low", {}).get("profile_ok") is not False
        or forbidden & retained_paths
        or not extra_locks <= retained_paths
        or any(size_by_path[path] != 0 for path in extra_locks)
        or "admission/fused_softmax_f1_f4c__low.profile.load.json" in retained_paths
        or any(path.startswith("timing/") for path in retained_paths)
        or launch.get("campaign_lock_sha256") != file_sha256(resolved["v5_campaign_lock"])
        or remote.get("lock_sha256") != file_sha256(resolved["v5_campaign_lock"])
        or len({
            remote.get("head_commit"),
            remote.get("upstream_commit"),
            remote.get("live_upstream_commit"),
        }) != 1
        or remote.get("head_equals_configured_upstream") is not True
        or not isinstance(remote.get("upstream_remote"), str)
        or not remote.get("upstream_remote")
        or not isinstance(remote.get("upstream_ref"), str)
        or not remote.get("upstream_ref", "").startswith("refs/")
        or remote.get("lock_path") != lock_relative
        or remote.get("lock_git_blob") != lock_blob
    ):
        raise ProtocolError("v5 failed admission census or remote authorization is inconsistent")

    status = read_json(resolved["v3_admission_run_status"])
    artifacts = status.get("artifact_sha256")
    if (
        status.get("campaign_id") != "tilelang-abstraction-v3-ada"
        or status.get("complete") is not True
        or not isinstance(artifacts, dict)
        or status.get("artifact_bundle_sha256") != canonical_sha256(artifacts)
    ):
        raise ProtocolError("v3 predecessor admission status is inconsistent")
    for relative, expected in artifacts.items():
        path = repo_path(relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtocolError(f"v3 predecessor admission artifact changed: {relative}")
    summary = read_json(resolved["v3_admission_summary"])
    predecessor_result = next(
        (row for row in summary.get("pairs", []) if row.get("pair_id") == "fused_softmax_f1_f4c"),
        None,
    )
    if (
        summary.get("campaign_id") != "tilelang-abstraction-v3-ada"
        or summary.get("campaign_lock_sha256") != file_sha256(resolved["v3_campaign_lock"])
        or summary.get("run_status_sha256") != file_sha256(resolved["v3_admission_run_status"])
        or summary.get("complete") is not True
        or summary.get("timing_eligible_pair_ids") != []
        or not isinstance(predecessor_result, dict)
        or predecessor_result.get("timing_eligible") is not False
        or predecessor_result.get("classification") != "excluded_fail_closed"
    ):
        raise ProtocolError("v3 predecessor summary is inconsistent")
    return resolved


def load_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    campaign, materials = read_json(CAMPAIGN_PATH), read_json(MATERIALS_PATH)
    validate_campaign(campaign)
    validate_materials(materials, campaign)
    return campaign, materials


def gpu_snapshot(index: int) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi", "-i", str(index),
            "--query-gpu=index,uuid,name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False, capture_output=True, text=True, timeout=15,
    )
    if query.returncode != 0:
        raise ProtocolError(f"nvidia-smi identity query failed: {query.stderr.strip()}")
    rows = [part.strip() for part in query.stdout.strip().split(",")]
    if len(rows) != 5:
        raise ProtocolError(f"unexpected nvidia-smi identity row: {query.stdout!r}")
    nvcc = subprocess.run(["/usr/local/cuda-13.1/bin/nvcc", "--version"], check=False, capture_output=True, text=True, timeout=15)
    ncu = subprocess.run(["/usr/local/cuda-13.1/bin/ncu", "--version"], check=False, capture_output=True, text=True, timeout=15)
    if nvcc.returncode or ncu.returncode:
        raise ProtocolError("frozen CUDA 13.1 nvcc/ncu tools are unavailable")
    return {
        "physical_index": int(rows[0]),
        "uuid": rows[1],
        "product_name": rows[2],
        "compute_capability": rows[3],
        "driver_version": rows[4],
        "nvcc_sha256": hashlib.sha256(nvcc.stdout.encode()).hexdigest(),
        "ncu_sha256": hashlib.sha256(ncu.stdout.encode()).hexdigest(),
    }


def ncu_executable_chain() -> dict[str, Any]:
    """Resolve the two NCU dispatch scripts and final ELF by their live shell traces."""
    dispatcher = NCU_DISPATCHER

    def traced(script: Path) -> subprocess.CompletedProcess[str]:
        if not script.is_file() or not os.access(script, os.X_OK):
            raise ProtocolError(f"NCU chain member is missing or not executable: {script}")
        run = subprocess.run(
            ["/bin/sh", "-x", str(script), "--version"], check=False,
            capture_output=True, text=True, timeout=30,
        )
        if run.returncode or not run.stdout:
            raise ProtocolError(f"NCU chain trace failed: {script}: {run.stderr[-1000:]}")
        return run

    first = traced(dispatcher)
    wrapper_candidates = []
    for line in first.stderr.splitlines():
        try:
            words = shlex.split(line.lstrip("+").strip())
        except ValueError:
            continue
        if len(words) >= 3 and words[0] == "exec" and words[-1] == "--version":
            wrapper_candidates.append(Path(words[1]).resolve())
    if len(wrapper_candidates) != 1:
        raise ProtocolError(f"NCU dispatcher resolved {len(wrapper_candidates)} wrappers")
    wrapper = wrapper_candidates[0]
    if wrapper != NCU_WRAPPER:
        raise ProtocolError(f"NCU dispatcher selection changed: {wrapper}")

    second = traced(wrapper)
    target_candidates = []
    for line in second.stderr.splitlines():
        try:
            words = shlex.split(line.lstrip("+").strip())
        except ValueError:
            continue
        if len(words) >= 2 and Path(words[0]).is_absolute() and words[-1] == "--version":
            target_candidates.append(Path(words[0]).resolve())
    if len(target_candidates) != 1 or first.stdout != second.stdout:
        raise ProtocolError("NCU wrapper did not resolve one version-identical target")
    target = target_candidates[0]
    if target != NCU_FINAL_ELF:
        raise ProtocolError(f"NCU final executable selection changed: {target}")

    rows = []
    for role, path, expected_magic in (
        ("cuda_dispatcher", dispatcher.resolve(), b"#!"),
        ("version_wrapper", wrapper, b"#!"),
        ("final_elf", target, b"\x7fELF"),
    ):
        with path.open("rb") as handle:
            magic = handle.read(len(expected_magic))
        if magic != expected_magic or not os.access(path, os.X_OK):
            raise ProtocolError(f"NCU {role} has an unexpected executable format: {path}")
        rows.append({
            "role": role,
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "format": "elf" if role == "final_elf" else "posix_shell",
        })
    if len({row["path"] for row in rows}) != 3:
        raise ProtocolError("NCU executable chain contains duplicate paths")
    return {
        "entries": rows,
        "entries_sha256": canonical_sha256(rows),
        "version_sha256": hashlib.sha256(first.stdout.encode()).hexdigest(),
    }


def live_toolchain() -> dict[str, Any]:
    initialize_compiler_environment()
    import tilelang
    import torch
    import triton

    executable = Path(sys.executable).resolve()
    if not executable.is_file():
        raise ProtocolError("Python executable is missing")
    package_modules = {}
    modules = (
        ("tilelang", tilelang),
        ("tilelang_env", importlib.import_module("tilelang.env")),
        ("tilelang_cache", importlib.import_module("tilelang.cache")),
        ("tilelang_kernel_cache", importlib.import_module("tilelang.cache.kernel_cache")),
        ("tilelang_jit", importlib.import_module("tilelang.jit")),
        ("tilelang_execution_backend", importlib.import_module("tilelang.jit.execution_backend")),
        ("tilelang_jit_kernel", importlib.import_module("tilelang.jit.kernel")),
        ("tilelang_adapter_base", importlib.import_module("tilelang.jit.adapter.base")),
        ("tilelang_adapter_kernel_cache", importlib.import_module("tilelang.jit.adapter.kernel_cache")),
        ("tilelang_tvm_ffi_adapter", importlib.import_module("tilelang.jit.adapter.tvm_ffi")),
        ("torch", torch),
        ("triton", triton),
    )
    for name, module in modules:
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not Path(raw).is_file():
            raise ProtocolError(f"toolchain package has no module file: {name}")
        path = Path(raw).resolve()
        package_modules[name] = {"path": str(path), "sha256": file_sha256(path)}
    local_modules = {
        name: {"path": relative, "sha256": file_sha256(repo_path(relative))}
        for name, relative in TOOLCHAIN_LOCAL_MODULES.items()
    }
    nvcc = CUDA_ROOT / "bin" / "nvcc"
    nvcc_run = subprocess.run(
        [str(nvcc), "--version"], check=False, capture_output=True, text=True, timeout=15,
    )
    if nvcc_run.returncode or not nvcc.is_file():
        raise ProtocolError("frozen CUDA 13.1 nvcc is unavailable")
    ncu_chain = ncu_executable_chain()
    cuda_tools = {
        "nvcc": {
            "path": str(nvcc),
            "binary_sha256": file_sha256(nvcc),
            "version_sha256": hashlib.sha256(nvcc_run.stdout.encode()).hexdigest(),
        },
        "ncu": {
            "path": str(NCU_DISPATCHER),
            "binary_sha256": file_sha256(NCU_DISPATCHER),
            "version_sha256": ncu_chain["version_sha256"],
            "executable_chain": ncu_chain["entries"],
            "executable_chain_sha256": ncu_chain["entries_sha256"],
        },
    }
    return {
        "cuda_tools": cuda_tools,
        "local_modules": local_modules,
        "package_modules": package_modules,
        "python_executable": str(executable),
        "python_executable_sha256": file_sha256(executable),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "tilelang_version": str(tilelang.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
    }


def validate_toolchain(value: Any) -> None:
    required = {
        "cuda_tools", "local_modules", "package_modules", "python_executable",
        "python_executable_sha256", "python_implementation", "python_version",
        "tilelang_version", "torch_cuda_version", "torch_version", "triton_version",
    }
    strings = required - {"cuda_tools", "local_modules", "package_modules"}
    if (
        not isinstance(value, dict)
        or set(value) != required
        or any(not isinstance(value.get(key), str) or not value[key] for key in strings)
        or set(value.get("local_modules", {})) != set(TOOLCHAIN_LOCAL_MODULES)
        or set(value.get("package_modules", {})) != {
            "tilelang", "tilelang_env", "tilelang_cache", "tilelang_kernel_cache",
            "tilelang_jit", "tilelang_execution_backend", "tilelang_jit_kernel",
            "tilelang_adapter_base", "tilelang_adapter_kernel_cache",
            "tilelang_tvm_ffi_adapter", "torch", "triton",
        }
        or set(value.get("cuda_tools", {})) != {"nvcc", "ncu"}
    ):
        raise ProtocolError("toolchain receipt is malformed")
    for group in ("local_modules", "package_modules"):
        for name, binding in value[group].items():
            if (
                not isinstance(binding, dict)
                or set(binding) != {"path", "sha256"}
                or not isinstance(binding["path"], str)
                or not binding["path"]
                or not HEX64.fullmatch(str(binding["sha256"]))
            ):
                raise ProtocolError("toolchain module binding is malformed")
            if group == "local_modules" and binding["path"] != TOOLCHAIN_LOCAL_MODULES[name]:
                raise ProtocolError("toolchain local-module path is malformed")
            if group == "package_modules" and not Path(binding["path"]).is_absolute():
                raise ProtocolError("toolchain package-module path is malformed")
    nvcc = value["cuda_tools"]["nvcc"]
    if (
        not isinstance(nvcc, dict)
        or set(nvcc) != {"path", "binary_sha256", "version_sha256"}
        or nvcc["path"] != str(CUDA_ROOT / "bin" / "nvcc")
        or not HEX64.fullmatch(str(nvcc["binary_sha256"]))
        or not HEX64.fullmatch(str(nvcc["version_sha256"]))
    ):
        raise ProtocolError("nvcc tool binding is malformed")
    ncu = value["cuda_tools"]["ncu"]
    entries = ncu.get("executable_chain") if isinstance(ncu, dict) else None
    expected_paths = [str(NCU_DISPATCHER), str(NCU_WRAPPER), str(NCU_FINAL_ELF)]
    if (
        not isinstance(ncu, dict)
        or set(ncu) != {
            "path", "binary_sha256", "version_sha256", "executable_chain",
            "executable_chain_sha256",
        }
        or ncu.get("path") != str(NCU_DISPATCHER)
        or not HEX64.fullmatch(str(ncu.get("binary_sha256")))
        or not HEX64.fullmatch(str(ncu.get("version_sha256")))
        or not isinstance(entries, list)
        or len(entries) != 3
        or [entry.get("path") for entry in entries if isinstance(entry, dict)] != expected_paths
        or [entry.get("role") for entry in entries if isinstance(entry, dict)] != [
            "cuda_dispatcher", "version_wrapper", "final_elf"
        ]
        or [entry.get("format") for entry in entries if isinstance(entry, dict)] != [
            "posix_shell", "posix_shell", "elf"
        ]
        or any(
            set(entry) != {"role", "path", "sha256", "bytes", "format"}
            or not HEX64.fullmatch(str(entry.get("sha256")))
            or not isinstance(entry.get("bytes"), int)
            or entry["bytes"] <= 0
            for entry in entries if isinstance(entry, dict)
        )
        or any(not isinstance(entry, dict) for entry in entries)
        or ncu.get("binary_sha256") != entries[0].get("sha256")
        or ncu.get("executable_chain_sha256") != canonical_sha256(entries)
    ):
        raise ProtocolError("NCU executable-chain binding is malformed")
    if (
        not Path(value["python_executable"]).is_absolute()
        or not HEX64.fullmatch(value["python_executable_sha256"])
    ):
        raise ProtocolError("Python executable hash is malformed")


def validate_gpu(snapshot: dict[str, Any], campaign: dict[str, Any]) -> None:
    expected = campaign["hardware"]
    if snapshot.get("physical_index") != expected["timing_gpu"]:
        raise ProtocolError("physical GPU index differs from campaign")
    if snapshot.get("product_name") != expected["product_name"] or snapshot.get("compute_capability") != expected["compute_capability"]:
        raise ProtocolError("GPU product or compute capability differs from frozen Ada contract")
    if snapshot.get("uuid") != GPU0_UUID:
        raise ProtocolError("physical GPU 0 UUID differs from the campaign host binding")


def _git(*args: str) -> str:
    run = subprocess.run(["git", *args], cwd=REPO, check=False, capture_output=True, text=True, timeout=30)
    if run.returncode:
        raise ProtocolError(f"git {' '.join(args)} failed: {run.stderr.strip()}")
    return run.stdout.strip()


def live_upstream_head() -> dict[str, str]:
    """Resolve the configured branch ref at the remote, preserving slash refs."""
    branch = _git("branch", "--show-current")
    if not branch:
        raise ProtocolError("live upstream verification requires an attached branch")
    remote = _git("config", "--get", f"branch.{branch}.remote")
    merge_ref = _git("config", "--get", f"branch.{branch}.merge")
    if not remote or not merge_ref.startswith("refs/"):
        raise ProtocolError("configured upstream remote/ref is missing")
    rows = _git("ls-remote", "--exit-code", remote, merge_ref).splitlines()
    fields = rows[0].split() if len(rows) == 1 else []
    if len(fields) != 2 or fields[1] != merge_ref or not re.fullmatch(r"[0-9a-f]{40,64}", fields[0]):
        raise ProtocolError("live upstream branch is missing, ambiguous, or malformed")
    return {"remote": remote, "ref": merge_ref, "commit": fields[0]}


def _frozen_paths(materials: dict[str, Any]) -> list[str]:
    return sorted(set(SOURCE_PATHS) | {
        binding["path"] for binding in materials["entries"].values()
    })


def _require_scoped_clean(paths: list[str]) -> None:
    if _git("status", "--porcelain=v1", "--untracked-files=all", "--", *paths):
        raise ProtocolError("commit the frozen source/material closure before continuing")


def build_lock(
    campaign: dict[str, Any],
    materials: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    git_commit: str,
    upstream_commit: str,
    toolchain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_campaign(campaign)
    validate_materials(materials, campaign)
    validate_gpu(snapshot, campaign)
    if git_commit != upstream_commit or not re.fullmatch(r"[0-9a-f]{40,64}", git_commit):
        raise ProtocolError("freezing requires HEAD to equal its configured upstream")
    toolchain = live_toolchain() if toolchain is None else toolchain
    validate_toolchain(toolchain)
    source_hashes = {relative: file_sha256(repo_path(relative)) for relative in SOURCE_PATHS}
    dependency_hashes = {binding["path"]: binding["sha256"] for binding in materials["entries"].values()}
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "state": "frozen_pending_remote_registration",
        "campaign_sha256": file_sha256(CAMPAIGN_PATH),
        "campaign_canonical_sha256": canonical_sha256(campaign),
        "materials_sha256": file_sha256(MATERIALS_PATH),
        "materials_canonical_sha256": canonical_sha256(materials),
        "source_sha256": source_hashes,
        "source_bundle_sha256": canonical_sha256(source_hashes),
        "dependency_sha256": dependency_hashes,
        "dependency_bundle_sha256": canonical_sha256(dependency_hashes),
        "gpu": snapshot,
        "toolchain": toolchain,
        "git_commit": git_commit,
        "upstream_commit": upstream_commit,
        "remote_preregistration": {
            "source_commit": upstream_commit,
            "source_head_equals_configured_upstream_at_freeze": True,
            "lock_must_be_committed_and_pushed_before_launch": True,
        },
        "authorized_stages": ["admission", "timing"],
    }


def freeze(output: Path, gpu: int) -> dict[str, Any]:
    campaign, materials = load_sources()
    if output.exists():
        raise FileExistsError(f"refusing existing lock: {output}")
    _require_scoped_clean(_frozen_paths(materials))
    head = _git("rev-parse", "HEAD")
    cached = _git("rev-parse", "@{upstream}")
    live = live_upstream_head()
    if head != cached or head != live["commit"]:
        raise ProtocolError("freezing requires HEAD to equal the cached and live configured upstream")
    lock = build_lock(
        campaign,
        materials,
        gpu_snapshot(gpu),
        git_commit=head,
        upstream_commit=live["commit"],
    )
    stable_write(output, lock)
    return lock


def remote_lock_receipt(path: Path = DEFAULT_LOCK_PATH) -> dict[str, Any]:
    """Verify the frozen lock itself is in the configured upstream commit."""
    if not path.is_file():
        raise ProtocolError("campaign lock is missing")
    try:
        relative = str(path.resolve().relative_to(REPO.resolve()))
    except ValueError as exc:
        raise ProtocolError("campaign lock must be retained inside the repository") from exc
    campaign, materials = load_sources()
    del campaign
    _require_scoped_clean(_frozen_paths(materials) + [relative])
    head, upstream = _git("rev-parse", "HEAD"), _git("rev-parse", "@{upstream}")
    live = live_upstream_head()
    if head != upstream or head != live["commit"]:
        raise ProtocolError("launch requires HEAD to equal the cached and live configured upstream")
    tracked = _git("ls-files", "--error-unmatch", "--", relative)
    if tracked != relative:
        raise ProtocolError("campaign lock is not tracked")
    worktree_blob = _git("hash-object", "--", relative)
    head_blob = _git("rev-parse", f"HEAD:{relative}")
    if worktree_blob != head_blob:
        raise ProtocolError("campaign lock differs from the pushed commit")
    lock = read_json(path)
    source_commit = lock.get("git_commit")
    if _git("merge-base", str(source_commit), head) != source_commit:
        raise ProtocolError("frozen source commit is not an ancestor of the lock commit")
    return {
        "head_commit": head,
        "upstream_commit": upstream,
        "lock_path": relative,
        "lock_sha256": file_sha256(path),
        "lock_git_blob": head_blob,
        "head_equals_configured_upstream": True,
        "live_upstream_commit": live["commit"],
        "upstream_ref": live["ref"],
        "upstream_remote": live["remote"],
    }


def load_lock(
    path: Path = DEFAULT_LOCK_PATH,
    *,
    check_gpu: bool = False,
    check_remote: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    campaign, materials = load_sources()
    if not path.is_file():
        raise ProtocolError("campaign lock is missing; no GPU stage is authorized")
    lock = read_json(path)
    if lock.get("campaign_id") != CAMPAIGN_ID or lock.get("state") != "frozen_pending_remote_registration":
        raise ProtocolError("unexpected campaign-lock identity/state")
    if lock.get("git_commit") != lock.get("upstream_commit") or lock.get("remote_preregistration") != {
        "source_commit": lock.get("upstream_commit"),
        "source_head_equals_configured_upstream_at_freeze": True,
        "lock_must_be_committed_and_pushed_before_launch": True,
    }:
        raise ProtocolError("campaign lock lacks the two-step remote-registration policy")
    if lock.get("campaign_sha256") != file_sha256(CAMPAIGN_PATH) or lock.get("campaign_canonical_sha256") != canonical_sha256(campaign):
        raise ProtocolError("campaign changed after freeze")
    if lock.get("materials_sha256") != file_sha256(MATERIALS_PATH) or lock.get("materials_canonical_sha256") != canonical_sha256(materials):
        raise ProtocolError("material registry changed after freeze")
    current_sources = {relative: file_sha256(repo_path(relative)) for relative in SOURCE_PATHS}
    if lock.get("source_sha256") != current_sources or lock.get("source_bundle_sha256") != canonical_sha256(current_sources):
        raise ProtocolError("successor source changed after freeze")
    dependencies = {binding["path"]: binding["sha256"] for binding in materials["entries"].values()}
    if lock.get("dependency_sha256") != dependencies or lock.get("dependency_bundle_sha256") != canonical_sha256(dependencies):
        raise ProtocolError("material dependency closure changed after freeze")
    validate_materials(materials, campaign)
    validate_toolchain(lock.get("toolchain"))
    if check_remote:
        remote_lock_receipt(path)
    if check_gpu:
        live = gpu_snapshot(campaign["hardware"]["timing_gpu"])
        validate_gpu(live, campaign)
        if live != lock.get("gpu") or live_toolchain() != lock["toolchain"]:
            raise ProtocolError("live GPU/toolchain identity differs from the frozen lock")
    return campaign, materials, lock


def make_timing_manifest(campaign: dict[str, Any], lock_sha256: str, admission: dict[str, Any]) -> dict[str, Any]:
    validate_campaign(campaign)
    if admission.get("campaign_id") != CAMPAIGN_ID or admission.get("campaign_lock_sha256") != lock_sha256 or admission.get("complete") is not True:
        raise ProtocolError("timing requires a complete admission summary bound to this lock")
    by_id = {row.get("pair_id"): row for row in admission.get("pairs", [])}
    if set(by_id) != {pair["pair_id"] for pair in campaign["pairs"]}:
        raise ProtocolError("admission summary pair denominator differs")
    eligible = set()
    for pair in campaign["pairs"]:
        row = by_id[pair["pair_id"]]
        if row.get("timing_eligible") is not True:
            continue
        if (
            not pair["gate"]["available"]
            or row.get("gate_available") is not True
            or row.get("classification") != "runtime_estimand"
            or set(row.get("artifact_identity_sha256", {})) != {"high", "low"}
        ):
            raise ProtocolError(f"{pair['pair_id']}: admission cannot authorize timing")
        identities = row["artifact_identity_sha256"]
        if (
            any(not HEX64.fullmatch(str(value)) for value in identities.values())
            or identities["high"] == identities["low"]
        ):
            raise ProtocolError(f"{pair['pair_id']}: admission artifact identities are invalid")
        eligible.add(pair["pair_id"])
    rows: list[dict[str, Any]] = []
    randomizer = random.Random(campaign["inference"]["randomization_seed"])
    for pair in campaign["pairs"]:
        if pair["pair_id"] not in eligible:
            continue
        for distribution in campaign["inference"]["distributions"]:
            for block in range(campaign["inference"]["confirmation_blocks"]):
                block_rows = [
                    ("high", "TL-H", "high"),
                    ("low", "TL-M", "low"),
                    ("sham_a", "sham_a", "high"),
                    ("sham_b", "sham_b", "high"),
                ]
                randomizer.shuffle(block_rows)
                for position, (role, label, implementation_side) in enumerate(block_rows):
                    arm = pair[implementation_side]
                    identity = [pair["pair_id"], distribution, block, role]
                    rows.append(
                        {
                            "row_id": hashlib.sha256(canonical_bytes(identity)).hexdigest(),
                            "pair_id": pair["pair_id"],
                            "family": pair["family"],
                            "distribution": distribution,
                            "block": block,
                            "position": position,
                            "role": role,
                            "label": label,
                            "implementation_side": implementation_side,
                            "variant": arm["variant"],
                            "set": arm["set"],
                            "physical_gpu": campaign["hardware"]["timing_gpu"],
                            "trials": campaign["inference"]["trials"],
                            "warmup_seconds": campaign["inference"]["warmup_seconds"],
                        }
                    )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": lock_sha256,
        "admission_summary_sha256": canonical_sha256(admission),
        "eligible_pair_ids": sorted(eligible),
        "rows": rows,
    }


def validate_timing_manifest(manifest: dict[str, Any], campaign: dict[str, Any], lock_sha256: str, admission: dict[str, Any]) -> None:
    expected = make_timing_manifest(campaign, lock_sha256, admission)
    if manifest != expected:
        raise ProtocolError("timing manifest differs from deterministic admission-bound projection")
    row_ids = [row["row_id"] for row in manifest["rows"]]
    if len(row_ids) != len(set(row_ids)):
        raise ProtocolError("timing manifest contains duplicate row IDs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "freeze", "verify-remote", "manifest"))
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "check":
        campaign, materials = load_sources()
        print(json.dumps({"ok": True, "materials": len(materials["entries"]), "pairs": len(campaign["pairs"])}, sort_keys=True))
        return 0
    if args.action == "freeze":
        output = args.output or args.lock
        freeze(output, args.gpu)
        print(output)
        return 0
    if args.action == "verify-remote":
        load_lock(args.lock)
        print(json.dumps(remote_lock_receipt(args.lock), sort_keys=True))
        return 0
    if args.admission is None or args.output is None:
        parser.error("manifest requires --admission and --output")
    campaign, _materials, _lock = load_lock(args.lock)
    from .analyze import load_verified_admission

    admission = load_verified_admission(args.admission, args.lock)
    manifest = make_timing_manifest(campaign, file_sha256(args.lock), admission)
    stable_write(args.output, manifest)
    print(json.dumps({"eligible_pairs": len(manifest["eligible_pair_ids"]), "rows": len(manifest["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
