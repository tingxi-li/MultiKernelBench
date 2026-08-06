#!/usr/bin/env python3
"""Fail-closed contract and plans for the local Ada candidate procedure."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONTRACT_PATH = HERE / "contract.json"
RESULTS_ROOT = HERE / "results"
EXECUTION_LOCK_PATH = HERE / "execution_lock.json"
CAMPAIGN_ID = "finite-frontier-ada-v5"
INSTRUMENT_ID = "fused-epilogue-crossed-v2"
SOURCE_TAG = "crossed_v2r3"
PREDECESSOR_INCIDENT_PATH = (
    "ako_runs/controlled_followup/finite_frontier_ada_v1/"
    "INCIDENT_SELECTION_20260806.json"
)
PREDECESSOR_INCIDENT_SHA256 = (
    "bef0a1e0eba76849420c918734d7837050e8b5a99a51d70b3307836cd081d9d2"
)
PREDECESSOR_ADMISSION_INCIDENT_PATH = (
    "ako_runs/controlled_followup/finite_frontier_ada_v2/"
    "INCIDENT_ADMISSION_20260806.json"
)
PREDECESSOR_ADMISSION_INCIDENT_SHA256 = (
    "89571304d78dcb6fa0e09e997218c66c6d6564af074af50d00e578571c1d4a6e"
)
PREDECESSOR_FRONTEND_INCIDENT_PATH = (
    "ako_runs/controlled_followup/finite_frontier_ada_v3/"
    "INCIDENT_ADMISSION_20260806.json"
)
PREDECESSOR_FRONTEND_INCIDENT_SHA256 = (
    "8791a13fa6bee8745c70a630f58ff5d5ba9894de6d12838f604b323be3f7f365"
)
PREDECESSOR_READINESS_INCIDENT_PATH = (
    "ako_runs/controlled_followup/finite_frontier_ada_v4/"
    "INCIDENT_SELECTION_READINESS_20260806.json"
)
PREDECESSOR_READINESS_INCIDENT_SHA256 = (
    "a5498dfbf32ba51da1639c642e4c5cdb3313d8b8f4fceffe8705f8f5f5d94d96"
)
QUESTION = (
    "Under the frozen two-stage screen/positive-selection procedure, does the "
    "selected TileLang or Triton candidate have lower independently confirmed latency?"
)
CLAIM_SCOPE = (
    "one frozen GEMM+bias+exact-GELU+row-softmax shape on physical RTX 6000 Ada "
    "GPU 0; Triton and TileLang only; two-stage procedure-selected candidates "
    "drawn from the finite registered space"
)
ESTIMAND = (
    "paired log latency ratio of the independently terminal-confirmed, "
    "procedure-selected TileLang and Triton candidates, TileLang over Triton; "
    "withheld-signed timing tests generalization of the positive-selected pair"
)
CLAIM_RULE = (
    "one procedure-selected candidate has lower local latency only if its "
    "terminal paired-ratio interval clears the sham floor in the same direction "
    "on the selection distribution and its withheld-signed generalization test"
)
DSLS = ("tilelang", "triton")
STRATEGIES = (
    "register_fused",
    "smem_staged",
    "global_intermediate",
    "register_common_postprocess",
)
GRID_IDS = tuple(f"g{index:02d}" for index in range(19))
DISTRIBUTIONS = ("positive", "withheld_signed")
SHAM_LABELS = ("sham_a", "sham_b")
TERMINAL_OUTCOMES = (
    "UNSUPPORTED",
    "BUILD_FAILED",
    "LAUNCH_FAILED",
    "GATE_FAILED",
    "GATE_PASSED",
)


class ProtocolError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON must contain an object: {path}")
    return value


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise ProtocolError(f"path escapes repository: {path}") from exc


def _verified_path(relative: Any) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ProtocolError("material path must be a safe repository-relative path")
    candidate = REPO_ROOT / relative
    if candidate.is_symlink():
        raise ProtocolError(f"material path must not be a symlink: {relative}")
    path = candidate.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ProtocolError(f"material path escapes repository: {relative}") from exc
    if not path.is_file():
        raise ProtocolError(f"material path is missing: {relative}")
    return path


def _tree_closure_paths(
    incident: dict[str, Any], expected_root: str, label: str,
) -> list[Path]:
    closure = incident.get("artifact_closure") if isinstance(incident, dict) else None
    if not isinstance(closure, dict) or set(closure) != {
        "algorithm",
        "excluded_runtime_files",
        "file_count",
        "root",
        "sha256",
        "total_bytes",
    }:
        raise ProtocolError(f"{label} incident closure schema changed")
    relative = closure["root"]
    if (
        relative != expected_root
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ProtocolError(f"{label} incident closure root changed")
    candidate = REPO_ROOT / relative
    if candidate.is_symlink() or not candidate.is_dir():
        raise ProtocolError(f"{label} incident closure root is missing or unsafe")
    root = candidate.resolve()
    try:
        root.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ProtocolError(f"{label} incident closure escapes the repository") from exc
    excluded = closure["excluded_runtime_files"]
    if excluded != ["active.lock"] or closure["algorithm"] != (
        "sha256(canonical_json(sorted([{path,sha256,size}])))"
    ):
        raise ProtocolError(f"{label} incident closure procedure changed")
    rows, paths = [], []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ProtocolError(f"{label} result closure contains a symlink: {path}")
        relative_path = path.relative_to(root).as_posix()
        if path.is_file() and relative_path not in excluded:
            paths.append(path)
            rows.append(
                {
                    "path": relative_path,
                    "sha256": file_sha256(path),
                    "size": path.stat().st_size,
                }
            )
    if (
        len(rows) != closure["file_count"]
        or sum(row["size"] for row in rows) != closure["total_bytes"]
        or canonical_sha256(rows) != closure["sha256"]
    ):
        raise ProtocolError(f"{label} non-controlling result closure changed")
    return paths


def local_launch_module():
    """Return only this campaign's launcher in package and script modes."""
    expected = (HERE / "launch.py").resolve()
    qualified = "ako_runs.controlled_followup.finite_frontier_ada_v5.launch"
    for name in ("__main__", qualified, "launch"):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str) and Path(module_file).resolve() == expected:
            return module
    module = importlib.import_module(qualified)
    if Path(getattr(module, "__file__", "")).resolve() != expected:
        raise ProtocolError("local launcher resolved outside the campaign root")
    return module


def predecessor_closure_paths(incident: dict[str, Any] | None = None) -> list[Path]:
    incident = incident or read_json(_verified_path(PREDECESSOR_INCIDENT_PATH))
    return _tree_closure_paths(
        incident,
        "ako_runs/controlled_followup/finite_frontier_ada_v1/results/selection_confirm",
        "v1 selection",
    )


def predecessor_frontend_closure_paths(
    incident: dict[str, Any] | None = None,
) -> list[Path]:
    incident = incident or read_json(_verified_path(PREDECESSOR_FRONTEND_INCIDENT_PATH))
    paths = _tree_closure_paths(
        incident,
        "ako_runs/controlled_followup/finite_frontier_ada_v3/results/artifact_admission",
        "v3 frontend admission",
    )
    if len(paths) != 15:
        raise ProtocolError("v3 frontend admission closure census changed")
    return paths


def predecessor_readiness_closure_paths(
    incident: dict[str, Any] | None = None,
) -> list[Path]:
    incident = incident or read_json(_verified_path(PREDECESSOR_READINESS_INCIDENT_PATH))
    paths = _tree_closure_paths(
        incident,
        "ako_runs/controlled_followup/finite_frontier_ada_v4/results/artifact_admission",
        "v4 selection readiness",
    )
    if len(paths) != 132:
        raise ProtocolError("v4 selection-readiness closure census changed")
    return paths


def predecessor_admission_closure_paths(
    incident: dict[str, Any] | None = None,
) -> list[Path]:
    incident = incident or read_json(_verified_path(PREDECESSOR_ADMISSION_INCIDENT_PATH))
    closure = incident.get("artifact_closure") if isinstance(incident, dict) else None
    if not isinstance(closure, dict) or set(closure) != {
        "excluded_runtime_files", "files", "retained_bytes", "retained_files"
    }:
        raise ProtocolError("predecessor admission incident closure schema changed")
    files = closure["files"]
    if (
        closure["excluded_runtime_files"] != ["active.lock"]
        or not isinstance(files, dict)
        or closure["retained_files"] != 2
        or len(files) != 2
    ):
        raise ProtocolError("predecessor admission incident closure census changed")
    paths = []
    retained_bytes = 0
    for relative, expected in sorted(files.items()):
        path = _verified_path(relative)
        if (
            not isinstance(expected, dict)
            or set(expected) != {"bytes", "sha256"}
            or path.stat().st_size != expected["bytes"]
            or file_sha256(path) != expected["sha256"]
        ):
            raise ProtocolError(f"predecessor admission artifact changed: {relative}")
        paths.append(path)
        retained_bytes += path.stat().st_size
    if retained_bytes != closure["retained_bytes"]:
        raise ProtocolError("predecessor admission incident byte census changed")
    return paths


def load_contract() -> dict[str, Any]:
    return validate_contract(read_json(CONTRACT_PATH))


def validate_contract(contract: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "campaign_id",
        "state",
        "question",
        "claim_scope",
        "estimand",
        "material_registry",
        "manifest",
        "broader_f1_outside_scope",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        raise ProtocolError("contract fields differ from the v2 schema")
    if (
        contract["schema_version"] != 1
        or contract["campaign_id"] != CAMPAIGN_ID
        or contract["state"] != "prepared_requires_execution_lock_and_remote_commit"
    ):
        raise ProtocolError("contract identity, schema, or state changed")
    if (
        contract["question"] != QUESTION
        or contract["claim_scope"] != CLAIM_SCOPE
        or contract["estimand"] != ESTIMAND
    ):
        raise ProtocolError("question, claim scope, or estimand changed")
    registry = contract["material_registry"]
    if not isinstance(registry, dict) or set(registry) != {
        "source_campaign_id",
        "source_result_tag",
        "roles",
        "sha256",
    }:
        raise ProtocolError("material registry schema changed")
    if (
        registry["source_campaign_id"] != INSTRUMENT_ID
        or registry["source_result_tag"] != SOURCE_TAG
    ):
        raise ProtocolError("foreign source campaign or result tag")
    roles, hashes = registry["roles"], registry["sha256"]
    if (
        not isinstance(roles, dict)
        or not isinstance(hashes, dict)
        or len(set(roles.values())) != len(roles)
        or set(hashes) != set(roles.values())
    ):
        raise ProtocolError("material roles and hashes must be one exact bijection")
    for relative, expected in hashes.items():
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ProtocolError("material digest is not lowercase SHA-256")
        if file_sha256(_verified_path(relative)) != expected:
            raise ProtocolError(f"material hash mismatch: {relative}")
    if (
        roles.get("predecessor_selection_incident") != PREDECESSOR_INCIDENT_PATH
        or hashes.get(PREDECESSOR_INCIDENT_PATH) != PREDECESSOR_INCIDENT_SHA256
    ):
        raise ProtocolError("successor does not bind the exact v1 selection incident")
    incident = read_json(_verified_path(PREDECESSOR_INCIDENT_PATH))
    if (
        incident.get("campaign_id") != "finite-frontier-ada-v1"
        or incident.get("result_state") != "complete_raw_noncontrolling"
        or incident.get("successor_policy", {}).get("corrective_result_tag")
        != "finite_frontier_ada_v2"
    ):
        raise ProtocolError("predecessor incident is not the frozen non-controlling failure")
    predecessor_closure_paths(incident)
    if (
        roles.get("predecessor_admission_incident")
        != PREDECESSOR_ADMISSION_INCIDENT_PATH
        or hashes.get(PREDECESSOR_ADMISSION_INCIDENT_PATH)
        != PREDECESSOR_ADMISSION_INCIDENT_SHA256
    ):
        raise ProtocolError("successor does not bind the exact v2 admission incident")
    admission_incident = read_json(_verified_path(PREDECESSOR_ADMISSION_INCIDENT_PATH))
    if (
        admission_incident.get("campaign_id") != "finite-frontier-ada-v2"
        or admission_incident.get("classification")
        != "NON_CONTROLLING_ARTIFACT_ADMISSION_INCIDENT"
        or admission_incident.get("policy", {}).get("successor_requires_new_lock") is not True
        or admission_incident.get("policy", {}).get(
            "successor_requires_same_filesystem_cache_temporary_directory"
        ) is not True
    ):
        raise ProtocolError("v2 admission incident is not the frozen non-controlling failure")
    predecessor_admission_closure_paths(admission_incident)
    if (
        roles.get("predecessor_frontend_incident")
        != PREDECESSOR_FRONTEND_INCIDENT_PATH
        or hashes.get(PREDECESSOR_FRONTEND_INCIDENT_PATH)
        != PREDECESSOR_FRONTEND_INCIDENT_SHA256
    ):
        raise ProtocolError("successor does not bind the exact v3 frontend incident")
    frontend_incident = read_json(_verified_path(PREDECESSOR_FRONTEND_INCIDENT_PATH))
    if (
        frontend_incident.get("campaign_id") != "finite-frontier-ada-v3"
        or frontend_incident.get("classification")
        != "NON_CONTROLLING_ARTIFACT_VERIFICATION_INCIDENT"
        or frontend_incident.get("policy", {}).get("successor_requires_new_lock") is not True
        or frontend_incident.get("policy", {}).get(
            "successor_requires_content_bound_frontend_key_normalization"
        ) is not True
        or frontend_incident.get("observations", {}).get("requested_output_index") != [-1]
        or frontend_incident.get("observations", {}).get("normalized_output_index") != [3]
    ):
        raise ProtocolError("v3 frontend incident is not the frozen non-controlling failure")
    predecessor_frontend_closure_paths(frontend_incident)
    if (
        roles.get("predecessor_readiness_incident")
        != PREDECESSOR_READINESS_INCIDENT_PATH
        or hashes.get(PREDECESSOR_READINESS_INCIDENT_PATH)
        != PREDECESSOR_READINESS_INCIDENT_SHA256
    ):
        raise ProtocolError("successor does not bind the exact v4 readiness incident")
    readiness_incident = read_json(_verified_path(PREDECESSOR_READINESS_INCIDENT_PATH))
    if (
        readiness_incident.get("campaign_id") != "finite-frontier-ada-v4"
        or readiness_incident.get("classification")
        != "NON_CONTROLLING_SELECTION_READINESS_INCIDENT"
        or readiness_incident.get("policy", {}).get("successor_requires_new_lock") is not True
        or readiness_incident.get("policy", {}).get(
            "successor_requires_exact_package_safe_local_launcher"
        ) is not True
        or readiness_incident.get("sealed_census", {}).get(
            "observed_artifact_admission_entries"
        ) != 7
        or readiness_incident.get("sealed_census", {}).get(
            "observed_selection_timing_records"
        ) != 0
    ):
        raise ProtocolError("v4 readiness incident is not the frozen non-controlling failure")
    predecessor_readiness_closure_paths(readiness_incident)

    manifest = contract["manifest"]
    factors = manifest.get("factors", {})
    if (
        tuple(factors.get("dsls", ())) != DSLS
        or tuple(factors.get("strategies", ())) != STRATEGIES
        or tuple(factors.get("grid_ids", ())) != GRID_IDS
    ):
        raise ProtocolError("finite candidate factors changed")
    audit, screen = manifest.get("audit", {}), manifest.get("screen", {})
    if audit.get("requested_cells") != 152 or sum(audit.get("expected_outcomes", {}).values()) != 152:
        raise ProtocolError("audit census is not exactly 2 x 4 x 19")
    if screen.get("gate_legal_cells") != 113 or screen.get("expected_records") != 226:
        raise ProtocolError("imported screen census changed")
    selection, terminal = manifest.get("selection_confirm", {}), manifest.get("terminal_confirm", {})
    candidates = selection.get("candidate_ids", {})
    if set(candidates) != set(DSLS) or any(len(candidates[dsl]) != 3 for dsl in DSLS):
        raise ProtocolError("selection-confirm requires three candidates per DSL")
    if len({item for values in candidates.values() for item in values}) != 6:
        raise ProtocolError("selection candidates must be six unique cells")
    if selection.get("expected_records") != 240 or terminal.get("expected_records") != 120:
        raise ProtocolError("fresh timing census changed")
    if tuple(selection.get("distributions", ())) != DISTRIBUTIONS or tuple(terminal.get("distributions", ())) != DISTRIBUTIONS:
        raise ProtocolError("timing distributions changed")
    if tuple(selection.get("sham_labels", ())) != SHAM_LABELS:
        raise ProtocolError("sham labels changed")
    admission = manifest.get("artifact_admission", {})
    loader_hashes = admission.get("cache_loader_sha256") if isinstance(admission, dict) else None
    static_admission = dict(admission) if isinstance(admission, dict) else {}
    static_admission.pop("cache_loader_sha256", None)
    if static_admission != {
        "candidate_artifacts": 6,
        "entry_count": 7,
        "fresh_cache_hit_verification_process": True,
        "generated_source_content_addressed": True,
        "loadable_code_objects_content_addressed": True,
        "negative_out_idx_fallback_content_bound": True,
        "performance_blind": True,
        "shared_sham_artifacts": 1,
        "timing_children_load_only": True,
    }:
        raise ProtocolError("artifact-admission design changed")
    loader_paths = {
        "tilelang/cache/__init__.py": "tilelang/cache/__init__.py",
        "tilelang/cache/kernel_cache.py": "tilelang/cache/kernel_cache.py",
        "tilelang/env.py": "tilelang/env.py",
        "tilelang/jit/__init__.py": "tilelang/jit/__init__.py",
        "tilelang/jit/adapter/base.py": "tilelang/jit/adapter/base.py",
        "tilelang/jit/adapter/kernel_cache.py": "tilelang/jit/adapter/kernel_cache.py",
        "tilelang/jit/adapter/tvm_ffi.py": "tilelang/jit/adapter/tvm_ffi.py",
        "tilelang/jit/execution_backend.py": "tilelang/jit/execution_backend.py",
        "tilelang/jit/kernel.py": "tilelang/jit/kernel.py",
        "triton/compiler/compiler.py": "triton/compiler/compiler.py",
        "triton/knobs.py": "triton/knobs.py",
        "triton/runtime/build.py": "triton/runtime/build.py",
        "triton/runtime/cache.py": "triton/runtime/cache.py",
    }
    roots = {}
    for package in ("tilelang", "triton"):
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise ProtocolError(f"frozen cache-loader package is unavailable: {package}")
        roots[package] = Path(next(iter(spec.submodule_search_locations))).parent
    observed_loader_hashes = {
        logical: file_sha256(roots[logical.split("/", 1)[0]] / relative)
        for logical, relative in loader_paths.items()
    }
    if loader_hashes != observed_loader_hashes:
        raise ProtocolError("native TileLang/Triton cache-loader implementation changed")
    timing = manifest.get("timing", {})
    if timing != {
        "flush_l2": True,
        "fresh_process_per_record": True,
        "trials": 100,
        "warmup_s": 2.0,
    }:
        raise ProtocolError("timing instrument changed")
    inference = manifest.get("inference", {})
    if (
        inference.get("primary_trials") != [60, 100]
        or inference.get("winner_count_per_dsl") != 1
        or inference.get("selection_metric")
        != "median of 15 positive-distribution process primary-tail medians"
        or inference.get("claim_rule") != CLAIM_RULE
        or inference.get("max_selection_sham_floor_log_ratio")
        != 0.048790164169432
        or inference.get("withheld_signed_role")
        != "generalization test for the positive-selected pair, not a signed-distribution frontier"
    ):
        raise ProtocolError("primary window or winner count changed")
    hardware = manifest.get("hardware", {})
    if hardware != {
        "compute_capability": "8.9",
        "driver_version": "610.43.02",
        "gpu_name": "NVIDIA RTX 6000 Ada Generation",
        "gpu_uuid": "GPU-45af34ad-0c74-74d0-ef3a-652090d837ae",
        "physical_gpu": 0,
    }:
        raise ProtocolError("current-Ada hardware binding changed")
    if manifest.get("toolchain") != {
        "nvcc_release": "V13.1.115",
        "python": "3.13.5",
        "tilelang": "0.1.11",
        "torch": "2.10.0+cu128",
        "triton": "3.6.0",
    }:
        raise ProtocolError("current-Ada toolchain binding changed")

    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core as instrument_core

    campaign, _cells, lock = instrument_core.load_contract()
    if campaign["campaign_id"] != INSTRUMENT_ID:
        raise ProtocolError("instrument contract has another campaign ID")
    index = read_json(_verified_path(roles["source_evidence_index"]))
    if (
        index.get("record_type") != "fused_crossed_v2_complete_evidence_index"
        or index.get("campaign_id") != INSTRUMENT_ID
        or index.get("launch_lock_sha256") != hashes[roles["instrument_launch_lock"]]
        or index.get("source_bundle_sha256") != lock.get("source_bundle_sha256")
    ):
        raise ProtocolError("source evidence index lost its campaign/lock binding")
    entries = index.get("entries")
    if not isinstance(entries, list) or index.get("entry_count") != len(entries):
        raise ProtocolError("source evidence entry census changed")
    by_path = {row.get("path"): row for row in entries if isinstance(row, dict)}
    if len(by_path) != len(entries):
        raise ProtocolError("source evidence entries are malformed or duplicated")
    index_role = roles["source_evidence_index"]
    successor_only = {
        PREDECESSOR_INCIDENT_PATH,
        PREDECESSOR_ADMISSION_INCIDENT_PATH,
        PREDECESSOR_FRONTEND_INCIDENT_PATH,
        PREDECESSOR_READINESS_INCIDENT_PATH,
    }
    for relative, expected in hashes.items():
        if relative != index_role and relative not in successor_only and by_path.get(relative, {}).get("sha256") != expected:
            raise ProtocolError(f"source evidence index does not bind material: {relative}")
    return contract


def _instrument_root(contract: dict[str, Any]) -> Path:
    audit = _verified_path(contract["material_registry"]["roles"]["imported_audit_summary"])
    return audit.parent


def derive_imported_frontier(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = validate_contract(contract or read_json(CONTRACT_PATH))
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import analyze as source_analyze
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core as source_core

    roles = contract["material_registry"]["roles"]
    root = _instrument_root(contract)
    audit_path = _verified_path(roles["imported_audit_summary"])
    screen_path = _verified_path(roles["imported_screen_selection"])
    audit = source_analyze.audit_summary(root)
    if audit != read_json(audit_path):
        raise ProtocolError("imported audit summary differs from source re-derivation")
    screen = source_analyze.screen_selection(root, audit_path)
    if screen != read_json(screen_path):
        raise ProtocolError("imported screen differs from source re-derivation")
    _campaign, cells, _lock = source_core.load_contract()
    by_id = {cell["cell_id"]: cell for cell in cells}
    candidate_ids = sorted(cell["cell_id"] for cell in cells if cell["lane"] in DSLS)
    manifest = contract["manifest"]
    if len(candidate_ids) != 152 or canonical_sha256(candidate_ids) != manifest["audit"]["candidate_ids_sha256"]:
        raise ProtocolError("imported finite candidate denominator changed")

    outcomes = Counter()
    audit_records = []
    for cell_id in candidate_ids:
        path = root / "audit" / "records" / source_analyze.cell_filename(cell_id)
        record = read_json(path)
        outcome = record.get("terminal_outcome")
        if outcome not in TERMINAL_OUTCOMES:
            raise ProtocolError(f"unknown imported audit outcome: {cell_id}")
        outcomes[outcome] += 1
        audit_records.append({"cell_id": cell_id, "sha256": file_sha256(path), "terminal_outcome": outcome})
    expected_outcomes = {name: outcomes[name] for name in TERMINAL_OUTCOMES}
    if expected_outcomes != manifest["audit"]["expected_outcomes"]:
        raise ProtocolError("imported audit outcome census changed")

    legal = sorted(cell_id for cell_id in audit["timing_eligible_cell_ids"] if by_id[cell_id]["lane"] in DSLS)
    if len(legal) != 113 or canonical_sha256(legal) != manifest["screen"]["gate_legal_ids_sha256"]:
        raise ProtocolError("imported gate-legal set changed")
    plan = [
        row for row in source_core.screen_plan(cells, set(audit["timing_eligible_cell_ids"]))
        if by_id[row["cell_id"]]["lane"] in DSLS
    ]
    if len(plan) != 226 or canonical_sha256(plan) != manifest["screen"]["plan_sha256"]:
        raise ProtocolError("imported screen plan changed")
    grouped: dict[str, list[float]] = defaultdict(list)
    screen_records = []
    for row in plan:
        path = root / "screen" / "raw" / source_core.timing_filename(
            row["cell_id"], "positive", row["rep"]
        )
        record = read_json(path)
        value = float(record["primary_tail_median_ms"])
        grouped[row["cell_id"]].append(value)
        screen_records.append(
            {
                "cell_id": row["cell_id"],
                "primary_tail_median_ms": value,
                "rep": row["rep"],
                "sha256": file_sha256(path),
            }
        )
    if canonical_sha256(screen_records) != manifest["screen"]["records_sha256"]:
        raise ProtocolError("imported screen record projection changed")
    selected = {
        dsl: [
            cell_id
            for _median, cell_id in sorted(
                (statistics.median(values), cell_id)
                for cell_id, values in grouped.items()
                if by_id[cell_id]["lane"] == dsl and len(values) == 2
            )[:3]
        ]
        for dsl in DSLS
    }
    if (
        selected != manifest["selection_confirm"]["candidate_ids"]
        or canonical_sha256(selected) != manifest["selection_confirm"]["candidate_ids_sha256"]
    ):
        raise ProtocolError("mechanical top-three selection changed")
    return {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_imported_frontier",
        "campaign_id": CAMPAIGN_ID,
        "complete": True,
        "contract_sha256": file_sha256(CONTRACT_PATH),
        "source_campaign_id": INSTRUMENT_ID,
        "source_result_tag": SOURCE_TAG,
        "source_audit_summary_sha256": file_sha256(audit_path),
        "source_screen_selection_sha256": file_sha256(screen_path),
        "audit_records": audit_records,
        "audit_outcome_counts": expected_outcomes,
        "gate_legal_cell_ids": legal,
        "screen_records": screen_records,
        "selection_confirm_candidate_ids": selected,
    }


def validate_imported_frontier(value: Any, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    expected = derive_imported_frontier(contract)
    if value != expected:
        raise ProtocolError("retained imported frontier differs from re-derivation")
    return expected


def timing_plan(stage: str, winners: dict[str, str] | None = None) -> list[dict[str, Any]]:
    contract = load_contract()
    manifest = contract["manifest"]
    if stage == "selection_confirm":
        candidates = [
            cell_id
            for dsl in DSLS
            for cell_id in manifest["selection_confirm"]["candidate_ids"][dsl]
        ]
        seed = manifest["selection_confirm"]["order_seed"]
        expected = manifest["selection_confirm"]["expected_records"]
    elif stage == "terminal_confirm":
        if not isinstance(winners, dict) or set(winners) != set(DSLS):
            raise ProtocolError("terminal-confirm requires one winner per DSL")
        allowed = manifest["selection_confirm"]["candidate_ids"]
        if any(winners[dsl] not in allowed[dsl] for dsl in DSLS) or len(set(winners.values())) != 2:
            raise ProtocolError("terminal winners are not distinct selection-confirm candidates")
        candidates = [winners[dsl] for dsl in DSLS]
        seed = manifest["terminal_confirm"]["order_seed"]
        expected = manifest["terminal_confirm"]["expected_records"]
    else:
        raise ProtocolError(f"unknown timing stage: {stage}")
    sham_base = manifest["selection_confirm"]["sham_base_cell"]
    randomizer = random.Random(seed)
    plan = []
    for block in range(15):
        rows = [
            {
                "block": block,
                "cell_id": cell_id,
                "distribution": distribution,
                "label": cell_id,
                "record_kind": "candidate",
                "stage": stage,
            }
            for cell_id in candidates
            for distribution in DISTRIBUTIONS
        ]
        rows.extend(
            {
                "block": block,
                "cell_id": sham_base,
                "distribution": distribution,
                "label": label,
                "record_kind": "sham",
                "stage": stage,
            }
            for label in SHAM_LABELS
            for distribution in DISTRIBUTIONS
        )
        randomizer.shuffle(rows)
        for block_position, row in enumerate(rows):
            row["block_position"] = block_position
            row["position"] = len(plan)
            plan.append(row)
    if len(plan) != expected or len(
        {(row["block"], row["label"], row["distribution"]) for row in plan}
    ) != expected:
        raise ProtocolError("timing plan census is not exact")
    return plan


def validate_raw_census(raw: Path, plan: list[dict[str, Any]]) -> None:
    expected = {timing_filename(row) for row in plan}
    if len(expected) != len(plan):
        raise ProtocolError("timing plan filenames are not unique")
    observed = {path.name for path in raw.iterdir()} if raw.is_dir() else set()
    if observed != expected:
        missing, extra = sorted(expected - observed), sorted(observed - expected)
        raise ProtocolError(
            f"raw timing census differs: missing={missing[:3]} extra={extra[:3]}"
        )


def timing_filename(row: dict[str, Any]) -> str:
    label = row["label"].replace(".", "__")
    return f"{label}__{row['distribution']}__block{row['block']:02d}.json"


def summarize_times(times: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in times]
    if len(values) != 100 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ProtocolError("timing requires 100 positive finite trials")
    first, last = statistics.median(values[:10]), statistics.median(values[90:])
    return {
        "full_median_ms": statistics.median(values),
        "primary_tail_median_ms": statistics.median(values[60:100]),
        "first_decile_median_ms": first,
        "last_decile_median_ms": last,
        "first_to_last_decile_ratio": last / first,
    }


def local_source_paths() -> list[Path]:
    names = (
        ".gitignore",
        "__init__.py",
        "README.md",
        "contract.json",
        "protocol.py",
        "artifacts.py",
        "admit.py",
        "launch.py",
        "analyze.py",
        "test_protocol.py",
    )
    paths = [HERE / name for name in names]
    if any(not path.is_file() for path in paths):
        raise ProtocolError("successor source set is incomplete")
    return paths
