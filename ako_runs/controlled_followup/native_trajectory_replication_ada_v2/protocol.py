#!/usr/bin/env python3
"""Material contract for same-campaign native-strategy recurrence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import analyze as crossed_analyze  # noqa: E402
from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core  # noqa: E402
from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import freeze as crossed_freeze  # noqa: E402


CROSSED = HERE.parent / "fused_epilogue_crossed_v2"
RESULT_ROOT = CROSSED / "results" / "crossed_v2r3"
CAMPAIGN_ID = "native-trajectory-replication-ada-v2"
LANES = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
STRATEGIES = (
    "global_intermediate",
    "register_common_postprocess",
    "register_fused",
)
STEP_NAMES = (
    "global_to_register_common_native_strategy_contrast",
    "register_common_to_register_fused_native_strategy_contrast",
)
DISTRIBUTIONS = ("positive", "withheld_signed")
SHAM_LABELS = ("same_config_sham_a", "same_config_sham_b")
SHAM_CELL_ID = "register_common_postprocess.tilelang.g01"
REFERENCE_LANE = "tilelang"
BLOCKS = 15
ROWS_PER_BLOCK = len(LANES) * len(STRATEGIES) * len(DISTRIBUTIONS) + len(SHAM_LABELS) * len(DISTRIBUTIONS)
RAW_RECORDS = BLOCKS * ROWS_PER_BLOCK
RANDOMIZATION_SEED = 2026080502
TIMING_TRIALS = 100
WARMUP_S = 2.0
TAIL_START = 60
TAIL_STOP = 100
WITHHELD_SEED = 2026073101
GPU0_UUID = "GPU-45af34ad-0c74-74d0-ef3a-652090d837ae"
CUDA_HOME = "/usr/local/cuda-13.1"
RESULTS_ROOT = HERE / "results"
CONTRACT_PATH = HERE / "contract.json"
MANIFEST_PATH = HERE / "manifest.json"
MATERIAL_LOCK_PATH = HERE / "material_lock.json"
PROVENANCE_PATH = HERE / "prelaunch_provenance.json"
EXECUTION_LOCK_PATH = HERE / "execution_lock.json"
PREDECESSOR = HERE.parent / "native_trajectory_replication_ada_v1"
PREDECESSOR_INCIDENT = PREDECESSOR / "INCIDENT_EXECUTION_20260806.json"
PREDECESSOR_INCIDENT_SHA256 = "fa23bbcfc7063c145c911585d809e82f51d14afe0b0a987608c1fe75a77a68b7"
CACHE_LOADER_FILES = (
    "tilelang/cache/__init__.py",
    "tilelang/cache/kernel_cache.py",
    "tilelang/env.py",
    "tilelang/jit/__init__.py",
    "tilelang/jit/adapter/base.py",
    "tilelang/jit/adapter/kernel_cache.py",
    "tilelang/jit/adapter/tvm_ffi.py",
    "tilelang/jit/execution_backend.py",
    "tilelang/jit/kernel.py",
    "triton/compiler/compiler.py",
    "triton/knobs.py",
    "triton/runtime/build.py",
    "triton/runtime/cache.py",
    "torch/utils/cpp_extension.py",
)
DEPENDENCIES = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/launch_lock.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/evidence/crossed_v2r3_complete_v1.index.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/core.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/candidates.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/campaign_runner.py",
    "ako_runs/controlled_followup/fused_grid/jobs/fused_gbgs_grid.json",
    "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r3/audit_summary.json",
)


class ProtocolError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise ProtocolError(f"path is outside the repository: {path}") from exc


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON must be an object: {path}")
    return value


def load_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    validate_contract(contract)
    return contract


def predecessor_incident() -> dict[str, Any]:
    if file_sha256(PREDECESSOR_INCIDENT) != PREDECESSOR_INCIDENT_SHA256:
        raise ProtocolError("predecessor incident receipt changed")
    incident = read_json(PREDECESSOR_INCIDENT)
    files = incident.get("artifact_closure", {}).get("files")
    if (
        incident.get("campaign_id") != "native-trajectory-replication-ada-v1"
        or incident.get("classification") != "NON_CONTROLLING_EXECUTION_INCIDENT"
        or incident.get("policy", {}).get("continuation_authorized") is not False
        or incident.get("policy", {}).get("reuse_authorized") is not False
        or incident.get("policy", {}).get("successor_requires_fresh_admission_and_new_lock") is not True
        or not isinstance(files, dict)
        or incident.get("artifact_closure", {}).get("retained_files") != 3
        or incident.get("artifact_closure", {}).get("retained_bytes") != 24462
    ):
        raise ProtocolError("predecessor incident is not the frozen non-controlling failure")
    observed = {}
    for path in sorted((PREDECESSOR / "results").rglob("*")):
        if path.is_symlink():
            raise ProtocolError(f"predecessor closure contains a symlink: {path}")
        if path.is_file() and path.name != "active.lock":
            relative = str(path.relative_to(REPO_ROOT))
            observed[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if observed != files:
        raise ProtocolError("predecessor three-file closure changed")
    return incident


def predecessor_closure_paths() -> list[Path]:
    files = predecessor_incident()["artifact_closure"]["files"]
    return [REPO_ROOT / relative for relative in sorted(files)]


def cache_loader_hashes() -> dict[str, str]:
    roots = {}
    for package in ("tilelang", "triton", "torch"):
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise ProtocolError(f"cache-loader package is unavailable: {package}")
        roots[package] = Path(next(iter(spec.submodule_search_locations))).parent
    return {
        relative: file_sha256(roots[relative.split("/", 1)[0]] / relative)
        for relative in CACHE_LOADER_FILES
    }


def _record_path(cell_id: str) -> Path:
    return RESULT_ROOT / "audit" / "records" / (cell_id.replace(".", "__") + ".json")


def _gate_path(cell_id: str) -> Path:
    return RESULT_ROOT / "audit" / "gate" / (cell_id.replace(".", "__") + ".jsonl")


@lru_cache(maxsize=1)
def _material_index() -> dict[str, Any]:
    dependency_sha256: dict[str, str] = {}
    for relative in DEPENDENCIES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ProtocolError(f"missing dependency: {relative}")
        dependency_sha256[relative] = file_sha256(path)

    lock_path = REPO_ROOT / DEPENDENCIES[0]
    evidence_index_path = REPO_ROOT / DEPENDENCIES[1]
    lock = read_json(lock_path)
    evidence_index = read_json(evidence_index_path)
    if (
        lock.get("campaign_id") != core.CAMPAIGN_ID
        or lock.get("lock_stage") != "campaign"
        or evidence_index.get("campaign_id") != core.CAMPAIGN_ID
        or evidence_index.get("record_type") != "fused_crossed_v2_complete_evidence_index"
        or evidence_index.get("launch_lock_sha256") != file_sha256(lock_path)
        or evidence_index.get("source_bundle_sha256") != lock.get("source_bundle_sha256")
    ):
        raise ProtocolError("crossed_v2r3 evidence does not bind the frozen campaign")
    current_lock = crossed_freeze.make_lock("campaign")
    if any(
        lock.get(key) != value
        for key, value in current_lock.items()
        if key != "created_utc"
    ):
        raise ProtocolError("crossed_v2 source/dependency closure differs from its launch lock")
    entries = evidence_index.get("entries")
    by_path = {
        row.get("path"): row for row in entries if isinstance(row, dict)
    } if isinstance(entries, list) else {}
    if (
        len(by_path) != evidence_index.get("entry_count")
        or len(by_path) != len(entries or [])
        or None in by_path
    ):
        raise ProtocolError("crossed_v2 evidence-index entries are malformed")

    def indexed(path: Path) -> dict[str, Any]:
        relative = str(path.relative_to(REPO_ROOT))
        observed = {
            "path": relative,
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        if by_path.get(relative) != observed:
            raise ProtocolError(f"evidence index does not bind current bytes: {relative}")
        return observed

    for relative in DEPENDENCIES:
        path = REPO_ROOT / relative
        if path != evidence_index_path:
            indexed(path)

    audit_path = RESULT_ROOT / "audit_summary.json"
    audit = read_json(audit_path)
    if (
        audit.get("complete") is not True
        or audit.get("requested_cells") != 304
        or audit.get("launch_lock_sha256") != file_sha256(lock_path)
        or crossed_analyze.audit_summary(RESULT_ROOT) != audit
    ):
        raise ProtocolError("crossed_v2r3 audit is incomplete or not evidence-derived")
    eligible = set(audit.get("timing_eligible_cell_ids", []))
    cells = {cell["cell_id"]: cell for cell in core.load_cells(require_resolved=True)}
    expected_ids = {
        f"{strategy}.{lane}.g01" for lane in LANES for strategy in STRATEGIES
    }
    selected = []
    for lane in LANES:
        for strategy in STRATEGIES:
            cell_id = f"{strategy}.{lane}.g01"
            cell = cells.get(cell_id)
            record_path, gate_path = _record_path(cell_id), _gate_path(cell_id)
            if cell is None or not record_path.is_file() or not gate_path.is_file():
                raise ProtocolError(f"selected native prefix evidence is missing: {cell_id}")
            record_binding, gate_binding = indexed(record_path), indexed(gate_path)
            record = read_json(record_path)
            metadata = record.get("build_metadata", {})
            implementation = metadata.get("implementation_sha256")
            gate_summary = record.get("gate_summary", {})
            if (
                cell_id not in eligible
                or record.get("terminal_outcome") != "GATE_PASSED"
                or record.get("cell") != cell
                or record.get("cell_sha256") != core.canonical_sha256(cell)
                or record.get("gate_jsonl_sha256") != gate_binding["sha256"]
                or gate_summary.get("complete") is not True
                or gate_summary.get("full_gate_pass") is not True
                or gate_summary.get("observed_records") != 512
                or metadata.get("n_kernels") != 2
                or not isinstance(implementation, str)
                or len(implementation) != 64
            ):
                raise ProtocolError(f"selected native prefix is not current-gate-legal: {cell_id}")
            selected.append(
                {
                    "cell_id": cell_id,
                    "cell_sha256": core.canonical_sha256(cell),
                    "gate_evidence": gate_binding,
                    "grid_id": "g01",
                    "implementation_sha256": implementation,
                    "lane": lane,
                    "record_evidence": record_binding,
                    "strategy": strategy,
                    "terminal_outcome": "GATE_PASSED",
                }
            )
    if len(selected) != 12 or {row["cell_id"] for row in selected} != expected_ids:
        raise ProtocolError("selected native-prefix census differs from 4 x 3")
    return {
        "campaign_id": CAMPAIGN_ID,
        "dependency_sha256": dependency_sha256,
        "instrument_campaign_id": core.CAMPAIGN_ID,
        "instrument_evidence_index_sha256": file_sha256(evidence_index_path),
        "instrument_launch_lock_sha256": file_sha256(lock_path),
        "instrument_source_bundle_sha256": lock["source_bundle_sha256"],
        "record_type": "native_trajectory_replication_ada_v2_material_index",
        "schema_version": 1,
        "selected_prefixes": selected,
    }


def material_index() -> dict[str, Any]:
    return deepcopy(_material_index())


def _build_contract(materials: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_admission": {
            "cache_loader_sha256": cache_loader_hashes(),
            "entry_count": 12,
            "fresh_build_and_load_verification_processes": True,
            "identity_scope": "generated_source_and_loadable_code_object_bytes_only",
            "performance_blind": True,
            "same_filesystem_entry_local_atomic_temp": True,
            "shared_sham_cell_artifact": SHAM_CELL_ID,
            "timing_load_only": True,
        },
        "blocks": BLOCKS,
        "campaign_id": CAMPAIGN_ID,
        "claim_scope": "same_campaign_cross_lane_recurrence_of_native_strategy_contrasts_at_fused_g01_on_ada",
        "claims_excluded": [
            "causal isolation of accumulator fusion from builder and schedule differences",
            "independent replication of a previously estimated TileLang effect",
            "literal source translation",
            "multiplicity-adjusted global all-destination conclusion",
            "optimization-order portability",
            "translator independence",
            "general trajectory transfer",
        ],
        "controlling": True,
        "distributions": list(DISTRIBUTIONS),
        "reference_lane": {
            "lane": REFERENCE_LANE,
            "performance_estimate_bound_before_campaign": False,
            "role_fixed_before_timing": True,
        },
        "grid_id": "g01",
        "hardware": {
            "compute_capability": "8.9",
            "gpu_name": "NVIDIA RTX 6000 Ada Generation",
            "physical_gpu": 0,
            "gpu_uuid": GPU0_UUID,
        },
        "lanes": list(LANES),
        "materials": materials,
        "materials_sha256": canonical_sha256(materials),
        "operator": "fused_gemm_bias_exact_gelu_row_softmax",
        "prefixes": [
            {"prefix_index": 0, "strategy": STRATEGIES[0], "associated_mechanism": None},
            {"prefix_index": 1, "strategy": STRATEGIES[1], "associated_mechanism": "accumulator_bias_exact_gelu_fusion"},
            {"prefix_index": 2, "strategy": STRATEGIES[2], "associated_mechanism": "lane_native_softmax"},
        ],
        "predecessor_incident": {
            "path": str(PREDECESSOR_INCIDENT.relative_to(REPO_ROOT)),
            "result_role": "noncontrolling",
            "sha256": PREDECESSOR_INCIDENT_SHA256,
        },
        "raw_record_census": {
            "cell_records": BLOCKS * len(LANES) * len(STRATEGIES) * len(DISTRIBUTIONS),
            "records_per_block": ROWS_PER_BLOCK,
            "sham_records": BLOCKS * len(SHAM_LABELS) * len(DISTRIBUTIONS),
            "total": RAW_RECORDS,
        },
        "record_type": "native_trajectory_replication_ada_v2_contract",
        "schema_version": 1,
        "sham": {
            "base_cell_id": SHAM_CELL_ID,
            "identity_scope": "same_frozen_cell_config_and_admitted_artifact_identity",
            "labels": list(SHAM_LABELS),
            "source_byte_identity_claimed": True,
        },
        "steps": [
            {
                "associated_mechanism": "accumulator_bias_exact_gelu_fusion",
                "causal_mechanism_isolation_claimed": False,
                "from_strategy": STRATEGIES[0],
                "name": STEP_NAMES[0],
                "to_strategy": STRATEGIES[1],
            },
            {
                "associated_mechanism": "lane_native_softmax",
                "causal_mechanism_isolation_claimed": False,
                "from_strategy": STRATEGIES[1],
                "name": STEP_NAMES[1],
                "to_strategy": STRATEGIES[2],
            },
        ],
        "timing": {
            "cuda_home": CUDA_HOME,
            "flush_l2": True,
            "fresh_process_per_record": True,
            "primary_trials": {"start_inclusive": TAIL_START, "stop_exclusive": TAIL_STOP},
            "randomization": "sha256_ranked_complete_blocks",
            "randomization_seed": RANDOMIZATION_SEED,
            "trials": TIMING_TRIALS,
            "warmup_s": WARMUP_S,
        },
    }


def make_contract() -> dict[str, Any]:
    predecessor_incident()
    contract = _build_contract(material_index())
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    predecessor_incident()
    if contract != _build_contract(material_index()):
        raise ProtocolError("contract differs from the current frozen material projection")


def _rank(block: int, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_bytes([RANDOMIZATION_SEED, block, row])
    ).hexdigest()


def _build_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    materials = {row["cell_id"]: row for row in contract["materials"]["selected_prefixes"]}
    rows: list[dict[str, Any]] = []
    global_position = 0
    for block in range(BLOCKS):
        block_rows: list[dict[str, Any]] = []
        for lane in LANES:
            for strategy in STRATEGIES:
                cell_id = f"{strategy}.{lane}.g01"
                for distribution in DISTRIBUTIONS:
                    block_rows.append(
                        {
                            "cell_id": cell_id,
                            "distribution": distribution,
                            "implementation_sha256": materials[cell_id]["implementation_sha256"],
                            "label": cell_id,
                            "lane": lane,
                            "record_kind": "cell",
                            "strategy": strategy,
                        }
                    )
        sham = materials[SHAM_CELL_ID]
        for label in SHAM_LABELS:
            for distribution in DISTRIBUTIONS:
                block_rows.append(
                    {
                        "cell_id": SHAM_CELL_ID,
                        "distribution": distribution,
                        "implementation_sha256": sham["implementation_sha256"],
                        "label": label,
                        "lane": sham["lane"],
                        "record_kind": "same_config_label_sham",
                        "strategy": sham["strategy"],
                    }
                )
        block_rows.sort(key=lambda row: _rank(block, row))
        for block_position, row in enumerate(block_rows, 1):
            row["predecessor_implementation_sha256"] = row.pop("implementation_sha256")
            global_position += 1
            coordinate = {
                **row,
                "block": block,
                "block_position": block_position,
                "global_position": global_position,
            }
            rows.append({**coordinate, "row_id": canonical_sha256([CAMPAIGN_ID, coordinate])})
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "contract_sha256": canonical_sha256(contract),
        "expected_raw_records": RAW_RECORDS,
        "plan_sha256": canonical_sha256(rows),
        "record_type": "native_trajectory_replication_ada_v2_manifest",
        "rows": rows,
        "rows_per_block": ROWS_PER_BLOCK,
        "schema_version": 1,
    }
    return manifest


def make_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    validate_contract(contract)
    manifest = _build_manifest(contract)
    validate_manifest(contract, manifest)
    return manifest


def validate_manifest(contract: dict[str, Any], manifest: dict[str, Any]) -> None:
    if manifest != _build_manifest(contract):
        raise ProtocolError("manifest differs from the deterministic randomized projection")
    rows = manifest["rows"]
    if (
        len(rows) != RAW_RECORDS
        or [row["global_position"] for row in rows] != list(range(1, RAW_RECORDS + 1))
        or len({row["row_id"] for row in rows}) != RAW_RECORDS
    ):
        raise ProtocolError("manifest global position/census drift")
    for block in range(BLOCKS):
        members = [row for row in rows if row["block"] == block]
        if (
            len(members) != ROWS_PER_BLOCK
            or sorted(row["block_position"] for row in members) != list(range(1, ROWS_PER_BLOCK + 1))
            or sum(row["record_kind"] == "cell" for row in members) != 24
            or sum(row["record_kind"] == "same_config_label_sham" for row in members) != 4
        ):
            raise ProtocolError("randomized complete block census drift")
        shams = [row for row in members if row["record_kind"] == "same_config_label_sham"]
        if (
            {row["label"] for row in shams} != set(SHAM_LABELS)
            or {row["distribution"] for row in shams} != set(DISTRIBUTIONS)
            or {row["cell_id"] for row in shams} != {SHAM_CELL_ID}
            or len({row["predecessor_implementation_sha256"] for row in shams}) != 1
        ):
            raise ProtocolError("same-config sham identity/census drift")


def raw_filename(row: dict[str, Any]) -> str:
    return f"position{row['global_position']:03d}__{row['row_id'][:16]}.json"


def position_receipt_filename(row: dict[str, Any]) -> str:
    return f"position{row['global_position']:03d}.json"
