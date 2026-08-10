#!/usr/bin/env python3
"""Material and manifest contract for the serialized local Ada C2 pilot."""
from __future__ import annotations

import hashlib
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

from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core  # noqa: E402
from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import analyze as crossed_analyze  # noqa: E402
from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import freeze as crossed_freeze  # noqa: E402


CROSSED = HERE.parent / "fused_epilogue_crossed_v2"
RESULT_ROOT = CROSSED / "results" / "crossed_v2r3"
CAMPAIGN_ID = "decision-complexity-ada-v2-pilot"
TARGET_CELL_ID = "register_fused.tilelang.g09"
AXES = ("stages", "BM", "BK", "BN")
REPLICATES = 4
GPU_UUIDS = (
    "GPU-45af34ad-0c74-74d0-ef3a-652090d837ae",
)
GPU_LOCK_PATHS = tuple(
    f"/tmp/multikernelbench-{gpu_uuid}-timing.lock"
    for gpu_uuid in sorted(GPU_UUIDS)
)
ARMS = (
    "c2_open_1",
    "c2_open_2",
    "c2_open_4",
    "label_sham_a",
    "label_sham_b",
    "sensitivity_target_hint",
)
ATTEMPT_STATUSES = (
    "AUDIT_FAILED",
    "BUILD_FAILED",
    "LAUNCH_FAILED",
    "GATE_FAILED",
    "GATE_PASSED",
    "TIMEOUT",
)
ATTEMPT_TIMEOUT_S = 900
TIMING_TRIALS = 100
WARMUP_S = 2.0
TAIL_START = 60
TAIL_STOP = 100
WITHHELD_SEED = 2026073101
RANDOMIZATION_SEED = "33c603ba76f105a59dc4fed88cccb3eecd1297493bd13c587f283855c23edb6e"
PLANNED_CONTRASTS = (
    ("c2_open_1", "c2_open_2"),
    ("c2_open_2", "c2_open_4"),
)
PAIRED_OUTCOMES = ("attempts_consumed", "active_s")
CACHE_DIRECTORIES = {
    "CUDA_CACHE_PATH": "cuda",
    "NUMBA_CACHE_DIR": "numba",
    "TILELANG_CACHE_DIR": "tilelang",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TRITON_CACHE_DIR": "triton",
    "TVM_CACHE_DIR": "tvm",
    "XDG_CACHE_HOME": "xdg",
}


def execution_schedule() -> dict[str, Any]:
    return {
        "gpu_lock_paths": list(GPU_LOCK_PATHS),
        "order": "serialized_manifest_order",
        "overlap_policy": "strict_previous_completion_before_next_launch",
        "resume_policy": (
            "new_successor_campaign_result_tag_and_root_after_any_result_evidence"
        ),
        "simultaneous_trajectories": 1,
    }


DEPENDENCIES = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/launch_lock.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/evidence/crossed_v2r3_complete_v1.index.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/core.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/candidates.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/campaign_runner.py",
    "ako_runs/controlled_followup/fused_grid/jobs/fused_gbgs_grid.json",
    "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r3/audit_summary.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r3/confirmation_selection.json",
)


class ProtocolError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


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
        raise ProtocolError(f"JSON must be an object: {path}")
    return value


def _record_path(cell_id: str) -> Path:
    return RESULT_ROOT / "audit" / "records" / (cell_id.replace(".", "__") + ".json")


def _gate_path(cell_id: str) -> Path:
    return RESULT_ROOT / "audit" / "gate" / (cell_id.replace(".", "__") + ".jsonl")


def _config(cell: dict[str, Any]) -> dict[str, int]:
    parsed = core.parse_set(cell["origin_job"]["set"])
    try:
        return {axis: int(parsed[axis]) for axis in AXES}
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"cell lacks the frozen search axes: {cell['cell_id']}") from exc


@lru_cache(maxsize=1)
def _material_index() -> dict[str, Any]:
    dependency_hashes = {}
    for relative in DEPENDENCIES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ProtocolError(f"missing dependency: {relative}")
        dependency_hashes[relative] = file_sha256(path)

    lock_path = REPO_ROOT / DEPENDENCIES[0]
    source_index_path = REPO_ROOT / DEPENDENCIES[1]
    lock = read_json(lock_path)
    source_index = read_json(source_index_path)
    if (
        lock.get("campaign_id") != core.CAMPAIGN_ID
        or lock.get("lock_stage") != "campaign"
        or source_index.get("campaign_id") != core.CAMPAIGN_ID
        or source_index.get("record_type") != "fused_crossed_v2_complete_evidence_index"
        or source_index.get("entry_count") != 2638
        or source_index.get("launch_lock_sha256") != file_sha256(lock_path)
        or source_index.get("source_bundle_sha256") != lock.get("source_bundle_sha256")
    ):
        raise ProtocolError("crossed_v2r3 evidence index does not bind the sealed campaign")
    current_lock = crossed_freeze.make_lock("campaign")
    if any(
        lock.get(key) != value
        for key, value in current_lock.items()
        if key != "created_utc"
    ):
        raise ProtocolError("crossed_v2 source/dependency closure differs from its launch lock")
    entries = source_index.get("entries")
    by_path = {
        row.get("path"): row for row in entries if isinstance(row, dict)
    } if isinstance(entries, list) else {}
    if len(by_path) != source_index.get("entry_count") or None in by_path:
        raise ProtocolError("crossed_v2 evidence index entries are malformed")

    def require_indexed(path: Path) -> None:
        relative = str(path.relative_to(REPO_ROOT))
        expected = {
            "path": relative,
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        if by_path.get(relative) != expected:
            raise ProtocolError(f"evidence index does not bind current bytes: {relative}")

    for relative in DEPENDENCIES:
        path = REPO_ROOT / relative
        if path != source_index_path:
            require_indexed(path)

    audit = read_json(RESULT_ROOT / "audit_summary.json")
    eligible = set(audit.get("timing_eligible_cell_ids", []))
    if (
        audit.get("complete") is not True
        or audit.get("requested_cells") != 304
        or audit.get("launch_lock_sha256") != file_sha256(lock_path)
    ):
        raise ProtocolError("crossed_v2r3 audit summary is incomplete or foreign")
    if crossed_analyze.audit_summary(RESULT_ROOT) != audit:
        raise ProtocolError("crossed_v2r3 audit summary is not re-derived from all 304 records")

    selection = read_json(RESULT_ROOT / "confirmation_selection.json")
    target_rows = [row for row in selection.get("selected", []) if row.get("cell_id") == TARGET_CELL_ID]
    if (
        selection.get("complete") is not True
        or len(target_rows) != 1
        or target_rows[0].get("screen_rank") != 1
        or target_rows[0].get("selection_reason") != "top_two"
    ):
        raise ProtocolError("target was not independently selected before this pilot")
    if crossed_analyze.screen_selection(RESULT_ROOT, RESULT_ROOT / "audit_summary.json") != selection:
        raise ProtocolError("target selection is not re-derived from retained screen evidence")

    cells = [
        cell
        for cell in core.load_cells(require_resolved=True)
        if cell["strategy"] == "register_fused" and cell["lane"] == "tilelang"
    ]
    if len(cells) != 19 or {cell["grid_id"] for cell in cells} != set(core.GRID_IDS):
        raise ProtocolError("candidate family differs from the frozen 19-grid census")
    candidates = []
    for cell in cells:
        cell_id = cell["cell_id"]
        record_path = _record_path(cell_id)
        gate_path = _gate_path(cell_id)
        if not record_path.is_file() or not gate_path.is_file():
            raise ProtocolError(f"candidate evidence is missing: {cell_id}")
        require_indexed(record_path)
        require_indexed(gate_path)
        record = read_json(record_path)
        metadata = record.get("build_metadata", {})
        artifacts = metadata.get("artifacts")
        gate_summary = record.get("gate_summary", {})
        implementation = metadata.get("implementation_sha256")
        if (
            cell_id not in eligible
            or record.get("terminal_outcome") != "GATE_PASSED"
            or record.get("cell") != cell
            or record.get("cell_sha256") != core.canonical_sha256(cell)
            or record.get("gate_jsonl_sha256") != file_sha256(gate_path)
            or gate_summary.get("complete") is not True
            or gate_summary.get("full_gate_pass") is not True
            or gate_summary.get("observed_records") != 512
            or metadata.get("n_kernels") != 2
            or not isinstance(artifacts, dict)
            or not isinstance(implementation, str)
            or len(implementation) != 64
        ):
            raise ProtocolError(f"candidate is not bound to current gate evidence: {cell_id}")
        candidates.append(
            {
                "cell_id": cell_id,
                "cell_sha256": core.canonical_sha256(cell),
                "config": _config(cell),
                "artifacts_sha256": canonical_sha256(artifacts),
                "implementation_sha256": implementation,
                "record_path": str(record_path.relative_to(REPO_ROOT)),
                "record_sha256": file_sha256(record_path),
                "gate_path": str(gate_path.relative_to(REPO_ROOT)),
                "gate_sha256": file_sha256(gate_path),
            }
        )
    candidates.sort(key=lambda row: row["cell_id"])
    return {
        "schema_version": 1,
        "record_type": "decision_complexity_ada_v2_material_index",
        "campaign_id": CAMPAIGN_ID,
        "instrument_campaign_id": core.CAMPAIGN_ID,
        "instrument_source_bundle_sha256": lock["source_bundle_sha256"],
        "instrument_launch_lock_sha256": file_sha256(lock_path),
        "instrument_evidence_index_sha256": file_sha256(source_index_path),
        "dependency_sha256": dependency_hashes,
        "target_cell_id": TARGET_CELL_ID,
        "candidates": candidates,
    }


def material_index() -> dict[str, Any]:
    return deepcopy(_material_index())


def _candidate_set(materials: dict[str, Any], open_count: int) -> list[str]:
    if open_count not in (1, 2, 4):
        raise ProtocolError(f"unsupported open-axis count: {open_count}")
    by_id = {row["cell_id"]: row for row in materials["candidates"]}
    target = by_id[TARGET_CELL_ID]["config"]
    open_axes = set(AXES[:open_count])
    fixed = set(AXES) - open_axes
    result = sorted(
        cell_id
        for cell_id, row in by_id.items()
        if all(row["config"][axis] == target[axis] for axis in fixed)
    )
    expected = {1: 3, 2: 6, 4: 19}[open_count]
    if len(result) != expected or TARGET_CELL_ID not in result:
        raise ProtocolError(f"nested candidate census changed for {open_count} axes")
    return result


def make_contract() -> dict[str, Any]:
    materials = material_index()
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "state": "pilot_prelaunch_not_authorized",
        "claim_scope": "local_tilelang_register_fused_uniform_search_decision_space_burden",
        "controlling": False,
        "hardware": {
            "name": "NVIDIA RTX 6000 Ada Generation",
            "compute_capability": "8.9",
            "gpu_uuids": list(GPU_UUIDS),
        },
        "replicates": REPLICATES,
        "arms": list(ARMS),
        "axis_order": list(AXES),
        "randomization_seed": RANDOMIZATION_SEED,
        "target_cell_id": TARGET_CELL_ID,
        "terminal_ratio": 1.05,
        "terminal_distribution": "withheld_signed",
        "settled_tail": {"start_inclusive": 60, "stop_exclusive": 100},
        "timing": {
            "distribution": "randn",
            "seed": WITHHELD_SEED,
            "trials": TIMING_TRIALS,
            "warmup_s": WARMUP_S,
            "flush_l2": True,
            "paired_candidate_target": True,
            "pair_order": "sha256_execution_contract_attempt",
        },
        "attempt_timeout_s": ATTEMPT_TIMEOUT_S,
        "attempt_statuses": list(ATTEMPT_STATUSES),
        "execution_schedule": execution_schedule(),
        "search_time_cache_policy": {
            "directory_environment": CACHE_DIRECTORIES,
            "mode": "fresh_empty_root_per_trajectory_attempt",
            "phase2_tl_cache": "0",
            "temporary_directory_environment": {"TMPDIR": "attempt_root"},
        },
        "pilot_analysis": {
            "control_checks": {
                "label_sham": {
                    "exact_match_fields_within_replicate": [
                        "attempts_consumed",
                        "event_observed",
                        "first_event_attempt",
                    ],
                },
                "target_hint": {
                    "event_at_first_attempt_required_replicates": REPLICATES,
                },
            },
            "multiplicity": "holm_across_all_four_planned_paired_tests",
            "paired_contrasts": [list(value) for value in PLANNED_CONTRASTS],
            "paired_outcomes": list(PAIRED_OUTCOMES),
            "paired_test": "exact_two_sided_sign_flip_of_mean_difference",
            "rmst_support_policy": "min_preregistered_tau_and_common_observed_support",
            "survival_tau": {
                "attempts": 3,
                "active_s": 3 * ATTEMPT_TIMEOUT_S,
            },
        },
        "materials": materials,
        "materials_sha256": canonical_sha256(materials),
    }


def timing_pair_order(row: dict[str, Any], attempt_index: int) -> list[str]:
    """Return the frozen within-attempt order, identical for both sham labels."""
    return sorted(
        ("candidate", "target"),
        key=lambda label: hashlib.sha256(
            f"{row['pair_order_seed_sha256']}:{attempt_index}:{label}".encode()
        ).digest(),
    )


def attempt_cache_receipt(
    row: dict[str, Any], candidate_id: str, attempt_index: int
) -> dict[str, Any]:
    return {
        "directory_environment": CACHE_DIRECTORIES,
        "mode": "fresh_empty_root_per_trajectory_attempt",
        "phase2_tl_cache": "0",
        "scope_sha256": canonical_sha256(
            [row["trajectory_id"], candidate_id, attempt_index]
        ),
    }


def _rank(seed: str, replicate: int, cell_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{replicate}\0{cell_id}".encode()).hexdigest()


def _build_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    materials = contract["materials"]
    seed = contract["randomization_seed"]
    rows = []
    for replicate in range(REPLICATES):
        arm_order = sorted(ARMS, key=lambda arm: _rank(seed, replicate, f"arm:{arm}"))
        common_order = sorted(
            (row["cell_id"] for row in materials["candidates"]),
            key=lambda cell_id: _rank(seed, replicate, cell_id),
        )
        for position, arm in enumerate(arm_order):
            if arm == "c2_open_1":
                open_count, execution_arm = 1, arm
            elif arm in {"c2_open_2", "label_sham_a", "label_sham_b"}:
                open_count = 2
                execution_arm = "label_sham" if arm.startswith("label_sham_") else arm
            else:
                open_count, execution_arm = 4, arm
            allowed = set(_candidate_set(materials, open_count))
            candidate_order = [cell_id for cell_id in common_order if cell_id in allowed]
            if arm == "sensitivity_target_hint":
                candidate_order.remove(TARGET_CELL_ID)
                candidate_order.insert(0, TARGET_CELL_ID)
            randomization_contract = {
                "searcher": "uniform_without_replacement_common_random_ranks_v1",
                "replicate": replicate,
                "gpu_slot": replicate,
                "open_axis_count": open_count,
                "open_axes": list(AXES[:open_count]),
                "candidate_order": candidate_order,
                "target_cell_id": TARGET_CELL_ID,
                "terminal_ratio": contract["terminal_ratio"],
                "terminal_distribution": contract["terminal_distribution"],
                "settled_tail": contract["settled_tail"],
                "execution_arm": execution_arm,
            }
            execution_contract = {**randomization_contract, "gpu_slot": 0}
            rows.append(
                {
                    "schema_version": 1,
                    "campaign_id": CAMPAIGN_ID,
                    "trajectory_id": canonical_sha256(
                        [CAMPAIGN_ID, replicate, arm, contract["materials_sha256"]]
                    ),
                    "replicate": replicate,
                    "gpu_slot": 0,
                    "gpu_uuid": GPU_UUIDS[0],
                    "block_position": position,
                    "launch_sequence": len(rows) + 1,
                    "arm": arm,
                    "hidden_label": arm[-1] if arm.startswith("label_sham_") else None,
                    "execution_contract": execution_contract,
                    "execution_contract_sha256": canonical_sha256(execution_contract),
                    "pair_order_seed_sha256": canonical_sha256(randomization_contract),
                    "materials_sha256": contract["materials_sha256"],
                    "failures_charge_attempt": True,
                }
            )
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "contract_sha256": canonical_sha256(contract),
        "requested_trajectories": REPLICATES * len(ARMS),
        "rows": rows,
    }
    return manifest


def make_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    validate_contract(contract)
    manifest = _build_manifest(contract)
    validate_manifest(contract, manifest)
    return manifest


def validate_contract(contract: dict[str, Any]) -> None:
    if (
        contract.get("schema_version") != 1
        or contract.get("campaign_id") != CAMPAIGN_ID
        or contract.get("state") != "pilot_prelaunch_not_authorized"
        or contract.get("controlling") is not False
        or contract.get("replicates") != REPLICATES
        or contract.get("arms") != list(ARMS)
        or contract.get("axis_order") != list(AXES)
        or contract.get("randomization_seed") != RANDOMIZATION_SEED
        or contract.get("target_cell_id") != TARGET_CELL_ID
        or contract.get("attempt_timeout_s") != ATTEMPT_TIMEOUT_S
        or contract.get("attempt_statuses") != list(ATTEMPT_STATUSES)
        or contract.get("execution_schedule") != execution_schedule()
        or contract.get("timing") != {
            "distribution": "randn",
            "seed": WITHHELD_SEED,
            "trials": TIMING_TRIALS,
            "warmup_s": WARMUP_S,
            "flush_l2": True,
            "paired_candidate_target": True,
            "pair_order": "sha256_execution_contract_attempt",
        }
        or contract.get("search_time_cache_policy") != {
            "directory_environment": CACHE_DIRECTORIES,
            "mode": "fresh_empty_root_per_trajectory_attempt",
            "phase2_tl_cache": "0",
            "temporary_directory_environment": {"TMPDIR": "attempt_root"},
        }
        or contract.get("pilot_analysis") != {
            "control_checks": {
                "label_sham": {
                    "exact_match_fields_within_replicate": [
                        "attempts_consumed",
                        "event_observed",
                        "first_event_attempt",
                    ],
                },
                "target_hint": {
                    "event_at_first_attempt_required_replicates": REPLICATES,
                },
            },
            "multiplicity": "holm_across_all_four_planned_paired_tests",
            "paired_contrasts": [list(value) for value in PLANNED_CONTRASTS],
            "paired_outcomes": list(PAIRED_OUTCOMES),
            "paired_test": "exact_two_sided_sign_flip_of_mean_difference",
            "rmst_support_policy": "min_preregistered_tau_and_common_observed_support",
            "survival_tau": {
                "attempts": 3,
                "active_s": 3 * ATTEMPT_TIMEOUT_S,
            },
        }
        or contract.get("materials_sha256") != canonical_sha256(contract.get("materials"))
        or contract.get("materials") != material_index()
    ):
        raise ProtocolError("contract differs from the current material projection")


def validate_manifest(contract: dict[str, Any], manifest: dict[str, Any]) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("contract_sha256") != canonical_sha256(contract)
        or manifest.get("requested_trajectories") != 24
    ):
        raise ProtocolError("manifest header differs from the pilot contract")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 24:
        raise ProtocolError("manifest requires exactly 24 trajectories")
    if len({row.get("trajectory_id") for row in rows}) != 24:
        raise ProtocolError("trajectory IDs are missing or duplicated")
    if [row.get("launch_sequence") for row in rows] != list(range(1, 25)):
        raise ProtocolError("trajectory launch sequence differs from manifest order")
    by_replicate = {replicate: [] for replicate in range(REPLICATES)}
    for row in rows:
        replicate = row.get("replicate")
        if replicate not in by_replicate:
            raise ProtocolError("manifest contains an unknown replicate")
        by_replicate[replicate].append(row)
    for replicate, block in by_replicate.items():
        if (
            {row.get("arm") for row in block} != set(ARMS)
            or sorted(row.get("block_position") for row in block) != list(range(len(ARMS)))
            or {row.get("gpu_slot") for row in block} != {0}
            or {row.get("gpu_uuid") for row in block} != {GPU_UUIDS[0]}
        ):
            raise ProtocolError("replicate block is incomplete or not GPU-bound")
        shams = sorted(
            (row for row in block if row["arm"].startswith("label_sham_")),
            key=lambda row: row["arm"],
        )
        if (
            len(shams) != 2
            or shams[0]["execution_contract"] != shams[1]["execution_contract"]
            or shams[0]["execution_contract_sha256"] != shams[1]["execution_contract_sha256"]
        ):
            raise ProtocolError("hidden-label sham execution contracts differ")
        sizes = {
            row["arm"]: len(row["execution_contract"]["candidate_order"])
            for row in block
        }
        if sizes["c2_open_1"] != 3 or sizes["c2_open_2"] != 6 or sizes["c2_open_4"] != 19:
            raise ProtocolError("treatment candidate-set sizes changed")
        hint = next(row for row in block if row["arm"] == "sensitivity_target_hint")
        if hint["execution_contract"]["candidate_order"][0] != TARGET_CELL_ID:
            raise ProtocolError("valid-hint control does not place the target first")
    expected = _build_manifest(contract)
    if manifest != expected:
        raise ProtocolError("manifest differs from the deterministic projection")


def refuse_launch() -> None:
    raise ProtocolError(
        "GPU launch requires the pilot runner, analyzer, tests, frozen execution lock, "
        "clean upstream commit, and fresh GPU-0 UUID preflight"
    )


if __name__ == "__main__":
    contract = make_contract()
    manifest = make_manifest(contract)
    print(json.dumps(manifest, indent=2, sort_keys=True))
