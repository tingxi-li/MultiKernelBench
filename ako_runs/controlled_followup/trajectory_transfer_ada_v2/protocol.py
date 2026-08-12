#!/usr/bin/env python3
"""Deterministic, fail-closed contract for the Ada trajectory-transfer study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CAMPAIGN_ID = "trajectory-transfer-ada-v2"
DESTINATIONS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
STRUCTURAL_ROUTES = ("direct_primitive_mapping", "manual_reconstruction")
ROUTE_BY_DESTINATION = {
    "tilelang": "direct_primitive_mapping",
    "triton": "direct_primitive_mapping",
    "cuda_noptx": "manual_reconstruction",
    "cuda_unlimited": "manual_reconstruction",
}
ADAPTATIONS = ("donor_fixed", "bounded_retune")
MECHANISM_STATES = ("off", "on")
DISTRIBUTIONS = ("positive", "withheld_signed")
GRID_IDS = tuple(f"g{index:02d}" for index in range(19))
TERMINAL_STATUSES = (
    "PRIMITIVE_ABSENT", "TRANSLATION_FAILED", "AUDIT_FAILED", "UNSUPPORTED",
    "BUILD_FAILED", "LAUNCH_FAILED", "GATE_FAILED", "GATE_PASSED",
)
CAMPAIGN_PATH = HERE / "campaign.json"
MECHANISM_CARD_PATH = HERE / "mechanism_card.json"
PRIMITIVE_MAP_PATH = HERE / "primitive_map.json"
MATERIALS_PATH = HERE / "materials.json"
ADMISSION_MANIFEST_PATH = HERE / "admission_manifest.json"
SCREEN_MANIFEST_PATH = HERE / "screen_manifest.json"
CONFIRMATION_MANIFEST_PATH = HERE / "confirmation_manifest.json"
CAMPAIGN_LOCK_PATH = HERE / "campaign_lock.json"
GRID_PATH = REPO_ROOT / "ako_runs/controlled_followup/fused_grid/jobs/fused_gbgs_grid.json"
FIXED_GRID = "g01"
ORIGIN_ID = "triton_native_softmax_g01"
DONOR_STEP_ID = "register_common_to_register_fused"
OFF_STRATEGY = "register_common_postprocess"
ON_STRATEGY = "register_fused"
BLOCKS = 15
SCREEN_REPLICATES = 2
SHAM_LABELS = ("same_artifact_sham_a", "same_artifact_sham_b")
RANDOMIZATION_SEED = 2026081201
INFERENCE_CONTRACT = {
    "aggregate_rule": "all donor/destination intervals clear the global sham floor on both distributions and all destination speedup sign tests survive Holm correction",
    "alpha": 0.05,
    "classification_rule": "fresh donor reference; otherwise destination and donor both speed up on both distributions, any destination slowdown, or unresolved transfer benefit",
    "controlling_trials": [60, 100],
    "effect_orientation": "log(off_primary_tail_median_ms/on_primary_tail_median_ms)",
    "interval_endpoint_rule": "speedup only when ci_lo > sham_floor; slowdown only when ci_hi < -sham_floor; equality is unresolved",
    "median_interval_order_rule": "reuse bound fused_epilogue_crossed_v1.core.exact_median_interval: choose maximum k with two-sided sign coverage >=0.95; endpoints ordered[k-1] and ordered[n-k], else k=1/full range",
    "destination_minus_donor_estimand": "destination_log_gain_minus_donor_log_gain",
    "interval": "exact_distribution_free_median_ratio_95",
    "multiplicity": "Holm within each adaptation and distribution across destinations",
    "primary_estimands": {
        "bounded_retune": "separately_optimized_mechanism_gain_under_equal_search_budgets",
        "donor_fixed": "transfer_mechanism_gain_at_donor_fixed_configuration",
    },
    "route_comparison_forbidden": True,
    "secondary_estimands": [
        "total_transferred_and_tuned_gain",
        "incremental_tuning_after_transfer",
        "generic_search_control",
    ],
    "sham_orientation": "same_artifact_sham_a_primary_tail_median_ms/same_artifact_sham_b_primary_tail_median_ms",
    "sham_floor": "maximum absolute log endpoint across every destination-distribution byte-identical sham interval",
    "sign_test": "exact one-sided sign test at global sham floor; block log-ratio <= floor counts nonpositive",
}
MATERIAL_PATH_CENSUS_SHA256 = "b445b453624c43656234300029b037ca23b371c9870dcf082a50c7f702d6b8a2"
SOURCE_RELATIVES = (
    ".gitignore",
    "README.md",
    "__init__.py",
    "analyze.py",
    "artifacts.py",
    "campaign.json",
    "implementations/__init__.py",
    "implementations/cuda_noptx.py",
    "implementations/cuda_unlimited.py",
    "implementations/tilelang.py",
    "implementations/triton.py",
    "materials.json",
    "mechanism_card.json",
    "primitive_map.json",
    "protocol.py",
    "runner.py",
    "test_execution.py",
    "test_artifacts.py",
    "test_protocol_contract.py",
)
REQUIRED_RUNTIME_MATERIALS = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/__init__.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/__init__.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/audit.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/common.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/validate.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/analyze.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/freeze.py",
    "ako_runs/controlled_followup/fused_grid/manifest.json",
    "ako_runs/controlled_followup/fused_grid/robust_adapter.py",
    "ako_runs/controlled_followup/fused_grid/robust_adapter_manifest.json",
    "ako_runs/controlled_followup/native_trajectory_replication_ada_v3/artifacts.py",
    "ako_runs/controlled_followup/native_trajectory_replication_ada_v3/__init__.py",
    "ako_runs/controlled_followup/native_trajectory_replication_ada_v3/protocol.py",
    "ako_runs/controlled_followup/native_trajectory_replication_ada_v3/runner.py",
    "ako_runs/controlled_followup/robust_gate/__init__.py",
    "ako_runs/controlled_followup/robust_gate/distributions.py",
    "ako_runs/controlled_followup/robust_gate/manifest.json",
    "ako_runs/controlled_followup/robust_gate/metrics.py",
    "ako_runs/controlled_followup/robust_gate/oracles.py",
    "ako_runs/controlled_followup/robust_gate/schema.py",
    "ako_runs/controlled_followup/robust_gate/seeds.py",
    "ako_runs/controlled_followup/robust_gate/validate.py",
    "ako_runs/phase2_fused_sdpa/runner2.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_tilelang_abstraction.py",
)


class ProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


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


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON must be an object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_if_absent_or_exact_json(path: Path, value: Any) -> None:
    """Create a frozen JSON file once; an exact retained value is idempotent."""
    def validate_retained() -> None:
        require(not path.is_symlink(), f"frozen JSON may not be a symlink: {path}")
        require(path.is_file(), f"frozen JSON is not a regular file: {path}")
        require(read_json(path) == value, f"refusing to replace changed frozen JSON: {path}")

    if path.exists() or path.is_symlink():
        validate_retained()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    created = False
    try:
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            created = True
        except FileExistsError as exc:
            raise ProtocolError(f"refusing existing freeze temporary: {temporary}") from exc
        try:
            os.link(temporary, path)
        except FileExistsError:
            validate_retained()
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def _base_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return tuple(read_json(path) for path in (
        CAMPAIGN_PATH, MECHANISM_CARD_PATH, PRIMITIVE_MAP_PATH, MATERIALS_PATH,
    ))  # type: ignore[return-value]


def _grid_jobs() -> dict[str, dict[str, dict[str, Any]]]:
    try:
        rows = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read grid: {exc}") from exc
    require(isinstance(rows, list), "grid must be a list")
    index: dict[str, dict[str, dict[str, Any]]] = {lane: {} for lane in DESTINATIONS}
    for row in rows:
        if isinstance(row, dict) and row.get("dsl") in index and row.get("grid_id") in GRID_IDS:
            index[row["dsl"]][row["grid_id"]] = row
    require(
        all(set(index[lane]) == set(GRID_IDS) for lane in DESTINATIONS),
        "grid destination/config census drift",
    )
    return index


def _validate_eligible_gate_record(
    record: dict[str, Any], destination: str, strategy: str,
) -> None:
    expected_cell_id = f"{strategy}.{destination}.{FIXED_GRID}"
    cell = record.get("cell")
    require(
        isinstance(cell, dict)
        and cell.get("cell_id") == expected_cell_id
        and cell.get("lane") == destination
        and cell.get("strategy") == strategy
        and cell.get("grid_id") == FIXED_GRID,
        f"{expected_cell_id}: bound gate-record cell identity drift",
    )
    origin_job = cell.get("origin_job")
    require(
        isinstance(origin_job, dict)
        and origin_job.get("dsl") == destination
        and origin_job.get("grid_id") == FIXED_GRID,
        f"{expected_cell_id}: bound origin job identity drift",
    )
    gate = record.get("gate_summary")
    require(
        record.get("terminal_outcome") == "GATE_PASSED"
        and record.get("build_attempted") is True
        and record.get("gate_attempted") is True
        and isinstance(gate, dict)
        and gate.get("complete") is True
        and gate.get("full_gate_pass") is True
        and gate.get("expected_records") == 512
        and gate.get("observed_records") == 512
        and gate.get("failed_records") == 0,
        f"{expected_cell_id}: reviewed implementation lacks a complete 512-row gate pass",
    )
    build = record.get("build_metadata")
    require(
        isinstance(build, dict) and build.get("n_kernels") == 2,
        f"{expected_cell_id}: reviewed implementation is not a two-kernel artifact",
    )


def _validate_tilelang_f1_gate_record(record: dict[str, Any], review: dict[str, Any]) -> None:
    gate = record.get("gate_summary")
    config = record.get("metadata", {}).get("config")
    artifact = record.get("metadata", {}).get("reported_artifacts")
    gate_path = REPO_ROOT / str(record.get("gate_path", ""))
    lock_path = REPO_ROOT / "ako_runs/controlled_followup/tilelang_abstraction_v7/campaign_lock.json"
    require(
        record.get("campaign_id") == "tilelang-abstraction-v7-ada"
        and record.get("variant") == "F1"
        and record.get("side") == "high"
        and record.get("set") == "x_soft_only=1,x_wcache=cached"
        and record.get("terminal_outcome") == "GATE_PASSED"
        and record.get("implementation_source_sha256") == review.get("source_sha256")
        and record.get("campaign_lock_sha256") == file_sha256(lock_path)
        and isinstance(gate, dict)
        and gate.get("complete") is True
        and gate.get("full_gate_pass") is True
        and gate.get("expected_records") == 512
        and gate.get("observed_records") == 512
        and gate.get("failed_records") == 0,
        "TileLang F1 lacks its bound complete 512-row gate pass",
    )
    require(
        isinstance(config, dict)
        and {
            key: config.get(key)
            for key in ("M", "N", "K", "BM", "BN", "BK", "threads", "stages", "kc", "arith", "cast", "dsl", "variant")
        } == {
            "M": 1024, "N": 8192, "K": 8192,
            "BM": 128, "BN": 128, "BK": 32,
            "threads": 256, "stages": 3, "kc": 2048,
            "arith": "fp16", "cast": "precast", "dsl": "tilelang_abs", "variant": "F1",
        }
        and config.get("extra") == {"soft_only": "1", "wcache": "cached"}
        and isinstance(artifact, dict)
        and artifact.get("n_kernels") == 1,
        "TileLang F1 predecessor coordinate drift",
    )
    require(
        gate_path.is_file() and file_sha256(gate_path) == record.get("gate_sha256"),
        "TileLang F1 gate JSONL changed",
    )


def validate_contract(*, rehash_materials: bool = True) -> None:
    campaign, mechanism, primitive_map, materials = _base_documents()
    require(campaign.get("schema_version") == 1 and campaign.get("campaign_id") == CAMPAIGN_ID, "invalid campaign")
    require(campaign.get("controlling") is True, "campaign controlling status drift")
    require(
        campaign.get("hardware") == {
            "compute_capability": "8.9",
            "driver_version": "610.43.02",
            "gpu_name": "NVIDIA RTX 6000 Ada Generation",
            "gpu_uuid": "GPU-45af34ad-0c74-74d0-ef3a-652090d837ae",
            "physical_gpu": 0,
        },
        "hardware identity drift",
    )
    require(campaign.get("destinations") == list(DESTINATIONS), "destination census/order drift")
    require(campaign.get("adaptations") == list(ADAPTATIONS), "adaptation census/order drift")
    require(campaign.get("routes") == list(STRUCTURAL_ROUTES), "structural-route census/order drift")
    require(campaign.get("route_by_destination") == ROUTE_BY_DESTINATION, "destination route drift")
    require(campaign.get("mechanism_states") == list(MECHANISM_STATES), "mechanism-state census/order drift")
    require(campaign.get("distributions") == list(DISTRIBUTIONS), "distribution census/order drift")
    require(campaign.get("failure_taxonomy") == list(TERMINAL_STATUSES), "terminal taxonomy drift")
    require(
        campaign.get("gates", {}).get("dynamic_work_audit") == {
            "expected_cuda_device_event_count": 2,
            "fresh_per_artifact": True,
            "method": "torch_profiler_cuda_activity_v1",
            "performance_observations_forbidden": True,
            "profiled_calls": 1,
            "retain_all_cuda_device_events": True,
        },
        "fresh per-artifact dynamic-work audit drift",
    )
    require(
        campaign.get("gates", {}).get("treatment_integrity") == {
            "field": "held_gemm_source_sha256",
            "off_on_equality_required": True,
            "projection": "generated first-kernel GEMM body and schedule, excluding the second softmax kernel and arm label",
            "sha256_required": True,
        },
        "held-GEMM treatment-integrity contract drift",
    )
    donor = campaign.get("donor", {})
    require(
        donor == {
            "config_id": FIXED_GRID,
            "donor_step_id": DONOR_STEP_ID,
            "from_strategy": OFF_STRATEGY,
            "origin_dsl": "triton",
            "origin_id": ORIGIN_ID,
            "to_strategy": ON_STRATEGY,
        },
        "donor step drift",
    )
    retune = campaign.get("retuning", {})
    require(
        retune.get("attempts_per_destination_and_state") == len(GRID_IDS)
        and retune.get("grid_ids") == list(GRID_IDS)
        and retune.get("origin_bound") is True
        and retune.get("build_failures_consume_attempts") is True
        and retune.get("screen_processes_per_gate_legal_artifact") == SCREEN_REPLICATES
        and retune.get("screen_distribution") == "positive"
        and retune.get("withheld_distribution_may_influence_selection") is False,
        "retune contract drift",
    )
    timing = campaign.get("timing", {})
    require(
        timing.get("blocks") == BLOCKS
        and timing.get("trials") == 100
        and timing.get("primary_trials") == {"start_inclusive": 60, "stop_exclusive": 100}
        and timing.get("fresh_process_per_record") is True,
        "timing contract drift",
    )
    require(
        campaign.get("inference") == INFERENCE_CONTRACT,
        "inference contract drift",
    )
    require(
        campaign.get("sequencing") == {
            "fixed_and_retuned_are_required_estimands": True,
            "outcome_contingent_branching": False,
            "retuned_confirmation_requires_fixed_failure": False,
        },
        "outcome-contingent sequencing is forbidden",
    )
    require(
        mechanism.get("campaign_id") == CAMPAIGN_ID
        and mechanism.get("mechanism_id") == "common_cuda_to_destination_native_softmax"
        and mechanism.get("controls", {}).get("treatment")
        == "replace the common CUDA softmax with the destination-native softmax",
        "mechanism card drift",
    )
    donor_analysis_path = REPO_ROOT / "ako_runs/controlled_followup/native_trajectory_replication_ada_v3/results/analysis.json"
    donor_analysis = read_json(donor_analysis_path)
    donor_effects = {
        row.get("distribution"): row.get("effect", {}).get("direction_above_sham_floor")
        for row in donor_analysis.get("effects", []) if isinstance(row, dict)
        and row.get("lane") == "triton"
        and row.get("strategy_step") == "register_common_to_register_fused_native_strategy_contrast"
    }
    require(
        donor_analysis.get("complete") is True
        and donor_effects == {"positive": "speedup", "withheld_signed": "speedup"},
        "donor selection is not rederived from two-distribution evidence",
    )
    mapping = primitive_map.get("destinations")
    require(isinstance(mapping, dict) and list(sorted(mapping)) == list(sorted(DESTINATIONS)), "primitive map census drift")
    absence = primitive_map.get("shared_receipts", {}).get("cuda_cpp_row_reduction_primitive_absence")
    require(
        isinstance(absence, dict)
        and canonical_sha256(absence) == "edb83b777edb4aadc3b7232353f281c66b4ecded7c25b12680c2491cc2304ef6"
        and absence.get("finding") == "PRIMITIVE_ABSENT"
        and absence.get("reviewed_before_campaign_freeze") is True
        and absence.get("available_lower_level_mechanisms")
        == ["__shfl_down_sync", "shared_memory", "__syncthreads"]
        and isinstance(absence.get("scope_limit"), str)
        and "CUB" in absence["scope_limit"]
        and "global CUDA-ecosystem absence claim" in absence["scope_limit"]
        and isinstance(absence.get("reconstruction_statement"), str)
        and "algorithmically equivalent" in absence["reconstruction_statement"]
        and "not evidence of a newly available destination primitive" in absence["reconstruction_statement"],
        "CUDA primitive-absence receipt drift",
    )
    require(
        primitive_map.get("programming_model_units", {}).get("cuda_cpp", {}).get("destination_lanes")
        == ["cuda_noptx", "cuda_unlimited"],
        "CUDA compiler-policy lanes were relabeled as independent models",
    )
    for destination in DESTINATIONS:
        row = mapping[destination]
        route = ROUTE_BY_DESTINATION[destination]
        direct = row.get("direct_primitive_mapping", {})
        manual = row.get("manual_reconstruction", {})
        review = row.get("review", {})
        require(
            row.get("route") == route
            and row.get("programming_model_unit")
            == ("cuda_cpp" if destination.startswith("cuda_") else destination)
            and isinstance(review.get("source_path"), str)
            and isinstance(review.get("source_sha256"), str)
            and isinstance(review.get("off_gate_record_path"), str)
            and isinstance(review.get("off_gate_record_sha256"), str)
            and isinstance(review.get("on_gate_record_path"), str)
            and isinstance(review.get("on_gate_record_sha256"), str),
            f"{destination}: source review is incomplete",
        )
        selected = row.get(route, {})
        recipe = selected.get("recipe", {})
        off_coordinate = f"{OFF_STRATEGY}.{destination}.g01"
        on_coordinate = f"{ON_STRATEGY}.{destination}.g01"
        on_implementation = (
            "tilelang_abstraction.F1.g01" if destination == "tilelang"
            else on_coordinate
        )
        require(
            selected.get("status") == "ELIGIBLE"
            and selected.get("off_coordinate_cell_id") == off_coordinate
            and selected.get("off_implementation_id") == off_coordinate
            and selected.get("on_coordinate_cell_id") == on_coordinate
            and selected.get("on_implementation_id") == on_implementation
            and recipe.get("adapter_entrypoint")
            == f"ako_runs.controlled_followup.trajectory_transfer_ada_v2.implementations.{destination}.build"
            and recipe.get("source_review_sha256") == canonical_sha256(review)
            and isinstance(recipe.get("authorized_change"), str)
            and isinstance(recipe.get("unchanged_controls"), str),
            f"{destination}: selected route recipe drift",
        )
        if route == "direct_primitive_mapping":
            require(
                direct.get("status") == "ELIGIBLE"
                and recipe.get("kind") == "direct_existing_destination_primitive_mapping"
                and recipe.get("new_destination_kernel_authored") is False
                and manual.get("status") == "NOT_APPLICABLE"
                and manual.get("primitive_absence_receipt_sha256") is None,
                f"{destination}: direct route classification drift",
            )
        else:
            require(
                direct.get("status") == "PRIMITIVE_ABSENT"
                and direct.get("primitive_absence_receipt_sha256") == canonical_sha256(absence)
                and manual.get("primitive_absence_receipt_sha256") == canonical_sha256(absence)
                and recipe.get("kind") == "manual_reconstruction_from_lower_level_primitives"
                and recipe.get("manual_source_path") == absence.get("manual_recipe_source_path")
                and recipe.get("manual_source_sha256") == absence.get("manual_recipe_source_sha256")
                and recipe.get("shared_manual_recipe_id") == "cuda_cpp_shuffle_shared_row_softmax_v1",
                f"{destination}: manual route lacks its primitive-absence/recipe binding",
            )
    require(materials.get("campaign_id") == CAMPAIGN_ID, "material campaign drift")
    files = materials.get("files")
    require(isinstance(files, dict) and files, "material closure is empty")
    require(
        len(files) == 59
        and canonical_sha256(sorted(files)) == MATERIAL_PATH_CENSUS_SHA256,
        "material path census drift",
    )
    missing_runtime = sorted(set(REQUIRED_RUNTIME_MATERIALS) - set(files))
    require(not missing_runtime, f"runtime material closure is incomplete: {missing_runtime}")
    require(
        files.get(absence["manual_recipe_source_path"]) == absence["manual_recipe_source_sha256"]
        and files.get(absence["baseline_source_path"]) == absence["baseline_source_sha256"],
        "CUDA baseline/manual recipe pair is outside the material closure",
    )
    if rehash_materials:
        for relative, expected in files.items():
            path = REPO_ROOT / relative
            require(path.is_file() and file_sha256(path) == expected, f"material changed: {relative}")
    for destination in DESTINATIONS:
        review = mapping[destination]["review"]
        require(
            files.get(review["source_path"]) == review["source_sha256"],
            f"{destination}: reviewed source is outside material closure",
        )
        require(
            files.get(review["off_gate_record_path"]) == review["off_gate_record_sha256"]
            and files.get(review["on_gate_record_path"]) == review["on_gate_record_sha256"],
            f"{destination}: gate review is not material-bound",
        )
        _validate_eligible_gate_record(
            read_json(REPO_ROOT / review["off_gate_record_path"]), destination, OFF_STRATEGY,
        )
        on_record = read_json(REPO_ROOT / review["on_gate_record_path"])
        if destination == "tilelang":
            _validate_tilelang_f1_gate_record(on_record, review)
        else:
            _validate_eligible_gate_record(on_record, destination, ON_STRATEGY)
    tilelang_source = (REPO_ROOT / mapping["tilelang"]["review"]["source_path"]).read_text(encoding="utf-8")
    f1_body = tilelang_source.split("def _f1", 1)[-1].split("def _f2", 1)[0]
    triton_source = (REPO_ROOT / mapping["triton"]["review"]["source_path"]).read_text(encoding="utf-8")
    cuda_source = (REPO_ROOT / absence["manual_recipe_source_path"]).read_text(encoding="utf-8")
    require(
        "T.reduce_max" in f1_body and "T.reduce_sum" in f1_body and "T.shfl_down" not in f1_body,
        "TileLang direct route does not use only its reviewed high-level reduction path",
    )
    require("tl.max" in triton_source and "tl.sum" in triton_source, "Triton direct primitive calls are absent")
    require(
        all(token in cuda_source for token in ("__shfl_down_sync", "__shared__", "__syncthreads")),
        "CUDA manual reconstruction source drift",
    )
    _grid_jobs()


def _primitive_graph_sha256(destination: str, state: str, route: str) -> str:
    require(route == ROUTE_BY_DESTINATION[destination], "primitive graph route differs from destination route")
    return canonical_sha256({
        "destination": destination,
        "route": route,
        "kernels": [
            "destination_native_register_gbg",
            "common_cuda_softmax" if state == "off" else {
                "direct_primitive_mapping": "destination_direct_primitive_softmax",
                "manual_reconstruction": "destination_manual_reconstructed_softmax",
            }[route],
        ],
        "kernel_count": 2,
    })


def _entry(destination: str, adaptation: str, state: str, grid_id: str, origin_job: dict[str, Any]) -> dict[str, Any]:
    enabled = state == "on"
    strategy = ON_STRATEGY if enabled else OFF_STRATEGY
    route = ROUTE_BY_DESTINATION[destination]
    primitive_map = read_json(PRIMITIVE_MAP_PATH)
    absence = (
        primitive_map["shared_receipts"]["cuda_cpp_row_reduction_primitive_absence"]
        if route == "manual_reconstruction" else None
    )
    absence_sha256 = canonical_sha256(absence) if absence is not None else None
    graph = _primitive_graph_sha256(destination, state, route)
    coordinate_cell_id = f"{strategy}.{destination}.{grid_id}"
    implementation_id = (
        f"tilelang_abstraction.F1.{grid_id}"
        if destination == "tilelang" and enabled else coordinate_cell_id
    )
    coordinate = [CAMPAIGN_ID, destination, adaptation, state, grid_id, route]
    entry_id = "tt2_" + canonical_sha256(coordinate)[:24]
    spec = {
        "config_id": grid_id,
        "coordinate_cell_id": coordinate_cell_id,
        "destination": destination,
        "donor_step_id": DONOR_STEP_ID,
        "grid_id": grid_id,
        "implementation_id": implementation_id,
        "origin_id": ORIGIN_ID,
        "origin_job": origin_job,
        "primitive_absence_receipt": absence,
        "primitive_absence_receipt_sha256": absence_sha256,
        "primitive_graph_sha256": graph,
        "route": route,
    }
    return {
        "adaptation": adaptation,
        "attempt_index": 1 if adaptation == "donor_fixed" else GRID_IDS.index(grid_id) + 1,
        "config_id": grid_id,
        "coordinate_cell_id": coordinate_cell_id,
        "destination": destination,
        "donor_step_id": DONOR_STEP_ID,
        "entry_id": entry_id,
        "grid_id": grid_id,
        "implementation_id": implementation_id,
        "mechanism_enabled": enabled,
        "mechanism_state": state,
        "origin_id": ORIGIN_ID,
        "origin_job": origin_job,
        "primitive_absence_receipt": absence,
        "primitive_absence_receipt_sha256": absence_sha256,
        "primitive_graph_sha256": graph,
        "route": route,
        "row_id": entry_id,
        "spec": spec,
    }


def make_admission_manifest() -> dict[str, Any]:
    validate_contract()
    jobs = _grid_jobs()
    rows: list[dict[str, Any]] = []
    for destination in DESTINATIONS:
        for state in MECHANISM_STATES:
            rows.append(_entry(destination, "donor_fixed", state, FIXED_GRID, jobs[destination][FIXED_GRID]))
            rows.extend(
                _entry(destination, "bounded_retune", state, grid_id, jobs[destination][grid_id])
                for grid_id in GRID_IDS
            )
    require(len(rows) == 160 and len({row["entry_id"] for row in rows}) == 160, "admission census/id drift")
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "expected_rows": len(rows),
        "record_type": "trajectory_transfer_ada_v2_admission_manifest",
        "rows": rows,
        "schema_version": 1,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _validate_admitted_ids(admission: dict[str, Any], admitted_entry_ids: Iterable[str] | None) -> set[str]:
    bounded = {row["entry_id"] for row in admission["rows"] if row["adaptation"] == "bounded_retune"}
    if admitted_entry_ids is None:
        return bounded
    selected = set(admitted_entry_ids)
    require(selected <= bounded, "screen selection contains a non-retuned admission entry")
    return selected


def make_screen_manifest(admitted_entry_ids: Iterable[str] | None = None) -> dict[str, Any]:
    admission = make_admission_manifest()
    admitted = _validate_admitted_ids(admission, admitted_entry_ids)
    rows: list[dict[str, Any]] = []
    for entry in admission["rows"]:
        if entry["entry_id"] not in admitted:
            continue
        for replicate in range(SCREEN_REPLICATES):
            coordinate = [entry["entry_id"], "positive", replicate]
            rows.append({
                "adaptation": "bounded_retune",
                "destination": entry["destination"],
                "distribution": "positive",
                "entry_id": entry["entry_id"],
                "mechanism_enabled": entry["mechanism_enabled"],
                "mechanism_state": entry["mechanism_state"],
                "record_id": "tts_" + canonical_sha256(coordinate)[:24],
                "replicate": replicate,
            })
    rows.sort(key=lambda row: canonical_sha256([RANDOMIZATION_SEED, row]))
    manifest = {
        "admission_manifest_sha256": admission["manifest_sha256"],
        "campaign_id": CAMPAIGN_ID,
        "expected_records": len(rows),
        "record_type": "trajectory_transfer_ada_v2_screen_manifest",
        "rows": rows,
        "schema_version": 1,
        "selection_distribution": "positive",
        "withheld_distribution_used": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_selection(selection: Any, admission: dict[str, Any] | None = None) -> dict[str, str]:
    require(isinstance(selection, dict), "selection must be an object")
    require(set(selection) == set(DESTINATIONS), "selection destination census drift")
    admission = admission or make_admission_manifest()
    by_id = {row["entry_id"]: row for row in admission["rows"]}
    normalized: dict[str, str] = {}
    for destination in DESTINATIONS:
        states = selection[destination]
        require(isinstance(states, dict) and set(states) == set(MECHANISM_STATES), f"{destination}: selection state census drift")
        for state in MECHANISM_STATES:
            entry_id = states[state]
            row = by_id.get(entry_id)
            require(
                row is not None
                and row["destination"] == destination
                and row["mechanism_state"] == state
                and row["adaptation"] == "bounded_retune",
                f"{destination}/{state}: invalid selected entry",
            )
            normalized[f"{destination}:{state}"] = entry_id
    return normalized


def make_confirmation_manifest(selection: Any) -> dict[str, Any]:
    admission = make_admission_manifest()
    selected = validate_selection(selection, admission)
    by_id = {row["entry_id"]: row for row in admission["rows"]}
    fixed = {
        f"{row['destination']}:{row['mechanism_state']}": row["entry_id"]
        for row in admission["rows"] if row["adaptation"] == "donor_fixed"
    }
    rows: list[dict[str, Any]] = []
    for block in range(BLOCKS):
        members: list[dict[str, Any]] = []
        for adaptation, entries in (("donor_fixed", fixed), ("bounded_retune", selected)):
            for destination in DESTINATIONS:
                for state in MECHANISM_STATES:
                    entry_id = entries[f"{destination}:{state}"]
                    entry = by_id[entry_id]
                    for distribution in DISTRIBUTIONS:
                        members.append({
                            "adaptation": adaptation,
                            "destination": destination,
                            "distribution": distribution,
                            "entry_id": entry_id,
                            "label": f"{adaptation}.{state}",
                            "mechanism_state": state,
                            "record_kind": "candidate",
                        })
        for destination in DESTINATIONS:
            sham_entry = by_id[fixed[f"{destination}:off"]]
            for label in SHAM_LABELS:
                for distribution in DISTRIBUTIONS:
                    members.append({
                        "adaptation": "donor_fixed",
                        "destination": destination,
                        "distribution": distribution,
                        "entry_id": sham_entry["entry_id"],
                        "label": label,
                        "mechanism_state": "off",
                        "record_kind": "same_artifact_label_sham",
                    })
        members.sort(key=lambda row: canonical_sha256([RANDOMIZATION_SEED, block, row]))
        for block_position, row in enumerate(members, 1):
            coordinate = {**row, "block": block, "block_position": block_position}
            rows.append({**coordinate, "record_id": "ttc_" + canonical_sha256(coordinate)[:24]})
    expected = BLOCKS * (2 * len(DESTINATIONS) * len(MECHANISM_STATES) * len(DISTRIBUTIONS)
                         + len(DESTINATIONS) * len(SHAM_LABELS) * len(DISTRIBUTIONS))
    require(expected == 720 and len(rows) == expected, "confirmation census drift")
    manifest = {
        "admission_manifest_sha256": admission["manifest_sha256"],
        "campaign_id": CAMPAIGN_ID,
        "expected_records": expected,
        "record_type": "trajectory_transfer_ada_v2_confirmation_manifest",
        "rows": rows,
        "schema_version": 1,
        "selection": selection,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def material_projection() -> dict[str, Any]:
    validate_contract()
    documents = {
        path.name: file_sha256(path)
        for path in (CAMPAIGN_PATH, MECHANISM_CARD_PATH, PRIMITIVE_MAP_PATH, MATERIALS_PATH)
    }
    return {
        "campaign_id": CAMPAIGN_ID,
        "documents": documents,
        "materials_sha256": file_sha256(MATERIALS_PATH),
        "record_type": "trajectory_transfer_ada_v2_material_projection",
        "schema_version": 1,
    }


def source_projection() -> dict[str, str]:
    sources: dict[str, str] = {}
    for relative in SOURCE_RELATIVES:
        path = HERE / relative
        require(path.is_file() and not path.is_symlink(), f"missing campaign source: {relative}")
        sources[relative] = file_sha256(path)
    return sources


def make_campaign_lock() -> dict[str, Any]:
    admission = make_admission_manifest()
    projection = material_projection()
    sources = source_projection()
    lock = {
        "admission_manifest_sha256": admission["manifest_sha256"],
        "campaign_id": CAMPAIGN_ID,
        "gpu_binding": read_json(CAMPAIGN_PATH)["hardware"],
        "material_projection": projection,
        "record_type": "trajectory_transfer_ada_v2_campaign_lock",
        "schema_version": 1,
        "source_bundle_sha256": canonical_sha256(sources),
        "source_sha256": sources,
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    return lock


def prepare() -> None:
    validate_contract()
    write_if_absent_or_exact_json(ADMISSION_MANIFEST_PATH, make_admission_manifest())


def freeze() -> None:
    prepare()
    write_if_absent_or_exact_json(CAMPAIGN_LOCK_PATH, make_campaign_lock())


def check_frozen() -> None:
    validate_contract()
    require(read_json(ADMISSION_MANIFEST_PATH) == make_admission_manifest(), "admission manifest drift")
    require(read_json(CAMPAIGN_LOCK_PATH) == make_campaign_lock(), "campaign lock drift")


def validate_terminal_outcome(value: Any) -> str:
    require(value in TERMINAL_STATUSES, "unknown terminal status")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "freeze", "check-frozen"))
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            validate_contract()
        elif args.command == "prepare":
            prepare()
        elif args.command == "freeze":
            freeze()
        else:
            check_frozen()
        return 0
    except ProtocolError as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
