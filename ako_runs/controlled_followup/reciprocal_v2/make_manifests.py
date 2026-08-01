#!/usr/bin/env python3
"""Generate the byte-deterministic reciprocal-v2 cards and manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import protocol


HERE = protocol.HERE
OLD = HERE.parent / "reciprocal"
PHASE1 = protocol.REPO_ROOT / "ako_runs/phase1_matmul"
CUDA_SOURCE = PHASE1 / "variants/cuda_unlimited_gemm.py"
CUDA_RECORD = PHASE1 / (
    "results/native_tuned/raw/"
    "cuda_unlimited__D__primary__BM128_BN128_BK32_stages3_kc2048__rand__rep0.json"
)
CUDA_INSPECTION = PHASE1 / "results/code_inspection.json"

CUDA_CARD = HERE / "recipe_cards/cuda_unlimited_native_d.json"
AUDIT_MANIFEST = HERE / "manifests/audit.json"
PRIMARY_MANIFEST = HERE / "manifests/primary.json"
SUMMARY = HERE / "manifests/summary.json"

ORIGIN_FILES = {
    "tilelang_phase1_confirmed": OLD / "recipe_cards/tilelang_phase1_confirmed.json",
    "triton_grouped_autotuned": OLD / "recipe_cards/triton_grouped_autotuned.json",
    "cuda_unlimited_native_d": CUDA_CARD,
}


def _evidence(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": protocol.repo_path(path),
        "sha256": protocol.file_sha256(path),
        "role": role,
    }


def cuda_card() -> dict[str, Any]:
    record = protocol.load_json(CUDA_RECORD)
    expected_key = "cuda_unlimited.D.128x128x32.kc2048.s3.fp16.precast"
    if record.get("key") != expected_key:
        raise ValueError("Phase-1 CUDA donor record no longer names the frozen recipe")
    config = record.get("cfg")
    if not isinstance(config, dict):
        raise ValueError("Phase-1 CUDA donor record lacks config")
    wanted = {"BM": 128, "BN": 128, "BK": 32, "kc": 2048, "stages": 3}
    if any(config.get(key) != value for key, value in wanted.items()):
        raise ValueError(f"Phase-1 CUDA donor config drift: {config!r}")
    return {
        "schema_version": 2,
        "recipe_id": "cuda_unlimited_native_d",
        "card_kind": "frozen_source_recipe",
        "workload": {
            "operation": "matmul",
            "shape": {"M": 2048, "K": 8192, "N": 4096},
            "operand_storage": "fp16_precast_outside_timed_region",
            "output_dtype": "fp32",
        },
        "origin": {
            "dsl": "cuda_unlimited",
            "historical_key": expected_key,
            "implementation": protocol.repo_path(CUDA_SOURCE),
        },
        "algorithm": {
            "BM": 128,
            "BN": 128,
            "BK": 32,
            "stages": 3,
            "threads": int(config.get("threads", 256)),
            "grid_mapping": "plain_2d_n_then_m",
            "tensor_core": True,
            "tensor_accumulator_dtype": "fp32",
            "outer_accumulator_dtype": "fp32",
            "cross_block_split_k": False,
            "inline_ptx_permitted_at_origin": True,
            "vendor_gemm": False,
        },
        "correctness_amendment": {
            "kind": "in_block_accumulator_flush_only",
            "historical_kc": 2048,
            "candidate_kc_order": list(protocol.KC_LADDER),
            "selection": "largest_common_all_destination_all_translator_v4_pass",
            "no_other_literal_change_permitted": True,
        },
        "literal_treatment": {
            "configs": [
                {
                    "BM": 128,
                    "BN": 128,
                    "BK": 32,
                    "stages": 3,
                    "threads": int(config.get("threads", 256)),
                    "kc": 2048,
                }
            ],
            "preserve": [
                "plain_2d_n_then_m_mapping",
                "tensor_core_fragment_mapping",
                "pipeline_depth",
                "thread_count",
                "in_block_chunk_then_fp32_outer_accumulate",
            ],
        },
        "retuned_treatment": {
            "attempted_candidates": protocol.RETUNE_ATTEMPTS,
            "build_failures_consume_budget": True,
            "preserve": [
                "plain_2d_n_then_m_mapping",
                "fp16_tensor_core_operands",
                "fp32_chunk_and_outer_accumulators",
                "no_cross_block_split_k",
                "no_vendor_gemm",
            ],
        },
        "evidence": [
            _evidence(CUDA_SOURCE, "parameterized Phase-1 implementation"),
            _evidence(CUDA_RECORD, "frozen native-search donor record"),
            _evidence(CUDA_INSPECTION, "generated-code inspection registry"),
        ],
    }


def _origin_binding(origin: str, generated_card: dict[str, Any]) -> dict[str, Any]:
    path = ORIGIN_FILES[origin]
    data = (
        protocol.stable_json_bytes(generated_card)
        if origin == "cuda_unlimited_native_d"
        else path.read_bytes()
    )
    return {
        "recipe_id": origin,
        "path": protocol.repo_path(path),
        "sha256": protocol.canonical_sha256(json.loads(data)),
        "file_sha256": __import__("hashlib").sha256(data).hexdigest(),
    }


def manifest(kind: str, generated_card: dict[str, Any]) -> dict[str, Any]:
    if kind not in protocol.KINDS:
        raise ValueError(kind)
    origins = {
        origin: _origin_binding(origin, generated_card) for origin in protocol.ORIGINS
    }
    jobs = []
    for ordinal, (origin, destination, mode, translator) in enumerate(
        protocol.expected_cells()
    ):
        cid = protocol.cell_id(origin, destination, mode, translator)
        jobs.append(
            {
                "ordinal": ordinal,
                "cell_id": cid,
                "job_id": f"reciprocal_v2.{kind}.{cid}",
                "recipe_origin": origin,
                "destination_dsl": destination,
                "transfer_mode": mode,
                "translator": translator,
                "recipe_card": origins[origin],
                "translator_source_root": f"translators/{translator}",
                "implementation_registry": "dependencies/implementation_registry.json",
                "retune_plan": (
                    f"retune_plans/{translator}/{origin}__to__{destination}.json"
                    if mode == "retuned"
                    else None
                ),
                "candidate_attempt_budget": (
                    protocol.RETUNE_ATTEMPTS if mode == "retuned" else 1
                ),
                "build_failures_consume_budget": True,
                "gate_in_loop": {
                    "gate": "matmul_v4",
                    "kc_ladder": list(protocol.KC_LADDER),
                    "threshold_changes_permitted": False,
                    "performance_inputs_visible": False,
                },
                "audit_receipt": f"results/audit/receipts/{cid}.json",
                "execution": (
                    {
                        "phase": "source_and_legality_audit",
                        "required_attempt_records": (
                            protocol.RETUNE_ATTEMPTS if mode == "retuned" else 1
                        ),
                    }
                    if kind == "audit"
                    else {
                        "phase": "same_gpu_randomized_confirmation",
                        "screen_repetitions": protocol.SCREEN_REPS,
                        "confirm_top_k": 2 if mode == "retuned" else 1,
                        "confirmation_blocks": protocol.CONFIRM_REPS,
                        "timing_gpu": 0,
                        "timing_distribution": "rand_seed0_precast",
                    }
                ),
            }
        )
    return {
        "schema_version": 2,
        "campaign_id": protocol.CAMPAIGN_ID,
        "manifest_kind": kind,
        "status": "preregistered_not_launched",
        "factor_order": [
            "recipe_origin",
            "destination_dsl",
            "transfer_mode",
            "translator",
        ],
        "factor_levels": {
            "recipe_origin": list(protocol.ORIGINS),
            "destination_dsl": list(protocol.DESTINATIONS),
            "transfer_mode": list(protocol.TRANSFER_MODES),
            "translator": list(protocol.TRANSLATORS),
        },
        "job_count": len(jobs),
        "jobs": jobs,
        "robust_gate": {
            "spec": protocol.repo_path(protocol.GATE_SPEC),
            "summary": protocol.repo_path(protocol.GATE_SUMMARY),
            "acceptance_receipt": protocol.repo_path(protocol.GATE_RECEIPT),
            "lock": "dependencies/gate_lock.json",
        },
        "required_locks": [
            "dependencies/recipe_resolution_lock.json",
            "dependencies/implementation_registry.json",
            "dependencies/prelaunch_provenance.json",
        ],
        "analysis": {
            "primary_estimands": [
                "destination_effect_within_origin_mode_translator",
                "origin_effect_within_destination_mode_translator",
                "origin_by_destination_interaction",
                "translator_bound_for_each_cell",
            ],
            "performance_repetitions": protocol.CONFIRM_REPS,
            "paired_randomized_blocks": True,
            "claim_limit": "finite_recipe_budget_on_one_shape_one_Ada_host",
        },
    }


def documents() -> dict[Path, bytes]:
    card = cuda_card()
    audit = manifest("audit", card)
    primary = manifest("primary", card)
    summary = {
        "schema_version": 2,
        "campaign_id": protocol.CAMPAIGN_ID,
        "origins": len(protocol.ORIGINS),
        "destinations": len(protocol.DESTINATIONS),
        "transfer_modes": len(protocol.TRANSFER_MODES),
        "translators": len(protocol.TRANSLATORS),
        "audit_cells": audit["job_count"],
        "primary_cells": primary["job_count"],
        "total_cells": audit["job_count"] + primary["job_count"],
        "retune_attempts_per_retuned_cell": protocol.RETUNE_ATTEMPTS,
        "confirmation_blocks": protocol.CONFIRM_REPS,
    }
    return {
        CUDA_CARD: protocol.stable_json_bytes(card),
        AUDIT_MANIFEST: protocol.stable_json_bytes(audit),
        PRIMARY_MANIFEST: protocol.stable_json_bytes(primary),
        SUMMARY: protocol.stable_json_bytes(summary),
    }


def write() -> None:
    for path, data in documents().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def check() -> None:
    stale = [
        protocol.repo_path(path)
        for path, data in documents().items()
        if not path.is_file() or path.read_bytes() != data
    ]
    if stale:
        raise SystemExit("stale generated documents: " + ", ".join(stale))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
