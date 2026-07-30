#!/usr/bin/env python3
"""Generate the frozen reciprocal-transfer recipe cards and job manifests.

The documents are derived from checked-in evidence, contain no host names or
timestamps, and are byte deterministic.  This module is CPU-only: it parses
Python source with :mod:`ast` and never imports Torch, Triton, or TileLang.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PHASE1 = REPO_ROOT / "ako_runs/phase1_matmul"

RECIPE_SCHEMA = HERE / "schemas/recipe_card.schema.json"
TILELANG_CARD = HERE / "recipe_cards/tilelang_phase1_confirmed.json"
TRITON_CARD = HERE / "recipe_cards/triton_grouped_autotuned.json"
PRIMARY_MANIFEST = HERE / "manifests/primary.json"
AUDIT_MANIFEST = HERE / "manifests/audit.json"
SUMMARY = HERE / "manifests/summary.json"

TILELANG_IMPL = PHASE1 / "variants/tilelang_gemm.py"
TRITON_SCHEDULE_IMPL = (
    REPO_ROOT
    / "ako_runs/matmul_gelu_softmax/triton/solution/matmul_gelu_softmax.py"
)
TRITON_CONTRACT_IMPL = PHASE1 / "variants/triton_gemm.py"
CONFIRM_JOBS = PHASE1 / "jobs/confirm.json"
PHASE1_SPEC = PHASE1 / "variants/SPEC.md"
NATIVE_GRID = PHASE1 / "jobs/native_tuned.json"

SCHEMA_VERSION = 1
CAMPAIGN_ID = "standard-matmul-reciprocal-transfer-v1"
GATE_SPEC_RELATIVE = "../robust_gate/calibration/gate_spec_matmul_v4.json"
GATE_VALIDATION_SUMMARY_RELATIVE = (
    "../robust_gate/validation/matmul_holdout_summary_v4.json"
)
GATE_ACCEPTANCE_RECEIPT_RELATIVE = (
    "../robust_gate/validation/v4_acceptance_receipt.json"
)
WORKLOAD = {
    "id": "phase1_standard_matmul_m2048_k8192_n4096",
    "operation": "matmul",
    "shape": {"M": 2048, "K": 8192, "N": 4096},
    "semantic_input_dtype": "fp32_from_robust_gate",
    "input_storage_contract": "fp16_operands_precast_outside_timed_kernel_region",
    "output_dtype": "fp32",
}
ORIGINS = ("tilelang_phase1_confirmed", "triton_grouped_autotuned")
DESTINATIONS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
TRANSFER_MODES = ("literal", "retuned")


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def parse_set(value: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for item in value.split(","):
        key, raw = item.split("=", 1)
        parsed[key.strip()] = int(raw.strip())
    return parsed


def tilelang_confirmed_config() -> dict[str, int]:
    """Read the first confirmed TileLang winner from Phase 1's frozen job list."""
    jobs = json.loads(CONFIRM_JOBS.read_text(encoding="utf-8"))
    matches = [
        job
        for job in jobs
        if job.get("dsl") == "tilelang" and job.get("variant") == "D"
    ]
    if len(matches) < 1:
        raise ValueError("Phase-1 confirm jobs contain no TileLang D winner")
    config = parse_set(matches[0]["set"])
    expected = {"BM", "BN", "BK", "stages", "kc"}
    if set(config) != expected:
        raise ValueError(f"unexpected TileLang winner fields: {config!r}")
    config["threads"] = 256
    return config


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def triton_native_configs() -> list[dict[str, int]]:
    """Extract the ordered native autotune family without importing Triton."""
    tree = ast.parse(TRITON_SCHEDULE_IMPL.read_text(encoding="utf-8"))
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_matmul_gelu_fp16_kernel"
        ),
        None,
    )
    if target is None:
        raise ValueError("Triton donor kernel _matmul_gelu_fp16_kernel is missing")
    autotune = next(
        (
            decorator
            for decorator in target.decorator_list
            if isinstance(decorator, ast.Call)
            and _call_name(decorator.func).endswith(".autotune")
        ),
        None,
    )
    if autotune is None:
        raise ValueError("Triton donor kernel no longer has @triton.autotune")
    keywords = {keyword.arg: keyword.value for keyword in autotune.keywords}
    config_nodes = keywords.get("configs")
    key_node = keywords.get("key")
    if not isinstance(config_nodes, (ast.List, ast.Tuple)):
        raise ValueError("Triton donor autotune configs are not a literal sequence")
    if ast.literal_eval(key_node) != ["M", "N", "K"]:
        raise ValueError("Triton donor autotune key changed")

    configs: list[dict[str, int]] = []
    for index, node in enumerate(config_nodes.elts):
        if not isinstance(node, ast.Call) or not _call_name(node.func).endswith(
            ".Config"
        ):
            raise ValueError(f"Triton donor config {index} is not triton.Config")
        if len(node.args) != 1:
            raise ValueError(f"Triton donor config {index} has unexpected arguments")
        values = ast.literal_eval(node.args[0])
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        config = {
            "BM": int(values["BLOCK_M"]),
            "BN": int(values["BLOCK_N"]),
            "BK": int(values["BLOCK_K"]),
            "group_m": int(values["GROUP_M"]),
            "stages": int(kwargs["num_stages"]),
            "warps": int(kwargs["num_warps"]),
        }
        configs.append(config)
    if len(configs) != 13:
        raise ValueError(f"expected 13 Triton donor configs, found {len(configs)}")
    if any(config["group_m"] != 8 for config in configs):
        raise ValueError("Triton donor family no longer uses GROUP_M=8 throughout")
    if len({tuple(sorted(config.items())) for config in configs}) != len(configs):
        raise ValueError("Triton donor family contains a duplicate configuration")
    return configs


def evidence(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": repo_path(path),
        "role": role,
        "sha256": sha256_file(path),
    }


def _common_retune(origin_mapping: str) -> dict[str, Any]:
    return {
        "candidate_budget": {
            "attempted_candidates": 19,
            "build_failures_consume_budget": True,
            "source_rationale": repo_path(NATIVE_GRID),
            "source_sha256": sha256_file(NATIVE_GRID),
        },
        "plan_contract": {
            "kind": "recipient_native_frozen_before_screening",
            "required_axes": ["BM", "BN", "BK", "stages", "threads_or_warps"],
            "plan_must_be_content_addressed": True,
            "plan_must_not_use_performance_split": True,
        },
        "preserve": [
            "fp16_tensor_core_operands",
            "fp32_chunk_accumulator",
            "fp32_outer_accumulator",
            "in_block_k_chunk_flush",
            origin_mapping,
            "no_cross_block_split_k",
            "no_vendor_gemm",
        ],
        "selection": {
            "screening_repetitions": 2,
            "confirm_top_k": 2,
            "confirmation_repetitions": 5,
            "selection_split": "robust_gate.tuning",
            "performance_split_hidden_until_winner_freeze": True,
        },
    }


def tilelang_recipe_card() -> dict[str, Any]:
    config = tilelang_confirmed_config()
    return {
        "$schema": "../schemas/recipe_card.schema.json",
        "schema_version": SCHEMA_VERSION,
        "recipe_id": ORIGINS[0],
        "card_kind": "frozen_source_recipe",
        "launch_state": "blocked_until_robust_gate_and_bindings_are_frozen",
        "workload": WORKLOAD,
        "origin": {
            "dsl": "tilelang",
            "artifact": repo_path(TILELANG_IMPL),
            "artifact_role": "phase1_confirmed_native_tuned_winner",
        },
        "algorithm": {
            "operand_dtype": "fp16",
            "tensor_core": True,
            "tensor_accumulator_dtype": "fp32",
            "outer_accumulator_dtype": "fp32",
            "output_dtype": "fp32",
            "k_reduction": "in_block_chunk_then_fp32_outer_accumulate",
            "grid_mapping": "plain_2d_n_then_m",
            "cross_block_split_k": False,
            "vendor_gemm": False,
        },
        "correctness_amendment": {
            "kind": "none_donor_already_has_chunk_flush",
            "kc": config["kc"],
            "gate_disposition": "literal_cell_fails_if_frozen_gate_rejects_kc",
        },
        "literal_treatment": {
            "schedule_kind": "single_configuration",
            "configs": [config],
            "frozen_fields": ["BM", "BN", "BK", "stages", "threads", "kc"],
            "translation_rule": (
                "Translate primitives only; preserve the exact tile, stage depth, "
                "thread count, plain 2-D mapping, and accumulator-flush structure."
            ),
        },
        "retuned_treatment": _common_retune("plain_2d_n_then_m_mapping"),
        "gate_dependency": {
            "operation": "matmul",
            "gate_spec": GATE_SPEC_RELATIVE,
            "validation_summary": GATE_VALIDATION_SUMMARY_RELATIVE,
            "acceptance_receipt": GATE_ACCEPTANCE_RECEIPT_RELATIVE,
            "local_lock": "dependencies/gate_lock.json",
            "required_state": "frozen",
        },
        "evidence": [
            evidence(PHASE1_SPEC, "matched arithmetic and work contract"),
            evidence(TILELANG_IMPL, "parameterized TileLang implementation"),
            evidence(CONFIRM_JOBS, "ordered confirmed-winner job list"),
            evidence(
                PHASE1 / "results/confirm/summary.json",
                "five-process confirmation records",
            ),
        ],
    }


def triton_recipe_card() -> dict[str, Any]:
    return {
        "$schema": "../schemas/recipe_card.schema.json",
        "schema_version": SCHEMA_VERSION,
        "recipe_id": ORIGINS[1],
        "card_kind": "frozen_source_recipe",
        "launch_state": "blocked_until_robust_gate_and_bindings_are_frozen",
        "workload": WORKLOAD,
        "origin": {
            "dsl": "triton",
            "artifact": repo_path(TRITON_SCHEDULE_IMPL),
            "artifact_role": "native_fp16_grouped_autotuned_gemm_subrecipe",
            "operation_projection": [
                "retain the fp16 tensor-core GEMM through its fp32 accumulator",
                "remove bias, GELU, softmax, and fp16 intermediate storage",
                "store the raw standard-GEMM result as fp32",
            ],
        },
        "algorithm": {
            "operand_dtype": "fp16",
            "tensor_core": True,
            "tensor_accumulator_dtype": "fp32",
            "outer_accumulator_dtype": "fp32",
            "output_dtype": "fp32",
            "k_reduction": "full_k_in_donor; gate_selected_in_block_chunk_amendment",
            "grid_mapping": "one_dimensional_grouped_m",
            "group_m": 8,
            "autotune_key": ["M", "N", "K"],
            "cross_block_split_k": False,
            "vendor_gemm": False,
        },
        "correctness_amendment": {
            "kind": "in_block_accumulator_flush_only",
            "candidate_kc_order": [8192, 4096, 2048, 1024, 512],
            "selection_rule": (
                "After the robust gate is frozen, choose the first (largest) KC "
                "passing every tuning case, once at origin level; bind that same "
                "KC in every destination before any performance input is opened."
            ),
            "same_value_across_destinations": True,
            "resolution_lock": "dependencies/recipe_resolution_lock.json",
            "no_other_donor_change_permitted": True,
        },
        "literal_treatment": {
            "schedule_kind": "ordered_native_autotune_family",
            "configs": triton_native_configs(),
            "frozen_fields": [
                "ordered_configs",
                "BM",
                "BN",
                "BK",
                "group_m",
                "stages",
                "warps",
            ],
            "translation_rule": (
                "Run the exact ordered 13-point family and GROUP_M=8 work mapping "
                "offline in every destination; DSL-native autotune machinery may "
                "execute it but may not add, delete, reorder, or rewrite points."
            ),
        },
        "retuned_treatment": _common_retune("grouped_m_mapping_with_group_m_8"),
        "gate_dependency": {
            "operation": "matmul",
            "gate_spec": GATE_SPEC_RELATIVE,
            "validation_summary": GATE_VALIDATION_SUMMARY_RELATIVE,
            "acceptance_receipt": GATE_ACCEPTANCE_RECEIPT_RELATIVE,
            "local_lock": "dependencies/gate_lock.json",
            "required_state": "frozen",
        },
        "evidence": [
            evidence(
                TRITON_SCHEDULE_IMPL,
                "ordered fp16 GROUP_M=8 native autotune family",
            ),
            evidence(
                TRITON_CONTRACT_IMPL,
                "controlled fp16 chunk-flush realization used for the sole amendment",
            ),
            evidence(PHASE1_SPEC, "standard-GEMM arithmetic and output contract"),
        ],
    }


def recipe_cards() -> dict[str, dict[str, Any]]:
    cards = {
        ORIGINS[0]: tilelang_recipe_card(),
        ORIGINS[1]: triton_recipe_card(),
    }
    if tuple(cards) != ORIGINS:
        raise AssertionError("recipe-card order drifted")
    return cards


def _requirements(kind: str, origin: str, destination: str, mode: str) -> list[str]:
    cell = f"{origin}.to.{destination}.{mode}"
    requirements = [
        "dependencies/gate_lock.json",
        "dependencies/recipe_resolution_lock.json",
        "implementations/registry.json",
        f"implementations/{origin}/to_{destination}/{mode}.py",
    ]
    if mode == "retuned":
        requirements.append(f"retune_plans/{origin}/to_{destination}.json")
    if kind == "primary":
        requirements.append(f"audit_receipts/{cell}.json")
    return requirements


def _protocol(kind: str) -> dict[str, Any]:
    if kind == "primary":
        return {
            "purpose": "independent_performance_confirmation",
            "input_split": "robust_gate.performance",
            "process_repetitions": 5,
            "trials_per_process": 100,
            "warmup_seconds": 2.0,
            "l2_thrash_between_trials": True,
            "one_variant_per_process": True,
            "randomized_process_order_seed": 20260729,
            "reporting_unit": "median_of_process_medians_with_process_bootstrap_ci",
        }
    if kind == "audit":
        return {
            "purpose": "translation_fidelity_and_dynamic_work_audit",
            "input_split": "robust_gate.validation",
            "process_repetitions": 1,
            "checks": [
                "exact_recipe_and_source_hashes",
                "robust_gate_all_cases",
                "one_hot_work_mapping",
                "dynamic_tensor_core_work_count",
                "no_cross_block_split_k_or_atomics",
                "no_vendor_gemm_calls",
                "literal_config_family_fidelity",
                "retune_plan_and_budget_fidelity",
                "sass_ptx_resource_capture",
            ],
        }
    raise ValueError(kind)


def manifest(kind: str, cards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if kind not in ("primary", "audit"):
        raise ValueError(kind)
    card_files = {
        ORIGINS[0]: TILELANG_CARD,
        ORIGINS[1]: TRITON_CARD,
    }
    card_hashes = {
        origin: sha256_bytes(stable_json_bytes(cards[origin])) for origin in ORIGINS
    }
    jobs: list[dict[str, Any]] = []
    for origin in ORIGINS:
        for destination in DESTINATIONS:
            for mode in TRANSFER_MODES:
                cell = f"{origin}.to.{destination}.{mode}"
                jobs.append(
                    {
                        "ordinal": len(jobs),
                        "job_id": f"{kind}.{cell}",
                        "cell_id": cell,
                        "manifest_kind": kind,
                        "operation": "matmul",
                        "workload_id": WORKLOAD["id"],
                        "recipe_origin": origin,
                        "destination_dsl": destination,
                        "transfer_mode": mode,
                        "recipe_card": card_files[origin]
                        .relative_to(HERE)
                        .as_posix(),
                        "recipe_card_sha256": card_hashes[origin],
                        "required_files": _requirements(
                            kind, origin, destination, mode
                        ),
                    }
                )
    protocol = _protocol(kind)
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "manifest_kind": kind,
        "launch_state": "blocked_pending_frozen_dependencies",
        "workload": WORKLOAD,
        "axes": {
            "recipe_origin": list(ORIGINS),
            "destination_dsl": list(DESTINATIONS),
            "transfer_mode": list(TRANSFER_MODES),
        },
        "axis_cardinality": {
            "recipe_origin": len(ORIGINS),
            "destination_dsl": len(DESTINATIONS),
            "transfer_mode": len(TRANSFER_MODES),
        },
        "job_count": len(jobs),
        "jobs_sha256": sha256_bytes(stable_json_bytes(jobs)),
        "recipe_card_sha256": card_hashes,
        "robust_gate_dependency": {
            "gate_spec": GATE_SPEC_RELATIVE,
            "validation_summary": GATE_VALIDATION_SUMMARY_RELATIVE,
            "acceptance_receipt": GATE_ACCEPTANCE_RECEIPT_RELATIVE,
            "local_lock": "dependencies/gate_lock.json",
            "required_hashes": [
                "manifest_sha256",
                "calibration_records_sha256",
                "gate_spec_sha256",
                "gate_spec_file_sha256",
                "validation_summary_sha256",
                "acceptance_receipt_sha256",
            ],
            "required_operation": "matmul",
            "required_state": "frozen",
        },
        "protocol": protocol,
        "jobs": jobs,
    }


def build_documents() -> dict[Path, bytes]:
    cards = recipe_cards()
    primary = manifest("primary", cards)
    audit = manifest("audit", cards)
    documents = {
        TILELANG_CARD: stable_json_bytes(cards[ORIGINS[0]]),
        TRITON_CARD: stable_json_bytes(cards[ORIGINS[1]]),
        PRIMARY_MANIFEST: stable_json_bytes(primary),
        AUDIT_MANIFEST: stable_json_bytes(audit),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "launch_state": "blocked_pending_frozen_dependencies",
        "documents": {
            path.relative_to(HERE).as_posix(): sha256_bytes(data)
            for path, data in documents.items()
        },
        "primary_jobs": primary["job_count"],
        "audit_jobs": audit["job_count"],
        "total_cells": primary["job_count"] + audit["job_count"],
    }
    documents[SUMMARY] = stable_json_bytes(summary)
    return documents


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def check_documents(verbose: bool = True) -> None:
    for path, expected in build_documents().items():
        if not path.is_file():
            raise SystemExit(f"missing generated document: {repo_path(path)}")
        if path.read_bytes() != expected:
            raise SystemExit(
                f"stale generated document: {repo_path(path)}; "
                "run reciprocal/make_manifests.py"
            )
    if verbose:
        print(
            "OK reciprocal documents: 2 origins x 4 destinations x 2 treatments "
            "= 16 primary + 16 audit cells"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="compare checked-in files without writing"
    )
    args = parser.parse_args()
    if args.check:
        check_documents()
        return 0
    for path, data in build_documents().items():
        atomic_write(path, data)
        print(f"wrote {repo_path(path)} sha256={sha256_bytes(data)}")
    check_documents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
