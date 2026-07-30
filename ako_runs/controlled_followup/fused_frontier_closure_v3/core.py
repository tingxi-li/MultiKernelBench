#!/usr/bin/env python3
"""Pure protocol, serialization, planning, and statistics helpers for v3."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ako_runs.controlled_followup.fused_closure_v2 import core as base


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CAMPAIGN_PATH = HERE / "campaign.json"
ELIGIBILITY_RECEIPT_PATH = HERE / "eligibility_receipt.json"
SOURCE_RECEIPT_PATH = HERE / "source_receipt.json"
RESULTS_ROOT = HERE / "results"

ClosureError = base.ClosureError
stable_json_bytes = base.stable_json_bytes
canonical_json_bytes = base.canonical_json_bytes
sha256_bytes = base.sha256_bytes
sha256_file = base.sha256_file
canonical_sha256 = base.canonical_sha256
read_json = base.read_json
atomic_json = base.atomic_json
exact_median_interval = base.exact_median_interval
exact_sign_test = base.exact_sign_test
holm_adjust = base.holm_adjust


EXPECTED_ORDER = [
    "torch_contract_fp32",
    "tilelang_full_g08",
    "triton_full_g05",
    "cuda_noptx_old_g04",
    "cuda_unlimited_old_g02",
    "cuda_noptx_streamed_g05",
    "cuda_noptx_streamed_g09",
    "cuda_unlimited_streamed_g07",
]


def validate_campaign(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 3:
        raise ClosureError("campaign must be schema v3")
    if value.get("campaign_id") != "fused-frontier-reachability-closure-v3":
        raise ClosureError("unexpected campaign ID")
    if value.get("candidate_order") != EXPECTED_ORDER:
        raise ClosureError("candidate order differs from preregistration")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or [
        row.get("candidate_id") for row in candidates
    ] != EXPECTED_ORDER:
        raise ClosureError("candidate definitions differ from frozen order")
    if len({row["candidate_id"] for row in candidates}) != 8:
        raise ClosureError("candidate IDs are not unique")
    for row in candidates:
        if row.get("implementation") == "closure_v2":
            if not row.get("source_candidate_id"):
                raise ClosureError("closure candidate lacks source candidate ID")
        elif row.get("implementation") == "reachability_v2":
            if not row.get("source_job_id"):
                raise ClosureError("reachability candidate lacks source job ID")
        else:
            raise ClosureError("unknown candidate implementation source")
    expected_protocol = {
        "blocks": 15,
        "dist": "rand",
        "flush_l2": True,
        "order_seed": 2026073005,
        "physical_gpu": 3,
        "seed": 0,
        "trials": 100,
        "warmup_s": 2.0,
    }
    if value.get("performance_protocol") != expected_protocol:
        raise ClosureError("performance protocol differs from preregistration")
    hardware = value.get("hardware", {})
    if hardware.get("physical_gpu") != 3 or hardware.get("required_uuid") != (
        "GPU-eafdd6ce-8857-40fd-f494-47a7240bf6b5"
    ):
        raise ClosureError("physical GPU binding differs")
    known = set(EXPECTED_ORDER)
    families = value.get("preregistered_families", {})
    if list(families) != [
        "old_vs_new_within_cuda",
        "compiler_vs_new_cuda_fixed",
        "new_cuda_vs_contract_torch",
    ]:
        raise ClosureError("comparison family order/set differs")
    for family, comparisons in families.items():
        if not comparisons:
            raise ClosureError(f"empty family {family}")
        for pair in comparisons:
            if not isinstance(pair, list) or len(pair) != 2 or any(
                candidate not in known for candidate in pair
            ):
                raise ClosureError(f"invalid comparison in {family}")
    expected_spread = [
        "tilelang_full_g08",
        "triton_full_g05",
        "cuda_noptx_streamed_g05",
        "cuda_unlimited_streamed_g07",
    ]
    if value.get("fixed_frontier_spread") != expected_spread:
        raise ClosureError("fixed frontier spread set differs")
    return value


def load_campaign(path: Path = CAMPAIGN_PATH) -> dict[str, Any]:
    return validate_campaign(read_json(path))


def candidates_by_id(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["candidate_id"]: row for row in campaign["candidates"]}


def candidate_sha256(candidate: dict[str, Any]) -> str:
    return canonical_sha256(candidate)


def protocol_sha256(campaign: dict[str, Any]) -> str:
    return canonical_sha256(campaign["performance_protocol"])


def block_plan(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = campaign["performance_protocol"]
    randomizer = random.Random(protocol["order_seed"])
    result = []
    for block in range(protocol["blocks"]):
        order = list(campaign["candidate_order"])
        randomizer.shuffle(order)
        for position, candidate_id in enumerate(order):
            result.append(
                {"block": block, "position": position, "candidate_id": candidate_id}
            )
    return result
