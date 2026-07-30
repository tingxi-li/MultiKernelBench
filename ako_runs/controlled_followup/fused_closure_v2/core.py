#!/usr/bin/env python3
"""Pure campaign validation, planning, serialization, and statistics helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CAMPAIGN_PATH = HERE / "campaign.json"
SOURCE_RECEIPT_PATH = HERE / "source_receipt.json"
RESULTS_ROOT = HERE / "results"


class ClosureError(ValueError):
    """A campaign input or evidence artifact violated its frozen contract."""


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureError(f"cannot read JSON {path}: {exc}") from exc


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(stable_json_bytes(value))
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def validate_campaign(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ClosureError("campaign must be a JSON object")
    if value.get("schema_version") != 2:
        raise ClosureError("campaign schema_version must be 2")
    if value.get("campaign_id") != "fused-gbgs-closure-v2":
        raise ClosureError("unexpected campaign_id")
    protocol = value.get("performance_protocol")
    if not isinstance(protocol, dict):
        raise ClosureError("missing performance_protocol")
    expected_protocol = {
        "blocks": 15,
        "dist": "rand",
        "flush_l2": True,
        "order_seed": 2026073002,
        "physical_gpu": 0,
        "seed": 0,
        "trials": 100,
        "warmup_s": 2.0,
    }
    if protocol != expected_protocol:
        raise ClosureError("performance protocol differs from preregistration")

    candidates = value.get("candidates")
    order = value.get("candidate_order")
    if not isinstance(candidates, list) or len(candidates) != 9:
        raise ClosureError("campaign must contain exactly nine candidates")
    ids = [candidate.get("candidate_id") for candidate in candidates]
    if ids != order or len(set(ids)) != len(ids):
        raise ClosureError("candidate order/IDs are inconsistent")
    allowed_implementations = {
        "historical_torch",
        "historical_torch_precast",
        "torch_contract_fp32",
        "phase2_custom",
    }
    for candidate in candidates:
        if candidate.get("implementation") not in allowed_implementations:
            raise ClosureError(
                f"unknown implementation for {candidate.get('candidate_id')!r}"
            )
        status = candidate.get("contract_adjudication")
        mismatches = candidate.get("structural_mismatches")
        if status not in ("diagnostic_nonconforming", "fused_v2_required"):
            raise ClosureError("invalid contract adjudication")
        if not isinstance(mismatches, list):
            raise ClosureError("structural_mismatches must be a list")
        if status == "fused_v2_required" and mismatches:
            raise ClosureError("required candidate declares a structural mismatch")
        if candidate["implementation"] == "phase2_custom":
            if candidate.get("dsl") not in {
                "tilelang",
                "triton",
                "cuda_noptx",
                "cuda_unlimited",
            }:
                raise ClosureError("custom candidate has invalid DSL")
            if not isinstance(candidate.get("set"), str) or not candidate["set"]:
                raise ClosureError("custom candidate has no Phase-2 set")

    common = value.get("selection", {}).get("strict_common_grid_ids")
    expected_common = [
        "g00",
        "g01",
        "g02",
        "g03",
        "g04",
        "g13",
        "g14",
        "g16",
        "g17",
    ]
    if common != expected_common:
        raise ClosureError("strict common intersection differs from nine-point freeze")

    gate = value.get("gate")
    if not isinstance(gate, dict):
        raise ClosureError("missing frozen-gate plan")
    if gate.get("split") != "validation":
        raise ClosureError("gate split must be validation")
    if gate.get("seed_indices") != {"start": 0, "stop_exclusive": 64}:
        raise ClosureError("gate validation must cover exactly seeds 0--63")
    if gate.get("gate_ids") != ["semantic_mixed", "conformance_mixed"]:
        raise ClosureError("fused-v2 gate order changed")
    if len(gate.get("case_ids", [])) != 4:
        raise ClosureError("fused-v2 adjudication must cover four cases")

    known = set(ids)
    for family, comparisons in value.get("preregistered_families", {}).items():
        if not comparisons:
            raise ClosureError(f"comparison family {family!r} is empty")
        for comparison in comparisons:
            if (
                not isinstance(comparison, list)
                or len(comparison) != 2
                or comparison[0] not in known
                or comparison[1] not in known
            ):
                raise ClosureError(f"invalid comparison in family {family!r}")
    return value


def load_campaign(path: Path = CAMPAIGN_PATH) -> dict[str, Any]:
    return validate_campaign(read_json(path))


def candidates_by_id(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["candidate_id"]: candidate for candidate in campaign["candidates"]}


def parse_set(value: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for field in value.split(","):
        key, text = field.split("=", 1)
        if key in {"arith", "cast"}:
            parsed[key] = text
        elif key.startswith("x_"):
            parsed.setdefault("extra", {})[key[2:]] = text
        else:
            parsed[key] = int(text)
    return parsed


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
                {
                    "block": block,
                    "position": position,
                    "candidate_id": candidate_id,
                }
            )
    return result


def exact_median_interval(values: Iterable[float], confidence: float = 0.95) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ClosureError("median interval requires finite observations")
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    choices = []
    for k in range(1, (n + 1) // 2 + 1):
        tail = sum(math.comb(n, index) for index in range(k)) / (2**n)
        coverage = 1.0 - 2.0 * tail
        if coverage >= confidence:
            choices.append((k, coverage))
    if choices:
        # The largest k gives the narrowest central finite interval meeting the target.
        k, coverage = max(choices)
        low = ordered[k - 1]
        high = ordered[n - k]
        finite = True
    else:
        # At n=5, even [min,max] is only 93.75%. Preserve this fact explicitly.
        k = 1
        coverage = 1.0 - 2.0 / (2**n)
        low, high, finite = ordered[0], ordered[-1], False
    return {
        "achieved_coverage": coverage,
        "confidence_target": confidence,
        "finite_interval_meets_target": finite,
        "interval_order_statistic_k": k,
        "median": median,
        "n": n,
        "ordered_values": ordered,
        "ci_lo": low,
        "ci_hi": high,
    }


def exact_sign_test(values: Iterable[float], null: float = 1.0) -> dict[str, Any]:
    observations = [float(value) for value in values]
    below = sum(value < null for value in observations)
    above = sum(value > null for value in observations)
    ties = len(observations) - below - above
    n = below + above
    if n == 0:
        p_value = 1.0
    else:
        extreme = min(below, above)
        p_value = min(
            1.0,
            2.0 * sum(math.comb(n, index) for index in range(extreme + 1)) / (2**n),
        )
    return {
        "above": above,
        "below": below,
        "n_nonties": n,
        "null": null,
        "p_value_two_sided": p_value,
        "ties": ties,
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    count = len(values)
    order = sorted(range(count), key=lambda index: values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted

