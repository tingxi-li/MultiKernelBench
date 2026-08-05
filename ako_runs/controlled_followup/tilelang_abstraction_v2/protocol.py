"""CPU-only, design-only controls for TileLang abstraction experiments A1/A2."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


MATCH_FIELDS = (
    "operator",
    "operator_family",
    "shape",
    "gate_lock_sha256",
    "input_contract_sha256",
    "algorithm",
    "dtype",
    "tile",
    "pipeline_depth",
    "threads",
    "instruction_family",
    "logical_work",
    "dynamic_work",
)
HASH_FIELDS = (
    "gate_lock_sha256",
    "input_contract_sha256",
    "implementation_sha256",
    "source_sha256",
    "gate_receipt_sha256",
    "ir_sha256",
    "ptx_sha256",
    "sass_sha256",
    "dynamic_work_receipt_sha256",
    "resource_receipt_sha256",
)
A2_ARMS = ("TL-H-only", "TL-M-only")
A2_EQUAL_FIELDS = (
    "candidate_attempt_budget",
    "wall_clock_seconds",
    "hardware_binding_sha256",
    "task_contract_sha256",
    "gate_feedback_contract_sha256",
    "searcher_lock_sha256",
    "prompt_lock_sha256",
    "tool_lock_sha256",
    "isolation_policy_sha256",
    "randomization_lock_sha256",
)
REQUIRED_LAUNCH_INPUTS = (
    "campaign_lock",
    "gate_lock",
    "hardware_lock",
    "manifest_lock",
    "remote_preregistration",
    "sham_protocol",
    "source_lock",
    "toolchain_lock",
)


class LaunchRefused(RuntimeError):
    """Raised because this design-only protocol cannot authorize execution."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def audit_manifest(pairs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify H/M pairs; malformed bindings fail and mismatches are excluded."""
    audited: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("each pair must be a mapping")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("pair_id must be a non-empty string")
        if pair_id in seen:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        seen.add(pair_id)
        implementer = pair.get("implementer_id")
        if not isinstance(implementer, str) or not implementer:
            raise ValueError(f"{pair_id}: implementer_id is required")
        if pair.get("implementation_order") not in {"H_then_M", "M_then_H"}:
            raise ValueError(f"{pair_id}: implementation_order is invalid")
        isolation = pair.get("isolation_receipt_sha256")
        if not _is_sha256(isolation):
            raise ValueError(f"{pair_id}: isolation receipt is required")
        if not _is_sha256(pair.get("randomization_receipt_sha256")):
            raise ValueError(f"{pair_id}: randomization receipt is required")

        high, low = pair.get("high"), pair.get("low")
        if not isinstance(high, Mapping) or not isinstance(low, Mapping):
            raise ValueError(f"{pair_id}: high and low must be mappings")
        if high.get("level") != "TL-H" or low.get("level") != "TL-M":
            raise ValueError(f"{pair_id}: expected TL-H/TL-M levels")
        for side, record in (("high", high), ("low", low)):
            missing = [field for field in MATCH_FIELDS if record.get(field) is None]
            if missing:
                raise ValueError(f"{pair_id}: {side} missing {','.join(missing)}")
            if record.get("terminal_status") != "GATE_PASSED":
                raise ValueError(f"{pair_id}: {side} must be GATE_PASSED")
            for field in HASH_FIELDS:
                value = record.get(field)
                if not _is_sha256(value):
                    raise ValueError(f"{pair_id}: {side} has invalid {field}")
        if high["implementation_sha256"] == low["implementation_sha256"]:
            raise ValueError(f"{pair_id}: H/M treatments must bind distinct implementations")

        mismatches = [field for field in MATCH_FIELDS if high[field] != low[field]]
        included = not mismatches
        audited.append(
            {
                "pair_id": pair_id,
                "classification": "runtime_estimand" if included else "capability_only",
                "included_in_runtime_estimand": included,
                "mismatch_fields": mismatches,
                "operator_family": high["operator_family"],
                "shape": high["shape"],
                "implementer_id": implementer,
                "implementation_order": pair["implementation_order"],
            }
        )

    audited.sort(key=lambda row: row["pair_id"])
    runtime = sum(row["included_in_runtime_estimand"] for row in audited)
    families: dict[str, set[str]] = {}
    implementers: set[str] = set()
    implementation_orders: set[str] = set()
    for row in audited:
        if row["included_in_runtime_estimand"]:
            families.setdefault(row["operator_family"], set()).add(repr(row["shape"]))
            implementers.add(row["implementer_id"])
            implementation_orders.add(row["implementation_order"])
    design_ready = (
        len(families) >= 3
        and all(len(shapes) >= 3 for shapes in families.values())
        and len(implementers) >= 2
        and implementation_orders == {"H_then_M", "M_then_H"}
    )
    return {
        "pairs": audited,
        "census": {
            "total_pairs": len(audited),
            "runtime_estimand_pairs": runtime,
            "capability_only_pairs": len(audited) - runtime,
        },
        # Hash syntax is not material receipt verification.  A frozen successor
        # must re-read those artifacts before any multi-family claim is allowed.
        "claim_scope": "local_only_unverified",
        "generalization_design_ready": design_ready,
        "material_receipts_verified": False,
    }


def classify_interval(
    lower: float, upper: float, *, delta_hw: float, epsilon: float
) -> dict[str, str]:
    """Classify a CI for log(T_low/T_high) using frozen practical thresholds."""
    lower, upper, delta_hw, epsilon = map(float, (lower, upper, delta_hw, epsilon))
    if not all(map(math.isfinite, (lower, upper, delta_hw, epsilon))):
        raise ValueError("interval and thresholds must be finite")
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    if delta_hw < 0:
        raise ValueError("delta_hw must be non-negative")
    if epsilon < delta_hw:
        raise ValueError("epsilon must be at least delta_hw")
    direction = "unresolved"
    if upper < -delta_hw:
        direction = "lower_level_faster"
    elif lower > delta_hw:
        direction = "higher_level_faster"
    equivalence = (
        "within_equivalence_bound"
        if lower >= -epsilon and upper <= epsilon
        else "not_demonstrated"
    )
    return {"direction": direction, "equivalence": equivalence}


def validate_a2_contract(contract: Mapping[str, Any]) -> None:
    """Validate the design-only equal-budget A2 search-arm contract."""
    if contract.get("status") != "design_only":
        raise ValueError("A2 must remain design_only")
    arms = contract.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(A2_ARMS):
        raise ValueError("A2 requires exactly TL-H-only and TL-M-only arms")
    for field in A2_EQUAL_FIELDS:
        values = []
        for arm in A2_ARMS:
            if not isinstance(arms[arm], Mapping) or arms[arm].get(field) is None:
                raise ValueError(f"{arm} missing {field}")
            values.append(arms[arm][field])
        if values[0] != values[1]:
            raise ValueError(f"A2 arms differ on {field}")
    attempts = arms[A2_ARMS[0]]["candidate_attempt_budget"]
    wall = arms[A2_ARMS[0]]["wall_clock_seconds"]
    if type(attempts) is not int or attempts <= 0:
        raise ValueError("candidate_attempt_budget must be a positive integer")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("wall_clock_seconds must be finite and positive")
    for field in A2_EQUAL_FIELDS[2:]:
        value = arms[A2_ARMS[0]][field]
        if not _is_sha256(value):
            raise ValueError(f"{field} must be a SHA-256 binding")
    reference = contract.get("terminal_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("A2 terminal_reference is required")
    if reference.get("visibility") != "hidden_until_search_complete":
        raise ValueError("terminal reference must stay hidden until search completes")
    if reference.get("gate_legal") is not True:
        raise ValueError("terminal reference must be gate-legal")
    if reference.get("target_ratio") != 1.05:
        raise ValueError("terminal endpoint must remain within 5% of reference")
    for field in ("reference_sha256", "gate_receipt_sha256", "tuning_dataset_sha256", "terminal_dataset_sha256"):
        value = reference.get(field)
        if not _is_sha256(value):
            raise ValueError(f"terminal reference missing {field}")
    if reference["tuning_dataset_sha256"] == reference["terminal_dataset_sha256"]:
        raise ValueError("tuning and terminal datasets must be distinct")


def authorize_launch(
    *, policy_authorized: bool = False, material_inputs: Iterable[str] = ()
) -> None:
    """Fail closed: this protocol is a design artifact, never an executor."""
    supplied = set(material_inputs)
    blockers = []
    if not policy_authorized:
        blockers.append("current policy does not authorize A1/A2 execution")
    blockers.extend(
        f"missing material input: {name}"
        for name in REQUIRED_LAUNCH_INPUTS
        if name not in supplied
    )
    if not blockers:
        blockers.append("design-only protocol has no launch path; freeze a successor campaign")
    raise LaunchRefused("; ".join(blockers))
