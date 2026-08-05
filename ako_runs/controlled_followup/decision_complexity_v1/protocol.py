#!/usr/bin/env python3
"""CPU-only protocol for decision-complexity experiments C1/C2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ako_runs.controlled_followup.convergence_v2.analyze import (
    SurvivalObservation,
    analyze,
)


CAMPAIGN_ID = "gpu-dsl-decision-complexity-v1"
STATE = "design_only_not_authorized"
LEVELS = (1, 3, 6)
HEX = set("0123456789abcdef")
ATTEMPT_STATUS = {
    "BUILD_FAILED",
    "LAUNCH_FAILED",
    "GATE_FAILED",
    "GATE_PASSED",
    "TIMEOUT",
}


def _digest(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("campaign_id") != CAMPAIGN_ID or contract.get("state") != STATE:
        raise ValueError("wrong campaign_id or state")
    if contract.get("launch_authorized") is not False:
        raise ValueError("this protocol must remain launch-forbidden")
    _hash(contract.get("randomization_seed_sha256"), "randomization_seed_sha256")

    replicates = contract.get("replicates")
    if type(replicates) is not int or replicates < 2:
        raise ValueError("replicates must be an integer >= 2")
    gpu_slots = contract.get("gpu_slots")
    if type(gpu_slots) is not int or gpu_slots <= 0 or replicates % gpu_slots:
        raise ValueError("gpu_slots must be positive and divide replicates")
    budgets = contract.get("budgets")
    if not isinstance(budgets, dict):
        raise ValueError("budgets are required")
    if type(budgets.get("max_attempts")) is not int or budgets["max_attempts"] <= 0:
        raise ValueError("max_attempts must be positive")
    if isinstance(budgets.get("tau_active_s"), bool) or not isinstance(budgets.get("tau_active_s"), (int, float)) or not math.isfinite(
        budgets["tau_active_s"]
    ) or budgets["tau_active_s"] <= 0:
        raise ValueError("tau_active_s must be finite and positive")

    tasks = contract.get("tasks")
    searchers = contract.get("searchers")
    if not isinstance(tasks, list) or not tasks or not isinstance(searchers, list) or not searchers:
        raise ValueError("nonempty tasks and searchers are required")
    task_ids = [task.get("task_id") for task in tasks if isinstance(task, dict)]
    searcher_ids = [row.get("searcher_id") for row in searchers if isinstance(row, dict)]
    if (
        len(task_ids) != len(tasks)
        or len(task_ids) != len(set(task_ids))
        or not all(isinstance(value, str) and value for value in task_ids)
    ):
        raise ValueError("task_id values must be unique and nonempty")
    if (
        len(searcher_ids) != len(searchers)
        or len(searcher_ids) != len(set(searcher_ids))
        or not all(isinstance(value, str) and value for value in searcher_ids)
    ):
        raise ValueError("searcher_id values must be unique and nonempty")
    if len(tasks) < 3:
        raise ValueError("C1 requires at least three preregistered tasks")

    for searcher in searchers:
        _hash(searcher.get("immutable_revision_sha256"), "immutable_revision_sha256")
        _hash(searcher.get("neutral_prompt_sha256"), "neutral_prompt_sha256")
        _hash(searcher.get("tool_contract_sha256"), "tool_contract_sha256")
    for task in tasks:
        if not all(isinstance(task.get(field), str) and task[field] for field in ("operator_family", "lane")):
            raise ValueError("operator_family and lane must be nonempty strings")
        for field in (
            "gate_sha256",
            "reference_sha256",
            "tuning_dataset_sha256",
            "terminal_dataset_sha256",
            "visible_search_contract_sha256",
            "valid_hint_sha256",
            "target_candidate_sha256",
            "target_gate_receipt_sha256",
            "terminal_evaluator_lock_sha256",
        ):
            _hash(task.get(field), field)
        if task["tuning_dataset_sha256"] == task["terminal_dataset_sha256"]:
            raise ValueError("tuning and terminal datasets must be distinct")
        axes = task.get("axis_order")
        target = task.get("target_values")
        if (
            not isinstance(axes, list)
            or len(axes) < max(LEVELS)
            or len(axes) != len(set(axes))
            or not all(isinstance(axis, str) and axis for axis in axes)
        ):
            raise ValueError("axis_order must contain at least six unique axes")
        if not isinstance(target, dict) or set(target) != set(axes):
            raise ValueError("target_values must bind every and only declared axis")
        domains = task.get("axis_domains")
        dependencies = task.get("dependency_graph")
        if not isinstance(domains, dict) or set(domains) != set(axes):
            raise ValueError("axis_domains must bind every and only declared axis")
        if not isinstance(dependencies, dict) or set(dependencies) != set(axes):
            raise ValueError("dependency_graph must bind every and only declared axis")
        for index, axis in enumerate(axes):
            if not isinstance(domains[axis], list) or not domains[axis] or target[axis] not in domains[axis]:
                raise ValueError("each axis domain must be nonempty and contain the target")
            if (
                not isinstance(dependencies[axis], list)
                or len(dependencies[axis]) != len(set(dependencies[axis]))
                or any(dependency not in axes[:index] for dependency in dependencies[axis])
            ):
                raise ValueError("dependency_graph must reference unique prior axes")
        complexity = task.get("observed_complexity")
        required = {"score", "axis_count", "dependency_depth", "correctness_constraints", "reachable_fraction"}
        if not isinstance(complexity, dict) or set(complexity) != required:
            raise ValueError("observed_complexity has the wrong fields")
        if any(type(complexity[key]) is not int or complexity[key] < 0 for key in required - {"reachable_fraction"}):
            raise ValueError("complexity counts must be nonnegative integers")
        if complexity["axis_count"] != len(axes):
            raise ValueError("observed axis_count must equal axis_order length")
        fraction = complexity["reachable_fraction"]
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
            raise ValueError("reachable_fraction must be in [0,1]")
        if any(task["valid_hint_sha256"] == searcher["neutral_prompt_sha256"] for searcher in searchers):
            raise ValueError("valid hint must differ from the neutral prompt")
    if len({task["observed_complexity"]["score"] for task in tasks}) < 2:
        raise ValueError("C1 requires at least two observed complexity scores")


def _build_manifest_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in contract["tasks"]:
        for searcher in contract["searchers"]:
            for replicate in range(contract["replicates"]):
                arms = [
                    ("c1_observed", task["observed_complexity"]["axis_count"], "neutral", None),
                    *((f"c2_open_{level}", level, "neutral", None) for level in LEVELS),
                    ("label_sham_a", 3, "neutral", "a"),
                    ("label_sham_b", 3, "neutral", "b"),
                    ("sensitivity_valid_hint", 6, "valid_hint", None),
                ]
                arms.sort(
                    key=lambda item: _digest(
                        contract["randomization_seed_sha256"],
                        task["task_id"],
                        searcher["searcher_id"],
                        replicate,
                        item[0],
                    )
                )
                for position, (arm, open_count, prompt_arm, hidden_label) in enumerate(arms):
                    identity = (task["task_id"], searcher["searcher_id"], replicate, arm)
                    open_axes = task["axis_order"][:open_count]
                    fixed_target_values = {
                        axis: task["target_values"][axis]
                        for axis in task["axis_order"][open_count:]
                    }
                    search_space = {
                        "open_axes": open_axes,
                        "axis_domains": {axis: task["axis_domains"][axis] for axis in open_axes},
                        "dependency_graph": {axis: task["dependency_graph"][axis] for axis in open_axes},
                        "fixed_target_values": fixed_target_values,
                        "shared_target_candidate_sha256": task["target_candidate_sha256"],
                    }
                    slot_arm = "label_sham" if arm.startswith("label_sham_") else arm
                    gpu_offset = int(_digest("gpu", task["task_id"], searcher["searcher_id"], slot_arm), 16)
                    rows.append(
                        {
                            "schema_version": 1,
                            "campaign_id": CAMPAIGN_ID,
                            "state": "planned_launch_blocked",
                            "trajectory_id": _digest(CAMPAIGN_ID, *identity),
                            "randomization_key": _digest(
                                "randomization", contract["randomization_seed_sha256"], CAMPAIGN_ID, *identity
                            ),
                            "position": position,
                            "gpu_slot": (gpu_offset + replicate) % contract["gpu_slots"],
                            "task_id": task["task_id"],
                            "operator_family": task["operator_family"],
                            "lane": task["lane"],
                            "searcher_id": searcher["searcher_id"],
                            "replicate": replicate,
                            "study": "C1" if arm == "c1_observed" else "C2",
                            "arm": arm,
                            "observed_complexity": task["observed_complexity"],
                            "open_axis_count": open_count,
                            "open_axes": open_axes,
                            "fixed_target_values": fixed_target_values,
                            "search_space_sha256": _json_sha256(search_space),
                            "target_candidate_sha256": task["target_candidate_sha256"],
                            "target_gate_receipt_sha256": task["target_gate_receipt_sha256"],
                            "terminal_evaluator_lock_sha256": task["terminal_evaluator_lock_sha256"],
                            "prompt_arm": prompt_arm,
                            "hidden_label": hidden_label,
                            "visible_search_contract_sha256": task["visible_search_contract_sha256"],
                            "gate_sha256": task["gate_sha256"],
                            "reference_sha256": task["reference_sha256"],
                            "tuning_dataset_sha256": task["tuning_dataset_sha256"],
                            "terminal_dataset_sha256": task["terminal_dataset_sha256"],
                            "immutable_revision_sha256": searcher["immutable_revision_sha256"],
                            "prompt_sha256": (
                                task["valid_hint_sha256"]
                                if prompt_arm == "valid_hint"
                                else searcher["neutral_prompt_sha256"]
                            ),
                            "tool_contract_sha256": searcher["tool_contract_sha256"],
                            "budgets": contract["budgets"],
                            "failed_attempts_charged": True,
                            "terminal_holdout_visibility": "offline_evaluator_only",
                            "gpu_identity_binding_required": True,
                        }
                    )
    return rows


def build_manifest(contract: dict[str, Any]) -> list[dict[str, Any]]:
    validate_contract(contract)
    rows = _build_manifest_rows(contract)
    validate_manifest(rows, contract)
    return rows


def validate_manifest(rows: list[dict[str, Any]], contract: dict[str, Any]) -> None:
    validate_contract(contract)
    if rows != _build_manifest_rows(contract):
        raise ValueError("manifest differs from the deterministic contract projection")
    expected = len(contract["tasks"]) * len(contract["searchers"]) * contract["replicates"] * 7
    if len(rows) != expected:
        raise ValueError(f"wrong manifest census: {len(rows)} != {expected}")
    ids = [row.get("trajectory_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trajectory_id")
    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["task_id"], row["searcher_id"], row["replicate"])
        by_cell.setdefault(key, []).append(row)
    expected_arms = {
        "c1_observed",
        "c2_open_1",
        "c2_open_3",
        "c2_open_6",
        "label_sham_a",
        "label_sham_b",
        "sensitivity_valid_hint",
    }
    for key, members in by_cell.items():
        if len(members) != 7 or sorted(row["position"] for row in members) != list(range(7)):
            raise ValueError(f"wrong block positions for {key}")
        if {row["arm"] for row in members} != expected_arms:
            raise ValueError(f"wrong arms for {key}")
        shams = [row for row in members if row["arm"].startswith("label_sham_")]
        if len({row["visible_search_contract_sha256"] for row in shams}) != 1 or len(
            {row["prompt_sha256"] for row in shams}
        ) != 1:
            raise ValueError(f"label sham is not byte-contract matched for {key}")
        ignored = {"trajectory_id", "randomization_key", "arm", "hidden_label", "position"}
        projections = [
            {field: value for field, value in row.items() if field not in ignored}
            for row in shams
        ]
        if projections[0] != projections[1]:
            raise ValueError(f"label sham differs beyond its hidden label for {key}")


def bind_outcome(manifest_row: dict[str, Any], attempt_ledger: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_row_sha256 = _json_sha256(manifest_row)
    receipt_value = {
        "manifest_row_sha256": manifest_row_sha256,
        "attempt_ledger": attempt_ledger,
    }
    return {
        "trajectory_id": manifest_row["trajectory_id"],
        "manifest_row_sha256": manifest_row_sha256,
        "attempt_ledger": attempt_ledger,
        "outcome_payload_sha256": _json_sha256(receipt_value),
    }


def _derive_observation(manifest_row: dict[str, Any], outcome: dict[str, Any]) -> tuple[int, float, bool]:
    allowed = {
        "trajectory_id",
        "manifest_row_sha256",
        "attempt_ledger",
        "outcome_payload_sha256",
    }
    if set(outcome) != allowed or outcome["manifest_row_sha256"] != _json_sha256(manifest_row):
        raise ValueError("outcome does not bind the manifest row")
    receipt = {
        "manifest_row_sha256": outcome["manifest_row_sha256"],
        "attempt_ledger": outcome["attempt_ledger"],
    }
    if outcome["outcome_payload_sha256"] != _json_sha256(receipt):
        raise ValueError("outcome payload hash mismatch")
    ledger = outcome["attempt_ledger"]
    max_attempts = manifest_row["budgets"]["max_attempts"]
    tau_active_s = float(manifest_row["budgets"]["tau_active_s"])
    if not isinstance(ledger, list) or not ledger or len(ledger) > max_attempts:
        raise ValueError("attempt ledger is empty or exceeds the frozen cap")
    cumulative = 0.0
    first_event: tuple[int, float] | None = None
    fields = {
        "attempt_index",
        "candidate_sha256",
        "terminal_status",
        "active_s",
        "gate_legal",
        "candidate_latency_ms",
        "reference_latency_ms",
    }
    for index, attempt in enumerate(ledger, 1):
        if first_event is not None:
            raise ValueError("attempt ledger continues after the terminal event")
        if not isinstance(attempt, dict) or set(attempt) != fields or attempt["attempt_index"] != index:
            raise ValueError("attempt ledger indexes or fields are invalid")
        _hash(attempt["candidate_sha256"], "candidate_sha256")
        if attempt["terminal_status"] not in ATTEMPT_STATUS:
            raise ValueError("unknown attempt terminal_status")
        active_s = attempt["active_s"]
        if isinstance(active_s, bool) or not isinstance(active_s, (int, float)) or not math.isfinite(active_s) or active_s < 0:
            raise ValueError("attempt active_s must be finite and nonnegative")
        if not isinstance(attempt["gate_legal"], bool) or attempt["gate_legal"] != (attempt["terminal_status"] == "GATE_PASSED"):
            raise ValueError("gate_legal must agree with terminal_status")
        latencies = (attempt["candidate_latency_ms"], attempt["reference_latency_ms"])
        if attempt["gate_legal"]:
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in latencies):
                raise ValueError("gate-legal attempts require positive finite latencies")
        elif latencies != (None, None):
            raise ValueError("failed attempts must not carry terminal latencies")
        within_target = attempt["gate_legal"] and latencies[0] <= 1.05 * latencies[1]
        cumulative += float(active_s)
        if first_event is None and within_target:
            first_event = (index, cumulative)
    if cumulative > tau_active_s and not math.isclose(cumulative, tau_active_s):
        raise ValueError("attempt ledger exceeds the active-time cap")
    if first_event is None:
        if len(ledger) != max_attempts and not math.isclose(cumulative, tau_active_s):
            raise ValueError("non-event trajectory is censored before either frozen cap")
        return len(ledger), cumulative, False
    if first_event[1] <= 0:
        raise ValueError("a terminal event must consume positive active time")
    return first_event[0], first_event[1], True


def analyze_complete_outcomes(
    contract: dict[str, Any],
    manifest: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    tau_attempts: int,
    tau_active_s: float,
) -> dict[str, Any]:
    validate_manifest(manifest, contract)
    expected = {row["trajectory_id"]: row for row in manifest}
    observed = [str(row.get("trajectory_id")) for row in outcomes]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError("outcomes must contain exactly one row per manifest trajectory")
    frozen_attempts = {row["budgets"]["max_attempts"] for row in manifest}
    frozen_active = {float(row["budgets"]["tau_active_s"]) for row in manifest}
    if frozen_attempts != {tau_attempts} or frozen_active != {float(tau_active_s)}:
        raise ValueError("analysis tau must match the frozen manifest budgets")

    grouped_attempts: dict[str, list[SurvivalObservation]] = {"c1": [], "c2": [], "controls": []}
    grouped_active: dict[str, list[SurvivalObservation]] = {"c1": [], "c2": [], "controls": []}
    for outcome in outcomes:
        manifest_row = expected[outcome["trajectory_id"]]
        arm = manifest_row["arm"]
        attempts, active_s, event = _derive_observation(manifest_row, outcome)
        if arm == "c1_observed":
            study = "c1"
            group = f"complexity_{manifest_row['observed_complexity']['score']}"
        elif arm.startswith("c2_open_"):
            study = "c2"
            group = arm
        else:
            study = "controls"
            group = arm
        grouped_attempts[study].append(SurvivalObservation(outcome["trajectory_id"], group, float(attempts), event))
        grouped_active[study].append(SurvivalObservation(outcome["trajectory_id"], group, float(active_s), event))

    analyses: dict[str, Any] = {}
    for study in grouped_attempts:
        attempt_analysis = analyze(grouped_attempts[study], float(tau_attempts))
        attempt_analysis["metric"] = "attempts"
        attempt_analysis["tau_attempts"] = attempt_analysis.pop("tau_s")
        active_analysis = analyze(grouped_active[study], tau_active_s)
        active_analysis["metric"] = "active_seconds"
        analyses[study] = {"attempts": attempt_analysis, "active_seconds": active_analysis}
    c2_rmst = analyses["c2"]["attempts"]["groups"]
    monotonic = all(
        c2_rmst[f"c2_open_{left}"]["rmst_s"] <= c2_rmst[f"c2_open_{right}"]["rmst_s"]
        for left, right in zip(LEVELS, LEVELS[1:])
    )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "design_only_noncontrolling",
        "contract_sha256": _json_sha256(contract),
        "manifest_sha256": _json_sha256(manifest),
        "c1_association": analyses["c1"],
        "c2_primary": analyses["c2"],
        "controls": analyses["controls"],
        "c2_monotonic_rmst_descriptive": monotonic,
        "claim_policy": "no controlling claim until a successor freezes clustered C1 and block-aware C2 inference",
    }


def launch() -> None:
    raise RuntimeError(
        "launch forbidden: decision_complexity_v1 needs a new authorization, frozen material inputs, and a new lock"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "manifest", "launch"))
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "launch":
        launch()
    if args.contract is None:
        parser.error("--contract is required")
    contract = json.loads(args.contract.read_text())
    rows = build_manifest(contract)
    if args.action == "check":
        print(json.dumps({"ok": True, "rows": len(rows)}, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required for manifest")
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
