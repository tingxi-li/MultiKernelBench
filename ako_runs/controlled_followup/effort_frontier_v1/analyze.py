#!/usr/bin/env python3
"""Fail-closed search-frontier and GPU0 confirmation analysis for RQ5."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from . import campaign, events, validate
except ImportError:  # direct script execution
    import campaign  # type: ignore
    import events  # type: ignore
    import validate  # type: ignore


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value > 0
    )


def _holm(p_values: Iterable[float]) -> list[float]:
    values = list(p_values)
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(values) - rank) * value))
        adjusted[index] = running
    return adjusted


def _midranks(values: list[float | None]) -> list[float]:
    # None is the preregistered right-censored/failure value: worse than every
    # finite legal latency, with all failures tied.
    order = sorted(range(len(values)), key=lambda i: (values[i] is None, values[i] or 0.0))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        key = values[order[cursor]]
        while end < len(order) and values[order[end]] == key:
            end += 1
        midrank = ((cursor + 1) + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = midrank
        cursor = end
    return ranks


def exact_mann_whitney(left: list[float | None], right: list[float | None]) -> dict[str, Any]:
    """Tie-aware exact randomization test over all pooled label assignments."""
    if len(left) != 5 or len(right) != 5:
        raise ValueError("effort-frontier Mann-Whitney inference requires 5 + 5 outcomes")
    pooled = left + right
    ranks = _midranks(pooled)
    n_left, n_right = len(left), len(right)
    expected = n_left * n_right / 2.0

    def u_for(indices: tuple[int, ...]) -> float:
        return sum(ranks[index] for index in indices) - n_left * (n_left + 1) / 2.0

    observed = u_for(tuple(range(n_left)))
    deviation = abs(observed - expected)
    universe = list(itertools.combinations(range(len(pooled)), n_left))
    extreme = sum(abs(u_for(indices) - expected) >= deviation - 1e-12 for indices in universe)
    return {
        "n_left": n_left,
        "n_right": n_right,
        "u_left": observed,
        "probability_left_slower_or_tied_half": observed / (n_left * n_right),
        "p_value_two_sided_exact": extreme / len(universe),
        "permutations": len(universe),
        "ties_present": len(set(ranks)) != len(ranks),
    }


def exact_sign_test(ratios: list[float]) -> dict[str, Any]:
    non_ties = [value for value in ratios if not math.isclose(value, 1.0, abs_tol=1e-15)]
    positives = sum(value > 1.0 for value in non_ties)
    n = len(non_ties)
    if n == 0:
        p_value = 1.0
    else:
        tail = min(positives, n - positives)
        p_value = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / 2**n)
    return {
        "n": len(ratios),
        "non_ties": n,
        "ratios_above_one": positives,
        "p_value_two_sided_exact": p_value,
    }


def _median_interval_15(values: list[float]) -> dict[str, float | int]:
    if len(values) != 15:
        raise ValueError("the frozen technical-block interval requires exactly 15 values")
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "ci_lo": ordered[3],
        "ci_hi": ordered[11],
        "coverage": 0.96484375,
        "order_statistics_one_based": [4, 12],
    }


def _validate_candidate_file(event_path: Path, relative: Any, digest: Any) -> None:
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ValueError("checkpoint lacks candidate binding")
    root = event_path.parent.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("candidate path escapes trajectory directory") from exc
    if not path.is_file() or campaign.file_sha256(path) != digest:
        raise ValueError("candidate bytes differ from event binding")


def _checkpoint_outcome(row: dict[str, Any], event_path: Path) -> float | None:
    terminal = row.get("terminal_holdout")
    if not isinstance(row.get("selection_eligible"), bool):
        raise ValueError("checkpoint selection_eligible is not boolean")
    eligible = row.get("selection_eligible") is True
    if not eligible:
        if terminal is not None:
            if not isinstance(terminal, dict):
                raise ValueError("terminal result is not an object")
            build_ok = terminal.get("build_ok")
            lane_policy_pass = terminal.get("lane_policy_pass")
            gate_pass = terminal.get("gate_pass")
            if not all(
                isinstance(value, bool)
                for value in (build_ok, lane_policy_pass, gate_pass)
            ):
                raise ValueError("terminal build/lane-policy/gate status is not boolean")
            if terminal.get("eligible") != bool(
                build_ok and lane_policy_pass and gate_pass
            ):
                raise ValueError("terminal eligibility is internally inconsistent")
            _validate_candidate_file(
                event_path,
                row.get("candidate_relative_path"),
                row.get("candidate_sha256"),
            )
            if gate_pass:
                evidence = terminal.get("gate_summary_sha256")
                if (
                    not isinstance(evidence, str)
                    or len(evidence) != 64
                    or any(character not in "0123456789abcdef" for character in evidence)
                ):
                    raise ValueError("gate-passing terminal result lacks evidence")
        elif any(
            row.get(key) is not None
            for key in (
                "candidate_relative_path",
                "candidate_sha256",
                "tuning_median_ms",
            )
        ):
            raise ValueError("checkpoint without a terminal action binds a candidate")
        return None
    if not isinstance(terminal, dict) or terminal.get("eligible") is not True:
        raise ValueError("eligible checkpoint lacks an eligible terminal result")
    if (
        terminal.get("build_ok") is not True
        or terminal.get("lane_policy_pass") is not True
        or terminal.get("gate_pass") is not True
    ):
        raise ValueError(
            "eligible checkpoint has inconsistent build/lane-policy/gate status"
        )
    evidence = terminal.get("gate_summary_sha256")
    if (
        not isinstance(evidence, str)
        or len(evidence) != 64
        or any(character not in "0123456789abcdef" for character in evidence)
    ):
        raise ValueError("eligible checkpoint lacks terminal gate evidence")
    latency = terminal.get("median_ms")
    if not _finite_positive(latency):
        raise ValueError("eligible checkpoint lacks a finite terminal median")
    relative, digest = row.get("candidate_relative_path"), row.get("candidate_sha256")
    _validate_candidate_file(event_path, relative, digest)
    return float(latency)


def validate_search(result_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate all 20 completed trajectories and return analysis + selections."""
    validate.validate_static()
    lock_blockers: list[str] = []
    validate.model_resolution(lock_blockers)
    validate.executor_registry(lock_blockers)
    validate.provenance_blockers(lock_blockers)
    if lock_blockers:
        raise ValueError("analysis refused because prelaunch locks fail: " + "; ".join(lock_blockers))
    result_root = result_root.resolve()
    by_lane_checkpoint: dict[tuple[str, str], list[float | None]] = defaultdict(list)
    trajectories_out: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    event_heads: dict[str, str] = {}
    common_bindings: dict[str, Any] | None = None
    for job in campaign.trajectories():
        identifier = job["trajectory_id"]
        path = result_root / identifier / "events.jsonl"
        rows = events.load_events(path, identifier)
        if not rows or rows[0].get("event_type") != "trajectory_start":
            raise ValueError(f"{identifier}: missing trajectory_start")
        if rows[-1].get("event_type") != "trajectory_complete":
            raise ValueError(f"{identifier}: trajectory is not complete")
        start = rows[0]
        for key in ("lane", "replicate", "search_seed", "physical_gpu", "required_gpu_uuid"):
            if start.get(key) != job[key]:
                raise ValueError(f"{identifier}: start {key} differs from manifest")
        bindings = start.get("bindings")
        if not isinstance(bindings, dict) or bindings.get("manifest_sha256") != campaign.file_sha256(campaign.MANIFEST):
            raise ValueError(f"{identifier}: manifest start binding differs")
        required_binding_hashes = (
            "manifest_sha256",
            "gate_spec_sha256",
            "gate_receipt_sha256",
            "model_resolution_lock_sha256",
            "executor_registry_sha256",
            "prelaunch_provenance_sha256",
        )
        for key in required_binding_hashes:
            value = bindings.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{identifier}: invalid start binding {key}")
        revision = bindings.get("immutable_model_revision")
        if (
            not isinstance(revision, str)
            or not revision
            or revision == campaign.MODEL["requested_alias"]
        ):
            raise ValueError(f"{identifier}: immutable model revision is not bound")
        if not isinstance(bindings.get("executor_source_hashes"), dict) or not bindings["executor_source_hashes"]:
            raise ValueError(f"{identifier}: executor source hashes are not bound")
        adapter = start.get("adapter_audit")
        if (
            not isinstance(adapter, dict)
            or adapter.get("provider") != "openai"
            or adapter.get("credential_source") != "environment"
            or adapter.get("sampling_seed_supported") is not False
        ):
            raise ValueError(f"{identifier}: provider adapter audit differs")
        if common_bindings is None:
            common_bindings = bindings
        else:
            # Source maps can differ by lane, but all global prelaunch bindings
            # and the immutable model revision must be identical.
            for key in (*required_binding_hashes, "immutable_model_revision"):
                if bindings.get(key) != common_bindings.get(key):
                    raise ValueError(f"{identifier}: global start binding {key} differs")

        requests = {row["iteration"]: row for row in rows if row["event_type"] == "provider_request"}
        responses = {row["iteration"]: row for row in rows if row["event_type"] == "provider_response"}
        if len(requests) != sum(row["event_type"] == "provider_request" for row in rows):
            raise ValueError(f"{identifier}: duplicate provider iteration")
        if sorted(requests) != list(range(len(requests))):
            raise ValueError(f"{identifier}: provider iterations are not contiguous from zero")
        if set(requests) != set(responses):
            raise ValueError(f"{identifier}: provider request/response set differs")
        for iteration, response in responses.items():
            request_row = requests[iteration]
            if response.get("request_id") != requests[iteration].get("request_id"):
                raise ValueError(f"{identifier}: provider request binding differs")
            if response["event_index"] != requests[iteration]["event_index"] + 1:
                raise ValueError(f"{identifier}: provider lifecycle is not contiguous")
            if response.get("resolved_model_revision") != revision:
                raise ValueError(f"{identifier}: provider response revision differs")
            if request_row.get("terminal_feedback_included") is not False:
                raise ValueError(f"{identifier}: provider request permits terminal feedback")
            prompt_hash = request_row.get("prompt_sha256")
            if (
                not isinstance(prompt_hash, str)
                or len(prompt_hash) != 64
                or any(character not in "0123456789abcdef" for character in prompt_hash)
            ):
                raise ValueError(f"{identifier}: provider prompt hash is invalid")
            previous_eligible = [
                candidate_row
                for candidate_row in rows
                if candidate_row["event_index"] < request_row["event_index"]
                and candidate_row.get("event_type") == "tuning_evaluation"
                and candidate_row.get("eligible") is True
            ]
            feedback_best = min(
                previous_eligible,
                default=None,
                key=lambda candidate_row: (
                    float(candidate_row["median_ms"]),
                    candidate_row["candidate_sha256"],
                ),
            )
            if feedback_best is None:
                if (
                    request_row.get("feedback_candidate_sha256") is not None
                    or request_row.get("feedback_tuning_median_ms") is not None
                ):
                    raise ValueError(f"{identifier}: provider request invents feedback")
            elif (
                request_row.get("feedback_candidate_sha256")
                != feedback_best["candidate_sha256"]
                or not math.isclose(
                    float(request_row.get("feedback_tuning_median_ms", -1)),
                    float(feedback_best["median_ms"]),
                    rel_tol=1e-12,
                )
            ):
                raise ValueError(f"{identifier}: provider tuning feedback differs")
            if response.get("parse_error") is None:
                relative, digest = response.get("candidate_relative_path"), response.get("candidate_sha256")
                if not isinstance(relative, str) or not isinstance(digest, str):
                    raise ValueError(f"{identifier}: parsed response lacks candidate binding")
                _validate_candidate_file(path, relative, digest)
            else:
                if not isinstance(response.get("parse_error"), str) or not response["parse_error"]:
                    raise ValueError(f"{identifier}: parse failure lacks an error")
                if (
                    response.get("candidate_relative_path") is not None
                    or response.get("candidate_sha256") is not None
                ):
                    raise ValueError(
                        f"{identifier}: parse failure unexpectedly binds a candidate"
                    )
        starts = [row for row in rows if row["event_type"] == "tuning_evaluation_start"]
        evaluations = [row for row in rows if row["event_type"] == "tuning_evaluation"]
        parsed_responses = [row for row in responses.values() if row.get("parse_error") is None]
        if len(starts) != len(evaluations) or len(starts) != len(parsed_responses):
            raise ValueError(f"{identifier}: tuning evaluator lifecycle is incomplete")
        for before, after in zip(starts, evaluations):
            if after["event_index"] != before["event_index"] + 1:
                raise ValueError(f"{identifier}: tuning evaluator lifecycle is not contiguous")
            for key in ("iteration", "candidate_relative_path", "candidate_sha256"):
                if before.get(key) != after.get(key):
                    raise ValueError(f"{identifier}: tuning evaluator {key} binding differs")
                response = responses.get(before["iteration"])
                if response is None or response.get(key) != before.get(key):
                    raise ValueError(
                        f"{identifier}: provider-to-evaluator {key} binding differs"
                    )
            if not isinstance(after.get("eligible"), bool):
                raise ValueError(f"{identifier}: tuning eligibility is not boolean")
            build_ok = after.get("build_ok")
            lane_policy_pass = after.get("lane_policy_pass")
            gate_pass = after.get("gate_pass")
            if not all(
                isinstance(value, bool)
                for value in (build_ok, lane_policy_pass, gate_pass)
            ):
                raise ValueError(
                    f"{identifier}: tuning build/lane-policy/gate status is not boolean"
                )
            if after["eligible"] != bool(build_ok and lane_policy_pass and gate_pass):
                raise ValueError(f"{identifier}: tuning eligibility is inconsistent")
            if after["eligible"] and not _finite_positive(after.get("median_ms")):
                raise ValueError(f"{identifier}: eligible tuning result lacks latency")
            if not after["eligible"] and after.get("median_ms") is not None:
                raise ValueError(f"{identifier}: ineligible tuning result retains latency")
            if gate_pass:
                evidence = after.get("gate_summary_sha256")
                if (
                    not isinstance(evidence, str)
                    or len(evidence) != 64
                    or any(character not in "0123456789abcdef" for character in evidence)
                ):
                    raise ValueError(f"{identifier}: tuning gate evidence is invalid")
            if gate_pass and after.get("gpu_uuid") != job["required_gpu_uuid"]:
                raise ValueError(f"{identifier}: tuning GPU UUID differs")

        checkpoints = [row for row in rows if row["event_type"] == "checkpoint_freeze"]
        if [row.get("checkpoint_label") for row in checkpoints] != list(campaign.CHECKPOINT_LABELS):
            raise ValueError(f"{identifier}: checkpoint labels/order differ")
        terminal_starts = [row for row in rows if row["event_type"] == "terminal_evaluation_start"]
        expected_terminal = [row for row in checkpoints if row.get("candidate_sha256") is not None]
        if len(terminal_starts) != len(expected_terminal):
            raise ValueError(f"{identifier}: terminal evaluator lifecycle differs")
        for before, after in zip(terminal_starts, expected_terminal):
            if after["event_index"] != before["event_index"] + 1:
                raise ValueError(f"{identifier}: terminal evaluator lifecycle is not contiguous")
            for key in ("checkpoint_label", "candidate_relative_path", "candidate_sha256"):
                if before.get(key) != after.get(key):
                    raise ValueError(f"{identifier}: terminal evaluator {key} binding differs")

        token_totals = {key: 0 for key in events.USAGE_KEYS if key != "raw_categories"}
        component_totals = {key: 0.0 for key in events.CLOCK_KEYS}
        checkpoint_rows = []
        checkpoint_index = 0
        for row in rows:
            for key in events.CLOCK_KEYS:
                component_totals[key] += float(row[key])
            if row["event_type"] == "provider_response":
                for key in token_totals:
                    token_totals[key] += int(row["usage"][key])
            if row["event_type"] != "checkpoint_freeze":
                continue
            label = campaign.CHECKPOINT_LABELS[checkpoint_index]
            target = campaign.CHECKPOINTS_S[checkpoint_index]
            checkpoint_index += 1
            if row.get("checkpoint_target_active_effort_s") != target:
                raise ValueError(f"{identifier}/{label}: checkpoint target differs")
            overshoot = float(row["cumulative_active_effort_s"]) - target
            if not math.isclose(overshoot, float(row.get("checkpoint_overshoot_s")), abs_tol=1e-6):
                raise ValueError(f"{identifier}/{label}: checkpoint overshoot differs")
            if row.get("terminal_feedback_exposed_to_future_search") is not False:
                raise ValueError(f"{identifier}/{label}: terminal feedback exposure differs")
            available = [
                candidate_row
                for candidate_row in rows
                if candidate_row["event_index"] < row["event_index"]
                and candidate_row.get("event_type") == "tuning_evaluation"
                and candidate_row.get("eligible") is True
            ]
            expected_best = min(
                available,
                default=None,
                key=lambda candidate_row: (
                    float(candidate_row["median_ms"]),
                    candidate_row["candidate_sha256"],
                ),
            )
            if expected_best is None:
                if row.get("candidate_sha256") is not None:
                    raise ValueError(f"{identifier}/{label}: checkpoint invents a candidate")
            elif (
                row.get("candidate_sha256") != expected_best["candidate_sha256"]
                or row.get("candidate_relative_path")
                != expected_best["candidate_relative_path"]
                or not math.isclose(
                    float(row.get("tuning_median_ms", -1)),
                    float(expected_best["median_ms"]),
                    rel_tol=1e-12,
                )
            ):
                raise ValueError(f"{identifier}/{label}: checkpoint did not freeze tuning best")
            outcome = _checkpoint_outcome(row, path)
            if outcome is not None and row["terminal_holdout"].get("gpu_uuid") != job["required_gpu_uuid"]:
                raise ValueError(f"{identifier}/{label}: terminal GPU UUID differs")
            by_lane_checkpoint[(job["lane"], label)].append(outcome)
            selection_id = f"{identifier}.{label}"
            if outcome is not None:
                selections.append(
                    {
                        "selection_id": selection_id,
                        "trajectory_id": identifier,
                        "lane": job["lane"],
                        "replicate": job["replicate"],
                        "checkpoint_label": label,
                        "checkpoint_target_active_effort_s": target,
                        "candidate_relative_path": row["candidate_relative_path"],
                        "candidate_sha256": row["candidate_sha256"],
                        "terminal_median_ms": outcome,
                    }
                )
            checkpoint_rows.append(
                {
                    "checkpoint_label": label,
                    "target_active_effort_s": target,
                    "completed_active_effort_s": row["cumulative_active_effort_s"],
                    "overshoot_s": overshoot,
                    "terminal_eligible": outcome is not None,
                    "terminal_median_ms": outcome,
                    "effort_categories_cumulative_s": dict(component_totals),
                    "token_categories_cumulative": dict(token_totals),
                }
            )
        if not math.isclose(
            sum(component_totals.values()),
            float(rows[-1]["cumulative_active_effort_s"]),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{identifier}: categorized effort does not close")
        trajectories_out.append(
            {
                "trajectory_id": identifier,
                "lane": job["lane"],
                "replicate": job["replicate"],
                "physical_gpu": job["physical_gpu"],
                "event_count": len(rows),
                "event_head_sha256": rows[-1]["event_sha256"],
                "checkpoints": checkpoint_rows,
            }
        )
        event_heads[identifier] = rows[-1]["event_sha256"]

    cells = []
    for label in campaign.CHECKPOINT_LABELS:
        for lane in campaign.PROGRAMMABLE_LANES:
            outcomes = by_lane_checkpoint[(lane, label)]
            if len(outcomes) != campaign.SEARCH_REPLICATES:
                raise ValueError(f"{lane}/{label}: expected five trajectory outcomes")
            finite = [value for value in outcomes if value is not None]
            cells.append(
                {
                    "lane": lane,
                    "checkpoint_label": label,
                    "n_trajectories": 5,
                    "terminal_successes": len(finite),
                    "terminal_failures": 5 - len(finite),
                    "terminal_median_ms_among_successes": (
                        statistics.median(finite) if finite else None
                    ),
                    "outcomes_by_replicate_ms_null_is_failure": outcomes,
                }
            )
    pairwise = []
    for label in campaign.CHECKPOINT_LABELS:
        family = []
        for left, right in itertools.combinations(campaign.PROGRAMMABLE_LANES, 2):
            test = exact_mann_whitney(
                by_lane_checkpoint[(left, label)], by_lane_checkpoint[(right, label)]
            )
            family.append(
                {
                    "checkpoint_label": label,
                    "left": left,
                    "right": right,
                    **test,
                }
            )
        adjusted = _holm(row["p_value_two_sided_exact"] for row in family)
        for row, value in zip(family, adjusted):
            row["holm_adjusted_p_value"] = value
            row["reject_holm_0_05"] = value <= 0.05
        pairwise.extend(family)
    summary = {
        "schema_version": 1,
        "record_type": "effort_frontier_search_analysis",
        "campaign_id": campaign.CAMPAIGN_ID,
        "complete": True,
        "trajectory_count": len(trajectories_out),
        "checkpoint_observation_count": len(trajectories_out) * 3,
        "eligible_confirmation_selection_count": len(selections),
        "failure_ordering": "null ranks worse than every finite latency; nulls tie",
        "trajectories": trajectories_out,
        "frontier_cells": cells,
        "pairwise_lane_tests": pairwise,
        "multiple_testing": "Holm over six lane pairs separately within each checkpoint",
        "event_heads": event_heads,
        "prelaunch_global_bindings": {
            key: common_bindings[key]
            for key in (
                "manifest_sha256",
                "gate_spec_sha256",
                "gate_receipt_sha256",
                "model_resolution_lock_sha256",
                "executor_registry_sha256",
                "prelaunch_provenance_sha256",
                "immutable_model_revision",
            )
        },
    }
    return summary, selections


def build_confirmation_plan(result_root: Path) -> dict[str, Any]:
    search, selections = validate_search(result_root)
    treatments = [
        {"treatment_type": "control", "treatment_id": campaign.CONTROL_LANE}
    ] + [
        {"treatment_type": "candidate", "treatment_id": row["selection_id"], **row}
        for row in selections
    ]
    records = []
    ordinal = 0
    namespace = "MKB-effort-frontier-v1-confirm-order-20260731"
    for distribution in campaign.TIMING_DISTRIBUTIONS:
        for block in range(campaign.CONFIRM_BLOCKS):
            ordered = sorted(
                treatments,
                key=lambda row: hashlib_sha(
                    f"{namespace}|{distribution}|{block}|{row['treatment_id']}"
                ),
            )
            for position, treatment in enumerate(ordered):
                records.append(
                    {
                        "record_index": ordinal,
                        "record_id": f"{distribution}.b{block:02d}.p{position:03d}",
                        "distribution": distribution,
                        "block": block,
                        "position": position,
                        **treatment,
                    }
                )
                ordinal += 1
    return {
        "schema_version": 1,
        "record_type": "effort_frontier_confirmation_plan",
        "campaign_id": campaign.CAMPAIGN_ID,
        "manifest_sha256": campaign.file_sha256(campaign.MANIFEST),
        "search_event_heads": search["event_heads"],
        "physical_gpu": 0,
        "required_gpu_uuid": campaign.GPU_UUIDS[0],
        "blocks": campaign.CONFIRM_BLOCKS,
        "warmup_iterations": campaign.CONFIRM_WARMUP,
        "timed_trials": campaign.CONFIRM_TRIALS,
        "distributions": list(campaign.TIMING_DISTRIBUTIONS),
        "eligible_selection_count": len(selections),
        "records": records,
    }


def hashlib_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            values.append(value)
    return values


def _validate_confirmation_records(plan: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    expected_plan_sha = campaign.canonical_sha256(plan)
    if len(rows) != len(plan["records"]):
        raise ValueError(
            f"confirmation record count differs: {len(rows)}/{len(plan['records'])}"
        )
    previous = events.GENESIS
    for expected, row in zip(plan["records"], rows):
        envelope = {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "record_index": expected["record_index"],
            "previous_record_sha256": previous,
            "confirmation_plan_sha256": expected_plan_sha,
        }
        mismatches = [key for key, value in envelope.items() if row.get(key) != value]
        for key, value in expected.items():
            if row.get(key) != value:
                mismatches.append(key)
        payload = {key: value for key, value in row.items() if key != "record_sha256"}
        digest = campaign.canonical_sha256(payload)
        if row.get("record_sha256") != digest:
            mismatches.append("record_sha256")
        if mismatches:
            raise ValueError(
                f"confirmation record {expected['record_index']} binding differs: {sorted(set(mismatches))}"
            )
        timestamp = row.get("completed_at_utc")
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise ValueError("confirmation record lacks UTC completion time")
        if row.get("ok") is True:
            if row.get("physical_gpu") != 0 or row.get("logical_device") != "cuda:0":
                raise ValueError("confirmation record was not pinned to logical GPU0")
            if row.get("gpu_uuid") != campaign.GPU_UUIDS[0]:
                raise ValueError("confirmation record GPU UUID differs")
            trials = row.get("trial_times_ms")
            if (
                not isinstance(trials, list)
                or len(trials) != campaign.CONFIRM_TRIALS
                or any(not _finite_positive(value) for value in trials)
            ):
                raise ValueError("successful confirmation record has invalid trials")
            median = statistics.median(float(value) for value in trials)
            if not math.isclose(median, float(row.get("median_ms", -1)), rel_tol=1e-12):
                raise ValueError("confirmation median does not match trial list")
            if row["treatment_type"] == "candidate" and (
                row.get("lane_policy_pass") is not True or row.get("gate_pass") is not True
            ):
                raise ValueError(
                    "timed candidate did not pass lane policy and the frozen fused-v2 gate"
                )
            if row["treatment_type"] == "control" and row.get("contract_pass") is not True:
                raise ValueError("timed control did not pass its exact contract check")
            evidence_key = (
                "gate_summary_sha256"
                if row["treatment_type"] == "candidate"
                else "contract_summary_sha256"
            )
            evidence = row.get(evidence_key)
            if (
                not isinstance(evidence, str)
                or len(evidence) != 64
                or any(character not in "0123456789abcdef" for character in evidence)
            ):
                raise ValueError("successful confirmation record lacks evidence hash")
        elif row.get("ok") is not False:
            raise ValueError("confirmation ok field is not boolean")
        elif not isinstance(row.get("error"), str) or not row["error"].strip():
            raise ValueError("failed confirmation record lacks an error")
        previous = digest
    return rows


def _average_ranks(values: list[tuple[int, float]]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = ((cursor + 1) + end) / 2
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _spearman_exact(left: list[float], right: list[float]) -> dict[str, Any]:
    def correlation(a: list[float], b: list[float]) -> float:
        mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
        numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        denominator = math.sqrt(
            sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
        )
        return numerator / denominator if denominator else 0.0

    observed = correlation(left, right)
    permutations = list(itertools.permutations(right))
    extreme = sum(abs(correlation(left, list(candidate))) >= abs(observed) - 1e-12 for candidate in permutations)
    return {
        "spearman_rho": observed,
        "p_value_two_sided_exact": extreme / len(permutations),
        "permutations": len(permutations),
    }


def analyze_confirmation(
    search: dict[str, Any], plan: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    control: dict[tuple[str, int], float] = {}
    candidate: dict[tuple[str, str, int], float] = {}
    failures = []
    for row in rows:
        if row.get("ok") is not True:
            failures.append({"record_id": row["record_id"], "error": row.get("error")})
            continue
        key = (row["distribution"], row["block"])
        if row["treatment_type"] == "control":
            if key in control:
                raise ValueError(f"duplicate confirmation control {key}")
            control[key] = float(row["median_ms"])
        else:
            candidate[(row["treatment_id"], *key)] = float(row["median_ms"])

    selection_defs = {
        row["treatment_id"]: row
        for row in plan["records"]
        if row["treatment_type"] == "candidate"
    }
    summaries = []
    trajectory_ratio: dict[tuple[str, str], float | None] = {}
    for selection_id, definition in sorted(selection_defs.items()):
        # The definition repeats in all blocks/distributions; equality was bound above.
        for distribution in campaign.TIMING_DISTRIBUTIONS:
            ratios = []
            for block in range(campaign.CONFIRM_BLOCKS):
                denominator = control.get((distribution, block))
                numerator = candidate.get((selection_id, distribution, block))
                if denominator is not None and numerator is not None:
                    ratios.append(numerator / denominator)
            complete = len(ratios) == campaign.CONFIRM_BLOCKS
            interval = _median_interval_15(ratios) if complete else None
            trajectory_ratio[(selection_id, distribution)] = (
                float(interval["median"]) if interval else None
            )
            summaries.append(
                {
                    "selection_id": selection_id,
                    "lane": definition["lane"],
                    "replicate": definition["replicate"],
                    "checkpoint_label": definition["checkpoint_label"],
                    "distribution": distribution,
                    "complete_blocks": len(ratios),
                    "complete": complete,
                    "candidate_over_control_ratio_interval": interval,
                }
            )

    # Reintroduce terminal-ineligible search outcomes as null failures so every
    # lane comparison retains five independent trajectories.
    eligible_ids = set(selection_defs)
    ratios_by_cell: dict[tuple[str, str, str], list[float | None]] = {}
    for lane in campaign.PROGRAMMABLE_LANES:
        for label in campaign.CHECKPOINT_LABELS:
            for distribution in campaign.TIMING_DISTRIBUTIONS:
                values = []
                for replicate in range(campaign.SEARCH_REPLICATES):
                    sid = f"effort_v1.{lane}.r{replicate}.{label}"
                    values.append(
                        trajectory_ratio.get((sid, distribution)) if sid in eligible_ids else None
                    )
                ratios_by_cell[(lane, label, distribution)] = values

    pairwise = []
    control_tests = []
    for label in campaign.CHECKPOINT_LABELS:
        for distribution in campaign.TIMING_DISTRIBUTIONS:
            family = []
            for left, right in itertools.combinations(campaign.PROGRAMMABLE_LANES, 2):
                family.append(
                    {
                        "checkpoint_label": label,
                        "distribution": distribution,
                        "left": left,
                        "right": right,
                        **exact_mann_whitney(
                            ratios_by_cell[(left, label, distribution)],
                            ratios_by_cell[(right, label, distribution)],
                        ),
                    }
                )
            for row, adjusted in zip(
                family, _holm(item["p_value_two_sided_exact"] for item in family)
            ):
                row["holm_adjusted_p_value"] = adjusted
                row["reject_holm_0_05"] = adjusted <= 0.05
            pairwise.extend(family)

            family_control = []
            for lane in campaign.PROGRAMMABLE_LANES:
                finite = [
                    value
                    for value in ratios_by_cell[(lane, label, distribution)]
                    if value is not None
                ]
                missing = 5 - len(finite)
                # A missing/gate-failed implementation is worse than the legal
                # control, consistent with the global frozen failure ordering.
                test = exact_sign_test(finite + [math.inf] * missing)
                ranked = sorted(finite) + [None] * missing
                ranked_median = ranked[2] if ranked[2] is not None else None
                family_control.append(
                    {
                        "checkpoint_label": label,
                        "distribution": distribution,
                        "lane": lane,
                        "terminal_and_timing_successes": len(finite),
                        "failures": missing,
                        "median_candidate_over_control_among_successes": (
                            statistics.median(finite) if finite else None
                        ),
                        "ranked_median_candidate_over_control_null_is_failure": ranked_median,
                        **test,
                    }
                )
            for row, adjusted in zip(
                family_control,
                _holm(item["p_value_two_sided_exact"] for item in family_control),
            ):
                row["holm_adjusted_p_value"] = adjusted
                row["reject_holm_0_05"] = adjusted <= 0.05
            control_tests.extend(family_control)

    stability = []
    for label in campaign.CHECKPOINT_LABELS:
        lane_scores: dict[str, dict[str, tuple[int, float]]] = defaultdict(dict)
        for lane in campaign.PROGRAMMABLE_LANES:
            for distribution in campaign.TIMING_DISTRIBUTIONS:
                values = ratios_by_cell[(lane, label, distribution)]
                finite = [value for value in values if value is not None]
                lane_scores[lane][distribution] = (
                    5 - len(finite),
                    statistics.median(finite) if finite else math.inf,
                )
        positive = _average_ranks(
            [lane_scores[lane][campaign.TIMING_DISTRIBUTIONS[0]] for lane in campaign.PROGRAMMABLE_LANES]
        )
        signed = _average_ranks(
            [lane_scores[lane][campaign.TIMING_DISTRIBUTIONS[1]] for lane in campaign.PROGRAMMABLE_LANES]
        )
        paired = []
        for lane in campaign.PROGRAMMABLE_LANES:
            ratios = []
            for replicate in range(5):
                sid = f"effort_v1.{lane}.r{replicate}.{label}"
                pos = trajectory_ratio.get((sid, campaign.TIMING_DISTRIBUTIONS[0]))
                sig = trajectory_ratio.get((sid, campaign.TIMING_DISTRIBUTIONS[1]))
                if pos is not None and sig is not None:
                    ratios.append(sig / pos)
            paired.append(
                {
                    "lane": lane,
                    "signed_over_positive_control_normalized_ratio_median": (
                        statistics.median(ratios) if ratios else None
                    ),
                    **exact_sign_test(ratios),
                }
            )
        for row, adjusted in zip(
            paired,
            _holm(item["p_value_two_sided_exact"] for item in paired),
        ):
            row["holm_adjusted_p_value"] = adjusted
            row["reject_holm_0_05"] = adjusted <= 0.05
        stability.append(
            {
                "checkpoint_label": label,
                "lane_order": list(campaign.PROGRAMMABLE_LANES),
                "positive_ranks": positive,
                "signed_ranks": signed,
                **_spearman_exact(positive, signed),
                "paired_distribution_ratios": paired,
            }
        )
    for row, adjusted in zip(
        stability,
        _holm(item["p_value_two_sided_exact"] for item in stability),
    ):
        row["holm_adjusted_p_value_across_checkpoints"] = adjusted
        row["reject_holm_0_05_across_checkpoints"] = adjusted <= 0.05

    return {
        "schema_version": 1,
        "record_type": "effort_frontier_confirmation_analysis",
        "campaign_id": campaign.CAMPAIGN_ID,
        "complete": not failures,
        "expected_records": len(plan["records"]),
        "observed_records": len(rows),
        "failed_records": failures,
        "selection_distribution_summaries": summaries,
        "pairwise_lane_tests": pairwise,
        "control_tests": control_tests,
        "distribution_rank_stability": stability,
        "analysis_unit_note": (
            "five search trajectories are inferential units; 15 GPU0 blocks are paired "
            "technical replicates summarized before lane inference"
        ),
        "claim_limit": "one implementation-specific workload on one RTX 6000 Ada GPU",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--confirmation-records", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    search, _ = validate_search(args.result_root)
    result: dict[str, Any] = {"search": search}
    if bool(args.confirmation_plan) != bool(args.confirmation_records):
        raise SystemExit("confirmation plan and records must be provided together")
    if args.confirmation_plan:
        supplied = campaign.load_json(args.confirmation_plan)
        expected = build_confirmation_plan(args.result_root)
        if supplied != expected or args.confirmation_plan.read_bytes() != campaign.stable_json_bytes(supplied):
            raise ValueError("confirmation plan differs from completed search or is noncanonical")
        rows = _validate_confirmation_records(expected, args.confirmation_records)
        result["confirmation"] = analyze_confirmation(search, expected, rows)
    campaign.atomic_write(args.out, campaign.stable_json_bytes(result))
    print(
        f"validated {search['trajectory_count']} trajectories and "
        f"{search['checkpoint_observation_count']} checkpoint observations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
