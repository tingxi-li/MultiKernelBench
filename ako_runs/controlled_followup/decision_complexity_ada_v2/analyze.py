#!/usr/bin/env python3
"""Validate and summarize the non-controlling Ada C2 pilot."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from . import protocol
except ImportError:  # direct script execution
    from ako_runs.controlled_followup.decision_complexity_ada_v2 import protocol

from ako_runs.controlled_followup.convergence_v2.analyze import (  # pure primitives only
    SurvivalObservation,
    holm_adjust,
    kaplan_meier,
    restricted_mean_survival_time,
)


def _runner_module():
    from ako_runs.controlled_followup.decision_complexity_ada_v2 import runner

    return runner


def _positive_times(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != protocol.TIMING_TRIALS:
        raise protocol.ProtocolError(f"{label} must contain 100 trials")
    rows = [float(item) for item in value]
    if any(not math.isfinite(item) or item <= 0 for item in rows):
        raise protocol.ProtocolError(f"{label} contains an invalid timing")
    return rows


def _survival_descriptives(
    observations: list[SurvivalObservation], preregistered_tau: float,
    duration_unit: str,
) -> dict[str, Any]:
    by_group = {
        group: [row for row in observations if row.group == group]
        for group in sorted({row.group for row in observations})
    }
    common_support = min(max(row.duration_s for row in rows) for rows in by_group.values())
    effective_tau = min(float(preregistered_tau), common_support)
    if effective_tau <= 0:
        raise protocol.ProtocolError("RMST common observed support is empty")
    groups = {}
    for group, members in by_group.items():
        curve = [
            {
                "duration": point["time_s"],
                "survival": point["survival"],
                "at_risk": point["at_risk"],
                "events": point["events"],
                "censored": point["censored"],
            }
            for point in kaplan_meier(members)
        ]
        groups[group] = {
            "n": len(members),
            "events": sum(row.event_observed for row in members),
            "right_censored": sum(not row.event_observed for row in members),
            "rmst": restricted_mean_survival_time(members, effective_tau),
            "kaplan_meier": curve,
        }
    return {
        "common_observed_support": common_support,
        "duration_unit": duration_unit,
        "effective_tau": effective_tau,
        "groups": groups,
        "preregistered_tau": float(preregistered_tau),
        "rmst_extrapolated": False,
        "support_policy": "min_preregistered_tau_and_common_observed_support",
    }


def _exact_sign_flip(differences: list[float]) -> dict[str, Any]:
    if not differences or any(not math.isfinite(value) for value in differences):
        raise protocol.ProtocolError("paired contrast differences are invalid")
    observed = abs(statistics.mean(differences))
    tolerance = 1e-12 * max(1.0, observed, *(abs(value) for value in differences))
    randomized = [
        abs(statistics.mean(sign * value for sign, value in zip(signs, differences)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return {
        "differences_by_replicate": differences,
        "mean_difference": statistics.mean(differences),
        "median_difference": statistics.median(differences),
        "n_paired_replicates": len(differences),
        "two_sided_exact_p_value": sum(
            value + tolerance >= observed for value in randomized
        ) / len(randomized),
    }


def _validate_retained_authorization(
    receipt: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    authorization = receipt.get("authorization")
    if (
        not isinstance(authorization, dict)
        or protocol.canonical_sha256(authorization) != receipt.get("authorization_sha256")
        or any(authorization.get(key) != value for key, value in expected.items())
        or not isinstance(authorization.get("authorization_nonce"), str)
        or len(authorization["authorization_nonce"]) != 64
        or authorization.get("spawn_requested_unix_ns")
        != receipt.get("child_launched_unix_ns")
        or authorization.get("authorized_unix_ns")
        != receipt.get("child_authorized_unix_ns")
    ):
        raise protocol.ProtocolError("retained parent authorization is incomplete or foreign")
    return authorization


def validate_attempt(contract: dict[str, Any], row: dict[str, Any], record: dict[str, Any]) -> None:
    index = record.get("attempt_index")
    if not isinstance(index, int) or index not in range(1, len(row["execution_contract"]["candidate_order"]) + 1):
        raise protocol.ProtocolError("attempt index is outside the frozen trajectory")
    expected = {
        "authorization_sha256": record.get("authorization_sha256"),
        "cache_isolation": protocol.attempt_cache_receipt(
            row,
            row["execution_contract"]["candidate_order"][index - 1],
            index,
        ),
        "campaign_id": protocol.CAMPAIGN_ID,
        "execution_lock_sha256": protocol.file_sha256(
            protocol.HERE / "execution_lock.json"
        ),
        "manifest_sha256": protocol.file_sha256(protocol.HERE / "manifest.json"),
        "record_type": "decision_complexity_ada_v2_attempt",
        "schema_version": 1,
        "trajectory_id": row["trajectory_id"],
        "candidate_cell_id": row["execution_contract"]["candidate_order"][index - 1],
        "gpu_slot": row["gpu_slot"],
        "gpu_uuid": row["gpu_uuid"],
        "gpu_lock_paths": list(protocol.GPU_LOCK_PATHS),
        "toolchain_sha256": protocol.read_json(
            protocol.HERE / "execution_lock.json"
        ).get("toolchain_sha256"),
    }
    if (
        any(record.get(key) != value for key, value in expected.items())
        or not isinstance(record.get("authorization_sha256"), str)
        or len(record["authorization_sha256"]) != 64
        or not isinstance(record.get("parent_pid"), int)
        or record["parent_pid"] <= 0
        or not isinstance(record.get("child_pid"), int)
        or record["child_pid"] <= 0
    ):
        raise protocol.ProtocolError("attempt lost its manifest/GPU coordinate")
    preflight = record.get("physical_gpu_preflight")
    expected_preflight = {
        "index": str(row["gpu_slot"]),
        "uuid": row["gpu_uuid"],
        "name": contract["hardware"]["name"],
        "compute_cap": contract["hardware"]["compute_capability"],
    }
    if (
        not isinstance(preflight, dict)
        or set(preflight) != {*expected_preflight, "driver_version"}
        or any(preflight.get(key) != value for key, value in expected_preflight.items())
        or not isinstance(preflight.get("driver_version"), str)
        or not preflight["driver_version"]
    ):
        raise protocol.ProtocolError("attempt lost its physical-GPU preflight binding")
    status = record.get("terminal_status")
    if status not in protocol.ATTEMPT_STATUSES:
        raise protocol.ProtocolError("unknown attempt terminal status")
    active = record.get("active_s")
    if isinstance(active, bool) or not isinstance(active, (int, float)) or not math.isfinite(active) or active <= 0 or active > contract["attempt_timeout_s"] + 30:
        raise protocol.ProtocolError("attempt active time is invalid")
    timing_fields = {
        "candidate_times_ms", "target_times_ms", "candidate_tail_median_ms",
        "target_tail_median_ms", "ratio_to_target",
    }
    if status != "GATE_PASSED":
        if any(field in record for field in timing_fields):
            raise protocol.ProtocolError("failed attempt carries timing outcomes")
        return
    candidate = _positive_times(record.get("candidate_times_ms"), "candidate_times_ms")
    target = _positive_times(record.get("target_times_ms"), "target_times_ms")
    c_tail = statistics.median(candidate[protocol.TAIL_START:protocol.TAIL_STOP])
    t_tail = statistics.median(target[protocol.TAIL_START:protocol.TAIL_STOP])
    if not (
        math.isclose(record.get("candidate_tail_median_ms", -1), c_tail)
        and math.isclose(record.get("target_tail_median_ms", -1), t_tail)
        and math.isclose(record.get("ratio_to_target", -1), c_tail / t_tail)
        and record.get("measurement_order") == protocol.timing_pair_order(row, index)
    ):
        raise protocol.ProtocolError("settled-tail paired timing is not re-derived")
    materials = {item["cell_id"]: item for item in contract["materials"]["candidates"]}
    material = materials[record["candidate_cell_id"]]
    target_material = materials[protocol.TARGET_CELL_ID]
    if (
        record.get("artifacts_sha256") != material["artifacts_sha256"]
        or record.get("implementation_sha256") != material["implementation_sha256"]
        or record.get("target_artifacts_sha256")
        != target_material["artifacts_sha256"]
        or record.get("target_implementation_sha256")
        != target_material["implementation_sha256"]
        or record.get("target_gate_evidence_reused")
        != {"path": target_material["gate_path"], "sha256": target_material["gate_sha256"]}
        or record.get("gate_evidence_reused") != {"path": material["gate_path"], "sha256": material["gate_sha256"]}
        or record.get("live_gate", {}).get("candidate", {}).get("gate_pass") is not True
        or record.get("live_gate", {}).get("target", {}).get("gate_pass") is not True
        or set(record.get("warmup_iterations", {})) != {"candidate", "target"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in record.get("warmup_iterations", {}).values()
        )
    ):
        raise protocol.ProtocolError("attempt lost implementation/gate evidence binding")


def validate_attempt_parent_receipt(
    contract: dict[str, Any], row: dict[str, Any], record: dict[str, Any],
    receipt: dict[str, Any], raw_path: Path, previous_completed: int,
    trajectory_authorization_sha256: str, trajectory_child_pid: int,
) -> int:
    runner = _runner_module()
    launched = receipt.get("child_launched_unix_ns")
    authorized = receipt.get("child_authorized_unix_ns")
    completed = receipt.get("child_completed_unix_ns")
    expected = {
        "attempt_index": record["attempt_index"],
        "authorization_sha256": record["authorization_sha256"],
        "campaign_id": protocol.CAMPAIGN_ID,
        "candidate_cell_id": record["candidate_cell_id"],
        "child_pid": record["child_pid"],
        "gpu_slot": row["gpu_slot"],
        "gpu_uuid": row["gpu_uuid"],
        "parent_pid": record["parent_pid"],
        "raw_path": str(raw_path.relative_to(protocol.REPO_ROOT)),
        "raw_sha256": protocol.file_sha256(raw_path),
        "record_type": "decision_complexity_ada_v2_attempt_parent_receipt",
        "schema_version": 1,
        "trajectory_authorization_sha256": trajectory_authorization_sha256,
        "trajectory_id": row["trajectory_id"],
    }
    authorization_record = _validate_retained_authorization(
        receipt,
        {
            "attempt_index": record["attempt_index"],
            "campaign_id": protocol.CAMPAIGN_ID,
            "candidate_cell_id": record["candidate_cell_id"],
            "canonical_output": str(raw_path.resolve()),
            "child_pid": record["child_pid"],
            "execution_lock_sha256": protocol.file_sha256(runner.LOCK_PATH),
            "gpu_lock_paths": [str(path) for path in runner.gpu_lock_paths()],
            "gpu_slot": row["gpu_slot"],
            "gpu_uuid": row["gpu_uuid"],
            "launch_receipt_sha256": protocol.file_sha256(
                runner.RESULTS / "launch_receipt.json"
            ),
            "manifest_sha256": protocol.file_sha256(runner.MANIFEST_PATH),
            "parent_pid": record["parent_pid"],
            "scope": "attempt",
            "trajectory_authorization_sha256": trajectory_authorization_sha256,
            "trajectory_id": row["trajectory_id"],
        },
    )
    staging_output = Path(str(authorization_record.get("staging_output", "")))
    if not staging_output.is_absolute() or staging_output == raw_path.resolve():
        raise protocol.ProtocolError("attempt authorization did not use isolated staging output")
    if (
        any(receipt.get(key) != value for key, value in expected.items())
        or record.get("parent_pid") != trajectory_child_pid
        or not all(isinstance(value, int) for value in (launched, authorized, completed))
        or launched <= 0
        or launched < previous_completed
        or not (launched <= authorized <= completed)
    ):
        raise protocol.ProtocolError("attempt parent receipt lost PID/timestamp/order binding")
    if record["terminal_status"] == "TIMEOUT":
        expected_code = 124
    else:
        expected_code = record.get("child_exit_code", 0)
    if receipt.get("returncode") != expected_code:
        raise protocol.ProtocolError("attempt parent receipt return code is inconsistent")
    runner.validate_idle_gpu_evidence(receipt.get("gpu_preflight"), row["gpu_slot"], contract)
    runner.validate_idle_gpu_evidence(receipt.get("gpu_postflight"), row["gpu_slot"], contract)
    pre_end = receipt["gpu_preflight"]["occupancy"]["query_completed_unix_ns"]
    post_start = receipt["gpu_postflight"]["occupancy"]["query_started_unix_ns"]
    if pre_end > launched or post_start < completed:
        raise protocol.ProtocolError("attempt occupancy evidence does not bracket the child")
    return completed


def derive_trajectory(
    contract: dict[str, Any], row: dict[str, Any], attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    if not attempts:
        raise protocol.ProtocolError("trajectory has no charged attempts")
    event_index = None
    active_s = 0.0
    for expected_index, attempt in enumerate(attempts, 1):
        validate_attempt(contract, row, attempt)
        if attempt["attempt_index"] != expected_index:
            raise protocol.ProtocolError("trajectory attempt indexes are not contiguous")
        if event_index is not None:
            raise protocol.ProtocolError("trajectory continued after its terminal event")
        active_s += float(attempt["active_s"])
        if attempt["terminal_status"] == "GATE_PASSED" and attempt["ratio_to_target"] <= contract["terminal_ratio"]:
            event_index = expected_index
    if event_index is None and len(attempts) != len(row["execution_contract"]["candidate_order"]):
        raise protocol.ProtocolError("non-event trajectory stopped before exhausting its frozen candidates")
    return {
        "active_s": active_s,
        "arm": row["arm"],
        "attempts_consumed": len(attempts),
        "event_observed": event_index is not None,
        "first_event_attempt": event_index,
        "gpu_slot": row["gpu_slot"],
        "manifest_row_sha256": protocol.canonical_sha256(row),
        "replicate": row["replicate"],
        "trajectory_id": row["trajectory_id"],
    }


def validate_trajectory_parent_receipt(
    contract: dict[str, Any], row: dict[str, Any], completion: dict[str, Any],
    receipt: dict[str, Any], completion_path: Path, launch_sha256: str,
    previous_completion: dict[str, Any] | None,
) -> tuple[int, int]:
    runner = _runner_module()
    launched = receipt.get("child_launched_unix_ns")
    authorized = receipt.get("child_authorized_unix_ns")
    completed = receipt.get("child_completed_unix_ns")
    started = completion.get("trajectory_started_unix_ns")
    ended = completion.get("trajectory_ended_unix_ns")
    expected = {
        "authorization_sha256": completion.get("trajectory_authorization_sha256"),
        "campaign_id": protocol.CAMPAIGN_ID,
        "child_pid": completion.get("trajectory_child_pid"),
        "completion_path": str(completion_path.relative_to(protocol.REPO_ROOT)),
        "completion_sha256": protocol.file_sha256(completion_path),
        "gpu_slot": row["gpu_slot"],
        "gpu_uuid": row["gpu_uuid"],
        "launch_receipt_sha256": launch_sha256,
        "launch_sequence": row["launch_sequence"],
        "parent_pid": completion.get("trajectory_parent_pid"),
        "previous_completion": previous_completion,
        "record_type": "decision_complexity_ada_v2_trajectory_parent_receipt",
        "returncode": 0,
        "schema_version": 1,
        "trajectory_id": row["trajectory_id"],
        "trajectory_ended_unix_ns": ended,
        "trajectory_started_unix_ns": started,
    }
    _validate_retained_authorization(
        receipt,
        {
            "campaign_id": protocol.CAMPAIGN_ID,
            "child_pid": completion.get("trajectory_child_pid"),
            "execution_lock_sha256": protocol.file_sha256(runner.LOCK_PATH),
            "gpu_lock_paths": [str(path) for path in runner.gpu_lock_paths()],
            "gpu_slot": row["gpu_slot"],
            "gpu_uuid": row["gpu_uuid"],
            "launch_receipt_sha256": launch_sha256,
            "launch_sequence": row["launch_sequence"],
            "manifest_sha256": protocol.file_sha256(runner.MANIFEST_PATH),
            "parent_pid": completion.get("trajectory_parent_pid"),
            "previous_completion": previous_completion,
            "scope": "trajectory",
            "trajectory_id": row["trajectory_id"],
        },
    )
    if (
        any(receipt.get(key) != value for key, value in expected.items())
        or not isinstance(expected["authorization_sha256"], str)
        or len(expected["authorization_sha256"]) != 64
        or not isinstance(expected["child_pid"], int)
        or not isinstance(expected["parent_pid"], int)
        or not all(
            isinstance(value, int)
            for value in (launched, authorized, completed, started, ended)
        )
        or launched <= 0
        or not launched <= authorized <= started < ended <= completed
        or completion.get("previous_completion") != previous_completion
        or (
            previous_completion is not None
            and launched <= previous_completion["trajectory_ended_unix_ns"]
        )
    ):
        raise protocol.ProtocolError("trajectory parent receipt lost PID/timestamp/order binding")
    for field in ("gpu_preflight", "gpu_postflight"):
        runner.validate_idle_gpu_evidence(receipt.get(field), row["gpu_slot"], contract)
    for field in ("trajectory_gpu_preflight", "trajectory_gpu_postflight"):
        runner.validate_idle_gpu_evidence(completion.get(field), row["gpu_slot"], contract)
    if (
        receipt["gpu_preflight"]["occupancy"]["query_completed_unix_ns"] > launched
        or receipt["gpu_postflight"]["occupancy"]["query_started_unix_ns"] < completed
        or completion["trajectory_gpu_preflight"]["occupancy"]["query_started_unix_ns"] < started
        or completion["trajectory_gpu_postflight"]["occupancy"]["query_completed_unix_ns"] > ended
    ):
        raise protocol.ProtocolError("trajectory occupancy evidence does not bracket the child")
    return launched, ended


def summarize(contract: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 24 or len({row["trajectory_id"] for row in rows}) != 24:
        raise protocol.ProtocolError("analysis requires exactly 24 trajectories")
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    if set(by_arm) != set(protocol.ARMS) or any(len(value) != 4 for value in by_arm.values()):
        raise protocol.ProtocolError("analysis arm/replicate census drift")
    attempt_observations = [
        SurvivalObservation(row["trajectory_id"], row["arm"], float(row["attempts_consumed"]), row["event_observed"])
        for row in rows
    ]
    active_observations = [
        SurvivalObservation(row["trajectory_id"], row["arm"], float(row["active_s"]), row["event_observed"])
        for row in rows
    ]
    arm_summaries = {}
    for arm, members in sorted(by_arm.items()):
        attempt_values = [float(row["attempts_consumed"]) for row in members]
        active_values = [float(row["active_s"]) for row in members]
        arm_observations = [item for item in attempt_observations if item.group == arm]
        cap = len(next(row for row in protocol.make_manifest(contract)["rows"] if row["arm"] == arm)["execution_contract"]["candidate_order"])
        rmst_tau = min(float(cap), max(attempt_values))
        arm_summaries[arm] = {
            "attempt_mean": statistics.mean(attempt_values),
            "attempt_variance": statistics.variance(attempt_values),
            "active_s_mean": statistics.mean(active_values),
            "active_s_variance": statistics.variance(active_values),
            "candidate_cap": cap,
            "events": sum(row["event_observed"] for row in members),
            "rmst_attempts": restricted_mean_survival_time(arm_observations, rmst_tau),
            "rmst_attempts_tau": rmst_tau,
        }
    by_key = {(row["replicate"], row["arm"]): row for row in rows}
    contrast_results: dict[str, dict[str, Any]] = {}
    raw_p_values: dict[str, float] = {}
    for left, right in contract["pilot_analysis"]["paired_contrasts"]:
        contrast_name = f"{right}_minus_{left}"
        contrast_results[contrast_name] = {
            "left_arm": left,
            "right_arm": right,
            "outcomes": {},
        }
        for outcome in contract["pilot_analysis"]["paired_outcomes"]:
            differences = [
                float(by_key[(replicate, right)][outcome])
                - float(by_key[(replicate, left)][outcome])
                for replicate in range(protocol.REPLICATES)
            ]
            test = _exact_sign_flip(differences)
            contrast_results[contrast_name]["outcomes"][outcome] = test
            raw_p_values[f"{contrast_name}.{outcome}"] = test[
                "two_sided_exact_p_value"
            ]
    adjusted = holm_adjust(raw_p_values)
    for name, value in adjusted.items():
        contrast, outcome = name.split(".", 1)
        contrast_results[contrast]["outcomes"][outcome][
            "holm_adjusted_p_value"
        ] = value

    control_rows = []
    sham_fields = contract["pilot_analysis"]["control_checks"]["label_sham"][
        "exact_match_fields_within_replicate"
    ]
    for replicate in range(protocol.REPLICATES):
        sham_a, sham_b = by_key[(replicate, "label_sham_a")], by_key[(replicate, "label_sham_b")]
        hint = by_key[(replicate, "sensitivity_target_hint")]
        control_rows.append({
            "replicate": replicate,
            "label_sham_exact_match": all(
                sham_a[field] == sham_b[field] for field in sham_fields
            ),
            "label_sham_b_minus_a_attempts": sham_b["attempts_consumed"] - sham_a["attempts_consumed"],
            "label_sham_b_minus_a_active_s": sham_b["active_s"] - sham_a["active_s"],
            "target_hint_event_at_first_attempt": hint["event_observed"] and hint["first_event_attempt"] == 1,
        })
    label_sham_passed = all(row["label_sham_exact_match"] for row in control_rows)
    target_hint_passed = sum(
        bool(row["target_hint_event_at_first_attempt"]) for row in control_rows
    ) == contract["pilot_analysis"]["control_checks"]["target_hint"][
        "event_at_first_attempt_required_replicates"
    ]
    pilot_valid = label_sham_passed and target_hint_passed
    tau = contract["pilot_analysis"]["survival_tau"]
    return {
        "arm_summaries": arm_summaries,
        "attempt_survival": _survival_descriptives(
            attempt_observations, float(tau["attempts"]), "attempts"
        ),
        "active_time_survival": _survival_descriptives(
            active_observations, float(tau["active_s"]), "seconds"
        ),
        "campaign_id": protocol.CAMPAIGN_ID,
        "claim_policy": (
            "noncontrolling paired variance/time pilot only; do not infer that "
            "complex kernels generally converge more slowly"
            if pilot_valid else
            "invalid pilot because a preregistered control failed; no treatment "
            "interpretation is permitted"
        ),
        "complete": True,
        "controlling": False,
        "control_validation": {
            "label_sham_passed": label_sham_passed,
            "passed": pilot_valid,
            "preregistered_checks": contract["pilot_analysis"]["control_checks"],
            "replicates": control_rows,
            "target_hint_passed": target_hint_passed,
        },
        "multiplicity": contract["pilot_analysis"]["multiplicity"],
        "paired_treatment_contrasts": contrast_results,
        "paired_test": contract["pilot_analysis"]["paired_test"],
        "interpretation_valid": pilot_valid,
        "pilot_valid": pilot_valid,
        "record_type": "decision_complexity_ada_v2_pilot_analysis",
        "schema_version": 1,
        "trajectory_count": 24,
    }


def validate_serial_intervals(intervals: list[dict[str, Any]]) -> None:
    if [item.get("launch_sequence") for item in intervals] != list(range(1, 25)):
        raise protocol.ProtocolError("serialized trajectory sequence is incomplete")
    previous = None
    for item in intervals:
        launched = item.get("child_launched_unix_ns")
        ended = item.get("trajectory_ended_unix_ns")
        if (
            not isinstance(launched, int)
            or not isinstance(ended, int)
            or launched <= 0
            or ended <= launched
            or item.get("previous_completion") != previous
            or (
                previous is not None
                and launched <= previous["trajectory_ended_unix_ns"]
            )
        ):
            raise protocol.ProtocolError("serialized trajectories overlap or lose predecessor binding")
        previous = item["completion_binding"]


def load_completed() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runner = _runner_module()
    contract, manifest, lock = runner.validate_frozen()
    launch_path, status_path = runner.RESULTS / "launch_receipt.json", runner.RESULTS / "run_status.json"
    provenance = protocol.read_json(runner.PROVENANCE_PATH)
    if (
        provenance.get("campaign_id") != protocol.CAMPAIGN_ID
        or provenance.get("execution_lock_sha256") != protocol.file_sha256(runner.LOCK_PATH)
        or provenance.get("record_type") != "decision_complexity_ada_v2_prelaunch_provenance"
        or provenance.get("remote_push_verified") is not True
        or provenance.get("schema_version") != 1
        or provenance.get("source_bundle_sha256") != lock["source_bundle_sha256"]
        or provenance.get("toolchain") != lock["toolchain"]
        or provenance.get("toolchain_sha256") != lock["toolchain_sha256"]
        or provenance.get("gpu_lock_paths") != [str(path) for path in runner.gpu_lock_paths()]
    ):
        raise protocol.ProtocolError("prelaunch provenance lost frozen toolchain/source binding")
    retained_preflights = provenance.get("gpu_preflights")
    if not isinstance(retained_preflights, list) or len(retained_preflights) != 1:
        raise protocol.ProtocolError("prelaunch provenance lacks idle-GPU evidence")
    for gpu, evidence in enumerate(retained_preflights):
        runner.validate_idle_gpu_evidence(evidence, gpu, contract)
    if provenance.get("gpus") != [item["gpu"] for item in retained_preflights]:
        raise protocol.ProtocolError("prelaunch GPU identities differ from retained evidence")
    launch, status = protocol.read_json(launch_path), protocol.read_json(status_path)
    launch_sha256 = protocol.file_sha256(launch_path)
    launch_expected = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "execution_lock_sha256": protocol.file_sha256(runner.LOCK_PATH),
        "gpu_lock_paths": [str(path) for path in runner.gpu_lock_paths()],
        "manifest_sha256": protocol.file_sha256(runner.MANIFEST_PATH),
        "prelaunch_provenance_sha256": protocol.file_sha256(runner.PROVENANCE_PATH),
        "record_type": "decision_complexity_ada_v2_launch_receipt",
        "requested_trajectories": 24,
        "schema_version": 1,
        "toolchain_sha256": lock["toolchain_sha256"],
    }
    if (
        any(launch.get(key) != value for key, value in launch_expected.items())
        or not isinstance(launch.get("created_utc"), str)
        or not isinstance(launch.get("launcher_pid"), int)
        or launch["launcher_pid"] <= 0
        or not isinstance(launch.get("campaign_gpu_preflight"), list)
        or len(launch["campaign_gpu_preflight"]) != 1
    ):
        raise protocol.ProtocolError("pilot launch receipt is incomplete or foreign")
    for gpu, evidence in enumerate(launch["campaign_gpu_preflight"]):
        runner.validate_idle_gpu_evidence(evidence, gpu, contract)
    expected_parent_names = {
        f"{row['trajectory_id']}.json" for row in manifest["rows"]
    }
    observed_parent_names = {
        path.name for path in (runner.RESULTS / "trajectory_parent_receipts").glob("*.json")
    }
    observed_trajectory_dirs = {
        path.name for path in (runner.RESULTS / "trajectories").iterdir() if path.is_dir()
    }
    if (
        observed_parent_names != expected_parent_names
        or observed_trajectory_dirs != {row["trajectory_id"] for row in manifest["rows"]}
    ):
        raise protocol.ProtocolError("trajectory/parent-receipt census differs from the manifest")
    summaries = []
    parent_hashes = {}
    completion_hashes = {}
    serial_intervals = []
    for row in manifest["rows"]:
        root = runner.RESULTS / "trajectories" / row["trajectory_id"]
        completion_path = root / "completion.json"
        parent_path = runner.RESULTS / "trajectory_parent_receipts" / f"{row['trajectory_id']}.json"
        completion = protocol.read_json(completion_path)
        parent = protocol.read_json(parent_path)
        receipts = completion.get("attempt_receipts")
        if not isinstance(receipts, list):
            raise protocol.ProtocolError("trajectory completion lacks attempts")
        attempts = []
        previous_completed = 0
        first_attempt_launched = None
        expected_raw_names = set()
        expected_attempt_parent_names = set()
        for attempt_index, binding in enumerate(receipts, 1):
            path = protocol.REPO_ROOT / binding.get("path", "")
            attempt_parent_path = protocol.REPO_ROOT / binding.get("parent_path", "")
            canonical_raw = root / "attempts" / f"attempt{attempt_index:02d}.json"
            canonical_parent = root / "attempt_parent_receipts" / f"attempt{attempt_index:02d}.json"
            if (
                path.resolve() != canonical_raw.resolve()
                or attempt_parent_path.resolve() != canonical_parent.resolve()
                or not path.is_file()
                or protocol.file_sha256(path) != binding.get("sha256")
                or not attempt_parent_path.is_file()
                or protocol.file_sha256(attempt_parent_path) != binding.get("parent_sha256")
            ):
                raise protocol.ProtocolError("attempt receipt hash changed")
            attempt = protocol.read_json(path)
            attempt_parent = protocol.read_json(attempt_parent_path)
            if first_attempt_launched is None:
                first_attempt_launched = attempt_parent.get("child_launched_unix_ns")
            validate_attempt(contract, row, attempt)
            previous_completed = validate_attempt_parent_receipt(
                contract, row, attempt, attempt_parent, path, previous_completed,
                completion.get("trajectory_authorization_sha256"),
                completion.get("trajectory_child_pid"),
            )
            attempts.append(attempt)
            expected_raw_names.add(path.name)
            expected_attempt_parent_names.add(attempt_parent_path.name)
        if (
            {path.name for path in (root / "attempts").glob("*.json")} != expected_raw_names
            or {path.name for path in (root / "attempt_parent_receipts").glob("*.json")}
            != expected_attempt_parent_names
        ):
            raise protocol.ProtocolError("attempt file census differs from completion bindings")
        derived = derive_trajectory(contract, row, attempts)
        for key, value in derived.items():
            if completion.get(key) != value:
                raise protocol.ProtocolError("trajectory completion is not attempt-derived")
        if (
            completion.get("launch_receipt_sha256") != launch_sha256
            or completion.get("record_type") != "decision_complexity_ada_v2_trajectory_completion"
            or completion.get("schema_version") != 1
            or completion.get("toolchain_sha256") != lock["toolchain_sha256"]
        ):
            raise protocol.ProtocolError("trajectory completion lost launch binding")
        previous_completion = runner.previous_completion_binding(manifest, row)
        launched, ended = validate_trajectory_parent_receipt(
            contract, row, completion, parent, completion_path, launch_sha256,
            previous_completion,
        )
        if (
            not isinstance(first_attempt_launched, int)
            or first_attempt_launched < launched
            or completion["trajectory_gpu_preflight"]["occupancy"][
                "query_completed_unix_ns"
            ] > first_attempt_launched
            or previous_completed > ended
        ):
            raise protocol.ProtocolError("attempt intervals escape their authorized trajectory")
        serial_intervals.append({
            "child_launched_unix_ns": launched,
            "completion_binding": {
                "path": str(completion_path.relative_to(protocol.REPO_ROOT)),
                "sha256": protocol.file_sha256(completion_path),
                "trajectory_ended_unix_ns": ended,
                "trajectory_id": row["trajectory_id"],
            },
            "launch_sequence": row["launch_sequence"],
            "previous_completion": previous_completion,
            "trajectory_ended_unix_ns": ended,
            "trajectory_id": row["trajectory_id"],
        })
        completion_hashes[row["trajectory_id"]] = protocol.file_sha256(completion_path)
        parent_hashes[row["trajectory_id"]] = protocol.file_sha256(parent_path)
        summaries.append(derived)
    validate_serial_intervals(serial_intervals)
    postflight = status.get("campaign_gpu_postflight")
    if not isinstance(postflight, list) or len(postflight) != 1:
        raise protocol.ProtocolError("pilot run status lacks final idle-GPU evidence")
    for gpu, evidence in enumerate(postflight):
        runner.validate_idle_gpu_evidence(evidence, gpu, contract)
    expected_status = {
        "campaign_gpu_postflight": postflight,
        "campaign_id": protocol.CAMPAIGN_ID,
        "complete": True,
        "launch_receipt_sha256": launch_sha256,
        "observed_trajectories": 24,
        "record_type": "decision_complexity_ada_v2_run_status",
        "schema_version": 1,
        "toolchain_sha256": lock["toolchain_sha256"],
        "trajectory_completion_bundle_sha256": protocol.canonical_sha256(completion_hashes),
        "trajectory_parent_bundle_sha256": protocol.canonical_sha256(parent_hashes),
    }
    if status != expected_status:
        raise protocol.ProtocolError("pilot run status is not exact-evidence-derived")
    return contract, summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        result = summarize(*load_completed())
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out:
            if args.out.exists():
                raise protocol.ProtocolError("refusing to overwrite analysis")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0
    except (OSError, json.JSONDecodeError, protocol.ProtocolError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
