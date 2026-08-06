#!/usr/bin/env python3
"""Validate evidence and estimate same-campaign native-strategy recurrence."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    from . import protocol
except ImportError:  # direct script execution
    import protocol


def _times(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != protocol.TIMING_TRIALS:
        raise protocol.ProtocolError("timing record must contain exactly 100 trials")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item <= 0 for item in result):
        raise protocol.ProtocolError("timing record contains a nonpositive/nonfinite trial")
    return result


def validate_timing_record(
    contract: dict[str, Any],
    row: dict[str, Any],
    record: dict[str, Any],
    *,
    execution_binding: dict[str, Any] | None = None,
    artifact_binding: dict[str, Any] | None = None,
    admitted_build: dict[str, Any] | None = None,
) -> None:
    material = next(
        item for item in contract["materials"]["selected_prefixes"]
        if item["cell_id"] == row["cell_id"]
    )
    if artifact_binding is None or admitted_build is None:
        raise protocol.ProtocolError("timing validation requires its admitted artifact")
    identity = artifact_binding.get("admitted_artifact_identity_sha256")
    if admitted_build.get("artifact_identity_sha256") != identity:
        raise protocol.ProtocolError("timing artifact binding differs from its admitted build")
    expected = {
        "block": row["block"],
        "block_position": row["block_position"],
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": row["cell_id"],
        "cell_sha256": material["cell_sha256"],
        "cell_sha256_expected": material["cell_sha256"],
        "compute_pids_preflight": [],
        "distribution": row["distribution"],
        "global_position": row["global_position"],
        "artifact_identity_sha256_expected": identity,
        "label": row["label"],
        "manifest_row_sha256": protocol.canonical_sha256(row),
        "physical_gpu": 0,
        "record_kind": row["record_kind"],
        "record_type": "native_trajectory_replication_ada_v3_timing_record",
        "row_id": row["row_id"],
        "schema_version": 1,
        "trials": protocol.TIMING_TRIALS,
        "warmup_s": protocol.WARMUP_S,
        "predecessor_implementation_sha256_expected": row["predecessor_implementation_sha256"],
        **artifact_binding,
    }
    if execution_binding is not None:
        expected.update(execution_binding)
    mismatch = [key for key, value in expected.items() if record.get(key) != value]
    if mismatch or record.get("ok") is not True:
        raise protocol.ProtocolError(f"timing record is failed or foreign: {mismatch}")
    preflight = record.get("gpu_preflight")
    expected_gpu = {
        "compute_cap": contract["hardware"]["compute_capability"],
        "index": "0",
        "name": contract["hardware"]["gpu_name"],
        "uuid": contract["hardware"]["gpu_uuid"],
    }
    if (
        not isinstance(preflight, dict)
        or set(preflight) != {*expected_gpu, "driver_version"}
        or any(preflight.get(key) != value for key, value in expected_gpu.items())
        or not isinstance(preflight.get("driver_version"), str)
        or not preflight["driver_version"]
    ):
        raise protocol.ProtocolError("timing record lost its physical-GPU0 binding")
    process_pid = record.get("process_pid")
    postflight = record.get("gpu_postflight")
    compute_pids = record.get("compute_pids_postflight")
    if (
        isinstance(process_pid, bool)
        or not isinstance(process_pid, int)
        or process_pid <= 0
        or postflight != preflight
        or not isinstance(compute_pids, list)
        or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in compute_pids)
        or not set(compute_pids) <= {process_pid}
    ):
        raise protocol.ProtocolError("timing record lost its idle-except-self GPU postflight")
    start, end = record.get("t_start_unix_ns"), record.get("t_end_unix_ns")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start <= 0
        or end < start
    ):
        raise protocol.ProtocolError("timing record timestamps are invalid")
    values = _times(record.get("times_ms"))
    primary = statistics.median(values[protocol.TAIL_START:protocol.TAIL_STOP])
    if (
        not math.isclose(float(record.get("primary_tail_median_ms", -1)), primary)
        or not math.isclose(float(record.get("full_median_ms", -1)), statistics.median(values))
    ):
        raise protocol.ProtocolError("timing summaries are not raw-trial-derived")
    metadata = record.get("build_metadata", {})
    compile_s = record.get("compile_s")
    warmups = record.get("warmup_iterations_actual")
    if (
        record.get("artifact_identity_sha256") != identity
        or metadata.get("artifact_identity_sha256") != identity
        or metadata.get("n_kernels") != 2
        or re.fullmatch(
            r"[0-9a-f]{64}", str(metadata.get("predecessor_implementation_sha256_observed"))
        ) is None
        or record.get("live_correctness", {}).get("gate_pass") is not True
        or isinstance(compile_s, bool)
        or not isinstance(compile_s, (int, float))
        or not math.isfinite(float(compile_s))
        or float(compile_s) < 0
        or isinstance(warmups, bool)
        or not isinstance(warmups, int)
        or warmups < 1
    ):
        raise protocol.ProtocolError("timing record lost its admitted implementation/gate binding")
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore
    artifacts.validate_load_evidence(record.get("load_evidence"), material, admitted_build)


def validate_position_receipt(
    row: dict[str, Any],
    record: dict[str, Any],
    position: dict[str, Any],
    raw_path: Path,
    previous_completed: int,
) -> int:
    launched, completed = position.get("child_launched_unix_ns"), position.get("child_completed_unix_ns")
    if (
        position.get("campaign_id") != protocol.CAMPAIGN_ID
        or position.get("global_position") != row["global_position"]
        or position.get("raw_path") != str(raw_path.relative_to(protocol.REPO_ROOT))
        or position.get("raw_sha256") != protocol.file_sha256(raw_path)
        or position.get("record_type") != "native_trajectory_replication_ada_v3_position_receipt"
        or position.get("returncode") != 0
        or position.get("row_id") != row["row_id"]
        or position.get("gpu_idle_after_child") != record.get("gpu_preflight")
        or position.get("compute_pids_after_child") != []
        or position.get("gpu_postflight_error") is not None
        or not isinstance(launched, int)
        or not isinstance(completed, int)
        or launched < previous_completed
        or not (launched <= record["t_start_unix_ns"] <= record["t_end_unix_ns"] <= completed)
    ):
        raise protocol.ProtocolError(f"position/timestamp receipt is invalid: {row['global_position']}")
    return completed


def _effect(ratios: list[float]) -> dict[str, Any]:
    interval = protocol.core.exact_median_interval(ratios)
    return {
        "block_speedup_ratios": ratios,
        "log_ratio_interval": {
            "ci_hi": math.log(interval["ci_hi"]),
            "ci_lo": math.log(interval["ci_lo"]),
            "median": math.log(interval["median"]),
        },
        "ratio_interval": interval,
    }


def _direction(effect: dict[str, Any], floor: float) -> str:
    interval = effect["log_ratio_interval"]
    if interval["ci_lo"] > floor:
        return "speedup"
    if interval["ci_hi"] < -floor:
        return "slowdown"
    return "unresolved_at_sham_floor"


def estimate(
    contract: dict[str, Any], manifest: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(records) != protocol.RAW_RECORDS or len({row.get("row_id") for row in records}) != protocol.RAW_RECORDS:
        raise protocol.ProtocolError("estimation requires the exact 420-record census")
    by_row = {row["row_id"]: row for row in records}
    if set(by_row) != {row["row_id"] for row in manifest["rows"]}:
        raise protocol.ProtocolError("estimation records differ from the frozen manifest")
    medians: dict[tuple[Any, ...], float] = {}
    sham_hashes: set[str] = set()
    for row in manifest["rows"]:
        record = by_row[row["row_id"]]
        key = (row["block"], row["label"], row["distribution"])
        if key in medians:
            raise protocol.ProtocolError("duplicate block/label/distribution timing")
        medians[key] = float(record["primary_tail_median_ms"])
        if row["record_kind"] == "same_config_label_sham":
            sham_hashes.add(str(record.get("artifact_identity_sha256")))
    if len(sham_hashes) != 1 or re.fullmatch(r"[0-9a-f]{64}", next(iter(sham_hashes), "")) is None:
        raise protocol.ProtocolError("sham labels do not share one admitted artifact identity")
    sham_intervals: dict[str, dict[str, Any]] = {}
    for distribution in protocol.DISTRIBUTIONS:
        ratios = [
            medians[(block, protocol.SHAM_LABELS[0], distribution)]
            / medians[(block, protocol.SHAM_LABELS[1], distribution)]
            for block in range(protocol.BLOCKS)
        ]
        sham_intervals[distribution] = protocol.core.exact_median_interval(ratios)
    floor = max(
        abs(math.log(endpoint))
        for interval in sham_intervals.values()
        for endpoint in (interval["ci_lo"], interval["ci_hi"])
    )
    effects: dict[tuple[str, str, str], dict[str, Any]] = {}
    effects_out = []
    definitions = (
        (protocol.STEP_NAMES[0], protocol.STRATEGIES[0], protocol.STRATEGIES[1]),
        (protocol.STEP_NAMES[1], protocol.STRATEGIES[1], protocol.STRATEGIES[2]),
        ("cumulative", protocol.STRATEGIES[0], protocol.STRATEGIES[2]),
    )
    for lane in protocol.LANES:
        for distribution in protocol.DISTRIBUTIONS:
            for name, before, after in definitions:
                ratios = [
                    medians[(block, f"{before}.{lane}.g01", distribution)]
                    / medians[(block, f"{after}.{lane}.g01", distribution)]
                    for block in range(protocol.BLOCKS)
                ]
                result = _effect(ratios)
                result["direction_above_sham_floor"] = _direction(result, floor)
                effects[(lane, distribution, name)] = result
                effects_out.append(
                    {
                        "distribution": distribution,
                        "effect": result,
                        "lane": lane,
                        "strategy_step": name,
                        "speedup_orientation": "before_strategy_time_over_after_strategy_time",
                    }
                )
    classifications = []
    for step in protocol.STEP_NAMES:
        reference_directions = {
            distribution: _direction(effects[(protocol.REFERENCE_LANE, distribution, step)], floor)
            for distribution in protocol.DISTRIBUTIONS
        }
        for lane in protocol.LANES:
            destination_directions = {
                distribution: _direction(effects[(lane, distribution, step)], floor)
                for distribution in protocol.DISTRIBUTIONS
            }
            directions = list(reference_directions.values()) + list(destination_directions.values())
            if lane == protocol.REFERENCE_LANE:
                classification = "precommitted_same_campaign_reference"
            elif "unresolved_at_sham_floor" in directions:
                classification = "unresolved_at_sham_floor"
            elif len(set(directions)) != 1:
                classification = "direction_conflict"
            elif directions[0] == "speedup":
                classification = "same_campaign_speedup_recurrence"
            else:
                classification = "same_campaign_slowdown_recurrence"
            classifications.append(
                {
                    "classification": classification,
                    "destination_lane": lane,
                    "destination_role": "precommitted_same_campaign_reference" if lane == protocol.REFERENCE_LANE else "native_destination",
                    "destination_directions": destination_directions,
                    "reference_directions": reference_directions,
                    "strategy_step": step,
                    "same_campaign_gain_recurrence": classification == "same_campaign_speedup_recurrence" if lane != protocol.REFERENCE_LANE else None,
                    "recurrence_rule": "speedup recurrence requires reference_and_destination_intervals_to_clear_global_sham_floor_in_the_speedup_direction_on_both_distributions",
                }
            )
    return {
        "campaign_id": protocol.CAMPAIGN_ID,
        "claim_boundary": {
            "eligible": "same-campaign cross-lane recurrence of native strategy contrasts at fused g01 on this Ada host",
            "not_eligible": contract["claims_excluded"],
        },
        "complete": True,
        "controlling": True,
        "effects": effects_out,
        "same_campaign_recurrence_classifications": classifications,
        "primary_trials": [protocol.TAIL_START, protocol.TAIL_STOP],
        "record_type": "native_trajectory_replication_ada_v3_final_analysis",
        "schema_version": 1,
        "sham_control": {
            "base_cell_id": protocol.SHAM_CELL_ID,
            "identity_scope": contract["sham"]["identity_scope"],
            "artifact_identity_sha256": next(iter(sham_hashes)),
            "intervals": sham_intervals,
            "labels": list(protocol.SHAM_LABELS),
            "resolution_floor_log_ratio": floor,
            "source_byte_identity_claimed": True,
        },
        "timing_records": len(records),
    }


def load_completed() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    runner = protocol.local_runner_module()

    contract, manifest, execution_lock, provenance = runner.validate_execution_frozen()
    artifact_state = runner.validate_artifact_admission(contract)
    execution_binding = {
        "artifact_admission_closure_sha256": artifact_state["artifact_admission_closure_sha256"],
        "artifact_admission_manifest_path": artifact_state["artifact_admission_manifest_path"],
        "artifact_admission_manifest_sha256": artifact_state["artifact_admission_manifest_sha256"],
        "execution_lock_sha256": protocol.file_sha256(runner.EXECUTION_LOCK_PATH),
        "manifest_sha256": protocol.file_sha256(runner.MANIFEST_PATH),
        "source_bundle_sha256": execution_lock["source_bundle_sha256"],
        "toolchain": execution_lock["toolchain"],
    }
    launch_path, status_path = runner.RESULTS / "launch_receipt.json", runner.RESULTS / "run_status.json"
    launch, status = protocol.read_json(launch_path), protocol.read_json(status_path)
    launch_expected = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "artifact_admission_closure_sha256": artifact_state["artifact_admission_closure_sha256"],
        "artifact_admission_manifest_path": artifact_state["artifact_admission_manifest_path"],
        "artifact_admission_manifest_sha256": artifact_state["artifact_admission_manifest_sha256"],
        "execution_lock_sha256": protocol.file_sha256(runner.EXECUTION_LOCK_PATH),
        "expected_raw_records": protocol.RAW_RECORDS,
        "gpu_preflight": provenance["gpu0"],
        "manifest_plan_sha256": manifest["plan_sha256"],
        "manifest_sha256": protocol.file_sha256(runner.MANIFEST_PATH),
        "physical_gpu": 0,
        "toolchain": execution_lock["toolchain"],
    }
    if (
        launch.get("record_type") != "native_trajectory_replication_ada_v3_launch_receipt"
        or launch.get("schema_version") != 1
        or any(launch.get("contract", {}).get(key) != value for key, value in launch_expected.items())
        or not isinstance(launch.get("contract", {}).get("git_commit"), str)
        or len(launch["contract"]["git_commit"]) not in (40, 64)
    ):
        raise protocol.ProtocolError("launch receipt is incomplete or foreign")
    raw_expected = {protocol.raw_filename(row) for row in manifest["rows"]}
    position_expected = {protocol.position_receipt_filename(row) for row in manifest["rows"]}
    raw_observed = {path.name for path in (runner.RESULTS / "raw").glob("*.json")}
    position_observed = {path.name for path in (runner.RESULTS / "position_receipts").glob("*.json")}
    if raw_observed != raw_expected or position_observed != position_expected:
        raise protocol.ProtocolError("raw/position-receipt file census differs from the exact manifest")
    records = []
    previous_completed = 0
    raw_hashes: dict[str, str] = {}
    position_hashes: dict[str, str] = {}
    evidence_hashes = [
        {"path": row["path"], "sha256": row["sha256"]}
        for row in artifact_state["closure"]
    ]
    for row in manifest["rows"]:
        artifact_cell = runner.artifact_cell_state(row["cell_id"], artifact_state)
        raw_path = runner.RESULTS / "raw" / protocol.raw_filename(row)
        position_path = runner.RESULTS / "position_receipts" / protocol.position_receipt_filename(row)
        record, position = protocol.read_json(raw_path), protocol.read_json(position_path)
        validate_timing_record(
            contract, row, record, execution_binding=execution_binding,
            artifact_binding=artifact_cell["binding"], admitted_build=artifact_cell["build"],
        )
        previous_completed = validate_position_receipt(
            row, record, position, raw_path, previous_completed,
        )
        records.append(record)
        raw_hashes[raw_path.name] = protocol.file_sha256(raw_path)
        position_hashes[position_path.name] = protocol.file_sha256(position_path)
        evidence_hashes.extend(
            {"path": str(path.relative_to(protocol.REPO_ROOT)), "sha256": protocol.file_sha256(path)}
            for path in (raw_path, position_path)
        )
    expected_status = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "artifact_admission_closure_sha256": artifact_state["artifact_admission_closure_sha256"],
        "artifact_admission_manifest_sha256": artifact_state["artifact_admission_manifest_sha256"],
        "complete": True,
        "expected_position_receipts": protocol.RAW_RECORDS,
        "expected_raw_records": protocol.RAW_RECORDS,
        "gpu_after": provenance["gpu0"],
        "launch_receipt_sha256": protocol.file_sha256(launch_path),
        "observed_position_receipts": protocol.RAW_RECORDS,
        "observed_raw_records": protocol.RAW_RECORDS,
        "position_receipt_bundle_sha256": protocol.canonical_sha256(position_hashes),
        "raw_record_bundle_sha256": protocol.canonical_sha256(raw_hashes),
        "record_type": "native_trajectory_replication_ada_v3_run_status",
        "schema_version": 1,
    }
    if status != expected_status:
        raise protocol.ProtocolError("run status is not exact-evidence-derived")
    evidence_hashes.extend(
        {"path": str(path.relative_to(protocol.REPO_ROOT)), "sha256": protocol.file_sha256(path)}
        for path in (launch_path, status_path)
    )
    bindings = {
        "artifact_admission_closure_sha256": artifact_state["artifact_admission_closure_sha256"],
        "artifact_admission_manifest_sha256": artifact_state["artifact_admission_manifest_sha256"],
        "evidence_hashes": evidence_hashes,
        "execution_lock_sha256": protocol.file_sha256(runner.EXECUTION_LOCK_PATH),
        "launch_receipt_sha256": protocol.file_sha256(launch_path),
        "material_lock_sha256": protocol.file_sha256(runner.MATERIAL_LOCK_PATH),
        "run_status_sha256": protocol.file_sha256(status_path),
        "source_bundle_sha256": execution_lock["source_bundle_sha256"],
    }
    return contract, manifest, records, bindings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        contract, manifest, records, bindings = load_completed()
        result = {**estimate(contract, manifest, records), **bindings}
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
