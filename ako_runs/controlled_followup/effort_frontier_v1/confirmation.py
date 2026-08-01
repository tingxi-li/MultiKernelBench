#!/usr/bin/env python3
"""Freeze and execute the 15-block, two-distribution GPU0 confirmation."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from . import analyze, campaign, events, validate
except ImportError:  # direct script execution
    import analyze  # type: ignore
    import campaign  # type: ignore
    import events  # type: ignore
    import validate  # type: ignore


def _read_partial(path: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    previous = events.GENESIS
    plan_sha = campaign.canonical_sha256(plan)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if len(rows) >= len(plan["records"]):
                raise ValueError("confirmation has more records than its frozen plan")
            expected = plan["records"][len(rows)]
            bindings = {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "record_index": len(rows),
                "previous_record_sha256": previous,
                "confirmation_plan_sha256": plan_sha,
                **expected,
            }
            mismatches = [key for key, value in bindings.items() if row.get(key) != value]
            digest = campaign.canonical_sha256(
                {key: value for key, value in row.items() if key != "record_sha256"}
            )
            if row.get("record_sha256") != digest:
                mismatches.append("record_sha256")
            if mismatches:
                raise ValueError(
                    f"{path}:{line_number}: record binding differs: {sorted(set(mismatches))}"
                )
            rows.append(row)
            previous = digest
    return rows


def _append(path: Path, plan: dict[str, Any], item: dict[str, Any], payload: dict[str, Any]) -> None:
    rows = _read_partial(path, plan)
    if item["record_index"] != len(rows):
        raise ValueError("confirmation append is not the next frozen record")
    row = {
        "schema_version": 1,
        "campaign_id": campaign.CAMPAIGN_ID,
        "confirmation_plan_sha256": campaign.canonical_sha256(plan),
        "previous_record_sha256": rows[-1]["record_sha256"] if rows else events.GENESIS,
        **item,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        **payload,
    }
    row["record_sha256"] = campaign.canonical_sha256(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _candidate_path(result_root: Path, item: dict[str, Any]) -> Path:
    root = (result_root.resolve() / item["trajectory_id"]).resolve()
    path = (root / item["candidate_relative_path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("confirmation candidate path escapes trajectory root") from exc
    if not path.is_file() or campaign.file_sha256(path) != item["candidate_sha256"]:
        raise ValueError("confirmation candidate bytes differ from frozen selection")
    return path


def _execute(
    command: list[str],
    timeout_s: int,
    request: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request, sort_keys=True) + "\n",
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.perf_counter() - started
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "error": f"confirmation executor infrastructure error: {exc}",
            "executor_wall_s": time.perf_counter() - started,
            "physical_gpu": 0,
            "logical_device": "cuda:0",
            "gpu_uuid": None,
        }
    assigned = {
        "physical_gpu": 0,
        "logical_device": "cuda:0",
        "gpu_uuid": None,
        "executor_wall_s": elapsed,
    }
    if completed.returncode != 0:
        return {
            **assigned,
            "ok": False,
            "error": completed.stderr[-4000:] or f"executor exit {completed.returncode}",
        }
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {**assigned, "ok": False, "error": f"invalid executor JSON: {exc}"}
    expected = {
        "schema_version": 1,
        "campaign_id": campaign.CAMPAIGN_ID,
        "record_id": item["record_id"],
        "treatment_id": item["treatment_id"],
        "distribution": item["distribution"],
        "block": item["block"],
        "position": item["position"],
        "physical_gpu": 0,
        "logical_device": "cuda:0",
        "gpu_uuid": campaign.GPU_UUIDS[0],
    }
    if not isinstance(raw, dict):
        return {**assigned, "ok": False, "error": "executor response is not an object"}
    mismatches = [key for key, value in expected.items() if raw.get(key) != value]
    if mismatches:
        return {
            **assigned,
            "ok": False,
            "error": f"executor response binding mismatch: {mismatches}",
        }
    trials = raw.get("trial_times_ms")
    if (
        not isinstance(trials, list)
        or len(trials) != campaign.CONFIRM_TRIALS
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in trials
        )
    ):
        return {**assigned, "ok": False, "error": "executor returned invalid timing trials"}
    base = {
        "physical_gpu": 0,
        "logical_device": "cuda:0",
        "gpu_uuid": campaign.GPU_UUIDS[0],
        "executor_wall_s": elapsed,
    }
    if item["treatment_type"] == "candidate":
        if raw.get("candidate_sha256") != item["candidate_sha256"]:
            return {**base, "ok": False, "error": "candidate hash echo differs"}
        if raw.get("lane_policy_pass") is not True:
            return {**base, "ok": False, "error": "candidate failed lane policy"}
        if raw.get("gate_spec_sha256") != campaign.file_sha256(campaign.GATE_SPEC):
            return {**base, "ok": False, "error": "fused-v2 gate hash echo differs"}
        if raw.get("gate_pass") is not True:
            return {**base, "ok": False, "error": "candidate failed confirmation gate"}
        gate_summary = raw.get("gate_summary_sha256")
        if (
            not isinstance(gate_summary, str)
            or len(gate_summary) != 64
            or any(character not in "0123456789abcdef" for character in gate_summary)
        ):
            return {**base, "ok": False, "error": "candidate gate evidence hash is invalid"}
        contract = {
            "lane_policy_pass": True,
            "gate_pass": True,
            "gate_summary_sha256": gate_summary,
            "contract_pass": None,
            "contract_summary_sha256": None,
        }
    else:
        if raw.get("contract_pass") is not True:
            return {**base, "ok": False, "error": "control failed contract check"}
        contract_summary = raw.get("contract_summary_sha256")
        if (
            not isinstance(contract_summary, str)
            or len(contract_summary) != 64
            or any(character not in "0123456789abcdef" for character in contract_summary)
        ):
            return {**base, "ok": False, "error": "control contract evidence hash is invalid"}
        contract = {
            "lane_policy_pass": None,
            "gate_pass": None,
            "gate_summary_sha256": None,
            "contract_pass": True,
            "contract_summary_sha256": contract_summary,
        }
    numeric = [float(value) for value in trials]
    return {
        **base,
        **contract,
        "ok": True,
        "trial_times_ms": numeric,
        "median_ms": statistics.median(numeric),
        "executor_metadata": raw.get("executor_metadata", {}),
    }


def run(result_root: Path, plan_path: Path, record_path: Path) -> None:
    expected = analyze.build_confirmation_plan(result_root)
    if not plan_path.is_file():
        raise RuntimeError("confirmation plan is missing; freeze it before launch")
    plan = campaign.load_json(plan_path)
    if plan != expected or plan_path.read_bytes() != campaign.stable_json_bytes(plan):
        raise RuntimeError("confirmation plan differs from completed search or is noncanonical")
    blockers = validate.confirmation_blockers()
    if blockers:
        raise RuntimeError("confirmation launch refused: " + "; ".join(blockers))
    registry_blockers: list[str] = []
    registry = validate.executor_registry(registry_blockers)
    if registry is None or registry_blockers:
        raise RuntimeError("confirmation registry refused: " + "; ".join(registry_blockers))

    record_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = (record_path.parent / ".confirmation.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another confirmation process holds the result lock") from exc
    completed = _read_partial(record_path, plan)
    for item in plan["records"][len(completed) :]:
        if item["treatment_type"] == "control":
            entry = registry["control"]
            candidate_path = None
        else:
            entry = registry["lanes"][item["lane"]]
            candidate_path = _candidate_path(result_root, item)
        request = {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "record_id": item["record_id"],
            "treatment_type": item["treatment_type"],
            "treatment_id": item["treatment_id"],
            "lane": item.get("lane"),
            "candidate_path": str(candidate_path) if candidate_path else None,
            "candidate_sha256": item.get("candidate_sha256"),
            "distribution": item["distribution"],
            "block": item["block"],
            "position": item["position"],
            "warmup_iterations": campaign.CONFIRM_WARMUP,
            "timed_trials": campaign.CONFIRM_TRIALS,
            "gate_spec": str(campaign.GATE_SPEC.resolve()),
            "gate_spec_sha256": campaign.file_sha256(campaign.GATE_SPEC),
            "physical_gpu": 0,
            "required_gpu_uuid": campaign.GPU_UUIDS[0],
            "logical_device": "cuda:0",
        }
        command = entry["confirmation_command"]
        payload = _execute(command, entry["timeout_s"], request, item)
        _append(record_path, plan, item, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--make-plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.make_plan:
        plan = analyze.build_confirmation_plan(args.result_root)
        if args.plan.exists():
            existing = campaign.load_json(args.plan)
            if existing != plan or args.plan.read_bytes() != campaign.stable_json_bytes(existing):
                raise RuntimeError("refusing to overwrite a different confirmation plan")
        else:
            campaign.atomic_write(args.plan, campaign.stable_json_bytes(plan))
        print(f"frozen {len(plan['records'])} confirmation records")
        return 0
    plan = analyze.build_confirmation_plan(args.result_root)
    if not args.plan.is_file() or campaign.load_json(args.plan) != plan:
        raise RuntimeError("confirmation plan is missing or stale")
    blockers = validate.confirmation_blockers()
    if blockers:
        for blocker in blockers:
            print("BLOCKED:", blocker)
        print("REFUSED: zero GPU confirmation processes started")
        return 2
    if not args.execute:
        print(f"confirmation launch-ready: {len(plan['records'])} records")
        return 0
    if args.records is None:
        parser.error("--records is required with --execute")
    run(args.result_root, args.plan, args.records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
