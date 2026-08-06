#!/usr/bin/env python3
"""Re-derive imported evidence, lock winners, and analyze terminal confirmation."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from . import launch, protocol
except ImportError:
    import launch  # type: ignore
    import protocol  # type: ignore

from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core as source_core


IMPORTED_PATH = protocol.RESULTS_ROOT / "imported_frontier.json"
SELECTION_PATH = protocol.RESULTS_ROOT / "selection_lock.json"
FINAL_PATH = protocol.RESULTS_ROOT / "final_summary.json"


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if protocol.read_json(path) != value:
            raise protocol.ProtocolError(f"immutable analysis differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def imported_frontier() -> dict[str, Any]:
    return protocol.derive_imported_frontier()


def _stage_records(
    stage: str, winners: dict[str, str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    input_path = (
        IMPORTED_PATH if stage == "selection_confirm" else SELECTION_PATH
    ).resolve()
    input_value = protocol.read_json(input_path)
    if stage == "selection_confirm":
        protocol.validate_imported_frontier(input_value)
        plan = protocol.timing_plan(stage)
    else:
        if input_value != selection_summary() or input_value.get("terminal_authorized") is not True:
            raise protocol.ProtocolError("terminal selection lock is not re-derived/authorized")
        if winners != input_value["winners"]:
            raise protocol.ProtocolError("terminal winner argument differs from the selection lock")
        plan = protocol.timing_plan(stage, winners)
    root = protocol.RESULTS_ROOT / stage
    receipt_path, status_path = root / "launch_receipt.json", root / "run_status.json"
    receipt = protocol.read_json(receipt_path)
    contract = protocol.load_contract()
    expected_launch = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "gpu_lock_id": launch.GPU_LOCK_ID,
        "input_artifact_path": protocol.repo_path(input_path),
        "input_artifact_sha256": protocol.file_sha256(input_path),
        "plan": plan,
        "plan_sha256": protocol.canonical_sha256(plan),
        "stage": stage,
        "timing": contract["manifest"]["timing"],
    }
    launch.validate_launch_receipt(receipt, expected_launch, contract)
    launch_binding = launch.launch_record_binding(receipt_path, receipt)
    status = protocol.read_json(status_path)
    launch.validate_run_status(
        status, stage, len(plan), protocol.file_sha256(receipt_path)
    )
    records, hashes = [], [
        {"path": protocol.repo_path(path), "sha256": protocol.file_sha256(path)}
        for path in (receipt_path, status_path)
    ]
    raw = root / "raw"
    protocol.validate_raw_census(raw, plan)
    for row in plan:
        path = raw / protocol.timing_filename(row)
        record = launch.validate_timing_record(
            path,
            launch._expected_record(
                stage, row, input_path, input_value, launch_binding
            ),
            contract,
        )
        records.append(record)
        hashes.append({"path": protocol.repo_path(path), "sha256": protocol.file_sha256(path)})
    launch.validate_record_sequence(records, plan)
    return records, hashes


def _group(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[tuple[int, float]]]:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        row = record["row"]
        grouped[(row["label"], row["distribution"])].append(
            (row["block"], float(record["primary_tail_median_ms"]))
        )
    for key in grouped:
        grouped[key].sort()
    return grouped


def _values(grouped: dict, label: str, distribution: str) -> list[float]:
    rows = grouped.get((label, distribution), [])
    if [block for block, _value in rows] != list(range(15)):
        raise protocol.ProtocolError(f"incomplete randomized blocks: {label}/{distribution}")
    return [value for _block, value in rows]


def _sham_floor(
    records: list[dict[str, Any]], grouped: dict[tuple[str, str], list[tuple[int, float]]]
) -> tuple[float, dict[str, Any], str]:
    implementations = defaultdict(set)
    for record in records:
        implementations[record["label"]].add(record["implementation_sha256"])
    if any(len(values) != 1 for values in implementations.values()):
        raise protocol.ProtocolError("one timing label resolved to multiple implementations")
    if (
        implementations[protocol.SHAM_LABELS[0]]
        != implementations[protocol.SHAM_LABELS[1]]
        or len(implementations[protocol.SHAM_LABELS[0]]) != 1
    ):
        raise protocol.ProtocolError(
            "sham labels do not bind one source/config implementation fingerprint"
        )
    intervals = {}
    for distribution in protocol.DISTRIBUTIONS:
        left = _values(grouped, protocol.SHAM_LABELS[0], distribution)
        right = _values(grouped, protocol.SHAM_LABELS[1], distribution)
        intervals[distribution] = source_core.exact_median_interval(
            a / b for a, b in zip(left, right)
        )
    return (
        source_core.resolution_floor(intervals.values()),
        intervals,
        next(iter(implementations[protocol.SHAM_LABELS[0]])),
    )


def selection_summary() -> dict[str, Any]:
    records, hashes = _stage_records("selection_confirm")
    grouped = _group(records)
    contract = protocol.load_contract()
    candidates = contract["manifest"]["selection_confirm"]["candidate_ids"]
    rows, winners = [], {}
    for dsl in protocol.DSLS:
        ranked = []
        for cell_id in candidates[dsl]:
            positive = _values(grouped, cell_id, "positive")
            signed = _values(grouped, cell_id, "withheld_signed")
            median = statistics.median(positive)
            ranked.append((median, cell_id))
            rows.append(
                {
                    "cell_id": cell_id,
                    "dsl": dsl,
                    "positive_interval": source_core.exact_median_interval(positive),
                    "positive_median_ms": median,
                    "withheld_signed_diagnostic_interval": source_core.exact_median_interval(signed),
                }
            )
        winners[dsl] = min(ranked)[1]
    floor, sham_intervals, implementation = _sham_floor(records, grouped)
    cap = contract["manifest"]["inference"]["max_selection_sham_floor_log_ratio"]
    terminal_plan = protocol.timing_plan("terminal_confirm", winners)
    return {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_selection_lock",
        "campaign_id": protocol.CAMPAIGN_ID,
        "complete": True,
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "selection_records": rows,
        "selection_stage_hashes": hashes,
        "selection_stage_plan_sha256": protocol.canonical_sha256(
            protocol.timing_plan("selection_confirm")
        ),
        "sham": {
            "source_config_implementation_sha256": implementation,
            "intervals": sham_intervals,
            "resolution_floor_log_ratio": floor,
        },
        "terminal_authorized": floor <= cap,
        "terminal_plan_sha256": protocol.canonical_sha256(terminal_plan),
        "winners": winners,
    }


def _ratio_result(left: list[float], right: list[float], floor: float) -> dict[str, Any]:
    ratios = [a / b for a, b in zip(left, right)]
    interval = source_core.exact_median_interval(ratios)
    lo, hi = math.log(interval["ci_lo"]), math.log(interval["ci_hi"])
    direction = "unresolved"
    if hi < -floor:
        direction = "tilelang_lower_latency"
    elif lo > floor:
        direction = "triton_lower_latency"
    return {
        "block_ratios_tilelang_over_triton": ratios,
        "interval": interval,
        "direction_beyond_sham_floor": direction,
        "reportable_above_sham_floor": direction != "unresolved",
    }


def final_summary() -> dict[str, Any]:
    selection = selection_summary()
    if protocol.read_json(SELECTION_PATH) != selection or selection["terminal_authorized"] is not True:
        raise protocol.ProtocolError("terminal analysis requires the re-derived authorized selection lock")
    records, hashes = _stage_records("terminal_confirm", selection["winners"])
    grouped = _group(records)
    floor, sham_intervals, implementation = _sham_floor(records, grouped)
    ratios = {}
    directions = []
    for distribution in protocol.DISTRIBUTIONS:
        tilelang = _values(grouped, selection["winners"]["tilelang"], distribution)
        triton = _values(grouped, selection["winners"]["triton"], distribution)
        ratios[distribution] = _ratio_result(tilelang, triton, floor)
        directions.append(ratios[distribution]["direction_beyond_sham_floor"])
    if directions == ["tilelang_lower_latency"] * 2:
        conclusion = "tilelang_lower_latency_for_procedure_selected_pair"
    elif directions == ["triton_lower_latency"] * 2:
        conclusion = "triton_lower_latency_for_procedure_selected_pair"
    elif set(directions) == {"tilelang_lower_latency", "triton_lower_latency"}:
        conclusion = "positive_selection_does_not_generalize_with_one_direction"
    else:
        conclusion = "unresolved_at_terminal_sham_resolution"
    contract = protocol.load_contract()
    return {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_final_summary",
        "campaign_id": protocol.CAMPAIGN_ID,
        "complete": True,
        "claim_scope": contract["claim_scope"],
        "conclusion": conclusion,
        "estimand": contract["estimand"],
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "paired_procedure_selected_ratios": ratios,
        "record_hashes": hashes,
        "selection_lock_sha256": protocol.file_sha256(SELECTION_PATH),
        "sham": {
            "source_config_implementation_sha256": implementation,
            "intervals": sham_intervals,
            "resolution_floor_log_ratio": floor,
        },
        "withheld_signed_interpretation": (
            "generalization test for the positive-selected pair; not a "
            "signed-distribution frontier"
        ),
        "winners": selection["winners"],
        "broader_f1_outside_scope": contract["broader_f1_outside_scope"],
    }


def verify() -> None:
    if protocol.read_json(IMPORTED_PATH) != imported_frontier():
        raise protocol.ProtocolError("imported frontier failed independent re-derivation")
    if protocol.read_json(SELECTION_PATH) != selection_summary():
        raise protocol.ProtocolError("selection lock failed independent re-derivation")
    if protocol.read_json(FINAL_PATH) != final_summary():
        raise protocol.ProtocolError("final summary failed independent re-derivation")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("import", "select", "final", "verify"))
    args = parser.parse_args(argv)
    if args.command == "import":
        value = imported_frontier()
        _write_once(IMPORTED_PATH, value)
        print(f"imported audit={len(value['audit_records'])} screen={len(value['screen_records'])}")
    elif args.command == "select":
        value = selection_summary()
        _write_once(SELECTION_PATH, value)
        print(f"selected={value['winners']} terminal_authorized={value['terminal_authorized']}")
    elif args.command == "final":
        value = final_summary()
        _write_once(FINAL_PATH, value)
        print(f"conclusion={value['conclusion']}")
    else:
        verify()
        print("verified=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
