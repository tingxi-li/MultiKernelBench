#!/usr/bin/env python3
"""Analyze only fresh v6 terminal confirmation against the sealed selection binding."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
try:
    from . import artifacts, launch, protocol
except ImportError:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ako_runs.controlled_followup.finite_frontier_ada_v6 import (
        artifacts, launch, protocol,
    )


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


def _terminal_records() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    selection = protocol.load_selection_binding()
    plan = protocol.timing_plan("terminal_confirm", selection["winners"])
    root = protocol.RESULTS_ROOT / "terminal_confirm"
    receipt_path, status_path = root / "launch_receipt.json", root / "run_status.json"
    receipt = protocol.read_json(receipt_path)
    contract = protocol.load_contract()
    manifest, _cells = launch._artifact_manifest()
    expected_launch = {
        "artifact_admission_manifest_path": protocol.repo_path(artifacts.MANIFEST_PATH),
        "artifact_admission_manifest_sha256": protocol.file_sha256(artifacts.MANIFEST_PATH),
        "campaign_id": protocol.CAMPAIGN_ID,
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "gpu_lock_id": launch.GPU_LOCK_ID,
        "input_artifact_path": protocol.repo_path(protocol.SELECTION_BINDING_PATH),
        "input_artifact_sha256": protocol.file_sha256(protocol.SELECTION_BINDING_PATH),
        "plan": plan,
        "plan_sha256": protocol.canonical_sha256(plan),
        "stage": "terminal_confirm",
        "timing": contract["manifest"]["timing"],
    }
    launch.validate_launch_receipt(receipt, expected_launch, contract)
    launch_binding = launch.launch_record_binding(receipt_path, receipt)
    launch.validate_run_status(
        protocol.read_json(status_path), "terminal_confirm", len(plan),
        protocol.file_sha256(receipt_path),
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
                "terminal_confirm", row, protocol.SELECTION_BINDING_PATH,
                selection, launch_binding, artifacts.binding(row["cell_id"], manifest),
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
    for rows in grouped.values():
        rows.sort()
    return grouped


def _values(grouped: dict, label: str, distribution: str) -> list[float]:
    rows = grouped.get((label, distribution), [])
    if [block for block, _value in rows] != list(range(15)):
        raise protocol.ProtocolError(f"incomplete randomized blocks: {label}/{distribution}")
    return [value for _block, value in rows]


def _sham_floor(
    records: list[dict[str, Any]], grouped: dict[tuple[str, str], list[tuple[int, float]]],
) -> tuple[float, dict[str, Any], str]:
    implementations: dict[str, set[str]] = defaultdict(set)
    for record in records:
        implementations[record["label"]].add(record["implementation_sha256"])
    if any(len(values) != 1 for values in implementations.values()):
        raise protocol.ProtocolError("one timing label resolved to multiple implementations")
    if implementations["sham_a"] != implementations["sham_b"]:
        raise protocol.ProtocolError("sham labels do not bind one byte-identical implementation")
    intervals = {
        distribution: protocol.exact_median_interval(
            left / right for left, right in zip(
                _values(grouped, "sham_a", distribution),
                _values(grouped, "sham_b", distribution),
            )
        )
        for distribution in protocol.DISTRIBUTIONS
    }
    return (
        protocol.resolution_floor(intervals.values()),
        intervals,
        next(iter(implementations["sham_a"])),
    )


def _ratio_result(left: list[float], right: list[float], floor: float) -> dict[str, Any]:
    ratios = [a / b for a, b in zip(left, right)]
    interval = protocol.exact_median_interval(ratios)
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
    selection = protocol.load_selection_binding()
    records, hashes = _terminal_records()
    grouped = _group(records)
    floor, sham_intervals, implementation = _sham_floor(records, grouped)
    ratios, directions = {}, []
    for distribution in protocol.DISTRIBUTIONS:
        ratios[distribution] = _ratio_result(
            _values(grouped, selection["winners"]["tilelang"], distribution),
            _values(grouped, selection["winners"]["triton"], distribution),
            floor,
        )
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
        "selection_binding_sha256": protocol.file_sha256(protocol.SELECTION_BINDING_PATH),
        "source_selection_lock_sha256": selection["source_selection_lock_sha256"],
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
    protocol.load_selection_binding()
    if protocol.read_json(FINAL_PATH) != final_summary():
        raise protocol.ProtocolError("final summary failed independent re-derivation")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("binding", "final", "verify"))
    args = parser.parse_args(argv)
    if args.command == "binding":
        value = protocol.derive_selection_binding()
        _write_once(protocol.SELECTION_BINDING_PATH, value)
        print(f"winners={value['winners']} terminal_authorized={value['terminal_authorized']}")
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
