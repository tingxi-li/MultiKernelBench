#!/usr/bin/env python3
"""Fail-closed feasibility, selection, interaction, and stability analyses."""
from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from core import (
    GRID_IDS,
    LANES,
    LOCK_PATH,
    REPO_ROOT,
    STRATEGIES,
    TERMINAL_AUDIT_OUTCOMES,
    canonical_sha256,
    cell_filename,
    confirmation_plan,
    exact_median_interval,
    file_sha256,
    load_contract,
    read_json,
    screen_plan,
    spearman_rank,
    stable_write,
    timing_filename,
)
from launch import validate_timing_record


def audit_summary(root: Path) -> dict[str, Any]:
    campaign, cells, lock = load_contract()
    receipt_dir = root / "audit" / "receipts"
    receipt_paths = sorted(
        path for path in receipt_dir.glob("shard*.json")
        if not path.name.endswith("_status.json")
    )
    if not receipt_paths:
        raise RuntimeError("audit incomplete: no shard receipts")
    assigned_to_gpu: dict[str, int] = {}
    shard_receipts = []
    for receipt_path in receipt_paths:
        receipt = read_json(receipt_path)
        contract = receipt.get("contract", {})
        if (
            receipt.get("record_type") != "fused_crossed_audit_receipt"
            or contract.get("campaign_id") != campaign["campaign_id"]
            or contract.get("launch_lock_sha256") != file_sha256(LOCK_PATH)
            or contract.get("source_bundle_sha256") != lock["source_bundle_sha256"]
        ):
            raise RuntimeError(f"foreign audit shard receipt: {receipt_path}")
        status_path = receipt_path.with_name(receipt_path.stem + "_status.json")
        status = read_json(status_path)
        if (
            status.get("complete") is not True
            or status.get("receipt_sha256") != file_sha256(receipt_path)
            or status.get("expected_cells") != len(contract.get("assigned_cell_ids", []))
            or status.get("observed_cells") != status.get("expected_cells")
        ):
            raise RuntimeError(f"audit shard is not complete: {receipt_path}")
        for cell_id in contract.get("assigned_cell_ids", []):
            if cell_id in assigned_to_gpu:
                raise RuntimeError(f"cell assigned by multiple audit shards: {cell_id}")
            assigned_to_gpu[cell_id] = contract["physical_gpu"]
        shard_receipts.append(
            {
                "path": str(receipt_path.relative_to(REPO_ROOT)),
                "physical_gpu": contract["physical_gpu"],
                "sha256": file_sha256(receipt_path),
                "status_path": str(status_path.relative_to(REPO_ROOT)),
                "status_sha256": file_sha256(status_path),
            }
        )
    if set(assigned_to_gpu) != {cell["cell_id"] for cell in cells}:
        raise RuntimeError("audit shard assignments do not cover exactly 228 cells")
    records = []
    for cell in cells:
        path = root / "audit" / "records" / cell_filename(cell["cell_id"])
        if not path.is_file():
            raise RuntimeError(f"audit incomplete: missing {path}")
        record = read_json(path)
        expected = {
            "campaign_id": campaign["campaign_id"],
            "cell": cell,
            "cell_sha256": canonical_sha256(cell),
            "launch_lock_sha256": file_sha256(LOCK_PATH),
            "source_bundle_sha256": lock["source_bundle_sha256"],
        }
        mismatches = [key for key, value in expected.items() if record.get(key) != value]
        if mismatches or record.get("terminal_outcome") not in TERMINAL_AUDIT_OUTCOMES:
            raise RuntimeError(f"invalid audit record {path}: {mismatches}")
        if record.get("physical_gpu") != assigned_to_gpu[cell["cell_id"]]:
            raise RuntimeError(f"audit record GPU differs from shard receipt: {cell['cell_id']}")
        outcome = record["terminal_outcome"]
        if outcome == "UNSUPPORTED":
            if cell["support_declared"] or record.get("build_attempted") is not False:
                raise RuntimeError(f"invalid unsupported outcome: {cell['cell_id']}")
        elif not cell["support_declared"]:
            raise RuntimeError(f"unsupported manifest cell has outcome {outcome}")
        if outcome in {"LAUNCH_FAILED", "GATE_FAILED", "GATE_PASSED"}:
            summary = record.get("gate_summary", {})
            gate_path = REPO_ROOT / record.get("gate_jsonl_path", "")
            if (
                summary.get("complete") is not True
                or summary.get("observed_records") != 512
                or not gate_path.is_file()
                or file_sha256(gate_path) != record.get("gate_jsonl_sha256")
            ):
                raise RuntimeError(f"incomplete or changed gate evidence: {cell['cell_id']}")
            if (outcome == "GATE_PASSED") != (summary.get("full_gate_pass") is True):
                raise RuntimeError(f"terminal gate verdict mismatch: {cell['cell_id']}")
        records.append(record)
    legal = {
        record["cell"]["cell_id"] for record in records
        if record["terminal_outcome"] == "GATE_PASSED"
    }
    outcomes = []
    for strategy in STRATEGIES:
        for lane in LANES:
            subset = [record for record in records if record["cell"]["strategy"] == strategy and record["cell"]["lane"] == lane]
            counts = Counter(record["terminal_outcome"] for record in subset)
            outcomes.append(
                {
                    "counts": {name: counts.get(name, 0) for name in TERMINAL_AUDIT_OUTCOMES},
                    "gate_legal_rate_over_requested": counts.get("GATE_PASSED", 0) / 19.0,
                    "lane": lane,
                    "requested_cells": 19,
                    "strategy": strategy,
                }
            )
    common_by_strategy = {
        strategy: [
            grid_id for grid_id in GRID_IDS
            if all(f"{strategy}.{lane}.{grid_id}" in legal for lane in LANES)
        ]
        for strategy in STRATEGIES
    }
    common_by_lane = {
        lane: [
            grid_id for grid_id in GRID_IDS
            if all(f"{strategy}.{lane}.{grid_id}" in legal for strategy in STRATEGIES)
        ]
        for lane in LANES
    }
    all_common = [
        grid_id for grid_id in GRID_IDS
        if all(f"{strategy}.{lane}.{grid_id}" in legal for strategy in STRATEGIES for lane in LANES)
    ]
    legal_rates = {
        (row["strategy"], row["lane"]): row["gate_legal_rate_over_requested"]
        for row in outcomes
    }
    feasibility_interactions = []
    for strategy in STRATEGIES[1:]:
        for lane in LANES[1:]:
            feasibility_interactions.append(
                {
                    "difference_in_differences_vs_register_tilelang": (
                        legal_rates[(strategy, lane)]
                        - legal_rates[("register_fused", lane)]
                        - legal_rates[(strategy, "tilelang")]
                        + legal_rates[("register_fused", "tilelang")]
                    ),
                    "lane": lane,
                    "strategy": strategy,
                }
            )
    cell_rows = []
    for record in records:
        summary = record.get("gate_summary", {})
        cell_rows.append(
            {
                "cell_id": record["cell"]["cell_id"],
                "gate_max_over_threshold_ratio": summary.get("max_over_threshold_ratio"),
                "gate_minimum_headroom_fraction": summary.get("minimum_headroom_fraction"),
                "record_sha256": file_sha256(root / "audit" / "records" / cell_filename(record["cell"]["cell_id"])),
                "terminal_outcome": record["terminal_outcome"],
                "timing_eligible": record["terminal_outcome"] == "GATE_PASSED",
            }
        )
    return {
        "all_factor_common_feasible_grid_ids": all_common,
        "campaign_id": campaign["campaign_id"],
        "cells": cell_rows,
        "common_feasible_grid_ids_by_lane": common_by_lane,
        "common_feasible_grid_ids_by_strategy": common_by_strategy,
        "complete": True,
        "feasibility_strategy_x_lane_interactions": feasibility_interactions,
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "outcome_counts": {name: sum(record["terminal_outcome"] == name for record in records) for name in TERMINAL_AUDIT_OUTCOMES},
        "outcomes_by_strategy_lane": outcomes,
        "record_type": "fused_crossed_audit_summary",
        "requested_cells": 228,
        "schema_version": 1,
        "shard_receipts": shard_receipts,
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "timing_eligible_cell_ids": sorted(legal),
    }


def choose_confirmation(
    cells: list[dict[str, Any]],
    process_medians: dict[str, list[float]],
    gate_legal_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    by_id = {cell["cell_id"]: cell for cell in cells}
    for strategy in STRATEGIES:
        for lane in LANES:
            candidates = []
            for grid_id in GRID_IDS:
                cell_id = f"{strategy}.{lane}.{grid_id}"
                values = process_medians.get(cell_id, [])
                if len(values) == 2:
                    candidates.append((statistics.median(values), grid_id, cell_id))
            ranked = sorted(candidates)
            chosen = [cell_id for _value, _grid, cell_id in ranked[:2]]
            g01 = f"{strategy}.{lane}.g01"
            g01_legal = (
                g01 in gate_legal_ids
                if gate_legal_ids is not None
                else len(process_medians.get(g01, [])) == 2
            )
            if g01_legal and g01 not in chosen:
                chosen.append(g01)
            for cell_id in chosen:
                rank = next(
                    (index + 1 for index, row in enumerate(ranked) if row[2] == cell_id),
                    None,
                )
                values = process_medians.get(cell_id, [])
                selected.append(
                    {
                        "cell_id": cell_id,
                        "grid_id": by_id[cell_id]["grid_id"],
                        "lane": lane,
                        "screen_median_of_two_process_medians_ms": statistics.median(values) if len(values) == 2 else None,
                        "screen_rank": rank,
                        "selection_reason": "top_two" if rank is not None and rank <= 2 else "g01_gate_legal_positive_control",
                        "strategy": strategy,
                    }
                )
    return selected


def screen_selection(root: Path, audit_path: Path) -> dict[str, Any]:
    campaign, cells, lock = load_contract()
    audit = read_json(audit_path)
    if audit.get("record_type") != "fused_crossed_audit_summary" or audit.get("complete") is not True or audit.get("launch_lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("screen analysis requires complete bound audit summary")
    legal = set(audit["timing_eligible_cell_ids"])
    plan = screen_plan(cells, legal)
    receipt_path = root / "screen" / "launch_receipt.json"
    receipt = read_json(receipt_path)
    contract = receipt.get("contract", {})
    if contract.get("execution_order") != plan or contract.get("physical_gpu") != 0 or contract.get("eligibility_sha256") != file_sha256(audit_path):
        raise RuntimeError("screen receipt differs from frozen plan/GPU/audit")
    status = read_json(root / "screen" / "run_status.json")
    if status.get("complete") is not True or status.get("expected_records") != len(plan) or status.get("observed_records") != len(plan):
        raise RuntimeError("screen run-status receipt is incomplete")
    by_id = {cell["cell_id"]: cell for cell in cells}
    grouped: dict[str, list[float]] = defaultdict(list)
    failed = []
    record_hashes = []
    raw = root / "screen" / "raw"
    for row in plan:
        path = raw / timing_filename(row["cell_id"], "positive", row["rep"])
        if not path.is_file():
            raise RuntimeError(f"screen missing record: {path}")
        record = validate_timing_record(path, cell=by_id[row["cell_id"]], phase="screen", distribution="positive", rep=row["rep"], gpu=0, eligibility_sha256=file_sha256(audit_path), lock=lock)
        record_hashes.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)})
        if record["ok"] is True and record.get("legacy_error", {}).get("gate_pass") is True:
            grouped[row["cell_id"]].append(float(record["median_ms"]))
        else:
            failed.append({"cell_id": row["cell_id"], "rep": row["rep"], "error": record.get("error", "legacy_gate_failure")})
    selected = choose_confirmation(cells, grouped, legal)
    selected_ids = [row["cell_id"] for row in selected]
    return {
        "audit_summary_path": str(audit_path.relative_to(REPO_ROOT)),
        "audit_summary_sha256": file_sha256(audit_path),
        "campaign_id": campaign["campaign_id"],
        "complete": True,
        "failed_screen_processes": failed,
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "record_hashes": record_hashes,
        "record_type": "fused_crossed_confirmation_selection",
        "schema_version": 1,
        "screen_receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
        "screen_receipt_sha256": file_sha256(receipt_path),
        "selected": selected,
        "selected_cell_ids": selected_ids,
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "timing_eligible_cell_ids": audit["timing_eligible_cell_ids"],
        "feasibility": {
            "all_factor_common_feasible_grid_ids": audit["all_factor_common_feasible_grid_ids"],
            "common_feasible_grid_ids_by_lane": audit["common_feasible_grid_ids_by_lane"],
            "common_feasible_grid_ids_by_strategy": audit["common_feasible_grid_ids_by_strategy"],
            "feasibility_strategy_x_lane_interactions": audit["feasibility_strategy_x_lane_interactions"],
            "outcome_counts": audit["outcome_counts"],
            "outcomes_by_strategy_lane": audit["outcomes_by_strategy_lane"],
        },
    }


def _performance_contrasts(cell_results: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    # The g01 positive control is deliberately included by every stratum when
    # gate-legal, making these expression-matched contrasts available without
    # post-hoc grid selection. Other overlapping selected grids are retained too.
    strategy_effects = []
    def paired_ratio(numerator: dict[str, Any], denominator: dict[str, Any]) -> dict[str, Any]:
        ratios = [
            n_value / d_value
            for n_value, d_value in zip(
                numerator["process_medians_ms"], denominator["process_medians_ms"]
            )
        ]
        return {"block_ratios": ratios, "interval": exact_median_interval(ratios)}

    for lane in LANES:
        grids = [grid for grid in GRID_IDS if all((f"{strategy}.{lane}.{grid}", "positive") in cell_results for strategy in STRATEGIES)]
        for grid in grids:
            baseline = cell_results[(f"register_fused.{lane}.{grid}", "positive")]["median_ms"]
            for strategy in STRATEGIES[1:]:
                point = cell_results[(f"{strategy}.{lane}.{grid}", "positive")]["median_ms"]
                strategy_effects.append({"grid_id": grid, "lane": lane, "paired_block_inference": paired_ratio(cell_results[(f"{strategy}.{lane}.{grid}", "positive")], cell_results[(f"register_fused.{lane}.{grid}", "positive")]), "ratio_to_register": point / baseline, "strategy": strategy})
    lane_effects = []
    for strategy in STRATEGIES:
        grids = [grid for grid in GRID_IDS if all((f"{strategy}.{lane}.{grid}", "positive") in cell_results for lane in LANES)]
        for grid in grids:
            baseline = cell_results[(f"{strategy}.tilelang.{grid}", "positive")]["median_ms"]
            for lane in LANES[1:]:
                point = cell_results[(f"{strategy}.{lane}.{grid}", "positive")]["median_ms"]
                lane_effects.append({"grid_id": grid, "lane": lane, "paired_block_inference": paired_ratio(cell_results[(f"{strategy}.{lane}.{grid}", "positive")], cell_results[(f"{strategy}.tilelang.{grid}", "positive")]), "ratio_to_tilelang": point / baseline, "strategy": strategy})
    # Interaction is the ratio-of-ratios for each strategy and non-TileLang lane
    # wherever a common grid exists. Values other than one indicate that the
    # strategy effect differs by lane.
    interactions = []
    for strategy in STRATEGIES[1:]:
        for lane in LANES[1:]:
            grids = [grid for grid in GRID_IDS if all((f"{s}.{l}.{grid}", "positive") in cell_results for s, l in (("register_fused", "tilelang"), (strategy, "tilelang"), ("register_fused", lane), (strategy, lane)))]
            for grid in grids:
                tile_ratio = cell_results[(f"{strategy}.tilelang.{grid}", "positive")]["median_ms"] / cell_results[(f"register_fused.tilelang.{grid}", "positive")]["median_ms"]
                lane_ratio = cell_results[(f"{strategy}.{lane}.{grid}", "positive")]["median_ms"] / cell_results[(f"register_fused.{lane}.{grid}", "positive")]["median_ms"]
                interaction_blocks = [
                    (strategy_lane / register_lane) / (strategy_tile / register_tile)
                    for strategy_lane, register_lane, strategy_tile, register_tile in zip(
                        cell_results[(f"{strategy}.{lane}.{grid}", "positive")]["process_medians_ms"],
                        cell_results[(f"register_fused.{lane}.{grid}", "positive")]["process_medians_ms"],
                        cell_results[(f"{strategy}.tilelang.{grid}", "positive")]["process_medians_ms"],
                        cell_results[(f"register_fused.tilelang.{grid}", "positive")]["process_medians_ms"],
                    )
                ]
                interactions.append({"grid_id": grid, "interaction_ratio_of_ratios": lane_ratio / tile_ratio, "lane": lane, "paired_block_inference": {"block_ratio_of_ratios": interaction_blocks, "interval": exact_median_interval(interaction_blocks)}, "strategy": strategy})
    return {"lane_effects_common_selected_grids": lane_effects, "strategy_effects_common_selected_grids": strategy_effects, "strategy_x_lane_interactions": interactions}


def confirmation_summary(root: Path, selection_path: Path) -> dict[str, Any]:
    campaign, cells, lock = load_contract()
    selection = read_json(selection_path)
    if selection.get("record_type") != "fused_crossed_confirmation_selection" or selection.get("complete") is not True or selection.get("launch_lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("confirmation analysis requires complete bound selection")
    selected = set(selection["selected_cell_ids"])
    plan = confirmation_plan(selected)
    receipt_path = root / "confirmation" / "launch_receipt.json"
    receipt = read_json(receipt_path)
    contract = receipt.get("contract", {})
    if contract.get("execution_order") != plan or contract.get("physical_gpu") != 0 or contract.get("eligibility_sha256") != file_sha256(selection_path):
        raise RuntimeError("confirmation receipt differs from frozen plan/GPU/selection")
    status = read_json(root / "confirmation" / "run_status.json")
    if status.get("complete") is not True or status.get("expected_records") != len(plan) or status.get("observed_records") != len(plan):
        raise RuntimeError("confirmation run-status receipt is incomplete")
    by_id = {cell["cell_id"]: cell for cell in cells}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    failures = []
    hashes = []
    raw = root / "confirmation" / "raw"
    for row in plan:
        path = raw / timing_filename(row["cell_id"], row["distribution"], row["rep"])
        if not path.is_file():
            raise RuntimeError(f"confirmation missing record: {path}")
        record = validate_timing_record(path, cell=by_id[row["cell_id"]], phase="confirmation", distribution=row["distribution"], rep=row["rep"], gpu=0, eligibility_sha256=file_sha256(selection_path), lock=lock)
        hashes.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)})
        if record["ok"] is True and record.get("legacy_error", {}).get("gate_pass") is True:
            grouped[(row["cell_id"], row["distribution"])].append(float(record["median_ms"]))
        else:
            failures.append({"cell_id": row["cell_id"], "distribution": row["distribution"], "rep": row["rep"], "error": record.get("error", "legacy_gate_failure")})
    results: dict[tuple[str, str], dict[str, Any]] = {}
    cells_out = []
    for cell_id in sorted(selected):
        for distribution in ("positive", "withheld_signed"):
            values = grouped[(cell_id, distribution)]
            row = {
                "cell_id": cell_id,
                "distribution": distribution,
                "eligible": len(values) == 15,
                "process_medians_ms": values,
            }
            if len(values) == 15:
                interval = exact_median_interval(values)
                row.update({"interval": interval, "median_ms": interval["median"]})
                results[(cell_id, distribution)] = row
            cells_out.append(row)
    stability = []
    for strategy in STRATEGIES:
        for lane in LANES:
            ids = sorted(cell_id for cell_id in selected if by_id[cell_id]["strategy"] == strategy and by_id[cell_id]["lane"] == lane and (cell_id, "positive") in results and (cell_id, "withheld_signed") in results)
            positive = [results[(cell_id, "positive")]["median_ms"] for cell_id in ids]
            signed = [results[(cell_id, "withheld_signed")]["median_ms"] for cell_id in ids]
            paired_intervals = {
                cell_id: exact_median_interval(
                    signed_process / positive_process
                    for positive_process, signed_process in zip(
                        grouped[(cell_id, "positive")],
                        grouped[(cell_id, "withheld_signed")],
                    )
                )
                for cell_id in ids
            }
            stability.append(
                {
                    "cell_ids": ids,
                    "lane": lane,
                    "per_cell_signed_over_positive": {cell_id: signed[index] / positive[index] for index, cell_id in enumerate(ids)},
                    "per_cell_signed_over_positive_paired_block_intervals": paired_intervals,
                    "positive_winner": ids[min(range(len(ids)), key=lambda index: positive[index])] if ids else None,
                    "rank_spearman": spearman_rank(positive, signed),
                    "signed_winner": ids[min(range(len(ids)), key=lambda index: signed[index])] if ids else None,
                    "strategy": strategy,
                    "winner_stable": bool(ids) and min(range(len(ids)), key=lambda index: positive[index]) == min(range(len(ids)), key=lambda index: signed[index]),
                }
            )
    return {
        "campaign_id": campaign["campaign_id"],
        "cell_results": cells_out,
        "complete": not failures and all(row["eligible"] for row in cells_out),
        "distribution_stability": stability,
        "failed_confirmation_processes": failures,
        "feasibility": selection["feasibility"],
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "performance_contrasts": _performance_contrasts(results),
        "record_hashes": hashes,
        "record_type": "fused_crossed_final_summary",
        "schema_version": 1,
        "selection_path": str(selection_path.relative_to(REPO_ROOT)),
        "selection_sha256": file_sha256(selection_path),
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "scope": campaign["claim_limit"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--result-root", required=True)
    audit_parser.add_argument("--out", required=True)
    screen_parser = subparsers.add_parser("screen")
    screen_parser.add_argument("--result-root", required=True)
    screen_parser.add_argument("--audit-summary", required=True)
    screen_parser.add_argument("--out", required=True)
    confirmation_parser = subparsers.add_parser("confirmation")
    confirmation_parser.add_argument("--result-root", required=True)
    confirmation_parser.add_argument("--selection", required=True)
    confirmation_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.result_root).resolve()
    if args.phase == "audit":
        value = audit_summary(root)
    elif args.phase == "screen":
        value = screen_selection(root, Path(args.audit_summary).resolve())
    else:
        value = confirmation_summary(root, Path(args.selection).resolve())
    stable_write(Path(args.out), value)
    print(f"{args.phase}: complete={value['complete']}")
    return 0 if value["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
