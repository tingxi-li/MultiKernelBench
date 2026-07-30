#!/usr/bin/env python3
"""Fail-closed screen/confirmation analysis for fused reachability v2."""
from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from protocol import (
    CAMPAIGN,
    JOBS,
    LOCK,
    V1_CONFIRMATION_SUMMARY,
    canonical_sha256,
    file_sha256,
    read_json,
    record_filename,
    stable_write,
    validate_launch_receipt,
    validate_process_record,
    verify_lock,
)


LANES = ("cuda_noptx", "cuda_unlimited")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "confirmation"), required=True)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selection", default="")
    return parser.parse_args()


def valid_process(record: dict) -> bool:
    error = record.get("legacy_error")
    median = record.get("median_ms")
    return bool(
        record.get("ok")
        and isinstance(error, dict)
        and error.get("gate_pass") is True
        and isinstance(median, (int, float))
        and math.isfinite(float(median))
        and median > 0
    )


def percentile_linear(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def expected_confirmation(selection_path: Path, lock: dict) -> tuple[dict, dict[str, set[str]]]:
    selection = read_json(selection_path)
    if (
        selection.get("record_type") != "fused_reachability_v2_confirmation_selection"
        or selection.get("campaign_id") != lock["campaign_id"]
        or selection.get("launch_lock_sha256") != file_sha256(LOCK)
        or selection.get("source_bundle_sha256") != lock["source_bundle_sha256"]
    ):
        raise RuntimeError("confirmation selection is not bound to this campaign")
    result = {lane: set() for lane in LANES}
    for row in selection.get("selected", []):
        if row.get("robust_eligible") is True:
            result[row["lane"]].add(row["job_id"])
    return selection, result


def analyze_confirmation_inference(
    cells: list[dict], grouped: dict[str, list[dict]], lock: dict
) -> dict:
    campaign = read_json(CAMPAIGN)
    inference = campaign["inference"]
    draws = inference["block_bootstrap_draws"]
    seed = inference["block_bootstrap_seed"]
    baselines = lock["v1_baseline"]["point_minima_ms"]
    lane_results = []
    for lane_index, lane in enumerate(LANES):
        lane_cells = [cell for cell in cells if cell["lane"] == lane and cell["eligible"]]
        if not lane_cells:
            lane_results.append({"lane": lane, "status": "NO_ROBUST_PASSING_CELL"})
            continue
        for cell in lane_cells:
            ordered = sorted(cell["process_medians_ms"])
            if len(ordered) != 15:
                raise RuntimeError(f"confirmation cell lacks n=15: {cell['job_id']}")
            cell["median_interval"] = {
                "coverage": 0.96484375,
                "method": "order_statistics_x4_x12",
                "lo_ms": ordered[3],
                "hi_ms": ordered[11],
            }
        point_winner = min(
            lane_cells,
            key=lambda row: (row["median_of_process_medians_ms"], row["job_id"]),
        )
        strict = all(
            point_winner["median_interval"]["hi_ms"]
            < other["median_interval"]["lo_ms"]
            for other in lane_cells
            if other["job_id"] != point_winner["job_id"]
        )

        by_job_rep = {
            cell["job_id"]: {
                record["rep"]: float(record["median_ms"])
                for record in grouped[cell["job_id"]]
            }
            for cell in lane_cells
        }
        generator = random.Random(seed + lane_index)
        minima = []
        winner_counts = {cell["job_id"]: 0 for cell in lane_cells}
        for _ in range(draws):
            sampled_reps = [generator.randrange(15) for _ in range(15)]
            estimates = {
                job_id: statistics.median(values[rep] for rep in sampled_reps)
                for job_id, values in by_job_rep.items()
            }
            winner = min(estimates, key=lambda job_id: (estimates[job_id], job_id))
            winner_counts[winner] += 1
            minima.append(estimates[winner])
        minima.sort()
        point = float(point_winner["median_of_process_medians_ms"])
        baseline = float(baselines[lane])
        lane_results.append(
            {
                "lane": lane,
                "status": "RESOLVED" if strict else "UNRESOLVED",
                "point_winner_job_id": point_winner["job_id"],
                "lane_minimum_ms": point,
                "lane_minimum_bootstrap_ci95_ms": [
                    percentile_linear(minima, 0.025),
                    percentile_linear(minima, 0.975),
                ],
                "bootstrap_draws": draws,
                "bootstrap_seed": seed + lane_index,
                "winner_probabilities": {
                    job_id: count / draws for job_id, count in winner_counts.items()
                },
                "v1_confirmed_minimum_ms": baseline,
                "descriptive_v1_over_v2_speedup_x": baseline / point,
                "v1_summary_path": str(V1_CONFIRMATION_SUMMARY),
                "v1_summary_sha256": lock["v1_baseline"]["summary_sha256"],
            }
        )
    return {
        "cell_estimator": inference["cell_point_estimator"],
        "cell_interval": inference["cell_interval"],
        "lane_minimum_rule": inference["lane_minimum"],
        "lanes": lane_results,
        "scope": (
            "v2 timing comparisons are within lane; v1 ratios are descriptive "
            "because they are cross-campaign. No cross-GPU lane ratio is inferred."
        ),
    }


def main() -> int:
    args = parse_args()
    lock = verify_lock()
    all_jobs = {row["job_id"]: row for row in read_json(JOBS)}
    expected_reps = lock["launch_policy"][args.phase]["reps"]
    selection = None
    expected_selected = None
    selection_path = None
    if args.phase == "confirmation":
        if not args.selection:
            raise ValueError("confirmation analysis requires --selection")
        selection_path = Path(args.selection)
        selection, expected_selected = expected_confirmation(selection_path, lock)
    elif args.selection:
        raise ValueError("screen analysis does not consume a confirmation selection")

    grouped: dict[str, list[dict]] = defaultdict(list)
    record_hashes = []
    seen_lanes = set()
    physical_gpus = {}
    for root_text in args.result:
        root = Path(root_text)
        lane = read_json(root / "launch_receipt.json").get("contract", {}).get("lane")
        if lane not in LANES or lane in seen_lanes:
            raise RuntimeError(f"invalid or duplicate lane receipt at {root}: {lane}")
        seen_lanes.add(lane)
        receipt, root_jobs = validate_launch_receipt(
            root, phase=args.phase, lane=lane, lock=lock
        )
        physical_gpu = receipt["contract"]["physical_gpu"]
        physical_gpus[lane] = physical_gpu
        root_job_ids = {job["job_id"] for job in root_jobs}
        frozen_lane_ids = {job_id for job_id, job in all_jobs.items() if job["lane"] == lane}
        if args.phase == "screen" and root_job_ids != frozen_lane_ids:
            raise RuntimeError(f"screen root does not cover all eight {lane} jobs")
        if args.phase == "confirmation":
            assert expected_selected is not None and selection_path is not None
            if root_job_ids != expected_selected[lane]:
                raise RuntimeError(f"confirmation root differs from robust selection for {lane}")
            if receipt["contract"].get("selection_sha256") != file_sha256(selection_path):
                raise RuntimeError(f"confirmation receipt selection mismatch for {lane}")
        expected_files = {
            record_filename(job["job_id"], rep)
            for job in root_jobs
            for rep in range(expected_reps)
        }
        observed_files = {path.name for path in (root / "raw").glob("*.json")}
        if observed_files != expected_files:
            raise RuntimeError(
                f"raw record set mismatch at {root}: "
                f"missing={sorted(expected_files-observed_files)}, "
                f"unexpected={sorted(observed_files-expected_files)}"
            )
        for job in root_jobs:
            for rep in range(expected_reps):
                path = root / "raw" / record_filename(job["job_id"], rep)
                record = validate_process_record(
                    path,
                    job=job,
                    phase=args.phase,
                    rep=rep,
                    physical_gpu=physical_gpu,
                    lock=lock,
                )
                if any(old.get("rep") == rep for old in grouped[job["job_id"]]):
                    raise RuntimeError(f"duplicate process record: {job['job_id']}/rep{rep}")
                grouped[job["job_id"]].append(record)
                record_hashes.append({"path": str(path), "sha256": file_sha256(path)})

    required_lanes = set(LANES) if args.phase == "screen" else {
        lane for lane in LANES if expected_selected and expected_selected[lane]
    }
    if seen_lanes != required_lanes:
        raise RuntimeError(f"result lane coverage mismatch: {seen_lanes} != {required_lanes}")

    cells = []
    for job_id, job in all_jobs.items():
        records = sorted(grouped.get(job_id, []), key=lambda row: row["rep"])
        if args.phase == "screen" or records:
            medians = [float(row["median_ms"]) for row in records if valid_process(row)]
            eligible = (
                len(records) == expected_reps
                and [row["rep"] for row in records] == list(range(expected_reps))
                and len(medians) == expected_reps
            )
            cells.append(
                {
                    "job_id": job_id,
                    "lane": job["lane"],
                    "grid_id": job["grid_id"],
                    "record_count": len(records),
                    "eligible": eligible,
                    "eligibility_scope": (
                        "launch_reachable_and_legacy_correct"
                        if args.phase == "screen" else "frozen_mixed_gate_and_legacy_correct"
                    ),
                    "process_medians_ms": medians,
                    "median_of_process_medians_ms": (
                        statistics.median(medians) if eligible else None
                    ),
                    "failures": [
                        row.get("error", "legacy_gate_failure")
                        for row in records if not valid_process(row)
                    ],
                }
            )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "record_type": "fused_reachability_v2_analysis",
        "campaign_id": lock["campaign_id"],
        "phase": args.phase,
        "launch_lock_sha256": file_sha256(LOCK),
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "expected_reps": expected_reps,
        "physical_gpus": physical_gpus,
        "cells": cells,
        "record_bundle": record_hashes,
        "record_bundle_sha256": canonical_sha256(record_hashes),
    }
    if args.phase == "confirmation":
        summary["confirmation_selection_path"] = str(selection_path)
        summary["confirmation_selection_sha256"] = file_sha256(selection_path)
        summary["inference"] = analyze_confirmation_inference(cells, grouped, lock)
    summary_path = output / f"{args.phase}_summary.json"
    stable_write(summary_path, summary)

    if args.phase == "screen":
        selected = []
        for lane in LANES:
            eligible = sorted(
                (row for row in cells if row["lane"] == lane and row["eligible"]),
                key=lambda row: (row["median_of_process_medians_ms"], row["job_id"]),
            )
            for rank, row in enumerate(eligible[:3], 1):
                selected.append(
                    {
                        "job_id": row["job_id"],
                        "lane": lane,
                        "grid_id": row["grid_id"],
                        "screen_rank": rank,
                        "screen_median_ms": row["median_of_process_medians_ms"],
                        "job_sha256": lock["job_sha256"][row["job_id"]],
                        "robust_eligible": False,
                    }
                )
        selection_value = {
            "schema_version": 1,
            "record_type": "fused_reachability_v2_screen_selection",
            "campaign_id": lock["campaign_id"],
            "launch_lock_sha256": file_sha256(LOCK),
            "source_bundle_sha256": lock["source_bundle_sha256"],
            "selection_rule": "top three legacy-correct screen cells per lane",
            "screen_summary_path": str(summary_path),
            "screen_summary_sha256": file_sha256(summary_path),
            "selected": selected,
            "screen_bundle_sha256": canonical_sha256(record_hashes),
        }
        stable_write(output / "selection.json", selection_value)
    print(f"wrote {len(cells)} cells to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
