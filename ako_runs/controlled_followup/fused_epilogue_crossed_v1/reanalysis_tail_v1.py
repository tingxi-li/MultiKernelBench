#!/usr/bin/env python3
"""Settled-tail re-analysis of the crossed_v1r1 confirmation records.

Motivation (DESIGN_REFLECTION_20260731.md §1.3): the published statistic is a median
over a 100-trial window that is not stationary. Processes enter the window in one of two
transient states and converge toward a common plateau from opposite directions, so the
per-process median estimates a point on a decay curve rather than a steady-state latency.

This module recomputes the campaign's own estimators -- unchanged in form -- over the
settled tail of each process's trial vector. It reads only sealed records, writes only to
a new file, and changes no frozen gate, threshold, selection, or cell definition.

The published full-window results remain controlling. This is a diagnostic overlay.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "crossed_v1r1")
RAW = os.path.join(RESULTS, "confirmation", "raw")

# n=15 exact central order-statistic interval, as preregistered in campaign.json:
# "n=15 uses [x4,x12] with coverage 0.96484375".
ORDER_K = 4
COVERAGE = 0.96484375


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def order_interval(values: list[float]) -> dict:
    ordered = sorted(values)
    if len(ordered) != 15:
        raise ValueError(f"expected 15 blocks, got {len(ordered)}")
    return {
        "median": statistics.median(ordered),
        "ci_lo": ordered[ORDER_K - 1],
        "ci_hi": ordered[len(ordered) - ORDER_K],
        "coverage": COVERAGE,
        "n": len(ordered),
        "order_k": ORDER_K,
    }


def load_trials() -> dict:
    """cell_id -> distribution -> rep -> times_ms."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        with open(path) as handle:
            record = json.load(handle)
        name = os.path.basename(path).split("__")
        cell_id = f"{name[0]}.{name[1]}.{name[2]}"
        out[cell_id][record["distribution"]][record["rep"]] = record["times_ms"]
    return out


def paired_ratio(num: dict, den: dict, lo: int, hi: int) -> dict | None:
    """Paired per-block ratio of medians over trials [lo, hi)."""
    reps = sorted(set(num) & set(den))
    if len(reps) != 15:
        return None
    ratios = [
        statistics.median(num[rep][lo:hi]) / statistics.median(den[rep][lo:hi])
        for rep in reps
    ]
    interval = order_interval(ratios)
    interval["excludes_unity"] = interval["ci_lo"] > 1.0 or interval["ci_hi"] < 1.0
    return interval


def drift(trials: dict) -> dict:
    """Within-window non-stationarity: first vs last decile, pooled over blocks."""
    head = statistics.mean(statistics.mean(v[:10]) for v in trials.values())
    tail = statistics.mean(statistics.mean(v[-10:]) for v in trials.values())
    return {
        "first_decile_ms": head,
        "last_decile_ms": tail,
        "drift_fraction": tail / head - 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-start", type=int, default=60,
                        help="first trial index of the settled tail (default 60)")
    parser.add_argument("--out", default=os.path.join(RESULTS, "reanalysis_tail_v1.json"))
    args = parser.parse_args()

    summary_path = os.path.join(RESULTS, "final_summary.json")
    with open(summary_path) as handle:
        published = json.load(handle)

    trials = load_trials()
    windows = {"full": (0, 100), "tail": (args.tail_start, 100)}

    # ---- performance contrasts, recomputed in both windows --------------------
    contrasts = []
    for family, rows, ref_key in (
        ("lane_effects", published["performance_contrasts"]["lane_effects_common_selected_grids"], "tilelang"),
        ("strategy_effects", published["performance_contrasts"]["strategy_effects_common_selected_grids"], "register_fused"),
    ):
        for row in rows:
            grid, lane, strategy = row["grid_id"], row["lane"], row["strategy"]
            if family == "lane_effects":
                num = f"{strategy}.{lane}.{grid}"
                den = f"{strategy}.{ref_key}.{grid}"
                published_point = row["ratio_to_tilelang"]
            else:
                num = f"{strategy}.{lane}.{grid}"
                den = f"{ref_key}.{lane}.{grid}"
                published_point = row["ratio_to_register"]
            entry = {
                "family": family,
                "grid_id": grid,
                "lane": lane,
                "strategy": strategy,
                "numerator_cell": num,
                "denominator_cell": den,
                "published_point_estimate": published_point,
            }
            for label, (lo, hi) in windows.items():
                got = paired_ratio(trials[num]["positive"], trials[den]["positive"], lo, hi)
                entry[label] = got
            if entry["full"] and entry["tail"]:
                fw = entry["full"]["ci_hi"] - entry["full"]["ci_lo"]
                tw = entry["tail"]["ci_hi"] - entry["tail"]["ci_lo"]
                entry["interval_width_full"] = fw
                entry["interval_width_tail"] = tw
                entry["precision_gain"] = fw / tw if tw > 0 else None
                entry["point_shift_fraction"] = entry["tail"]["median"] / entry["full"]["median"] - 1.0
            contrasts.append(entry)

    # ---- distribution stability, paired, recomputed in both windows -----------
    stability = []
    for cell_id, by_dist in sorted(trials.items()):
        if "positive" not in by_dist or "withheld_signed" not in by_dist:
            continue
        entry = {"cell_id": cell_id}
        for label, (lo, hi) in windows.items():
            entry[label] = paired_ratio(by_dist["withheld_signed"], by_dist["positive"], lo, hi)
        entry["published_unpaired_signed_over_positive"] = None
        stability.append(entry)

    published_ratios = {}
    for row in published["distribution_stability"]:
        published_ratios.update(row.get("per_cell_signed_over_positive") or {})
    for entry in stability:
        entry["published_unpaired_signed_over_positive"] = published_ratios.get(entry["cell_id"])

    # ---- non-stationarity census --------------------------------------------
    drifts = []
    for cell_id, by_dist in sorted(trials.items()):
        for dist, blocks in sorted(by_dist.items()):
            row = drift(blocks)
            row.update({"cell_id": cell_id, "distribution": dist})
            drifts.append(row)

    out = {
        "record_type": "fused_crossed_settled_tail_reanalysis",
        "schema_version": 1,
        "campaign_id": published["campaign_id"],
        "result_tag": published["result_tag"],
        "status": "diagnostic_overlay",
        "claim_limit": (
            "The published full-window final_summary.json remains the controlling artifact. "
            "This overlay recomputes the same preregistered estimators over a settled tail of "
            "each process's trial vector to quantify how much of the reported interval width is "
            "within-window non-stationarity. It changes no gate, threshold, selection, or cell."
        ),
        "tail_start_trial": args.tail_start,
        "windows": {k: {"lo": v[0], "hi": v[1]} for k, v in windows.items()},
        "source_final_summary": {
            "path": os.path.relpath(summary_path, os.path.dirname(HERE) + "/../.."),
            "sha256": sha256_file(summary_path),
        },
        "recovery_git_commit": published["recovery_git_commit"],
        "recovery_lock_sha256": published["recovery_lock_sha256"],
        "parent_launch_lock_sha256": published["parent_launch_lock_sha256"],
        "performance_contrasts": contrasts,
        "distribution_stability": stability,
        "within_window_drift": drifts,
    }

    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
    print(f"wrote {args.out}")
    print(f"sha256 {sha256_file(args.out)}")


if __name__ == "__main__":
    main()
