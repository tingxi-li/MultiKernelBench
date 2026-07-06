#!/usr/bin/env python3
"""
check_gate.py — protect the validated 12-op artifact from a redo regression.

A redo of an already-committed (op, dsl) cell must LAND AT OR ABOVE its committed
speedup (within a clock-noise margin). If it doesn't, the redo did not improve
anything and the orchestrator MUST keep the committed solution (git checkout the
prior bytes) rather than accept a slower one.

Speedup is a ref/solution RATIO measured in the same run, so it is far more
clock-robust than absolute ms — a legitimate cross-run floor. Even so, the
authoritative signal per the project's orchestration rule is the redo's OWN
same-GPU baseline->final delta; this gate is the second guard, not the only one.

Usage:
    python check_gate.py --op layer_norm --dsl cuda_noptx --speedup 1.72
    # exit 0 = PASS (keep redo), exit 1 = FAIL (revert to committed)

Ops with no committed floor (the 28 new ones) PASS with a note — nothing to
protect yet.
"""
import argparse
import csv
import os
import sys

BASELINE = os.path.join(os.path.dirname(__file__), "committed_baseline.csv")


def load_floor():
    floor = {}
    with open(BASELINE) as f:
        for row in csv.DictReader(f):
            floor[(row["op"], row["dsl"])] = float(row["committed_speedup"])
    return floor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True)
    ap.add_argument("--dsl", required=True)
    ap.add_argument("--speedup", type=float, required=True,
                    help="the redo's verdict SPEEDUP (ref/solution, full bench)")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="fractional clock-noise tolerance (default 0.03 = 3%%)")
    args = ap.parse_args()

    floor = load_floor()
    key = (args.op, args.dsl)
    if key not in floor:
        print(f"GATE: PASS — no committed floor for {args.op}/{args.dsl} "
              f"(new op, nothing to protect). redo speedup={args.speedup:.4f}x")
        sys.exit(0)

    committed = floor[key]
    threshold = committed * (1.0 - args.margin)
    ok = args.speedup >= threshold
    verdict = "PASS" if ok else "FAIL"
    print(f"GATE: {verdict} — {args.op}/{args.dsl}: "
          f"redo={args.speedup:.4f}x vs committed={committed:.4f}x "
          f"(floor={threshold:.4f}x @ {args.margin:.0%} margin)")
    if not ok:
        print("  -> KEEP COMMITTED: git checkout <committed-sha> -- solution/ "
              "and re-verify; do NOT accept the regression.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
