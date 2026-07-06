#!/usr/bin/env python3
"""
record_committed_baseline.py — refresh a cell's gate floor with an AGENT-FREE
re-bench of the exact committed bytes (the project's "trust the committed bytes,
not the trajectory logs" rule). Seeds in committed_baseline.csv came from
RESULTS.md, which mixes clock states; a fresh same-host re-bench is truer.

Runs the workspace's OWN scripts/bench.sh (single source of truth for per-op
flags like scatter's --deterministic), captures SPEEDUP, and optionally writes
it back into committed_baseline.csv.

Usage:
    # one cell, pinned to GPU 3, just print:
    python record_committed_baseline.py --op layer_norm --dsl cuda_noptx --gpu 3
    # ... and update the floor table:
    python record_committed_baseline.py --op layer_norm --dsl cuda_noptx --gpu 3 --write

Fan out across GPUs with the 4-lane pattern (one cell per GPU at a time) to
refresh all 48 without OOM. Do NOT run agents for this — a plain bench cannot
contaminate the number.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
AKO_RUNS = os.path.dirname(TOOLS)
BASELINE = os.path.join(TOOLS, "committed_baseline.csv")


def run_bench(op, dsl, gpu):
    bench = os.path.join(AKO_RUNS, op, dsl, "scripts", "bench.sh")
    if not os.path.exists(bench):
        sys.exit(f"no bench.sh at {bench}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    out = subprocess.run(
        ["bash", bench, "baseline_refresh"],
        env=env, capture_output=True, text=True,
    )
    text = out.stdout + out.stderr
    m_sp = re.search(r"^SPEEDUP:\s*([0-9.]+)x", text, re.M)
    m_ok = re.search(r"^CORRECT:\s*(True|False)", text, re.M)
    speedup = float(m_sp.group(1)) if m_sp else None
    correct = (m_ok.group(1) == "True") if m_ok else False
    return speedup, correct, text


def update_csv(op, dsl, speedup):
    rows = []
    with open(BASELINE) as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else ["op", "dsl", "committed_speedup", "source"]
    hit = False
    for r in rows:
        if r["op"] == op and r["dsl"] == dsl:
            r["committed_speedup"] = f"{speedup:.4f}"
            r["source"] = "rebench"
            hit = True
    if not hit:
        rows.append({"op": op, "dsl": dsl,
                     "committed_speedup": f"{speedup:.4f}", "source": "rebench"})
    with open(BASELINE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True)
    ap.add_argument("--dsl", required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--write", action="store_true", help="update committed_baseline.csv")
    args = ap.parse_args()

    speedup, correct, text = run_bench(args.op, args.dsl, args.gpu)
    if speedup is None or not correct:
        print(f"{args.op}/{args.dsl}: FAILED to get a clean number "
              f"(correct={correct}, speedup={speedup}). Not writing.")
        print(text[-800:])
        sys.exit(1)
    print(f"{args.op}/{args.dsl}: SPEEDUP={speedup:.4f}x CORRECT={correct} (gpu {args.gpu})")
    if args.write:
        update_csv(args.op, args.dsl, speedup)
        print(f"  -> committed_baseline.csv updated")


if __name__ == "__main__":
    main()
