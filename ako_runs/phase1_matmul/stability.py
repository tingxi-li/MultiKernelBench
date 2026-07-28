#!/usr/bin/env python3
"""Characterize timing stability before trusting any number from this host.

An adversarial verifier observed tilelang variant D returning 0.657 / 1.033 /
1.034 ms across three identical invocations -- a 57% swing. Averaging over that
without understanding it would produce a confident-looking number that is a
coin flip. This probes the mechanism.

Two things are tested, one process per point:

  1. WITHIN-PROCESS drift: report the timed trials in order, so a clock ramp
     (monotone downward) is distinguishable from bimodal switching (two levels)
     and from noise (unstructured).
  2. BETWEEN-PROCESS spread: N independent processes, same everything, so the
     per-process medians can be compared.

Also sweeps warmup depth, since these cards idle at 210 MHz and the harness's
200-iteration warmup was chosen for memory-bound ops, not for a 1 ms GEMM.

usage:
  python stability.py --dsl tilelang --variant D --procs 8 --gpu 0
  python stability.py --dsl tilelang --variant D --warmups 20,50,200,500 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402


def one(dsl, variant, gpu, warmup, trials, tag, warmup_s=0.0):
    out = os.path.join(common.RESULTS_DIR, "stability", f"{tag}.json")
    cmd = [sys.executable, os.path.join(HERE, "runner.py"), "--dsl", dsl,
           "--variant", variant, "--warmup", str(warmup), "--warmup-s", str(warmup_s), "--trials", str(trials),
           "--time-only", "--out", out]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = HERE + ":" + env.get("PYTHONPATH", "")
    subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE, timeout=1800)
    if not os.path.exists(out):
        return None
    with open(out) as f:
        return json.load(f)


def describe(ts):
    """Split the trial series in halves to expose a ramp, and count clusters."""
    n = len(ts)
    h1 = statistics.median(ts[:n // 2])
    h2 = statistics.median(ts[n // 2:])
    lo, hi = min(ts), max(ts)
    mid = 0.5 * (lo + hi)
    n_lo = sum(1 for t in ts if t < mid)
    return {"first_half_median": h1, "second_half_median": h2,
            "drift_pct": 100.0 * (h2 - h1) / h1 if h1 else 0.0,
            "below_midpoint": n_lo, "above_midpoint": n - n_lo,
            "min": lo, "max": hi, "range_pct": 100.0 * (hi - lo) / statistics.median(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsl", default="tilelang")
    ap.add_argument("--variant", default="D")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--warmups", default="200")
    ap.add_argument("--warmup-s", dest="warmup_s", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(os.path.join(common.RESULTS_DIR, "stability"), exist_ok=True)
    report = {"dsl": args.dsl, "variant": args.variant, "by_warmup": {}}

    for w in [int(x) for x in args.warmups.split(",")]:
        meds, series = [], []
        print(f"\n== warmup={w}, {args.procs} independent processes ==")
        for i in range(args.procs):
            r = one(args.dsl, args.variant, args.gpu, w, args.trials,
                    f"{args.dsl}_{args.variant}_w{w}s{args.warmup_s}_p{i}", args.warmup_s)
            if not r or not r.get("ok"):
                print(f"  proc {i}: FAILED")
                continue
            ts = r["times_ms"]
            m = r["timing"]["median_ms"]
            d = describe(ts)
            meds.append(m)
            series.append(ts)
            print(f"  proc {i}: median={m:.4f}  1st-half={d['first_half_median']:.4f} "
                  f"2nd-half={d['second_half_median']:.4f} drift={d['drift_pct']:+.1f}%  "
                  f"range={d['range_pct']:.1f}%  split={d['below_midpoint']}/{d['above_midpoint']}")
        if meds:
            ci = common.median_ci(meds)
            print(f"  -> median of medians {ci['median_of_medians_ms']:.4f} ms, "
                  f"process spread {ci.get('rel_spread_pct', 0):.1f}%  "
                  f"[{ci['min_ms']:.4f} .. {ci['max_ms']:.4f}]")
            report["by_warmup"][w] = {"process_medians": meds, "ci": ci,
                                      "per_process": [describe(s) for s in series]}

    common.write_json(os.path.join(common.RESULTS_DIR, "stability",
                                   f"summary_{args.dsl}_{args.variant}.json"), report)
    print(f"\n-> results/stability/summary_{args.dsl}_{args.variant}.json")


if __name__ == "__main__":
    main()
