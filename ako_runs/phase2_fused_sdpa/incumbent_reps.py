#!/usr/bin/env python3
"""Run the two incumbent checks as N independent processes and aggregate.

Both `fused_incumbent_check.py` and `sdpa_incumbent_check.py` were written to
run once, in a single process, and report a bare millisecond per subject. That
is out of step with the protocol §1 states for everything else -- five
independent processes, median of per-process medians, t-based 95% CI -- and it
matters: a re-run of the fused check moved the tilelang figure by 9.9%, which is
more than twice the 4.10% cross-campaign drift §1.2 calls the floor, and it is
enough to flip the sign of the §2.1 torch-fp16-vs-tilelang ordering.

Rather than caveat that, this runs each script the protocol's five times, one
process each, and folds the results into the same JSON shape the report already
consumes -- with `ci95_lo_ms`/`ci95_hi_ms`/`n_procs` added so the tables can show
what is and is not resolvable. Ratios are recomputed from the aggregated
medians, never averaged across reps.

usage: python incumbent_reps.py --gpu 0 --reps 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402
import common  # noqa: E402

# Timing fields per subject, per script. Everything else (gate_pass,
# max_abs_err, notes) is taken from the first successful rep -- those are
# deterministic given the fixed seed and do not vary across processes.
TIMED = ("ms", "ms_flush0", "ms_flush1")


def run_reps(script, tag, gpu, reps, outdir):
    paths = []
    for r in range(reps):
        out = os.path.join(outdir, f"{tag}_rep{r}.json")
        if not os.path.exists(out):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = HERE + ":" + env.get("PYTHONPATH", "")
            env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, script), "--out", out],
                capture_output=True, text=True, env=env, cwd=HERE, timeout=3600)
            if not os.path.exists(out):
                print(f"  rep{r} FAILED: {(p.stderr or p.stdout)[-300:]}")
                continue
        print(f"  rep{r} ok -> {os.path.basename(out)}")
        paths.append(out)
    return paths


def fold(paths):
    """Median-of-per-process-medians + CI for every timed field, keyed by `who`."""
    docs = [json.load(open(p)) for p in paths]
    base = json.loads(json.dumps(docs[0]))          # deep copy, keeps non-timed fields
    by_who = {}
    for d in docs:
        for rec in d.get("records", []):
            k = (rec.get("who"), rec.get("flush_l2"))
            for f in TIMED:
                if isinstance(rec.get(f), (int, float)):
                    by_who.setdefault(k, {}).setdefault(f, []).append(rec[f])
    for rec in base.get("records", []):
        k = (rec.get("who"), rec.get("flush_l2"))
        for f, vals in (by_who.get(k) or {}).items():
            st = common.median_ci(vals)
            rec[f] = st["median_of_medians_ms"]
            if "ci95_lo_ms" in st:
                rec[f + "_ci95"] = [st["ci95_lo_ms"], st["ci95_hi_ms"]]
            rec[f + "_n"] = len(vals)
    # Ratios must come from the aggregated medians, not from averaging per-rep
    # ratios -- a mean of ratios is not the ratio of the medians.
    den = {r.get("who"): r for r in base.get("records", [])}
    f32 = den.get("torch_fp32") or den.get("torch_sdpa_fp32")
    f16 = den.get("torch_fp16") or den.get("torch_sdpa_fp16")

    def ms_of(r):
        for f in ("ms_flush0", "ms"):
            if isinstance(r.get(f), (int, float)):
                return r[f]
        return None

    b32, b16 = (ms_of(f32) if f32 else None), (ms_of(f16) if f16 else None)
    for rec in base.get("records", []):
        m = ms_of(rec)
        if m is None or not m:
            continue
        if b32 and "vs_fp32" in rec:
            rec["vs_fp32"] = b32 / m
        if b16 and "vs_fp16" in rec:
            rec["vs_fp16"] = b16 / m
        if b32 and "vs_fp32_ref" in rec:
            rec["vs_fp32_ref"] = b32 / m
        if b16 and "vs_fp16_ref" in rec:
            rec["vs_fp16_ref"] = b16 / m
    base["n_procs"] = len(docs)
    base["aggregation"] = ("median of per-process medians, t-based 95% CI, "
                           f"{len(docs)} independent processes")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    outdir = os.path.join(common2.RESULTS_DIR, "incumbent_reps")
    os.makedirs(outdir, exist_ok=True)
    for script, tag, final in (
            ("fused_incumbent_check.py", "fused", "fused_incumbent_check.json"),
            ("sdpa_incumbent_check.py", "sdpa", "sdpa_incumbent_check.json")):
        print(f"=== {script} x{a.reps} on GPU {a.gpu} ===", flush=True)
        paths = run_reps(script, tag, a.gpu, a.reps, outdir)
        if not paths:
            print(f"  no reps succeeded for {tag}")
            continue
        common2.write_json(os.path.join(common2.RESULTS_DIR, final), fold(paths))
        print(f"  -> {final}  ({len(paths)} processes)")


if __name__ == "__main__":
    main()
