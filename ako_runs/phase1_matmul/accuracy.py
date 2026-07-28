#!/usr/bin/env python3
"""Precision control: is the fp16 path generally valid, or a specialization on
the benchmark's all-positive input distribution?

Runs one variant over N seeds x {rand, randn-scaled-to-same-RMS}, recording
max/mean error, error quantiles, signed bias, and the percentage of elements
that violate the harness gate. Also compares BOTH the kernel and the fp32
oracle against an fp64 ground truth, so "error" can be attributed rather than
just observed.

The two distributions are RMS-matched by construction (common.RAND_RMS), so the
operands carry the same energy; the only difference is the mean. Under rand,
|C| ~ 2048 and the 1e-4 relative gate is a ~0.205 absolute budget. Under randn,
|C| ~ 24 and the same gate is a ~0.0025 budget -- 82x tighter. That ratio is the
confound, and it is a property of the benchmark, not of any kernel.

usage:
  python accuracy.py --dsl tilelang --variant D --seeds 20
  python accuracy.py --dsl tilelang --variant C --set kc=512 --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
common.setup_cuda_env()

import torch  # noqa: E402
import variants  # noqa: E402
from runner import parse_set  # noqa: E402


def harness_oracle_pass(ref, got) -> bool:
    """Exactly the harness predicate (AKO bench.py:_allclose_and_maxdiff)."""
    d = (ref - got).abs()
    return bool((d <= common.GATE_ATOL + common.GATE_RTOL * ref.abs()).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsl", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--geom", default="primary")
    ap.add_argument("--set", dest="setstr", default="")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--dists", default="rand,randn")
    ap.add_argument("--fp64", action="store_true", default=True)
    ap.add_argument("--no-fp64", dest="fp64", action="store_false")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = common.make_config(args.dsl, args.variant, args.geom, **parse_set(args.setstr))
    built = variants.build(cfg)

    rec = {"cfg": cfg.to_dict(), "key": cfg.key(), "dsl": cfg.dsl,
           "variant": cfg.variant, "geom": args.geom,
           "compile_s": built.compile_s, "seeds": args.seeds,
           "seed0": args.seed0, "per_seed": [], "env": common.env_fingerprint()}

    t0 = time.time()
    for dist in args.dists.split(","):
        dist = dist.strip()
        if not dist:
            continue
        for i in range(args.seeds):
            seed = args.seed0 + i
            A32, B32 = common.load_inputs(dist, seed, device="cuda")
            with torch.no_grad():
                ref = common.reference_fp32(A32, B32)
                truth = common.reference_fp64(A32, B32) if args.fp64 else None

                if built.input_dtype == torch.float16:
                    A, B = A32.half().contiguous(), B32.half().contiguous()
                else:
                    A, B = A32, B32
                got = built.run(A, B).float()
                torch.cuda.synchronize()

            st = common.gate_stats(ref, got, truth)
            st["dist"] = dist
            st["seed"] = seed
            # the harness's own 5-trial verdict is all-or-nothing per trial
            st["harness_trial_pass"] = harness_oracle_pass(ref, got)
            rec["per_seed"].append(st)
            del ref, got, A, B, A32, B32, truth
            torch.cuda.empty_cache()

    # roll up per distribution
    rec["by_dist"] = {}
    for dist in {s["dist"] for s in rec["per_seed"]}:
        rows = [s for s in rec["per_seed"] if s["dist"] == dist]
        n = len(rows)

        def agg(k, f=max):
            return f(r[k] for r in rows)

        def mean(k):
            return sum(r[k] for r in rows) / n

        rec["by_dist"][dist] = {
            "n_seeds": n,
            "trials_passing_gate": sum(1 for r in rows if r["harness_trial_pass"]),
            "pass_rate_pct": 100.0 * sum(1 for r in rows if r["harness_trial_pass"]) / n,
            "max_abs_err_worst": agg("max_abs_err", max),
            "max_abs_err_mean": mean("max_abs_err"),
            "mean_abs_err_mean": mean("mean_abs_err"),
            "signed_mean_err_mean": mean("signed_mean_err"),
            "err_q50_mean": mean("err_q50"),
            "err_q90_mean": mean("err_q90"),
            "err_q99_mean": mean("err_q99"),
            "err_q999_mean": mean("err_q999"),
            "pct_elems_failing_gate_mean": mean("pct_elems_failing_gate"),
            "pct_elems_failing_gate_worst": agg("pct_elems_failing_gate", max),
            "budget_mean": mean("budget_mean"),
            "ref_abs_mean": mean("ref_abs_mean"),
        }
        if args.fp64:
            rec["by_dist"][dist].update({
                "oracle_max_err_vs_fp64_mean": mean("oracle_max_abs_err_vs_fp64"),
                "kernel_max_err_vs_fp64_mean": mean("kernel_max_abs_err_vs_fp64"),
                "oracle_bias_vs_fp64_mean": mean("oracle_signed_mean_err_vs_fp64"),
                "kernel_bias_vs_fp64_mean": mean("kernel_signed_mean_err_vs_fp64"),
            })

    rec["wall_s"] = time.time() - t0
    out = args.out or os.path.join(common.RESULTS_DIR, "accuracy", cfg.key() + ".json")
    common.write_json(out, rec)

    print(f"== {cfg.key()} ==")
    for dist, d in sorted(rec["by_dist"].items()):
        print(f"  {dist:6s} pass {d['trials_passing_gate']}/{d['n_seeds']} "
              f"({d['pass_rate_pct']:.0f}%)  maxerr(worst)={d['max_abs_err_worst']:.4g}  "
              f"budget={d['budget_mean']:.4g}  "
              f"failing_elems={d['pct_elems_failing_gate_mean']:.3f}%  "
              f"bias={d['signed_mean_err_mean']:+.4g}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
