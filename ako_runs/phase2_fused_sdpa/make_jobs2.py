#!/usr/bin/env python3
"""Emit the Phase-2 job lists.

  fused_matched   the ladder, at the Phase-1 matched configuration.
                  4 DSLs x 4 arms x {cached, uncached}, plus both torch
                  denominators on the same ladder and the same weight factor.
                  This is the table that answers "what does each feature cost,
                  and how much of the published gain is the weight cache".

  fused_native    the third weight level, where the lane can express it:
                  the kernel consumes W in its stored (N, K) layout and never
                  materializes a transposed fp16 copy at all. Without this the
                  "uncached" arm is a straw man -- it pays for a transpose that
                  a competent uncached kernel simply would not do.

  fused_epilogue  cuda_unlimited only: register epilogue vs the shared-memory
                  epilogue the WMMA lane is forced into. Measures what knowing
                  your own fragment layout is worth.

  fused_cast      the activation-cast control, x fp32 -> fp16 inside the timed
                  region instead of outside it. Phase 1 found this worth 0.21 ms
                  on a 1.05 ms kernel, so it cannot be left implicit.

usage: python make_jobs2.py [--outdir jobs]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402

ARMS = list(common2.FUSED_ARMS)          # G, GB, GBG, GBGS
DSLS = list(common2.DSLS)                # tilelang, triton, cuda_noptx, cuda_unlimited


def j(dsl, variant, **extra):
    s = ",".join(f"{k}={v}" if k in ("arith",) else f"x_{k}={v}"
                 for k, v in sorted(extra.items()))
    return {"dsl": dsl, "variant": variant, "set": s}


def fused_matched():
    jobs = []
    for w in ("cached", "uncached"):
        for d in DSLS:
            for a in ARMS:
                jobs.append(j(d, a, wcache=w))
        # both denominators climb the same ladder under the same weight factor
        for arith in ("fp32", "fp16"):
            for a in ARMS:
                jobs.append({"dsl": "torch", "variant": a,
                             "set": f"arith={arith},x_wcache={w}"})
    return jobs


def fused_native():
    # tilelang and triton express a transposed-B, fp32-global GEMM as a config
    # change; the two hand-written CUDA lanes would need a different ldmatrix /
    # WMMA fragment staging, which is a rewrite rather than a factor level, so
    # they are absent here and SPEC2.md says so.
    jobs = []
    for d in ("tilelang", "triton"):
        for a in ARMS:
            jobs.append(j(d, a, wcache="native"))
    for a in ARMS:
        jobs.append({"dsl": "torch", "variant": a,
                     "set": "arith=fp16,x_wcache=native"})
    return jobs


def fused_epilogue():
    return [j("cuda_unlimited", a, wcache="cached", epilogue=e)
            for e in ("smem", "regs") for a in ARMS]


def fused_cast():
    # cast=in_region on the full op only: the question is how much of the
    # published number is the activation cast, not how it interacts with bias.
    return [{"dsl": d, "variant": "GBGS", "set": "cast=in_region,x_wcache=cached"}
            for d in DSLS] + \
           [{"dsl": "torch", "variant": "GBGS", "set": "arith=fp16,x_wcache=cached"}]


def fused_abstraction():
    """TileLang-only softmax-abstraction arms F1..F4, with the weight factor
    run as the separate two-way factor the study specifies."""
    jobs = [j("tilelang_abs", a, wcache=w)
            for w in ("cached", "uncached") for a in common2.FUSED_ABS_ARMS]
    # ...and the same arms timed with the GEMM excluded. Without this the study
    # cannot answer its own question: the softmax is ~5% of the op, so a 3%
    # difference between reduction styles is invisible in the whole-op number.
    jobs += [j("tilelang_abs", a, wcache="cached", soft_only=1)
             for a in common2.FUSED_ABS_ARMS]
    return jobs


def sdpa_cross():
    """The cross-DSL SDPA matrix: two identical algorithms, three head dims,
    the two dtypes fixed independently. K2 is deliberately absent -- the study
    puts the two-kernel decomposition in the TileLang algorithm axis (S2), not
    in the cross-DSL comparison."""
    jobs = []
    for d in common2.S_HEAD_DIMS:
        for sd, pd in common2.SDPA_DTYPES:
            for dsl in DSLS:
                for algo in ("K3", "FLASH"):
                    jobs.append(j(dsl, algo, d=d, sdtype=sd, pdtype=pd))
        for who in ("TORCH_F32", "TORCH_F16", "TORCH_MATH"):
            jobs.append(j("torch", who, d=d, sdtype="fp32", pdtype="fp32"))
    return jobs


def sdpa_abstraction():
    """TileLang only. Two axes kept apart on purpose: S3-H/M/MP/L is the
    within-kernel abstraction axis, S1/S2/S3 is the algorithmic decomposition
    axis. The report must not compare S1 against S3-H and call it abstraction."""
    jobs = []
    for d in common2.S_HEAD_DIMS:
        for arm in ("S3-H", "S3-M", "S3-MP", "S3-L", "S1", "S2"):
            jobs.append(j("tilelang_abs", arm, d=d, sdtype="fp32", pdtype="fp16"))
    return jobs


SETS = {
    "fused_matched": fused_matched,
    "fused_abstraction": fused_abstraction,
    "sdpa_cross": sdpa_cross,
    "sdpa_abstraction": sdpa_abstraction,
    "fused_native": fused_native,
    "fused_epilogue": fused_epilogue,
    "fused_cast": fused_cast,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "jobs"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for name, fn in SETS.items():
        jobs = fn()
        p = os.path.join(a.outdir, name + ".json")
        with open(p, "w") as f:
            json.dump(jobs, f, indent=1)
        print(f"{name:<16s} {len(jobs):3d} jobs -> {p}")


if __name__ == "__main__":
    main()
