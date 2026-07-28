#!/usr/bin/env python3
"""Minimal launcher for Nsight Compute: build one Phase-2 variant, run it N times.

Kept separate from runner2.py because ncu serializes and replays every kernel it
profiles; the timing protocol (warm-up, L2 flush, trial loop) would multiply the
profiling cost by hundreds and change nothing about the counters.

usage: python profile_target2.py --op sdpa --dsl tilelang --variant FLASH \
           --set x_d=1024,x_sdtype=fp32,x_pdtype=fp16 [--iters 3]
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402
common2.setup_cuda_env()

import torch  # noqa: E402
import variants2  # noqa: E402
from runner2 import parse_set  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=("fused", "sdpa"))
    ap.add_argument("--dsl", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--set", dest="setstr", default="")
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args()

    over = parse_set(a.setstr)
    if a.op == "fused":
        cfg = common2.make_fused_config(a.dsl, a.variant, **over)
        x, W, b = common2.fused_inputs(seed=0)
        built = variants2.build("fused", cfg)
        if built.x_dtype == torch.float16:
            x = x.half().contiguous()
        operands = (x, W, b)
    else:
        cfg = common2.make_sdpa_config(a.dsl, a.variant, **over)
        operands = common2.sdpa_inputs(cfg.extra["d"], seed=0)
        built = variants2.build("sdpa", cfg)

    # One untimed call first so lazy module loads and any host-side cache are
    # not what ncu ends up profiling.
    with torch.no_grad():
        built.run(*operands)
        torch.cuda.synchronize()
        for _ in range(a.iters):
            built.run(*operands)
        torch.cuda.synchronize()
    print(f"OK {cfg.key()}")


if __name__ == "__main__":
    main()
