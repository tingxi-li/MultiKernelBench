#!/usr/bin/env python3
"""Minimal ncu profiling target: build one variant, launch it N times, exit.

Kept separate from runner.py so the profiled process contains nothing but the
kernel under test -- no reference matmul, no error check, no L2-thrash kernel
polluting the metric set.

usage (under ncu):
  ncu --kernel-name-base function --launch-skip 3 --launch-count 1 ... \
      python profile_target.py --dsl tilelang --variant D
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
common.setup_cuda_env()

import torch  # noqa: E402
import variants  # noqa: E402
from runner import parse_set  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsl", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--geom", default="primary")
    ap.add_argument("--set", dest="setstr", default="")
    ap.add_argument("--dist", default="rand")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    cfg = common.make_config(args.dsl, args.variant, args.geom, **parse_set(args.setstr))
    A32, B32 = common.load_inputs(args.dist, args.seed, device="cuda")
    built = variants.build(cfg)
    if built.input_dtype == torch.float16:
        A, B = A32.half().contiguous(), B32.half().contiguous()
    else:
        A, B = A32, B32
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(args.iters):
            built.run(A, B)
            torch.cuda.synchronize()
    print(f"profiled {cfg.key()} x{args.iters}")


if __name__ == "__main__":
    main()
