"""The denominators for the fused ladder.

Two of them, deliberately, because Phase 1 showed that the single published
number divides a fp16 kernel by a fp32 reference and so reports an arithmetic
change as a code-generation win:

  arith=fp32   what the benchmark actually compares against: cuBLAS fp32
               (TF32 disabled, per the Phase-1 environment) + F.gelu + softmax.
  arith=fp16   the precision-matched denominator: the same torch ops with fp16
               operands and fp32 accumulate, i.e. what the DSL kernels do.

Both climb the same ladder, so `torch @ G` is a pure cuBLAS number and the
increments are torch's own fusion costs -- which is the only fair thing to
compare the DSLs' increments against.

The weight factor applies here too. `cached` hands torch a prebuilt fp16 (K, N)
weight; `uncached` rebuilds it per call; `native` lets torch consume the stored
(N, K) layout directly (`x @ W.t()` with no `.contiguous()`), which is what
`nn.Linear` itself does.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

import common2

_INV_SQRT2 = 0.70710678118654752440


def build(cfg) -> common2.Built2:
    arm = common2.FUSED_ARMS[cfg.variant]
    wmode = cfg.extra.get("wcache", "cached")
    fp16 = cfg.arith == "fp16"
    t0 = time.perf_counter()

    if wmode == "cached":
        box = {}

        def wprep(W):
            # address-keyed, so the build-time warm-up's dummy cannot become
            # the weight for the rest of the campaign. `!=` not `is not`: see
            # common2.weight_fn.
            if box.get("src") != W.data_ptr():
                box["src"] = W.data_ptr()
                box["w"] = (W.half() if fp16 else W).t().contiguous()
            return box["w"]
    elif wmode == "uncached":
        def wprep(W):
            return (W.half() if fp16 else W).t().contiguous()
    else:  # native: no materialization, `nn.Linear`'s own access pattern
        def wprep(W):
            return (W.half() if fp16 else W).t()

    def run(x, W, b):
        xx = x.half() if fp16 else x
        y = xx @ wprep(W)
        if arm["bias"]:
            y = y + (b.half() if fp16 else b)
        if arm["gelu"]:
            y = F.gelu(y, approximate="none")
        if arm["softmax"]:
            y = torch.softmax(y, dim=1)
        return y.float()

    xw = torch.zeros((cfg.M, cfg.K), dtype=torch.float32, device="cuda")
    Ww = torch.zeros((cfg.N, cfg.K), dtype=torch.float32, device="cuda")
    bw = torch.zeros((cfg.N,), dtype=torch.float32, device="cuda")
    _ = run(xw, Ww, bw)
    torch.cuda.synchronize()
    del xw, Ww, bw, _
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    return common2.Built2(
        run=run, compile_s=compile_s,
        artifacts={"backend_detail": f"torch {torch.__version__} "
                                     f"arith={cfg.arith} wcache={wmode}",
                   "wcache": wmode, "n_kernels": None},
        notes=f"torch {cfg.arith} fused arm {cfg.variant} wcache={wmode}",
        # torch always receives the fp32 activation and does its own cast, so
        # the fp16 lane pays for `x.half()` inside the timer. That is what a
        # torch user actually writes, and it keeps the two torch denominators
        # differing in exactly one thing: arithmetic.
        x_dtype=torch.float32)
