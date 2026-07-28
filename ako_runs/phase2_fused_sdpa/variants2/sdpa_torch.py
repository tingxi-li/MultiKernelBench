"""SDPA denominators.

`sdpa_reference_audit.py` established what the benchmark's reference actually
runs at the published shape: at head_dim 1024 FlashAttention is unavailable
(its head-dim cap is 256) but the mem-efficient backend IS selected, so the
often-repeated "the SDPA reference falls back to the naive math backend" is
false as stated. The reference is a real tiled kernel.

Three denominators are carried, and which one a speedup is divided by decides
most of the answer:

  algo=TORCH_F32   F.scaled_dot_product_attention on fp32 operands. This is the
                   published denominator.
  algo=TORCH_F16   the same call on fp16 operands. Precision-matched to every
                   DSL kernel in the study, all of which use fp16 tensor cores.
  algo=TORCH_MATH  the math backend forced, i.e. the fallback the folklore
                   claims is being measured. Carried so the claim can be
                   checked rather than repeated.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

import common2

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    SDPBackend = sdpa_kernel = None

ALGOS = {
    "TORCH_F32": "F.scaled_dot_product_attention, fp32 in (published denominator)",
    "TORCH_F16": "F.scaled_dot_product_attention, fp16 in (precision-matched)",
    "TORCH_MATH": "F.scaled_dot_product_attention with the math backend forced",
}


def build(cfg) -> common2.Built2:
    algo = cfg.variant
    if algo not in ALGOS:
        raise KeyError(f"unknown torch sdpa algo {algo!r}")
    fp16 = algo == "TORCH_F16"
    t0 = time.perf_counter()

    if algo == "TORCH_MATH":
        if sdpa_kernel is None:
            raise RuntimeError("this torch build cannot force an SDPA backend")

        def run(q, k, v):
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(q, k, v)
    elif fp16:
        def run(q, k, v):
            return F.scaled_dot_product_attention(
                q.half(), k.half(), v.half()).float()
    else:
        def run(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

    d = cfg.extra["d"]
    w = torch.zeros((2, 2, common2.S_S, d), dtype=torch.float32, device="cuda")
    _ = run(w, w, w)
    torch.cuda.synchronize()
    del w, _
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    return common2.Built2(
        run=run, compile_s=compile_s, n_kernels=None,
        artifacts={"backend_detail": ALGOS[algo], "algo": algo,
                   "score_dtype": "fp16" if fp16 else "fp32",
                   "prob_dtype": "fp16" if fp16 else "fp32"},
        notes=f"torch sdpa {algo} at d={d}", x_dtype=torch.float32)
