"""Phase-2 variant registry.

A fused module exposes `build(cfg) -> common2.Built2` where `Built2.run(x, W, b)`
takes x fp32 (M, K), W fp32 (N, K) exactly as `nn.Linear` stores it, and b fp32
(N,), and returns fp32 (M, N). Every host-side weight preparation the variant
needs lives inside `run`, so the weight-cache factor is timed identically in
every lane.

An SDPA module exposes `build(cfg) -> common2.Built2` where `run(q, k, v)` takes
fp32 (B, H, S, D) and returns fp32 (B, H, S, D).
"""
from __future__ import annotations

import importlib

FUSED = {
    "torch": "variants2.fused_torch",
    "tilelang": "variants2.fused_tilelang",
    "triton": "variants2.fused_triton",
    "cuda_noptx": "variants2.fused_cuda_noptx",
    "cuda_unlimited": "variants2.fused_cuda_unlimited",
    # TileLang-only softmax-abstraction study (F1..F4), reported separately.
    "tilelang_abs": "variants2.fused_tilelang_abstraction",
}

SDPA = {
    "torch": "variants2.sdpa_torch",
    "tilelang": "variants2.sdpa_tilelang",
    "triton": "variants2.sdpa_triton",
    "cuda_noptx": "variants2.sdpa_cuda_noptx",
    "cuda_unlimited": "variants2.sdpa_cuda_unlimited",
    "tilelang_abs": "variants2.sdpa_tilelang_abstraction",
}


def get_module(op: str, dsl: str):
    table = {"fused": FUSED, "sdpa": SDPA}[op]
    if dsl not in table:
        raise KeyError(f"unknown dsl {dsl!r} for op {op!r}; known: {sorted(table)}")
    return importlib.import_module(table[dsl])


def build(op: str, cfg):
    return get_module(op, cfg.dsl).build(cfg)
