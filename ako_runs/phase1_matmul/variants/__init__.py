"""Variant registry.

Each DSL module must expose exactly one entry point:

    def build(cfg: common.Config) -> common.Built

`Built.run(A, B) -> C` where the dtypes of A and B are `cfg.input_dtype` and C
is fp32 of shape (M, N). `run` allocates its own output, the way a real
KernelBench `forward()` does, so allocation cost is inside the timed region for
every DSL equally.

`Built.compile_s` is wall-clock spent turning source into a launchable kernel,
measured by the module and reported separately from execution time.

`Built.artifacts` is free-form; the code-inspection pass looks for the keys
`cuda_source`, `ptx`, `cubin_path`, `n_regs`, `n_spills`, `shared_bytes`.
"""
from __future__ import annotations

import importlib

_MODULES = {
    "torch": "variants.torch_ref",
    "tilelang": "variants.tilelang_gemm",
    "triton": "variants.triton_gemm",
    "cuda_noptx": "variants.cuda_noptx_gemm",
    "cuda_unlimited": "variants.cuda_unlimited_gemm",
    # TileLang-only abstraction study; variants H1/H2/M1/M2/S1, reported
    # separately from the cross-DSL transfer study.
    "tilelang_abs": "variants.tilelang_abstraction",
}


def get_module(dsl: str):
    if dsl not in _MODULES:
        raise KeyError(f"unknown dsl {dsl!r}; known: {sorted(_MODULES)}")
    return importlib.import_module(_MODULES[dsl])


def build(cfg):
    return get_module(cfg.dsl).build(cfg)
