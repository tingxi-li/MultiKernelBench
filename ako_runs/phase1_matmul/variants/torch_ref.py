"""The benchmark's own reference, wrapped in the variant API.

Not a DSL under test -- this is the denominator. Two forms:
  variant 'A'  : torch.matmul on fp32 (exactly what reference/matmul/
                 standard_matrix_multiplication.py::Model.forward does)
  variant 'B'  : torch.matmul on fp16 operands with fp32 output, i.e. what
                 cuBLAS does when handed the same precision latitude the fp16
                 kernels take. Useful as a "vendor library at fp16" ceiling.
"""
from __future__ import annotations

import time

import torch

import common


def build(cfg: common.Config) -> common.Built:
    t0 = time.perf_counter()

    if cfg.arith == "fp32":
        def run(A, B):
            return torch.matmul(A, B)
        dtype = torch.float32
    else:
        if cfg.cast == "precast":
            def run(A, B):
                return torch.matmul(A, B).float()
            dtype = torch.float16
        else:
            def run(A, B):
                return torch.matmul(A.half(), B.half()).float()
            dtype = torch.float32

    # force cuBLAS handle/heuristic setup out of the timed region
    a = torch.zeros(16, 16, device="cuda", dtype=dtype)
    run(a, a)
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0

    return common.Built(run=run, compile_s=compile_s, input_dtype=dtype,
                        artifacts={"library": "cuBLAS via torch.matmul"},
                        notes="reference denominator, not a DSL under test")
