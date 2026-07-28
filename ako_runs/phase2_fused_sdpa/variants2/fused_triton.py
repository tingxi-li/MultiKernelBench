"""Triton lane of the Phase-2 fused ladder.

The GEMM body is Phase 1's `_gemm_kernel` with the ladder's epilogue bolted on:
same plain 2-D grid (no swizzle, no group-M), same `KC` chunk accumulator, same
`num_stages`/`num_warps` taken from the Config rather than from an autotuner.
No `@triton.autotune` anywhere -- the shipped incumbent for this cell carries 13
autotune configs, which is a search the published runtime does not account for
and which would make "matched configuration" meaningless.

Ladder:
  G     tl.store(acc)
  GB    acc += bias[None, :]
  GBG   exact erf GELU on the accumulator: v*0.5*(1+erf(v/sqrt2)).
        `tl.math.erf` is used, NOT the tanh approximation -- the reference is
        F.gelu(approximate='none') and the tanh form is a different function.
  GBGS  + a separate row-softmax kernel

The softmax kernel is held algorithmically identical to the TileLang lane's:
one row per program, 256 threads' worth of work, exponentials computed once and
kept in registers. In Triton that falls out of a single `BLOCK_N == N` tile, so
the reduction is `tl.max`/`tl.sum` over one tile rather than a hand-written
shuffle -- Triton does not expose warp shuffles, and pretending otherwise by
writing a fake tree would measure the fake, not the language.
"""
from __future__ import annotations

import os
import time

import torch
import triton
import triton.language as tl

import common2

_INV_SQRT2 = 0.70710678118654752440
# Triton refuses to close over a plain Python global from inside @jit; it has to
# be a tl.constexpr object.
_TL_INV_SQRT2 = tl.constexpr(0.70710678118654752440)


@triton.jit
def _fused_gemm_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    KC: tl.constexpr,
    TO_FP16: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_GELU: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if KC == 0:
        for _ in range(0, K // BK):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            if TO_FP16:
                b = b.to(tl.float16)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK * stride_ak
            b_ptrs += BK * stride_bk
    else:
        for _ in range(0, K // KC):
            chunk = tl.zeros((BM, BN), dtype=tl.float32)
            for _ in range(0, KC // BK):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                if TO_FP16:
                    b = b.to(tl.float16)
                chunk = tl.dot(a, b, chunk)
                a_ptrs += BK * stride_ak
                b_ptrs += BK * stride_bk
            acc += chunk

    if HAS_BIAS:
        acc += tl.load(Bias + offs_n)[None, :]
    if HAS_GELU:
        acc = acc * 0.5 * (1.0 + tl.math.erf(acc * _TL_INV_SQRT2))

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)


@triton.jit
def _softmax_kernel(X, Y, M, N, stride_m, BN: tl.constexpr):
    """One row per program. BN == N, so the row is a single tile and the
    exponentials stay in registers between the sum and the divide -- the same
    'compute exp once' property the TileLang F4 kernel gets from its `lexp`
    local buffer."""
    row = tl.program_id(0)
    offs = tl.arange(0, BN)
    x = tl.load(X + row * stride_m + offs).to(tl.float32)
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(Y + row * stride_m + offs, e / tl.sum(e, axis=0))


def build(cfg) -> common2.Built2:
    M, N, K = cfg.M, cfg.N, cfg.K
    BM, BN, BK = cfg.BM, cfg.BN, cfg.BK
    arm = common2.FUSED_ARMS[cfg.variant]
    wmode = cfg.extra.get("wcache", "cached")
    wspec = common2.weight_kernel_spec(wmode)
    num_warps = cfg.threads // 32
    num_stages = cfg.stages
    xcast = cfg.cast == "in_region"

    t0 = time.perf_counter()
    grid = (N // BN, M // BM)
    soft_grid = (M,)
    # Triton needs a power-of-two tile; N == 8192 already is one.
    assert N & (N - 1) == 0, "softmax tile assumes N is a power of two"
    soft_warps = max(4, min(32, common2.SOFT_THREADS // 32))

    wf = common2.weight_fn(wmode)
    xf = (lambda x: x.half()) if xcast else (lambda x: x)

    def run(x, W, b):
        xx = xf(x)
        Bt = wf(W)
        C = torch.empty((M, N), dtype=torch.float32, device=xx.device)
        _fused_gemm_kernel[grid](
            xx, Bt, b, C, M, N, K,
            xx.stride(0), xx.stride(1),
            # native mode hands us W as (N, K); the dot still wants (K, N), which
            # is exactly a stride swap -- no transpose kernel, no materialization.
            *((Bt.stride(1), Bt.stride(0)) if wspec["transpose_b"]
              else (Bt.stride(0), Bt.stride(1))),
            C.stride(0), C.stride(1),
            BM=BM, BN=BN, BK=BK, KC=cfg.kc,
            TO_FP16=wspec["b_dtype"] == "float32",
            HAS_BIAS=arm["bias"], HAS_GELU=arm["gelu"],
            num_warps=num_warps, num_stages=num_stages)
        if not arm["softmax"]:
            return C
        Y = torch.empty_like(C)
        _softmax_kernel[soft_grid](C, Y, M, N, C.stride(0), BN=N,
                                   num_warps=soft_warps, num_stages=1)
        return Y

    # Warm the JIT at the real shape, bypassing `run` so the cached-weight box
    # is not primed with a dummy.
    x_dtype = torch.float32 if xcast else torch.float16
    xw = torch.zeros((M, K), dtype=torch.float16, device="cuda")
    Bw = torch.zeros((N, K) if wspec["transpose_b"] else (K, N),
                     dtype=torch.float16 if wspec["b_dtype"] == "float16"
                     else torch.float32, device="cuda")
    bw = torch.zeros((N,), dtype=torch.float32, device="cuda")
    Cw = torch.empty((M, N), dtype=torch.float32, device="cuda")
    _fused_gemm_kernel[grid](
        xw, Bw, bw, Cw, M, N, K, xw.stride(0), xw.stride(1),
        *((Bw.stride(1), Bw.stride(0)) if wspec["transpose_b"]
          else (Bw.stride(0), Bw.stride(1))),
        Cw.stride(0), Cw.stride(1),
        BM=BM, BN=BN, BK=BK, KC=cfg.kc,
        TO_FP16=wspec["b_dtype"] == "float32",
        HAS_BIAS=arm["bias"], HAS_GELU=arm["gelu"],
        num_warps=num_warps, num_stages=num_stages)
    if arm["softmax"]:
        Yw = torch.empty_like(Cw)
        _softmax_kernel[soft_grid](Cw, Yw, M, N, Cw.stride(0), BN=N,
                                   num_warps=soft_warps, num_stages=1)
        del Yw
    torch.cuda.synchronize()
    del xw, Bw, bw, Cw
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    artifacts = {
        "grid": list(grid) + [1], "block": [cfg.threads, 1, 1],
        "triton_version": triton.__version__,
        "num_warps": num_warps, "num_stages": num_stages,
        "wcache": wmode, "b_global_dtype": wspec["b_dtype"],
        "transpose_b": wspec["transpose_b"],
        "n_kernels": 2 if arm["softmax"] else 1,
        "backend_detail": (
            f"tl.dot on fp16 operands -> mma.sync.m16n8k16.f32.f16.f16.f32; "
            f"KC={cfg.kc}; num_stages={num_stages} num_warps={num_warps}; "
            f"epilogue bias={arm['bias']} gelu={arm['gelu']} (tl.math.erf, exact "
            f"form); NO autotune"
            + ("; + row-softmax kernel, one row per program, BN=N so exp is "
               "computed once in registers" if arm["softmax"] else "")),
    }
    notes = (f"triton {triton.__version__} fused arm {cfg.variant} "
             f"({arm['label']}) wcache={wmode}: {BM}x{BN}x{BK} kc={cfg.kc} "
             f"stages={num_stages} warps={num_warps}")
    return common2.Built2(run=run, compile_s=compile_s, artifacts=artifacts,
                          notes=notes, n_kernels=artifacts["n_kernels"],
                          x_dtype=x_dtype)
