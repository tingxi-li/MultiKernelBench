"""Triton lane of the Phase-1 matched-GEMM study.

One `@triton.jit` kernel body covers all four variants; every axis the study
varies is a `tl.constexpr` so the differences are resolved at compile time and
nothing shows up as runtime branching in the inner loop:

    KC        0 -> single fp32 accumulator over the whole K extent (variants A,B)
              >0 -> outer loop over K/KC, an inner tl.dot chain into a *fresh*
                    chunk accumulator, `acc += chunk` after the inner loop
                    (variants C,D)
    PREC      "ieee" -> tl.dot on fp32 operands lowers to CUDA-core FMA, i.e.
                        NO tensor cores at all (variant A).
              "tf32" -> the default; with fp16 operands it is ignored and the
                        dot lowers to `mma.sync ... .f32.f16.f16.f32` (B,C,D).
    TO_FP16   True -> `cast=on_load`: global pointers are fp32, tiles are
                      converted to fp16 on the way to the dot operand.

The pipeline depth is the launch-time `num_stages=cfg.stages`.  `num_stages=1`
turns Triton's software pipeliner off; that is verifiable in the PTX (zero
`cp.async` at stages=1, non-zero at stages=3) and in the SASS (`LDGSTS`).

No `@triton.autotune`, no group-M / block swizzle (plain 2-D grid
`(cdiv(N,BN), cdiv(M,BM))`), no cross-block split-K, no atomics, no cuBLAS.
"""
from __future__ import annotations

import os
import time

import torch
import triton
import triton.language as tl

import common


# --------------------------------------------------------------- the kernel ---
@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    KC: tl.constexpr,
    PREC: tl.constexpr,
    TO_FP16: tl.constexpr,
):
    # plain 2-D grid, no swizzle: axis0 -> N tiles, axis1 -> M tiles
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    if KC == 0:
        # ---- variants A, B: one accumulator chain across the whole K extent --
        for _ in range(0, K // BK):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            if TO_FP16:
                a = a.to(tl.float16)
                b = b.to(tl.float16)
            acc = tl.dot(a, b, acc, input_precision=PREC)
            a_ptrs += BK * stride_ak
            b_ptrs += BK * stride_bk
    else:
        # ---- variants C, D: chunk accumulator flushed every KC of K ----------
        for _ in range(0, K // KC):
            chunk = tl.zeros((BM, BN), dtype=tl.float32)
            for _ in range(0, KC // BK):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                if TO_FP16:
                    a = a.to(tl.float16)
                    b = b.to(tl.float16)
                chunk = tl.dot(a, b, chunk, input_precision=PREC)
                a_ptrs += BK * stride_ak
                b_ptrs += BK * stride_bk
            acc += chunk

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)


# ------------------------------------------------------------------ helpers ---
def _dump(cfg: common.Config, handle):
    """Write ptx/cubin next to the study's other artifacts, return the paths."""
    d = os.path.join(common.ARTIFACTS_DIR, "triton")
    try:
        os.makedirs(d, exist_ok=True)
        stem = os.path.join(d, cfg.key())
        ptx = handle.asm["ptx"]
        with open(stem + ".ptx", "w") as f:
            f.write(ptx)
        cubin = handle.asm.get("cubin")
        if cubin:
            with open(stem + ".cubin", "wb") as f:
                f.write(cubin)
            return stem + ".ptx", stem + ".cubin", ptx
        return stem + ".ptx", "", ptx
    except Exception:  # artifacts are never allowed to break a measurement
        try:
            return "", "", handle.asm["ptx"]
        except Exception:
            return "", "", ""


def build(cfg: common.Config) -> common.Built:
    t0 = time.perf_counter()

    M, N, K = cfg.M, cfg.N, cfg.K
    BM, BN, BK = cfg.BM, cfg.BN, cfg.BK
    num_warps = cfg.threads // 32
    num_stages = cfg.stages

    if cfg.threads % 32:
        raise ValueError(f"threads={cfg.threads} is not a multiple of 32")
    if M % BM or N % BN or K % BK:
        raise ValueError(f"geometry {BM}x{BN}x{BK} does not divide {M}x{N}x{K}")
    if cfg.kc:
        if K % cfg.kc or cfg.kc % BK:
            raise ValueError(f"kc={cfg.kc} must divide K={K} and be a multiple of BK={BK}")

    # ---- what the variant means, in Triton terms ----------------------------
    if cfg.arith == "fp32":
        # variant A: fp32 operands + input_precision="ieee".  The Triton default
        # is "tf32", which silently uses tensor cores; "ieee" forces the
        # FMA-based dot on the CUDA cores.
        prec = "ieee"
        to_fp16 = False
        ker_dtype = torch.float32
        arith_detail = 'tl.dot(..., input_precision="ieee") on fp32 operands -> CUDA-core FMA'
    else:
        prec = "tf32"          # ignored for fp16 operands; dot -> mma.sync f16
        to_fp16 = (cfg.cast == "on_load")
        ker_dtype = torch.float32 if to_fp16 else torch.float16
        arith_detail = "tl.dot on fp16 operands -> mma.sync.m16n8k16.f32.f16.f16.f32"

    in_region = (cfg.arith == "fp16" and cfg.cast == "in_region")
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))

    launch_kw = dict(BM=BM, BN=BN, BK=BK, KC=cfg.kc, PREC=prec, TO_FP16=to_fp16,
                     num_warps=num_warps, num_stages=num_stages)

    def run(A, B):
        if in_region:
            A = A.half()
            B = B.half()
        C = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.float32)
        return _launch(A, B, C)

    def _launch(A, B, C):
        _gemm_kernel[grid](
            A, B, C,
            A.shape[0], B.shape[1], A.shape[1],
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            **launch_kw)
        return C

    # ---- force the JIT inside the compile window ----------------------------
    # Full-size dummies, so the compiled binary is bit-identical to the one the
    # timed calls use (Triton specializes on the runtime int arguments).
    host_dtype = cfg.input_dtype
    a0 = torch.zeros((M, K), device="cuda", dtype=host_dtype)
    b0 = torch.zeros((K, N), device="cuda", dtype=host_dtype)
    c0 = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    try:
        handle = _gemm_kernel[grid](
            a0.half() if in_region else a0,
            b0.half() if in_region else b0,
            c0, M, N, K,
            a0.stride(0), a0.stride(1), b0.stride(0), b0.stride(1),
            c0.stride(0), c0.stride(1), **launch_kw)
        torch.cuda.synchronize()
    except triton.runtime.errors.OutOfResources as e:
        raise RuntimeError(
            f"triton could not compile variant {cfg.variant} at "
            f"BM={BM} BN={BN} BK={BK} stages={num_stages} arith={cfg.arith} "
            f"cast={cfg.cast}: {e}. Operand tiles need "
            f"{num_stages} * (BM*BK + BK*BN) * {2 if ker_dtype==torch.float16 else 4} B "
            f"of shared memory and sm_89 allows 101376 B/block."
        ) from e
    # the real call path, exactly once, so the first timed call is not a compile
    run(a0, b0)
    torch.cuda.synchronize()
    del a0, b0, c0
    torch.cuda.empty_cache()

    compile_s = time.perf_counter() - t0

    # ------------------------------------------------------------ artifacts ---
    ptx_path, cubin_path, ptx = _dump(cfg, handle)
    md = handle.metadata
    n_mma = ptx.count("mma.sync")
    n_cpasync = ptx.count("cp.async")
    art = {
        "ptx": ptx,
        "ptx_path": ptx_path,
        "cubin_path": cubin_path,
        "n_regs": handle.n_regs,
        "n_spills": handle.n_spills,
        "shared_bytes": getattr(md, "shared", None),
        "grid": list(grid),
        "block": [cfg.threads, 1, 1],
        "num_warps": num_warps,
        "num_stages": num_stages,
        "ptx_mma_sync_count": n_mma,
        "ptx_cp_async_count": n_cpasync,
        "backend_detail": (
            f"triton {triton.__version__}; {arith_detail}; "
            f"KC={cfg.kc} ({'single fp32 accumulator over all K' if cfg.kc == 0 else f'chunk accumulator flushed into acc every {cfg.kc} of K'}); "
            f"num_stages={num_stages} ({'pipeline OFF' if num_stages == 1 else 'software pipeline ON'}), "
            f"ptx cp.async={n_cpasync}, ptx mma.sync={n_mma}; "
            f"cast={cfg.cast}; grid={grid} (plain 2-D, no swizzle); "
            f"num_warps={num_warps}; no autotune, no split-K, no atomics"
        ),
    }

    notes = (
        f"variant {cfg.variant}: {common.VARIANT_SPECS[cfg.variant]['label']}. "
        f"tensor cores: {'NO (ieee fp32 dot, 0 mma.sync in PTX)' if cfg.arith == 'fp32' else f'YES ({n_mma} mma.sync in PTX)'}. "
        f"pipeline: {'off' if num_stages == 1 else f'{num_stages} stages'} "
        f"({n_cpasync} cp.async in PTX). "
        f"regs={handle.n_regs} spills={handle.n_spills} smem={getattr(md, 'shared', None)}B."
    )

    return common.Built(run=run, compile_s=compile_s,
                        input_dtype=cfg.input_dtype,
                        artifacts=art, notes=notes)
