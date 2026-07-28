"""TileLang-only abstraction study — standard matmul (DSL key `tilelang_abs`).

Reported separately from the cross-DSL transfer study, because abstraction
level, algorithm and hardware instruction path are otherwise confounded.  Here
*only* the abstraction level at which the inner-K loop is expressed changes;
everything else in variants/ABSTRACTION_SPEC.md is held fixed:

    fp16 tensor-core operands (S1 excepted), fp32 output, KC=2048 chunk flush,
    BM=128 BN=128 BK=32, 256 threads, pre-cast fp16 inputs, plain 2-D grid,
    no swizzle, no cross-block split-K, no autotuning.

`cfg.stages` and `cfg.arith` come from `common.ABSTRACTION_SPECS`; nothing in
this file hardcodes them.

The five arms
-------------

H1  TL-H.  `T.Pipelined(KC//BK, num_stages=cfg.stages)` (=3) wrapping
    `T.copy` + `T.copy` + `T.gemm`.  The compiler multi-buffers the shared
    tiles, picks the cp.async groups and inserts the waits/barriers.

H2  TL-H.  *Byte-identical kernel source to H1* -- literally the same builder
    function -- with `num_stages=cfg.stages` == 1, i.e. the software pipeline
    off.  H1-vs-H2 therefore isolates the pipeline and nothing else.

M1  TL-M.  Plain `T.serial` K loop, one shared buffer per operand, explicit
    `T.copy` into shared, explicit `T.sync_threads()` on both sides of the
    `T.gemm` (fill->use and use->overwrite).  No pipelining construct anywhere.

M2  TL-M.  A hand-written software pipeline: two shared buffers per operand,
    `T.async_copy` (raw `cp.async`, no auto-wait) + `T.ptx_commit_group()` to
    *issue* the load of tile i+1, `T.ptx_wait_group(0)` + `T.sync_threads()` to
    *consume* tile i, and `T.gemm` on the other buffer.  The K loop is unrolled
    by two so the buffer parity is a compile-time constant.  Per K tile this is
    one barrier and one cp.async group, with the copy of tile i+1 in flight
    across the gemm of tile i -- the classic 2-stage double buffer, written out
    by hand.  It is NOT `T.Pipelined` with a different flag: `T.Pipelined` never
    appears in the M2 kernel, and the ordering/commit/wait structure below is
    the schedule, not a hint to a scheduler.

S1  TL-SIMT *hardware control, not an abstraction measurement*.  Exactly the M1
    loop structure (serial K loop, explicit copies, explicit barriers, same
    KC=2048 fp32 chunk flush) with `T.gemm` replaced by a
    `T.Parallel(BM,BN) x T.serial(BK)` scalar FMA chain on fp32 shared tiles.
    M1 - S1 is therefore the tensor-core contribution: the only difference
    between the two kernels is the arithmetic instruction.

Deliberately excluded: `T.wgmma_gemm` / `T.tcgen05_gemm` (Hopper/Blackwell
manual async interfaces; this host is Ada sm_89).
"""
#   NOTE: no `from __future__ import annotations` -- TileLang's eager builder
#   resolves the `T.Tensor((M, K), dtype)` parameter annotations with
#   typing.get_type_hints() against *module* globals, so stringised annotations
#   would fail to see the closure variables M/N/K/BM/BN.
import os
import re
import time

import torch

import tilelang
import tilelang.language as T

import common

# TileLang keeps a disk cache of compiled kernels.  With it on, compile_s would
# be "cold" for whichever process ran first and "warm" for the rest -- and the
# abstraction study's second conclusion (search cost) is *about* compile time.
# Set PHASE1_TL_CACHE=1 to restore the default behaviour.
_CACHE_ENABLED = os.environ.get("PHASE1_TL_CACHE", "0") == "1"
if not _CACHE_ENABLED:
    tilelang.disable_cache()

# Dump PTX/SASS next to the study's artifacts (slow: invokes cuobjdump).
_DUMP_ASM = os.environ.get("PHASE1_TL_ASM", "0") == "1"


# --------------------------------------------------------------- TL-H (H1/H2) ---
def _kernel_H(M, N, K, BM, BN, BK, threads, stages, kc):
    """H1 and H2 -- same source, `stages` is the only thing that differs.

    Everything about the inner K loop is delegated: T.Pipelined decides the
    number of shared buffers, whether to use cp.async, where the commit/wait
    groups go and where the barriers go.
    """
    NC = K // kc
    KI = kc // BK

    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        def main(A: T.Tensor((M, K), "float16"),
                 B: T.Tensor((K, N), "float16"),
                 C: T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
                As = T.alloc_shared((BM, BK), "float16")
                Bs = T.alloc_shared((BK, BN), "float16")
                Cchunk = T.alloc_fragment((BM, BN), "float32")
                Cacc = T.alloc_fragment((BM, BN), "float32")
                T.clear(Cacc)
                for c in T.serial(NC):
                    T.clear(Cchunk)
                    for ko in T.Pipelined(KI, num_stages=stages):
                        T.copy(A[by * BM, c * kc + ko * BK], As)
                        T.copy(B[c * kc + ko * BK, bx * BN], Bs)
                        T.gemm(As, Bs, Cchunk)
                    for i, j in T.Parallel(BM, BN):
                        Cacc[i, j] += Cchunk[i, j]
                T.copy(Cacc, C[by * BM, bx * BN])
        return main

    return _k()


# ------------------------------------------------------------------ TL-M (M1) ---
def _kernel_M1(M, N, K, BM, BN, BK, threads, kc):
    """M1 -- regular K loop, one shared buffer, explicit copies and barriers.

    Two barriers per K tile, by construction:
      fill -> gemm   (the tile must be visible to every warp before the mma)
      gemm -> refill (every warp must be done with the tile before it is
                      overwritten by the next iteration's copy)
    That is the price of a single buffer, and it is exactly what M2 removes.
    """
    NC = K // kc
    KI = kc // BK

    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        def main(A: T.Tensor((M, K), "float16"),
                 B: T.Tensor((K, N), "float16"),
                 C: T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
                As = T.alloc_shared((BM, BK), "float16")
                Bs = T.alloc_shared((BK, BN), "float16")
                Cchunk = T.alloc_fragment((BM, BN), "float32")
                Cacc = T.alloc_fragment((BM, BN), "float32")
                T.clear(Cacc)
                for c in T.serial(NC):
                    T.clear(Cchunk)
                    for ko in T.serial(KI):
                        T.copy(A[by * BM, c * kc + ko * BK], As)
                        T.copy(B[c * kc + ko * BK, bx * BN], Bs)
                        T.sync_threads()
                        T.gemm(As, Bs, Cchunk)
                        T.sync_threads()
                    for i, j in T.Parallel(BM, BN):
                        Cacc[i, j] += Cchunk[i, j]
                T.copy(Cacc, C[by * BM, bx * BN])
        return main

    return _k()


# ------------------------------------------------------------------ TL-M (M2) ---
def _kernel_M2(M, N, K, BM, BN, BK, threads, kc):
    """M2 -- hand-written double-buffered software pipeline.

    Two shared buffers per operand.  The K loop is unrolled by two so buffer
    parity is static; the steady-state body of each half-iteration is

        wait_group(0)          # the cp.async group issued one half-iteration
                               # ago (tile g) has landed
        sync_threads()         # (a) tile g visible to every warp
                               # (b) every warp is past the gemm on the *other*
                               #     buffer, so it is safe to overwrite it
        async_copy(tile g+1 -> other buffer); commit_group()
        gemm(this buffer)      # runs with the copy of tile g+1 in flight

    One barrier and one cp.async group per K tile (M1 needs two barriers and has
    nothing in flight during the mma).  The prologue issues tile 0; the epilogue
    prefetch is clamped to the last tile so the tail never reads out of bounds
    (one redundant 8 KB tile load for the whole kernel).

    The K tile index is global (0..K/BK-1), so a buffer prefetched at the end of
    chunk c is consumed at the start of chunk c+1 -- the fp32 chunk flush is
    register-only and does not disturb the shared-memory pipeline.  KI is even
    (2048/32 = 64), so the parity is consistent across chunk boundaries.
    """
    NC = K // kc
    KI = kc // BK          # K tiles per chunk (even)
    assert KI % 2 == 0, f"M2 unrolls the K loop by 2, so kc/BK={KI} must be even"

    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        def main(A: T.Tensor((M, K), "float16"),
                 B: T.Tensor((K, N), "float16"),
                 C: T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
                # two explicit shared stages per operand
                As0 = T.alloc_shared((BM, BK), "float16")
                Bs0 = T.alloc_shared((BK, BN), "float16")
                As1 = T.alloc_shared((BM, BK), "float16")
                Bs1 = T.alloc_shared((BK, BN), "float16")
                Cchunk = T.alloc_fragment((BM, BN), "float32")
                Cacc = T.alloc_fragment((BM, BN), "float32")
                T.clear(Cacc)

                # ---- prologue: issue tile 0 into stage 0, do not wait ----
                T.async_copy(A[by * BM, 0], As0)
                T.async_copy(B[0, bx * BN], Bs0)
                T.ptx_commit_group()

                for c in T.serial(NC):
                    T.clear(Cchunk)
                    for kk in T.serial(KI // 2):
                        g = c * KI + 2 * kk          # global K-tile index

                        # ---- consume stage 0, prefetch tile g+1 into stage 1 ----
                        T.ptx_wait_group(0)
                        T.sync_threads()
                        k1 = T.min((g + 1) * BK, K - BK)
                        T.async_copy(A[by * BM, k1], As1)
                        T.async_copy(B[k1, bx * BN], Bs1)
                        T.ptx_commit_group()
                        T.gemm(As0, Bs0, Cchunk)

                        # ---- consume stage 1, prefetch tile g+2 into stage 0 ----
                        T.ptx_wait_group(0)
                        T.sync_threads()
                        k2 = T.min((g + 2) * BK, K - BK)
                        T.async_copy(A[by * BM, k2], As0)
                        T.async_copy(B[k2, bx * BN], Bs0)
                        T.ptx_commit_group()
                        T.gemm(As1, Bs1, Cchunk)

                    for i, j in T.Parallel(BM, BN):
                        Cacc[i, j] += Cchunk[i, j]
                T.copy(Cacc, C[by * BM, bx * BN])
        return main

    return _k()


# --------------------------------------------------------------- TL-SIMT (S1) ---
def _kernel_S1(M, N, K, BM, BN, BK, threads, kc):
    """S1 -- M1's loop structure with scalar FMA instead of T.gemm.

    `T.gemm` is deliberately NOT used: on sm_89 it lowers fp32 operands to
    `mma.sync ... tf32`, which would make the "no tensor core" control a
    tensor-core kernel.  The inner product is a `T.Parallel(BM,BN)` register
    tile with a `T.serial(BK)` FMA chain, which lowers to plain FFMA.
    """
    NC = K // kc
    KI = kc // BK

    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        def main(A: T.Tensor((M, K), "float32"),
                 B: T.Tensor((K, N), "float32"),
                 C: T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
                As = T.alloc_shared((BM, BK), "float32")
                Bs = T.alloc_shared((BK, BN), "float32")
                Cchunk = T.alloc_fragment((BM, BN), "float32")
                Cacc = T.alloc_fragment((BM, BN), "float32")
                T.clear(Cacc)
                for c in T.serial(NC):
                    T.clear(Cchunk)
                    for ko in T.serial(KI):
                        T.copy(A[by * BM, c * kc + ko * BK], As)
                        T.copy(B[c * kc + ko * BK, bx * BN], Bs)
                        T.sync_threads()
                        for i, j in T.Parallel(BM, BN):
                            for kk in T.serial(BK):
                                Cchunk[i, j] += As[i, kk] * Bs[kk, j]
                        T.sync_threads()
                    for i, j in T.Parallel(BM, BN):
                        Cacc[i, j] += Cchunk[i, j]
                T.copy(Cacc, C[by * BM, bx * BN])
        return main

    return _k()


# ---------------------------------------------------------------- artifacts ---
_SHMEM_RE = re.compile(r"\[(\d+)\]\.v_int64\)\s*=\s*\(\(int64_t\)(\d+)\)")


def _shared_bytes_from_host(host_src: str):
    """The dynamic-shared-memory argument of the packed kernel launch.

    TileLang's host stub pushes (ptrs..., grid, block, dyn_smem) onto an FFI
    stack; the dynamic shared size is the last integer pushed.
    """
    best = None
    for m in _SHMEM_RE.finditer(host_src):
        idx, val = int(m.group(1)), int(m.group(2))
        if best is None or idx > best[0]:
            best = (idx, val)
    return best[1] if best else None


def _sass_mma_counts(path: str) -> dict:
    try:
        with open(path, "r", errors="ignore") as f:
            s = f.read()
    except OSError:
        return {}
    return {
        "sass_hmma": len(re.findall(r"\bHMMA\b", s)),
        "sass_imma": len(re.findall(r"\bIMMA\b", s)),
        "sass_ffma": len(re.findall(r"\bFFMA\b", s)),
        "sass_ldgsts": len(re.findall(r"\bLDGSTS\b", s)),
        "sass_bar_sync": len(re.findall(r"BAR\.SYNC", s)),
        "sass_lines": s.count("\n"),
    }


_BACKEND_DETAIL = {
    "H1": ("TL-H: T.Pipelined(kc/BK={KI}, num_stages={stages}) wrapping "
           "T.copy(A)->As, T.copy(B)->Bs, T.gemm(As,Bs,Cchunk). The compiler "
           "owns the schedule: it multi-buffers the shared tiles {stages} deep, "
           "chooses cp.async, and places every commit_group/wait_group/barrier. "
           "Chunk flush Cchunk->Cacc every kc={kc} elements of K, in fp32 "
           "registers. Source is byte-identical to H2."),
    "H2": ("TL-H: identical kernel source to H1 (same builder function), only "
           "num_stages={stages}. One shared buffer, synchronous copy + wait "
           "before each T.gemm -- the software pipeline is off. Chunk flush "
           "Cchunk->Cacc every kc={kc} elements of K."),
    "M1": ("TL-M: plain T.serial(kc/BK={KI}) K loop, ONE shared buffer per "
           "operand, explicit T.copy into shared, explicit T.sync_threads() on "
           "both sides of T.gemm (fill->use and use->overwrite). No T.Pipelined "
           "and no cp.async: 2 barriers per K tile, nothing in flight across "
           "the mma. Chunk flush Cchunk->Cacc every kc={kc}."),
    "M2": ("TL-M: hand-written double buffer. TWO shared stages per operand "
           "(As0/Bs0, As1/Bs1); K loop unrolled by 2 so parity is static. Per "
           "half-iteration: T.ptx_wait_group(0) + T.sync_threads() to consume "
           "the tile that landed, then T.async_copy(tile g+1)->other stage + "
           "T.ptx_commit_group() to ISSUE the next load, then T.gemm on the "
           "current stage while that cp.async is in flight. 1 barrier and 1 "
           "cp.async group per K tile. T.Pipelined does not appear; the "
           "prologue issues tile 0 and the tail prefetch is clamped to the last "
           "tile. Chunk flush Cchunk->Cacc every kc={kc}."),
    "S1": ("TL-SIMT hardware control (NOT an abstraction measurement): M1's "
           "exact loop structure -- T.serial(kc/BK={KI}), one shared buffer, "
           "explicit T.copy, explicit T.sync_threads() both sides -- with "
           "T.gemm replaced by a T.Parallel(BM,BN) x T.serial(BK) scalar FMA "
           "register tile on fp32 shared tiles. No tensor cores: T.gemm is "
           "avoided because it would emit tf32 mma.sync on sm_89. Same fp32 "
           "chunk flush every kc={kc}."),
}


# -------------------------------------------------------------------- build ---
def build(cfg: common.Config) -> common.Built:
    M, N, K = cfg.M, cfg.N, cfg.K
    BM, BN, BK, threads = cfg.BM, cfg.BN, cfg.BK, cfg.threads
    kc, stages = cfg.kc, cfg.stages
    v = cfg.variant

    if v not in common.ABSTRACTION_SPECS:
        raise ValueError(f"tilelang_abs takes variants {sorted(common.ABSTRACTION_SPECS)}, got {v!r}")
    spec = common.ABSTRACTION_SPECS[v]
    # cfg.stages / cfg.arith are pre-set from ABSTRACTION_SPECS; refuse to run
    # if the driver has drifted, because any drift voids the study.
    assert cfg.arith == spec["arith"], f"{v}: arith drifted {cfg.arith!r} != {spec['arith']!r}"
    # The spec pins each arm's pipeline depth and any accidental drift voids the
    # study, so the default is to refuse. But H1 is 3-stage and M2 is 2-stage,
    # which means the headline H1-vs-M2 comparison confounds "compiler-generated
    # vs hand-written pipeline" with depth. Resolving that requires deliberately
    # running one arm off its spec depth, so there is an explicit opt-in --
    # `--set depth_control=1` -- rather than a silently permissive check.
    #
    # Note this only makes H1 movable. M2's double buffer is written by hand with
    # a static parity unroll (see _kernel_M2), so "M2 at 3 stages" is a rewrite,
    # not a parameter. That asymmetry is itself part of the abstraction finding.
    depth_control = str(cfg.extra.get("depth_control", "")).lower() in ("1", "true", "yes")
    if cfg.stages != spec["stages"]:
        if not depth_control:
            raise AssertionError(
                f"{v}: stages drifted {cfg.stages} != {spec['stages']} "
                f"(pass depth_control=1 if this is the deliberate depth control)")
        if v not in ("H1", "H2"):
            raise AssertionError(
                f"{v}: depth is not a free parameter for this arm -- its pipeline "
                f"is hand-written at depth {spec['stages']}; changing it is a "
                f"rewrite, not a config change")
    assert kc > 0 and K % kc == 0 and kc % BK == 0, f"{v}: kc={kc} must be a positive multiple of BK={BK} dividing K={K}"
    assert K % BK == 0, f"BK={BK} must divide K={K}"
    if cfg.cast != "precast":
        raise ValueError("the abstraction study holds cast=precast fixed; got " + repr(cfg.cast))

    t0 = time.perf_counter()

    if v in ("H1", "H2"):
        kern = _kernel_H(M, N, K, BM, BN, BK, threads, stages, kc)
    elif v == "M1":
        kern = _kernel_M1(M, N, K, BM, BN, BK, threads, kc)
    elif v == "M2":
        kern = _kernel_M2(M, N, K, BM, BN, BK, threads, kc)
    else:  # S1
        kern = _kernel_S1(M, N, K, BM, BN, BK, threads, kc)

    def run(A, B):
        return kern(A, B)

    # Force the lazy CUDA module load / first launch out of the timed region.
    # The JIT is shape-specialised, so the warm launch must be at the real shape.
    warm_dtype = torch.float32 if cfg.arith == "fp32" else torch.float16
    wa = torch.zeros((M, K), dtype=warm_dtype, device="cuda")
    wb = torch.zeros((K, N), dtype=warm_dtype, device="cuda")
    _ = run(wa, wb)
    torch.cuda.synchronize()
    del wa, wb, _
    torch.cuda.empty_cache()

    compile_s = time.perf_counter() - t0

    # ---- artifacts (measured after the compile clock stops) ----
    grid = (N // BN + (1 if N % BN else 0), M // BM + (1 if M % BM else 0), 1)
    block = (threads, 1, 1)
    how = _BACKEND_DETAIL[v].format(KI=kc // BK, stages=stages, kc=kc)

    artifacts = {
        "grid": list(grid),
        "block": list(block),
        "backend_detail": how,
        "abstraction_level": spec["level"],
        "abstraction_label": spec["label"],
        "tilelang_version": tilelang.__version__,
        "tilelang_disk_cache": _CACHE_ENABLED,
        "global_dtype": "float32" if cfg.arith == "fp32" else "float16",
        "smem_dtype": "float32" if cfg.arith == "fp32" else "float16",
        "smem_stages": 2 if v == "M2" else (stages if v == "H1" else 1),
        "uses_T_Pipelined": v in ("H1", "H2"),
        "uses_T_gemm": v != "S1",
    }
    try:
        artifacts["cuda_source"] = kern.get_kernel_source()
    except Exception as e:  # noqa: BLE001
        artifacts["cuda_source_error"] = repr(e)
    try:
        artifacts["shared_bytes"] = _shared_bytes_from_host(kern.get_host_source())
    except Exception as e:  # noqa: BLE001
        artifacts["shared_bytes_error"] = repr(e)
    for attr in ("n_regs", "n_spills"):
        val = getattr(kern, attr, None)
        if val is not None:
            artifacts[attr] = val

    if _DUMP_ASM:
        d = os.path.join(common.ARTIFACTS_DIR, "tilelang_abs")
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, cfg.key().replace("/", "_"))
        try:
            kern.export_ptx(base + ".ptx")
            artifacts["ptx_path"] = base + ".ptx"
            with open(base + ".ptx", "r", errors="ignore") as f:
                artifacts["ptx"] = f.read()
        except Exception as e:  # noqa: BLE001
            artifacts["ptx_error"] = repr(e)
        try:
            kern.export_sass(base + ".sass")
            artifacts["sass_path"] = base + ".sass"
            artifacts.update(_sass_mma_counts(base + ".sass"))
        except Exception as e:  # noqa: BLE001
            artifacts["sass_error"] = repr(e)
        with open(base + ".cu", "w") as f:
            f.write(artifacts.get("cuda_source", ""))

    notes = (f"tilelang {tilelang.__version__} abstraction variant {v} "
             f"({spec['level']}: {spec['label']}): {BM}x{BN}x{BK}/{threads}thr "
             f"kc={kc} stages={stages} arith={cfg.arith} cast={cfg.cast}; disk "
             f"cache {'ON' if _CACHE_ENABLED else 'OFF (compile_s is a cold compile)'}")

    return common.Built(run=run, compile_s=compile_s,
                        input_dtype=cfg.input_dtype,
                        artifacts=artifacts, notes=notes)
