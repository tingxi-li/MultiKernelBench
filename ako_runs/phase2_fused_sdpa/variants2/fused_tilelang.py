"""TileLang lane of the Phase-2 fused ladder.

The GEMM body is Phase 1's matched variant D, character for character: fp16
operands into `T.gemm`, an fp32 `Cchunk` fragment flushed into `Cacc` every
`kc` elements of K, `T.Pipelined(num_stages=stages)` on the global->shared
loads. Only the epilogue changes as the ladder climbs, and the softmax rung
adds a second kernel because a BN=128 block owns 1/64th of an 8192-wide row and
cannot normalize it alone.

  G     T.copy(Cacc, C)                              -- the Phase-1 kernel
  GB    Cacc[i,j] += Bias[...]        then copy
  GBG   v = Cacc+bias; Cacc = v*0.5*(1+erf(v/sqrt2)); then copy
  GBGS  as GBG, then a separate row-softmax kernel over the fp32 scratch

The softmax kernel is the incumbent's: one row per block, 256 threads x 32
elements held in a local buffer so `exp` is evaluated once, warp shuffles into a
3-level shared-memory tree. It is held FIXED across the whole cross-DSL ladder
-- varying it is the separate TileLang abstraction study (F1..F4), not this one.

`wcache=native` swaps the GEMM for a transposed-B, fp32-global variant so the
(N, K) weight is consumed in place. Everything else is identical.
"""
#   No `from __future__ import annotations`: TileLang resolves the T.Tensor
#   annotations with typing.get_type_hints() against module globals, so
#   stringised annotations would not see the closure variables.
import math
import os
import time

import torch

import tilelang
import tilelang.language as T

import common
import common2

_CACHE_ENABLED = os.environ.get("PHASE2_TL_CACHE", "0") == "1"
if not _CACHE_ENABLED:
    tilelang.disable_cache()

_DUMP_ASM = os.environ.get("PHASE2_TL_ASM", "0") == "1"

_INV_SQRT2 = 0.70710678118654752440


# ------------------------------------------------------------------- GEMM ---
def _kernel_gemm(M, N, K, BM, BN, BK, threads, stages, kc,
                 bias, gelu, b_dtype, transpose_b):
    """Phase-1 variant D plus an optional bias/GELU epilogue.

    `transpose_b` selects the `native` weight mode: B arrives as (N, K) and
    `T.gemm` is told to transpose it, so no host-side transpose exists. With
    b_dtype="float32" the fp32->fp16 conversion rides the global->shared copy,
    which is Phase 1's `cast=on_load` path.
    """
    use_chunk = kc > 0
    NC = K // kc if use_chunk else 1
    KI = (kc // BK) if use_chunk else (K // BK)
    Bshape = (N, K) if transpose_b else (K, N)

    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        def main(A: T.Tensor((M, K), "float16"),
                 B: T.Tensor(Bshape, b_dtype),
                 Bias: T.Tensor((N,), "float32"),
                 C: T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
                As = T.alloc_shared((BM, BK), "float16")
                if transpose_b:
                    Bs = T.alloc_shared((BN, BK), "float16")
                else:
                    Bs = T.alloc_shared((BK, BN), "float16")
                Cacc = T.alloc_fragment((BM, BN), "float32")
                Cchunk = T.alloc_fragment((BM, BN), "float32")
                T.clear(Cacc)

                for c in T.serial(NC):
                    T.clear(Cchunk)
                    for ko in T.Pipelined(KI, num_stages=stages):
                        T.copy(A[by * BM, c * kc + ko * BK], As)
                        if transpose_b:
                            T.copy(B[bx * BN, c * kc + ko * BK], Bs)
                        else:
                            T.copy(B[c * kc + ko * BK, bx * BN], Bs)
                        T.gemm(As, Bs, Cchunk, transpose_B=transpose_b)
                    for i, j in T.Parallel(BM, BN):
                        Cacc[i, j] += Cchunk[i, j]

                # ---- epilogue: the only thing the ladder changes -------------
                if bias and gelu:
                    for i, j in T.Parallel(BM, BN):
                        v = Cacc[i, j] + Bias[bx * BN + j]
                        Cacc[i, j] = v * T.float32(0.5) * (
                            T.float32(1.0) + T.erf(v * T.float32(_INV_SQRT2)))
                elif bias:
                    for i, j in T.Parallel(BM, BN):
                        Cacc[i, j] = Cacc[i, j] + Bias[bx * BN + j]
                elif gelu:
                    for i, j in T.Parallel(BM, BN):
                        v = Cacc[i, j]
                        Cacc[i, j] = v * T.float32(0.5) * (
                            T.float32(1.0) + T.erf(v * T.float32(_INV_SQRT2)))
                T.copy(Cacc, C[by * BM, bx * BN])
        return main

    return _k()


# ---------------------------------------------------------------- softmax ---
def _kernel_softmax(M, N, th):
    """Row softmax, F4 form: warp-shuffle reduction + locally cached exponentials.

    Held fixed for the whole cross-DSL ladder. Its abstraction level is the
    subject of the separate F1..F4 study; here it is a constant so that the
    ladder measures fusion and caching, not reduction style.
    """
    ept = N // th
    nwarps = th // 32
    nlevels = nwarps.bit_length() - 1

    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        def main(X: T.Tensor((M, N), "float32"),
                 Out: T.Tensor((M, N), "float32")):
            with T.Kernel(M, threads=th) as bx:
                tid = T.get_thread_binding(0)
                wid = tid >> 5
                lane = tid & 31
                smem_m = T.alloc_shared((nwarps,), "float32")
                smem_s = T.alloc_shared((nwarps,), "float32")
                lmax = T.alloc_local((1,), "float32")
                lsum = T.alloc_local((1,), "float32")
                lexp = T.alloc_local((ept,), "float32")

                lmax[0] = T.float32(-3.402823466e+38)
                for k in T.serial(ept):
                    v = X[bx, tid * ept + k]
                    if v > lmax[0]:
                        lmax[0] = v
                # The shuffle chain is written out rather than looped: TileLang's
                # eager builder intercepts `for` and only accepts its own loop
                # constructs, so a Python-level tuple loop is a build error.
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 16))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 8))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 4))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 2))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 1))
                if lane == 0:
                    smem_m[wid] = lmax[0]
                T.sync_threads()
                for _lvl in range(nlevels):
                    stride = nwarps >> (_lvl + 1)
                    if tid < stride:
                        smem_m[tid] = T.max(smem_m[tid], smem_m[tid + stride])
                    T.sync_threads()
                row_max = smem_m[0]

                lsum[0] = T.float32(0.0)
                for k in T.serial(ept):
                    e = T.exp(X[bx, tid * ept + k] - row_max)
                    lexp[k] = e
                    lsum[0] = lsum[0] + e
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 16)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 8)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 4)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 2)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 1)
                if lane == 0:
                    smem_s[wid] = lsum[0]
                T.sync_threads()
                for _lvl in range(nlevels):
                    stride = nwarps >> (_lvl + 1)
                    if tid < stride:
                        smem_s[tid] = smem_s[tid] + smem_s[tid + stride]
                    T.sync_threads()
                inv_sum = T.float32(1.0) / smem_s[0]

                for k in T.serial(ept):
                    Out[bx, tid * ept + k] = lexp[k] * inv_sum
        return main

    return _k()


# ------------------------------------------------------------------ build ---
def build(cfg) -> common2.Built2:
    M, N, K = cfg.M, cfg.N, cfg.K
    BM, BN, BK, threads = cfg.BM, cfg.BN, cfg.BK, cfg.threads
    kc, stages = cfg.kc, cfg.stages
    arm = common2.FUSED_ARMS[cfg.variant]
    wmode = cfg.extra.get("wcache", "cached")
    wspec = common2.weight_kernel_spec(wmode)

    assert K % BK == 0 and (not kc or (K % kc == 0 and kc % BK == 0))
    assert N % common2.SOFT_THREADS == 0

    t0 = time.perf_counter()
    kg = _kernel_gemm(M, N, K, BM, BN, BK, threads, stages, kc,
                      bias=arm["bias"], gelu=arm["gelu"],
                      b_dtype=wspec["b_dtype"], transpose_b=wspec["transpose_b"])
    ks = _kernel_softmax(M, N, common2.SOFT_THREADS) if arm["softmax"] else None
    wf = common2.weight_fn(wmode)
    # cast=precast -> the runner hands us fp16 x (Phase-1 variant D, verbatim).
    # cast=in_region -> we get fp32 and pay for .half() inside the timer.
    xcast = cfg.cast == "in_region"
    xf = (lambda x: x.half()) if xcast else (lambda x: x)

    if ks is None:
        def run(x, W, b):
            return kg(xf(x), wf(W), b)
    else:
        def run(x, W, b):
            return ks(kg(xf(x), wf(W), b))

    # Force JIT module load / first-launch cost out of the timed region, at the
    # real shape (the JIT is shape-specialised) and through `run` so the weight
    # path warms too. For wcache=cached this is also where the cache fills, which
    # is the point: a cached weight is by definition not paid for per call.
    x_dtype = torch.float32 if xcast else torch.float16
    # Warm the KERNELS, not `run`. Going through `run` would prime the cached-
    # weight box with this dummy and every timed call would then multiply by
    # zeros. The weight cache is filled instead by the caller's first real
    # call, which the runner makes before the timed region either way.
    xw = torch.zeros((M, K), dtype=torch.float16, device="cuda")
    Bw = torch.zeros((N, K) if wspec["transpose_b"] else (K, N),
                     dtype=torch.float16 if wspec["b_dtype"] == "float16"
                     else torch.float32, device="cuda")
    bw = torch.zeros((N,), dtype=torch.float32, device="cuda")
    scratch = kg(xw, Bw, bw)
    if ks is not None:
        _ = ks(scratch)
        del _
    torch.cuda.synchronize()
    del xw, Bw, bw, scratch
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    artifacts = {
        "grid": [N // BN, M // BM, 1], "block": [threads, 1, 1],
        "tilelang_version": tilelang.__version__,
        "tilelang_disk_cache": _CACHE_ENABLED,
        "wcache": wmode, "b_global_dtype": wspec["b_dtype"],
        "transpose_b": wspec["transpose_b"],
        "n_kernels": 2 if ks is not None else 1,
        "backend_detail": (
            f"T.gemm(fp16 smem -> fp32 fragment) kc={kc} "
            f"T.Pipelined(num_stages={stages}); epilogue "
            f"bias={arm['bias']} gelu={arm['gelu']} (T.erf, exact form)"
            + ("; + row-softmax kernel (F4: warp shuffle + cached exp, "
               f"{common2.SOFT_THREADS} thr)" if ks is not None else "")),
    }
    try:
        artifacts["cuda_source"] = kg.get_kernel_source()
        if ks is not None:
            artifacts["cuda_source_softmax"] = ks.get_kernel_source()
    except Exception as e:  # noqa: BLE001
        artifacts["cuda_source_error"] = repr(e)

    if _DUMP_ASM:
        d = os.path.join(common2.ARTIFACTS_DIR, "tilelang")
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, cfg.key().replace("/", "_"))
        for tag, kern in (("gemm", kg), ("softmax", ks)):
            if kern is None:
                continue
            try:
                kern.export_ptx(f"{base}.{tag}.ptx")
                kern.export_sass(f"{base}.{tag}.sass")
                artifacts[f"sass_path_{tag}"] = f"{base}.{tag}.sass"
            except Exception as e:  # noqa: BLE001
                artifacts[f"asm_error_{tag}"] = repr(e)

    notes = (f"tilelang {tilelang.__version__} fused arm {cfg.variant} "
             f"({arm['label']}) wcache={wmode}: {BM}x{BN}x{BK}/{threads}thr "
             f"kc={kc} stages={stages}")
    return common2.Built2(run=run, compile_s=compile_s, artifacts=artifacts,
                          notes=notes, n_kernels=artifacts["n_kernels"],
                          x_dtype=x_dtype)
