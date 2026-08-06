"""TileLang-only abstraction study for the fused op: the softmax reduction level.

Reported separately from the cross-DSL ladder, for the reason Phase 1 gave:
abstraction level, algorithm and hardware instruction path are otherwise
confounded, and a cross-DSL table cannot separate them.

Everything is held fixed except the reduction: the GEMM is the Phase-1 matched
kernel with bias and exact erf GELU in a `T.Parallel` epilogue (arm GBG's
kernel, unchanged), the softmax is one row per block at 256 threads, and the
output is fp32. Only how the row max and the row sum are computed changes.

  F1  T.reduce_max / T.reduce_sum on a (1, N) fragment. The whole reduction is
      one call; TileLang picks the schedule.
  F2  Manual shared-memory tree. No warp shuffles anywhere -- every partial
      goes through smem and a __syncthreads().
  F3  Manual warp-shuffle reduction (T.shfl_down) feeding a short smem tree
      across the 8 warps. Exponentials are RECOMPUTED in the write pass.
  F4  F3 plus a per-thread local buffer that caches the exponentials, so
      T.exp is evaluated once per element instead of twice. This is the
      incumbent solution's kernel.

F3 -> F4 is not an abstraction step at all -- it is an algorithmic change
(trading 32 registers per thread for one pass of transcendentals). It is carried
in the same ladder on purpose, so the report can show what fraction of the F1->F4
distance is abstraction and what fraction is that one algorithmic idea.

The weight-cache factor is orthogonal and is run as a separate two-way factor,
per the study spec.
"""
import os
import time

import torch

import tilelang
import tilelang.language as T

import common2
from .fused_tilelang import _kernel_gemm

if os.environ.get("PHASE2_TL_CACHE", "0") != "1":
    tilelang.disable_cache()

ABS_ARMS = {
    "F1": "T.reduce_max / T.reduce_sum over a (1,N) fragment",
    "F2": "manual shared-memory tree, no warp shuffles",
    "F3": "manual T.shfl_down warp reduction + smem tree, exp recomputed",
    "F4": "F3 + per-thread local buffer caching the exponentials (incumbent)",
    # ...and the same three manual arms with the ONE access pattern the manual
    # form makes it possible to get wrong. F2/F3/F4 index X[bx, tid*ept + k]:
    # each thread walks 32 CONSECUTIVE floats, so a warp touches 32 separate
    # 128 B segments spread over 4 KB and the loads do not coalesce. That is the
    # indexing the shipped incumbent uses. F2c/F3c/F4c are identical in every
    # other respect and index X[bx, k*th + tid], where a warp reads one
    # contiguous 128 B line per step.
    #
    # Carrying both is the only way to separate the study's two questions:
    # F1 vs F4c asks what the abstraction COSTS when the manual code is written
    # well; F1 vs F4 asks what the abstraction PREVENTS.
    "F2c": "F2 with coalesced (stride-threads) indexing",
    "F3c": "F3 with coalesced (stride-threads) indexing",
    "F4c": "F4 with coalesced (stride-threads) indexing",
}


def _f1(M, N, th):
    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        def main(X: T.Tensor((M, N), "float32"),
                 Out: T.Tensor((M, N), "float32")):
            with T.Kernel(M, threads=th) as bx:
                xs = T.alloc_fragment((1, N), "float32")
                mx = T.alloc_fragment((1,), "float32")
                sm = T.alloc_fragment((1,), "float32")
                T.copy(X[bx, 0], xs)
                T.reduce_max(xs, mx, dim=1, clear=True)
                for i, jj in T.Parallel(1, N):
                    xs[i, jj] = T.exp(xs[i, jj] - mx[0])
                T.reduce_sum(xs, sm, dim=1)
                for i, jj in T.Parallel(1, N):
                    Out[bx, jj] = xs[i, jj] / sm[0]
        return main
    return _k()


def _f2(M, N, th, coalesced=False):
    """Everything through shared memory. The per-thread partials are written to
    a (th,) smem array and reduced by a log2(th)-deep tree; no shuffle
    instruction appears anywhere in the generated code."""
    ept = N // th
    nlev = th.bit_length() - 1
    # index of this thread's k-th element
    idx = (lambda tid, k: k * th + tid) if coalesced else (lambda tid, k: tid * ept + k)

    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        def main(X: T.Tensor((M, N), "float32"),
                 Out: T.Tensor((M, N), "float32")):
            with T.Kernel(M, threads=th) as bx:
                tid = T.get_thread_binding(0)
                sm = T.alloc_shared((th,), "float32")
                acc = T.alloc_local((1,), "float32")

                acc[0] = T.float32(-3.402823466e+38)
                for k in T.serial(ept):
                    v = X[bx, idx(tid, k)]
                    if v > acc[0]:
                        acc[0] = v
                sm[tid] = acc[0]
                T.sync_threads()
                for _lvl in range(nlev):
                    stride = th >> (_lvl + 1)
                    if tid < stride:
                        sm[tid] = T.max(sm[tid], sm[tid + stride])
                    T.sync_threads()
                row_max = sm[0]
                T.sync_threads()

                acc[0] = T.float32(0.0)
                for k in T.serial(ept):
                    acc[0] = acc[0] + T.exp(X[bx, idx(tid, k)] - row_max)
                sm[tid] = acc[0]
                T.sync_threads()
                for _lvl in range(nlev):
                    stride = th >> (_lvl + 1)
                    if tid < stride:
                        sm[tid] = sm[tid] + sm[tid + stride]
                    T.sync_threads()
                inv = T.float32(1.0) / sm[0]

                for k in T.serial(ept):
                    Out[bx, idx(tid, k)] = T.exp(
                        X[bx, idx(tid, k)] - row_max) * inv
        return main
    return _k()


def _f3_f4(M, N, th, cache_exp, coalesced=False):
    """F3 and F4 differ only in `cache_exp`; sharing the body keeps the two arms
    from drifting in anything else."""
    ept = N // th
    nwarps = th // 32
    nlev = nwarps.bit_length() - 1
    idx = (lambda tid, k: k * th + tid) if coalesced else (lambda tid, k: tid * ept + k)

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
                lexp = T.alloc_local((ept if cache_exp else 1,), "float32")

                lmax[0] = T.float32(-3.402823466e+38)
                for k in T.serial(ept):
                    v = X[bx, idx(tid, k)]
                    if v > lmax[0]:
                        lmax[0] = v
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 16))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 8))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 4))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 2))
                lmax[0] = T.max(lmax[0], T.shfl_down(lmax[0], 1))
                if lane == 0:
                    smem_m[wid] = lmax[0]
                T.sync_threads()
                for _lvl in range(nlev):
                    stride = nwarps >> (_lvl + 1)
                    if tid < stride:
                        smem_m[tid] = T.max(smem_m[tid], smem_m[tid + stride])
                    T.sync_threads()
                row_max = smem_m[0]

                lsum[0] = T.float32(0.0)
                if cache_exp:
                    for k in T.serial(ept):
                        e = T.exp(X[bx, idx(tid, k)] - row_max)
                        lexp[k] = e
                        lsum[0] = lsum[0] + e
                else:
                    for k in T.serial(ept):
                        lsum[0] = lsum[0] + T.exp(X[bx, idx(tid, k)] - row_max)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 16)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 8)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 4)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 2)
                lsum[0] = lsum[0] + T.shfl_down(lsum[0], 1)
                if lane == 0:
                    smem_s[wid] = lsum[0]
                T.sync_threads()
                for _lvl in range(nlev):
                    stride = nwarps >> (_lvl + 1)
                    if tid < stride:
                        smem_s[tid] = smem_s[tid] + smem_s[tid + stride]
                    T.sync_threads()
                inv = T.float32(1.0) / smem_s[0]

                if cache_exp:
                    for k in T.serial(ept):
                        Out[bx, idx(tid, k)] = lexp[k] * inv
                else:
                    for k in T.serial(ept):
                        Out[bx, idx(tid, k)] = T.exp(
                            X[bx, idx(tid, k)] - row_max) * inv
        return main
    return _k()


_BUILDERS = {
    "F1": lambda M, N, th: _f1(M, N, th),
    "F2": lambda M, N, th: _f2(M, N, th),
    "F3": lambda M, N, th: _f3_f4(M, N, th, cache_exp=False),
    "F4": lambda M, N, th: _f3_f4(M, N, th, cache_exp=True),
    "F2c": lambda M, N, th: _f2(M, N, th, coalesced=True),
    "F3c": lambda M, N, th: _f3_f4(M, N, th, cache_exp=False, coalesced=True),
    "F4c": lambda M, N, th: _f3_f4(M, N, th, cache_exp=True, coalesced=True),
}


def build(cfg) -> common2.Built2:
    arm = cfg.variant
    if arm not in ABS_ARMS:
        raise KeyError(f"unknown abstraction arm {arm!r}; known: {sorted(ABS_ARMS)}")
    M, N, K = cfg.M, cfg.N, cfg.K
    wmode = cfg.extra.get("wcache", "cached")
    wspec = common2.weight_kernel_spec(wmode)
    th = common2.SOFT_THREADS

    t0 = time.perf_counter()
    # held fixed across every arm: the matched GEMM with a T.Parallel bias+GELU
    # epilogue -- i.e. exactly arm GBG of the cross-DSL ladder
    kg = _kernel_gemm(M, N, K, cfg.BM, cfg.BN, cfg.BK, cfg.threads,
                      cfg.stages, cfg.kc, bias=True, gelu=True,
                      b_dtype=wspec["b_dtype"], transpose_b=wspec["transpose_b"])
    ks = _BUILDERS[arm](M, N, th)
    # Timing the whole op cannot resolve this study's question. The softmax is
    # ~0.07 ms of a ~1.5 ms op, so a 3% difference between reduction styles is
    # 0.002 ms -- below the noise floor. `soft_only` times ONLY the softmax
    # kernel, on the real post-GELU scratch (so the value distribution the
    # reduction sees is the real one, not a synthetic tensor).
    #
    # The scratch is built on the first call for an input triple and cached.
    # That first call is the runner's correctness check, which is outside the
    # timed region; every timed call afterwards launches only the softmax.
    # Correctness is still checked against the full GBGS reference, which is
    # exactly right:
    # scratch is the GEMM+bias+GELU output, so softmax(scratch) IS the full op.
    soft_only = str(cfg.extra.get("soft_only", "")).lower() in ("1", "true", "yes")

    if soft_only:
        box = {}

        def run(x, W, b):
            inputs = (x, W, b)
            versions = tuple(t._version for t in inputs)
            previous = box.get("inputs")
            if (previous is None
                    or any(old is not new for old, new in zip(previous, inputs))
                    or box["versions"] != versions):
                # Retaining the tensors prevents allocator pointer reuse from
                # making a different gate input look like the cached one.
                scratch = kg(x, common2.weight_fn(wmode)(W), b)
                box.update(inputs=inputs, versions=versions, scratch=scratch)
            return ks(box["scratch"])
    else:
        wf = common2.weight_fn(wmode)

        def run(x, W, b):
            return ks(kg(x, wf(W), b))

    xw = torch.zeros((M, K), dtype=torch.float16, device="cuda")
    Bw = torch.zeros((N, K) if wspec["transpose_b"] else (K, N),
                     dtype=torch.float16 if wspec["b_dtype"] == "float16"
                     else torch.float32, device="cuda")
    bw = torch.zeros((N,), dtype=torch.float32, device="cuda")
    scratch = kg(xw, Bw, bw)
    _ = ks(scratch)
    torch.cuda.synchronize()
    del xw, Bw, bw, scratch, _
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    artifacts = {
        "grid": [N // cfg.BN, M // cfg.BM, 1], "block": [cfg.threads, 1, 1],
        "tilelang_version": tilelang.__version__,
        "wcache": wmode, "abstraction_arm": arm,
        "soft_only": soft_only,
        "n_kernels": 1 if soft_only else 2,
        "backend_detail": (f"GEMM held fixed (matched Phase-1 D + T.Parallel "
                           f"bias/erf-GELU epilogue); softmax = {ABS_ARMS[arm]}; "
                           f"{th} threads, {N // th} elements/thread"),
    }
    try:
        artifacts["cuda_source"] = ks.get_kernel_source()
    except Exception as e:  # noqa: BLE001
        artifacts["cuda_source_error"] = repr(e)
    capture_dir = os.environ.get("TILELANG_ABSTRACTION_CAPTURE_DIR", "")
    if capture_dir:
        os.makedirs(capture_dir, exist_ok=True)
        base = os.path.join(capture_dir, cfg.key().replace("/", "_"))
        for extension, exporter in (("ptx", ks.export_ptx), ("sass", ks.export_sass)):
            path = base + "." + extension
            try:
                exporter(path)
                artifacts[extension + "_path"] = path
            except Exception as e:  # noqa: BLE001 - retained as admission evidence
                artifacts[extension + "_error"] = repr(e)
        with open(base + ".cu", "w", encoding="utf-8") as handle:
            handle.write(artifacts.get("cuda_source", ""))
        artifacts["cuda_source_path"] = base + ".cu"

    return common2.Built2(
        run=run, compile_s=compile_s, artifacts=artifacts,
        n_kernels=1 if soft_only else 2,
        notes=f"tilelang softmax-abstraction arm {arm} ({ABS_ARMS[arm]}) "
              f"wcache={wmode}{' SOFTMAX-ONLY' if soft_only else ''}",
        x_dtype=torch.float16)
