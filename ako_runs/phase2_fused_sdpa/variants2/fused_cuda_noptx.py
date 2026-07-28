"""cuda_noptx lane of the Phase-2 fused ladder: WMMA C++ intrinsics, no inline PTX.

The GEMM is Phase 1's, reused rather than rewritten. This module imports
`variants.cuda_noptx_gemm` from the Phase-1 tree and takes its `_make_source`
output verbatim -- the same warp grid, the same shared-memory padding search,
the same `__pipeline_memcpy_async` staging, the same KC flush into a plain
`float[NMF][NNF][8]` register array. The `__device__ __forceinline__` helpers
(`loadA_h`, `loadB_h`, `loadA_async`, `loadB_async`, `compute_h`, `flush_acc`)
come along with it and this lane's fused kernel calls exactly those.

What is added: a `fused_kernel` whose main loop is copied from Phase 1's
`gemm_kernel` and whose only difference is the ending -- instead of
`wmma::store_matrix_sync` straight to global, the fragments are staged into
shared memory so the bias (a column-indexed vector) and the exact erf GELU can
be applied with known indices. `wcache=native` is not implemented in this lane;
see SPEC2.md.

The build asserts, as Phase 1 did, that no inline asm reaches nvcc.
"""
from __future__ import annotations

import os
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

import common
import common2
from . import cuda_fused_common as cfc

_P1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "phase1_matmul")
if _P1 not in sys.path:
    sys.path.insert(0, _P1)
from variants import cuda_noptx_gemm as p1  # noqa: E402


# The main loop below is Phase 1's `gemm_kernel` body, copied without edit down
# to the final store. Keeping the copy explicit (rather than macro-splicing into
# the Phase-1 string) is what lets the report claim the inner loop is unchanged
# and lets a reader check it by diffing the two files.
FUSED_KERNEL = r"""
__global__ __launch_bounds__(THREADS)
void fused_kernel(const half* __restrict__ Ag,
                  const half* __restrict__ Bg,
                  const float* __restrict__ Bias,
                  float* __restrict__ Cg) {
    extern __shared__ __align__(16) char smem_raw[];
    half* As = reinterpret_cast<half*>(smem_raw);
    half* Bs = As + STAGES * BM * LDA;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    FragC acc[NMF][NNF];
#pragma unroll
    for (int i = 0; i < NMF; ++i) {
#pragma unroll
        for (int j = 0; j < NNF; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }
#if KC_TILES > 0
    float c2[NMF][NNF][FRAG_ELEMS];
#pragma unroll
    for (int i = 0; i < NMF; ++i) {
#pragma unroll
        for (int j = 0; j < NNF; ++j) {
#pragma unroll
            for (int e = 0; e < FRAG_ELEMS; ++e) c2[i][j][e] = 0.0f;
        }
    }
#endif

    constexpr int NK = K_ / BK;

#if STAGES == 1
    for (int i = 0; i < NK; ++i) {
        __syncthreads();
        loadA_h(Ag, As, bm, i * BK, tid);
        loadB_h(Bg, Bs, bn, i * BK, tid);
        __syncthreads();
        compute_h(As, Bs, acc, wm, wn);
#if KC_TILES > 0
        if (((i + 1) % KC_TILES) == 0) flush_acc(acc, c2);
#endif
    }
#elif USE_ASYNC
#pragma unroll
    for (int j = 0; j < STAGES - 1; ++j) {
        loadA_async(Ag, As + j * BM * LDA, bm, j * BK, tid);
        loadB_async(Bg, Bs + j * BK * LDB, bn, j * BK, tid);
        __pipeline_commit();
    }
    for (int i = 0; i < NK; ++i) {
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();
        compute_h(As + (i % STAGES) * BM * LDA,
                  Bs + (i % STAGES) * BK * LDB, acc, wm, wn);
        __syncthreads();
        const int nxt = i + STAGES - 1;
        if (nxt < NK) {
            const int b = nxt % STAGES;
            loadA_async(Ag, As + b * BM * LDA, bm, nxt * BK, tid);
            loadB_async(Bg, Bs + b * BK * LDB, bn, nxt * BK, tid);
        }
        __pipeline_commit();
#if KC_TILES > 0
        if (((i + 1) % KC_TILES) == 0) flush_acc(acc, c2);
#endif
    }
#else
#pragma unroll
    for (int j = 0; j < STAGES - 1; ++j) {
        loadA_h(Ag, As + j * BM * LDA, bm, j * BK, tid);
        loadB_h(Bg, Bs + j * BK * LDB, bn, j * BK, tid);
    }
    __syncthreads();
    for (int i = 0; i < NK; ++i) {
        compute_h(As + (i % STAGES) * BM * LDA,
                  Bs + (i % STAGES) * BK * LDB, acc, wm, wn);
        const int nxt = i + STAGES - 1;
        if (nxt < NK) {
            const int b = nxt % STAGES;
            loadA_h(Ag, As + b * BM * LDA, bm, nxt * BK, tid);
            loadB_h(Bg, Bs + b * BK * LDB, bn, nxt * BK, tid);
        }
        __syncthreads();
#if KC_TILES > 0
        if (((i + 1) % KC_TILES) == 0) flush_acc(acc, c2);
#endif
    }
#endif

#if KC_TILES > 0
    if ((NK % KC_TILES) != 0) flush_acc(acc, c2);
#pragma unroll
    for (int i = 0; i < NMF; ++i) {
#pragma unroll
        for (int j = 0; j < NNF; ++j) {
#pragma unroll
            for (int e = 0; e < FRAG_ELEMS; ++e) acc[i][j].x[e] = c2[i][j][e];
        }
    }
#endif

    /* ---- the only departure from Phase 1: stage, then epilogue ---------- */
    __syncthreads();                       /* the A/B buffers are dead now */
    float* Cs = reinterpret_cast<float*>(smem_raw);
#pragma unroll
    for (int i = 0; i < NMF; ++i) {
#pragma unroll
        for (int j = 0; j < NNF; ++j) {
            wmma::store_matrix_sync(
                &Cs[(wm * WMT + i * 16) * CSTRIDE + wn * WNT + j * 16],
                acc[i][j], CSTRIDE, wmma::mem_row_major);
        }
    }
    __syncthreads();
    fused_epilogue(Cs, Bias, Cg, bm, bn, tid, N_);
}
"""


def build(cfg) -> common2.Built2:
    common.setup_cuda_env()
    arm = common2.FUSED_ARMS[cfg.variant]
    wmode = cfg.extra.get("wcache", "cached")
    if wmode == "native":
        raise NotImplementedError(
            "cuda_noptx lane implements the two-way cached/uncached factor only; "
            "the native (N,K) layout would need a transposed WMMA B-fragment "
            "load, which is a different kernel and would confound the factor")

    # Phase-1 schedule logic, unmodified, at the fused shape. common.M/N/K are
    # module-level constants there, so they are swapped for the duration of the
    # call rather than the function being forked.
    saved = (common.M, common.N, common.K)
    common.M, common.N, common.K = cfg.M, cfg.N, cfg.K
    try:
        gen = p1._make_source(cfg)
    finally:
        common.M, common.N, common.K = saved

    src = (gen["kernel_src"] + cfc.arm_defines(arm, cfg.N) + cfc.EPILOGUE_DEVICE
           + FUSED_KERNEL + (cfc.SOFTMAX_KERNEL if arm["softmax"] else ""))
    full = src + cfc.WRAPPER
    for bad in ("asm(", "asm (", "asm volatile", "__asm"):
        if bad in full:
            raise RuntimeError(f"inline asm {bad!r} found in cuda_noptx source")

    name = ("p2noptx_%s_%dx%dx%d_t%d_kc%d_s%d_%s"
            % (cfg.variant, cfg.BM, cfg.BN, cfg.BK, cfg.threads,
               cfg.kc, cfg.stages, wmode))

    t0 = time.perf_counter()
    mod = load_inline(
        name=name, cpp_sources=cfc.CPP_DECL, cuda_sources=full,
        functions=["fused"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-Xptxas=-v",
                           "-gencode=arch=compute_89,code=sm_89"],
        verbose=False)

    wf = common2.weight_fn(wmode)
    xcast = cfg.cast == "in_region"
    xf = (lambda x: x.half()) if xcast else (lambda x: x)

    def run(x, W, b):
        return mod.fused(xf(x), wf(W), b)

    xw = torch.zeros((cfg.M, cfg.K), dtype=torch.float16, device="cuda")
    Bw = torch.zeros((cfg.K, cfg.N), dtype=torch.float16, device="cuda")
    bw = torch.zeros((cfg.N,), dtype=torch.float32, device="cuda")
    mod.fused(xw, Bw, bw)
    torch.cuda.synchronize()
    del xw, Bw, bw
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    art = {
        "cuda_source": full, "kernel_source": src,
        "shared_bytes": max(gen["smem"], cfg.BM * (cfg.BN + 4) * 4),
        "grid": gen["info"]["grid"], "block": gen["info"]["block"],
        "ext_name": name, "wcache": wmode,
        "n_kernels": 2 if arm["softmax"] else 1,
        "backend_detail": (
            "nvcuda::wmma 16x16x16 half x half -> float; warp grid %s, warp tile "
            "%s (%s frags/warp); loads = %s; KC flush every %d BK-tiles; epilogue "
            "stages fragments through %d B of smem, then bias=%s gelu=%s (erff, "
            "exact form). Zero inline PTX (asserted at build)."
            % (gen["info"]["warp_grid"], gen["info"]["warp_tile"],
               gen["info"]["frags"],
               "__pipeline_memcpy_async x %d buffers" % gen["info"]["stages_smem"]
               if gen["info"]["use_async"] else "synchronous",
               gen["info"]["kc_tiles"], cfg.BM * (cfg.BN + 4) * 4,
               arm["bias"], arm["gelu"])
            + ("; + row-softmax kernel (256 thr, 32 elem/thread cached in "
               "registers, __shfl_down_sync + smem tree)" if arm["softmax"] else "")),
        "geom_info": gen["info"],
    }
    art.update(p1._side_compile(src, name))
    notes = "cuda_noptx fused arm %s (%s) wcache=%s" % (
        cfg.variant, arm["label"], wmode)
    if gen["deviations"]:
        notes += " | DEVIATION: " + "; ".join(gen["deviations"])

    return common2.Built2(run=run, compile_s=compile_s, artifacts=art,
                          notes=notes, n_kernels=art["n_kernels"],
                          x_dtype=torch.float32 if xcast else torch.float16)
