"""cuda_noptx lane -- hand-written CUDA C++ GEMM, ZERO inline PTX.

The scientific point of this lane: can plain WMMA C++ (`<mma.h>`, half operands
accumulating into a float accumulator) plus CUDA *intrinsics*
(`__pipeline_memcpy_async` from `<cuda_pipeline.h>`) reach the tensor-core
frontier without a single `asm()` statement?  Nothing in the generated CUDA
below contains inline assembly; `grep -c asm` on `artifacts["cuda_source"]` is 0
by construction (a build-time assertion enforces it).

Variant map (SPEC.md):
  A  arith=fp32, kc=0,    stages=1  -> scalar register-tiled SGEMM, no `wmma::`
  B  arith=fp16, kc=0,    stages=1  -> wmma 16x16x16 half->float, ONE accumulator
                                       chain over all K, synchronous loads
  C  arith=fp16, kc=2048, stages=1  -> B + flush the wmma accumulator fragment
                                       into a second fp32 register accumulator
                                       every kc elements of K
  D  arith=fp16, kc=2048, stages=3  -> C + `stages`-deep __pipeline_memcpy_async
                                       software pipeline on global->shared

cast modes:
  precast    global operands are half        (uint4 / cp.async 16B copies)
  in_region  `run` calls .half() itself, same kernel as precast
  on_load    global operands are float; float4 loaded, converted with
             __floats2half2_rn on the way into the half smem tile.  cp.async
             cannot convert, so `on_load` with stages>1 uses a multi-buffered
             load-ahead loop instead (documented in backend_detail).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time

import torch
from torch.utils.cpp_extension import load_inline

import common

# sm_89 hardware cap on opt-in dynamic shared memory per block (verified on this
# host: torch.cuda.get_device_properties(0).shared_memory_per_block_optin).
SMEM_LIMIT = 101376


# --------------------------------------------------------------- geometry ---
def _pick_thread_tile(BM: int, BN: int, threads: int):
    """Deterministic thread-tile choice for the scalar SGEMM (variant A).

    Not autotuning: a fixed rule, evaluated once, no measurement in the loop.
    Prefer the squarest thread tile; break ties toward the wider N tile so the
    epilogue and the Bs read stay float4.
    """
    best = None
    tnt = 1
    while tnt <= threads:
        if threads % tnt == 0:
            tmt = threads // tnt
            if BM % tmt == 0 and BN % tnt == 0:
                TM, TN = BM // tmt, BN // tnt
                if TN % 4 == 0 and TM >= 1:
                    cand = (abs(TM - TN), -TN, tmt, tnt, TM, TN)
                    if best is None or cand < best:
                        best = cand
        tnt *= 2
    if best is None:
        raise RuntimeError(f"no thread tile for BM={BM} BN={BN} threads={threads}")
    _, _, tmt, tnt, TM, TN = best
    return tmt, tnt, TM, TN


def _pick_warp_grid(BM: int, BN: int, threads: int):
    """Deterministic warp grid for the wmma path (squarest warp tile)."""
    nwarps = threads // 32
    best = None
    for wm in range(1, nwarps + 1):
        if nwarps % wm:
            continue
        wn = nwarps // wm
        if BM % (wm * 16) or BN % (wn * 16):
            continue
        WMT, WNT = BM // wm, BN // wn
        cand = (abs(WMT - WNT), wm, wn, WMT, WNT)
        if best is None or cand < best:
            best = cand
    if best is None:
        raise RuntimeError(f"no warp grid for BM={BM} BN={BN} threads={threads}")
    _, wm, wn, WMT, WNT = best
    return wm, wn, WMT, WNT


# ------------------------------------------------------------ CUDA source ---
_KERNEL_BODY = r"""
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

#if ARITH_FP32
/* =======================================================================
   Variant A -- scalar register-tiled SGEMM on the CUDA cores.
   No <mma.h> type is instantiated anywhere on this path: the multiply is a
   plain `acc += a*b` on floats, so ptxas emits FFMA and the SASS contains
   zero HMMA.
   ===================================================================== */

__device__ __forceinline__ void loadA_f32(const float* __restrict__ Ag,
                                          float* __restrict__ As,
                                          int bm, int k0, int tid) {
    constexpr int VPR = BK / 4;          /* float4 per A-tile row */
    constexpr int NV  = BM * BK / 4;
    for (int g = tid; g < NV; g += THREADS) {
        const int r = g / VPR, c = g % VPR;
        const float4 v = *reinterpret_cast<const float4*>(&Ag[(bm + r) * K_ + k0 + c * 4]);
        /* transposed into As[k][m] so the inner loop reads m contiguously */
        As[(c * 4 + 0) * BM + r] = v.x;
        As[(c * 4 + 1) * BM + r] = v.y;
        As[(c * 4 + 2) * BM + r] = v.z;
        As[(c * 4 + 3) * BM + r] = v.w;
    }
}

__device__ __forceinline__ void loadB_f32(const float* __restrict__ Bg,
                                          float* __restrict__ Bs,
                                          int bn, int k0, int tid) {
    constexpr int VPR = BN / 4;
    constexpr int NV  = BK * BN / 4;
    for (int g = tid; g < NV; g += THREADS) {
        const int r = g / VPR, c = g % VPR;
        const float4 v = *reinterpret_cast<const float4*>(&Bg[(k0 + r) * N_ + bn + c * 4]);
        *reinterpret_cast<float4*>(&Bs[r * BN + c * 4]) = v;
    }
}

__device__ __forceinline__ void compute_f32(const float* __restrict__ As,
                                            const float* __restrict__ Bs,
                                            float (&acc)[TM][TN],
                                            int trow, int tcol) {
#pragma unroll
    for (int k = 0; k < BK; ++k) {
        float ra[TM], rb[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) ra[i] = As[k * BM + trow * TM + i];
#pragma unroll
        for (int j = 0; j < TN; ++j) rb[j] = Bs[k * BN + tcol * TN + j];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];   /* FFMA */
        }
    }
}

__global__ __launch_bounds__(THREADS)
void gemm_kernel(const float* __restrict__ Ag,
                 const float* __restrict__ Bg,
                 float* __restrict__ Cg) {
    extern __shared__ __align__(16) char smem_raw[];
    float* As = reinterpret_cast<float*>(smem_raw);
    float* Bs = As + STAGES * BK * BM;

    const int tid  = threadIdx.x;
    const int trow = tid / TNT;
    const int tcol = tid % TNT;
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    constexpr int NK = K_ / BK;

#if STAGES == 1
    /* pipeline OFF: single buffer, __syncthreads()-separated */
    for (int i = 0; i < NK; ++i) {
        __syncthreads();
        loadA_f32(Ag, As, bm, i * BK, tid);
        loadB_f32(Bg, Bs, bn, i * BK, tid);
        __syncthreads();
        compute_f32(As, Bs, acc, trow, tcol);
    }
#else
#pragma unroll
    for (int j = 0; j < STAGES - 1; ++j) {
        loadA_f32(Ag, As + j * BK * BM, bm, j * BK, tid);
        loadB_f32(Bg, Bs + j * BK * BN, bn, j * BK, tid);
    }
    __syncthreads();
    for (int i = 0; i < NK; ++i) {
        compute_f32(As + (i % STAGES) * BK * BM, Bs + (i % STAGES) * BK * BN,
                    acc, trow, tcol);
        const int nxt = i + STAGES - 1;
        if (nxt < NK) {
            const int b = nxt % STAGES;
            loadA_f32(Ag, As + b * BK * BM, bm, nxt * BK, tid);
            loadB_f32(Bg, Bs + b * BK * BN, bn, nxt * BK, tid);
        }
        __syncthreads();
    }
#endif

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = bm + trow * TM + i;
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 v;
            v.x = acc[i][j + 0]; v.y = acc[i][j + 1];
            v.z = acc[i][j + 2]; v.w = acc[i][j + 3];
            *reinterpret_cast<float4*>(&Cg[r * N_ + bn + tcol * TN + j]) = v;
        }
    }
}

#else  /* ---------------------------------------------------------------- */
/* =======================================================================
   Variants B / C / D -- nvcuda::wmma 16x16x16, half operands, float
   accumulator.  No inline PTX: mma.sync is emitted by the <mma.h> intrinsics,
   cp.async by __pipeline_memcpy_async.
   ===================================================================== */
using namespace nvcuda;

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> FragA;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> FragB;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float>              FragC;

#if ON_LOAD
/* fp32 global -> fp16 shared, converted on the way in (cast=on_load) */
__device__ __forceinline__ void loadA_h(const float* __restrict__ Ag,
                                        half* __restrict__ As,
                                        int bm, int k0, int tid) {
    constexpr int VPR = BK / 4;
    constexpr int NV  = BM * BK / 4;
    for (int g = tid; g < NV; g += THREADS) {
        const int r = g / VPR, c = g % VPR;
        const float4 v = *reinterpret_cast<const float4*>(&Ag[(bm + r) * K_ + k0 + c * 4]);
        half* d = As + r * LDA + c * 4;
        *reinterpret_cast<__half2*>(d + 0) = __floats2half2_rn(v.x, v.y);
        *reinterpret_cast<__half2*>(d + 2) = __floats2half2_rn(v.z, v.w);
    }
}
__device__ __forceinline__ void loadB_h(const float* __restrict__ Bg,
                                        half* __restrict__ Bs,
                                        int bn, int k0, int tid) {
    constexpr int VPR = BN / 4;
    constexpr int NV  = BK * BN / 4;
    for (int g = tid; g < NV; g += THREADS) {
        const int r = g / VPR, c = g % VPR;
        const float4 v = *reinterpret_cast<const float4*>(&Bg[(k0 + r) * N_ + bn + c * 4]);
        half* d = Bs + r * LDB + c * 4;
        *reinterpret_cast<__half2*>(d + 0) = __floats2half2_rn(v.x, v.y);
        *reinterpret_cast<__half2*>(d + 2) = __floats2half2_rn(v.z, v.w);
    }
}
#else
/* fp16 global -> fp16 shared, 16-byte granules */
__device__ __forceinline__ void loadA_h(const half* __restrict__ Ag,
                                        half* __restrict__ As,
                                        int bm, int k0, int tid) {
    constexpr int VPR = BK / 8;
    constexpr int NV  = BM * BK / 8;
    for (int g = tid; g < NV; g += THREADS) {
        const int r = g / VPR, c = g % VPR;
        *reinterpret_cast<uint4*>(As + r * LDA + c * 8) =
            *reinterpret_cast<const uint4*>(&Ag[(bm + r) * K_ + k0 + c * 8]);
    }
}
__device__ __forceinline__ void loadB_h(const half* __restrict__ Bg,
                                        half* __restrict__ Bs,
                                        int bn, int k0, int tid) {
    constexpr int VPR = BN / 8;
    constexpr int NV  = BK * BN / 8;
    for (int g = tid; g < NV; g += THREADS) {
        const int r = g / VPR, c = g % VPR;
        *reinterpret_cast<uint4*>(Bs + r * LDB + c * 8) =
            *reinterpret_cast<const uint4*>(&Bg[(k0 + r) * N_ + bn + c * 8]);
    }
}
#if USE_ASYNC
__device__ __forceinline__ void loadA_async(const half* __restrict__ Ag,
                                            half* __restrict__ As,
                                            int bm, int k0, int tid) {
    constexpr int VPR = BK / 8;
    constexpr int NV  = BM * BK / 8;
    for (int g = tid; g < NV; g += THREADS) {
        const int r = g / VPR, c = g % VPR;
        __pipeline_memcpy_async(As + r * LDA + c * 8,
                                &Ag[(bm + r) * K_ + k0 + c * 8], 16);
    }
}
__device__ __forceinline__ void loadB_async(const half* __restrict__ Bg,
                                            half* __restrict__ Bs,
                                            int bn, int k0, int tid) {
    constexpr int VPR = BN / 8;
    constexpr int NV  = BK * BN / 8;
    for (int g = tid; g < NV; g += THREADS) {
        const int r = g / VPR, c = g % VPR;
        __pipeline_memcpy_async(Bs + r * LDB + c * 8,
                                &Bg[(k0 + r) * N_ + bn + c * 8], 16);
    }
}
#endif
#endif

__device__ __forceinline__ void compute_h(const half* __restrict__ As,
                                          const half* __restrict__ Bs,
                                          FragC (&acc)[NMF][NNF],
                                          int wm, int wn) {
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
        FragA af[NMF];
        FragB bf[NNF];
#pragma unroll
        for (int i = 0; i < NMF; ++i)
            wmma::load_matrix_sync(af[i], As + (wm * WMT + i * 16) * LDA + kk * 16, LDA);
#pragma unroll
        for (int j = 0; j < NNF; ++j)
            wmma::load_matrix_sync(bf[j], Bs + (kk * 16) * LDB + wn * WNT + j * 16, LDB);
#pragma unroll
        for (int i = 0; i < NMF; ++i) {
#pragma unroll
            for (int j = 0; j < NNF; ++j)
                wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
    }
}

#if KC_TILES > 0
/* the chunk flush: drain the tensor-core accumulator fragment into a plain
   fp32 register array and zero the fragment.  fragment::x[] is public API. */
__device__ __forceinline__ void flush_acc(FragC (&acc)[NMF][NNF],
                                          float (&c2)[NMF][NNF][FRAG_ELEMS]) {
#pragma unroll
    for (int i = 0; i < NMF; ++i) {
#pragma unroll
        for (int j = 0; j < NNF; ++j) {
#pragma unroll
            for (int e = 0; e < FRAG_ELEMS; ++e) c2[i][j][e] += acc[i][j].x[e];
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }
}
#endif

__global__ __launch_bounds__(THREADS)
void gemm_kernel(const GTYPE* __restrict__ Ag,
                 const GTYPE* __restrict__ Bg,
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
    /* pipeline OFF: one shared buffer, synchronous __syncthreads()-separated */
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
    /* STAGES-deep software pipeline built from __pipeline_memcpy_async
       (a CUDA intrinsic; no inline PTX).  STAGES-1 tiles are in flight. */
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
    /* on_load + stages>1: cp.async cannot convert fp32->fp16 in flight, so the
       pipeline is STAGES shared buffers with the tile-(i+STAGES-1) global load
       issued at iteration i (the converting store lands in a buffer nobody is
       reading).  Same depth, register-staged by the scheduler instead of DMA. */
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

#pragma unroll
    for (int i = 0; i < NMF; ++i) {
#pragma unroll
        for (int j = 0; j < NNF; ++j) {
            wmma::store_matrix_sync(
                &Cg[(bm + wm * WMT + i * 16) * N_ + bn + wn * WNT + j * 16],
                acc[i][j], N_, wmma::mem_row_major);
        }
    }
}
#endif
"""

_WRAPPER = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "operands must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "operands must be contiguous");
    TORCH_CHECK(A.scalar_type() == TORCH_GTYPE && B.scalar_type() == TORCH_GTYPE,
                "operand dtype mismatch for this variant");
    TORCH_CHECK(A.size(0) == M_ && A.size(1) == K_, "A shape");
    TORCH_CHECK(B.size(0) == K_ && B.size(1) == N_, "B shape");

    auto C = torch::empty({M_, N_},
        torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

    static bool attr_done = false;
    if (!attr_done) {
        cudaError_t e = cudaFuncSetAttribute(
            (const void*)gemm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        TORCH_CHECK(e == cudaSuccess,
                    "cudaFuncSetAttribute(", SMEM_BYTES, " bytes smem) failed: ",
                    cudaGetErrorString(e));
        attr_done = true;
    }

    dim3 grid(N_ / BN, M_ / BM);          /* plain 2-D grid, no swizzle */
    dim3 block(THREADS);
    gemm_kernel<<<grid, block, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const GTYPE*>(A.data_ptr()),
        reinterpret_cast<const GTYPE*>(B.data_ptr()),
        C.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
"""


def _make_source(cfg: common.Config) -> dict:
    BM, BN, BK, TH = cfg.BM, cfg.BN, cfg.BK, cfg.threads
    fp32 = cfg.arith == "fp32"
    on_load = (not fp32) and cfg.cast == "on_load"

    if common.M % BM or common.N % BN or common.K % BK:
        raise RuntimeError("problem shape not divisible by the tile")

    info = dict(BM=BM, BN=BN, BK=BK, threads=TH)
    deviations = []

    if fp32:
        tmt, tnt, TM, TN = _pick_thread_tile(BM, BN, TH)
        stages = cfg.stages
        smem = stages * (BK * BM + BK * BN) * 4
        if smem > SMEM_LIMIT:
            raise RuntimeError(f"variant A needs {smem} B smem > {SMEM_LIMIT} cap")
        defs = {
            "ARITH_FP32": 1, "ON_LOAD": 0, "USE_ASYNC": 0,
            "KC_TILES": 0, "STAGES": stages,
            "TMT": tmt, "TNT": tnt, "TM": TM, "TN": TN,
        }
        info.update(thread_grid=f"{tmt}x{tnt}", thread_tile=f"{TM}x{TN}",
                    stages_smem=stages)
        gtype, torch_gtype = "float", "torch::kFloat32"
    else:
        wm, wn, WMT, WNT = _pick_warp_grid(BM, BN, TH)
        NMF, NNF = WMT // 16, WNT // 16
        if BK % 16:
            raise RuntimeError("BK must be a multiple of 16 for wmma")
        if cfg.kc:
            if cfg.kc % BK or common.K % cfg.kc:
                raise RuntimeError(f"kc={cfg.kc} must be a multiple of BK={BK} "
                                   f"and divide K={common.K}")
        kc_tiles = (cfg.kc // BK) if cfg.kc else 0

        # pick padding + number of shared buffers that fit the 99 KB cap
        stages = cfg.stages
        pad = None
        for p in (8, 0):                     # ldm must stay a multiple of 8 halfs
            if ((BK + p) % 8) or ((BN + p) % 8):
                continue
            if stages * (BM * (BK + p) + BK * (BN + p)) * 2 <= SMEM_LIMIT:
                pad, chosen_stages = p, stages
                break
        if pad is None:
            # cannot honour cfg.stages on this hardware -- shrink and SAY SO
            for s in range(stages - 1, 0, -1):
                for p in (8, 0):
                    if ((BK + p) % 8) or ((BN + p) % 8):
                        continue
                    if s * (BM * (BK + p) + BK * (BN + p)) * 2 <= SMEM_LIMIT:
                        pad, chosen_stages = p, s
                        break
                if pad is not None:
                    break
            need = stages * (BM * (BK + 0) + BK * (BN + 0)) * 2
            deviations.append(
                f"cfg.stages={stages} needs {need} B of shared memory "
                f"({stages} x {need // stages} B/buffer) but sm_89 caps a block at "
                f"{SMEM_LIMIT} B; ran with {chosen_stages} shared buffers instead")
        if pad is None:
            raise RuntimeError("no shared-memory configuration fits")
        stages = chosen_stages
        LDA, LDB = BK + pad, BN + pad
        smem = stages * (BM * LDA + BK * LDB) * 2
        use_async = 1 if (stages > 1 and not on_load) else 0
        defs = {
            "ARITH_FP32": 0, "ON_LOAD": 1 if on_load else 0,
            "USE_ASYNC": use_async, "KC_TILES": kc_tiles, "STAGES": stages,
            "WARPS_M": wm, "WARPS_N": wn, "WMT": WMT, "WNT": WNT,
            "NMF": NMF, "NNF": NNF, "LDA": LDA, "LDB": LDB, "FRAG_ELEMS": 8,
        }
        info.update(warp_grid=f"{wm}x{wn}", warp_tile=f"{WMT}x{WNT}",
                    frags=f"{NMF}x{NNF}", pad_halfs=pad, stages_smem=stages,
                    kc_tiles=kc_tiles, use_async=bool(use_async))
        gtype = "float" if on_load else "half"
        torch_gtype = "torch::kFloat32" if on_load else "torch::kHalf"

    defs.update({"M_": common.M, "N_": common.N, "K_": common.K,
                 "BM": BM, "BN": BN, "BK": BK, "THREADS": TH,
                 "SMEM_BYTES": smem})

    const_block = "\n".join(f"#define {k} {v}" for k, v in defs.items())
    const_block += f"\n#define GTYPE {gtype}\n#define TORCH_GTYPE {torch_gtype}\n"

    kernel_src = const_block + _KERNEL_BODY
    full_src = kernel_src + _WRAPPER

    # hard guarantee for this lane: not one line of inline asm
    for bad in ("asm(", "asm (", "asm volatile", "__asm"):
        if bad in full_src:
            raise RuntimeError(f"inline asm {bad!r} found in cuda_noptx source")

    info.update(smem_bytes=smem, grid=f"({common.N // BN},{common.M // BM})",
                block=f"({TH},1,1)")
    return dict(kernel_src=kernel_src, full_src=full_src, info=info,
                deviations=deviations, smem=smem)


# ------------------------------------------------------------- artifacts ---
_REG_RE = re.compile(r"Used (\d+) registers")
_SPILL_RE = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
_STACK_RE = re.compile(r"(\d+) bytes stack frame")
_SMEM_RE = re.compile(r"(\d+) bytes smem")


def _side_compile(kernel_src: str, name: str) -> dict:
    """Untimed: nvcc -cubin -Xptxas=-v for registers/spills + a SASS-able cubin."""
    out = {}
    try:
        adir = os.path.join(common.ARTIFACTS_DIR, "cuda_noptx")
        os.makedirs(adir, exist_ok=True)
        cu = os.path.join(adir, name + ".cu")
        with open(cu, "w") as f:
            f.write(kernel_src)
        cubin = os.path.join(adir, name + ".cubin")
        nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.1"),
                            "bin", "nvcc")
        r = subprocess.run([nvcc, "-arch=sm_89", "-O3", "-cubin", "-Xptxas=-v",
                            "-o", cubin, cu],
                           capture_output=True, text=True, timeout=300)
        log = r.stdout + r.stderr
        out["ptxas_log"] = log.strip()
        out["cu_path"] = cu
        if r.returncode == 0:
            out["cubin_path"] = cubin
        m = _REG_RE.search(log)
        if m:
            out["n_regs"] = int(m.group(1))
        m = _SPILL_RE.search(log)
        out["n_spills"] = int(m.group(1)) if m else 0
        m = _STACK_RE.search(log)
        out["stack_frame_bytes"] = int(m.group(1)) if m else 0
        m = _SMEM_RE.search(log)
        if m:
            out["static_smem_bytes"] = int(m.group(1))
        # HMMA census straight from the SASS -- the variant-A guarantee
        if r.returncode == 0:
            dis = subprocess.run(
                [os.path.join(os.path.dirname(nvcc), "cuobjdump"), "-sass", cubin],
                capture_output=True, text=True, timeout=300)
            sass = dis.stdout
            out["sass_hmma_count"] = sass.count("HMMA")
            out["sass_imma_count"] = sass.count("IMMA")
            out["sass_ffma_count"] = sass.count("FFMA")
            out["sass_ldgsts_count"] = sass.count("LDGSTS")
            sp = os.path.join(adir, name + ".sass")
            with open(sp, "w") as f:
                f.write(sass)
            out["sass_path"] = sp
    except Exception as e:  # noqa: BLE001
        out["artifact_error"] = f"{type(e).__name__}: {e}"
    return out


# ------------------------------------------------------------------ build ---
def build(cfg: common.Config) -> common.Built:
    common.setup_cuda_env()
    gen = _make_source(cfg)

    name = ("noptx_%s_%dx%dx%d_t%d_kc%d_s%d_%s_%s"
            % (cfg.variant, cfg.BM, cfg.BN, cfg.BK, cfg.threads,
               cfg.kc, cfg.stages, cfg.arith, cfg.cast))

    t0 = time.perf_counter()
    mod = load_inline(
        name=name,
        cpp_sources="#include <torch/extension.h>\n"
                    "torch::Tensor gemm(torch::Tensor A, torch::Tensor B);",
        cuda_sources=gen["full_src"],
        functions=["gemm"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-Xptxas=-v",
                           "-gencode=arch=compute_89,code=sm_89"],
        verbose=False,
    )

    dtype = cfg.input_dtype
    kdtype = torch.float32 if cfg.arith == "fp32" else (
        torch.float32 if cfg.cast == "on_load" else torch.float16)

    if cfg.arith == "fp32" or cfg.cast != "in_region":
        def run(A, B):
            return mod.gemm(A, B)
    else:
        def run(A, B):                      # cast=in_region: pay it in the timer
            return mod.gemm(A.half(), B.half())

    # force CUDA module load / JIT-to-SASS inside the compile window
    a = torch.empty(common.M, common.K, device="cuda", dtype=kdtype)
    b = torch.empty(common.K, common.N, device="cuda", dtype=kdtype)
    mod.gemm(a, b)
    torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    art = {
        "cuda_source": gen["full_src"],
        "kernel_source": gen["kernel_src"],
        "shared_bytes": gen["smem"],
        "grid": gen["info"]["grid"],
        "block": gen["info"]["block"],
        "ext_name": name,
    }
    art.update(_side_compile(gen["kernel_src"], name))

    if cfg.arith == "fp32":
        how = ("scalar register-tiled SGEMM: acc[i][j] += a*b on floats, "
               "thread grid %s, thread tile %s, transposed As[k][m] smem, "
               "float4 global loads and float4 epilogue. No <mma.h> type is "
               "instantiated on this path." % (gen["info"]["thread_grid"],
                                               gen["info"]["thread_tile"]))
    else:
        how = ("nvcuda::wmma 16x16x16 half x half -> float accumulator; warp "
               "grid %s, warp tile %s (%s fragments/warp), smem ldm A=%d B=%d "
               "halfs; kc flush every %d BK-tiles into a plain float[%s][8] "
               "register array via fragment.x[] + fill_fragment(0); loads = %s"
               % (gen["info"]["warp_grid"], gen["info"]["warp_tile"],
                  gen["info"]["frags"], gen["info"].get("pad_halfs", 0) + cfg.BK,
                  gen["info"].get("pad_halfs", 0) + cfg.BN,
                  gen["info"]["kc_tiles"], gen["info"]["frags"],
                  ("__pipeline_memcpy_async x %d buffers"
                   % gen["info"]["stages_smem"]) if gen["info"]["use_async"]
                  else ("synchronous single-buffer" if gen["info"]["stages_smem"] == 1
                        else "multi-buffer converting loads (no cp.async: it cannot "
                             "convert fp32->fp16 in flight)")))
    art["backend_detail"] = how + " | zero inline PTX (asserted at build time)"
    art["geom_info"] = gen["info"]

    notes = "cuda_noptx / load_inline / no inline asm"
    if gen["deviations"]:
        notes += " | DEVIATION: " + "; ".join(gen["deviations"])

    return common.Built(run=run, compile_s=compile_s, input_dtype=dtype,
                        artifacts=art, notes=notes)
