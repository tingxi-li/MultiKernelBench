"""cuda_noptx lane of the Phase-2 SDPA study: hand-written CUDA C++, ZERO inline PTX.

Everything below is generated CUDA C that goes through `torch.utils.cpp_extension.
load_inline`. The tensor-core path is the `nvcuda::wmma` C++ API from `<mma.h>`
only -- no `asm()`, no `mma.sync`, no `cp.async` PTX. A build-time scan of the
generated source asserts it (the same check the fused lane uses).

Two algorithms, identical semantics (scale = 1/sqrt(d), no mask, no dropout):

  K3     three kernels, S materialized to global memory.
           k3_qk       S = Q@K^T * scale        -> (B,H,S,S) in `sdtype`
           k3_softmax  P = softmax(S, dim=-1)   -> (B,H,S,S) in `pdtype`
           k3_pv       O = P@V                  -> (B,H,S,d) fp32
  FLASH  one kernel, tiled online softmax with a running max / running sum;
         the score tile lives in registers (wmma accumulator) and shared memory
         and never reaches global memory. The output accumulator is fp32.

THE TWO DTYPE AXES
------------------
`sdtype` is the dtype the SCORE is kept in. In every arm the QK^T *operands* are
fp16 (that is what a tensor core takes) and the mma accumulator is fp32; sdtype
decides whether the scaled fp32 accumulator is then rounded to fp16 before the
softmax reads it. In K3 that rounding is also literally the dtype of the (B,H,S,S)
tensor in DRAM, so sdtype changes 1.07 GB of traffic into 0.54 GB. In FLASH the
score never leaves the SM, so sdtype is *only* an arithmetic change: the value is
put through `__float2half_rn` and back, while the shared-memory container stays
fp32 so that the smem footprint, the occupancy and the tile are bit-identical
across the three dtype pairs and nothing but the arithmetic moves.

`pdtype` is the dtype the PROBABILITIES are in when they feed the PV matmul.

  pdtype=fp16   P is rounded to fp16, V is converted fp32->fp16 on the
                global->shared path, and PV is `wmma::mma_sync` on fp16 tensor
                cores with an fp32 accumulator.
  pdtype=fp32   P stays fp32, so no fp16 tensor core can be used. This lane does
                NOT silently fall back to tf32: on sm_89 a tf32 tensor-core mma
                and the plain fp32 FFMA pipe have the same 91.1 TFLOP/s dense
                peak, so tf32 would cost 13 mantissa bits for no speed. The fp32
                arm therefore runs a classic register-tiled FFMA GEMM on the CUDA
                cores -- real fp32, recorded as such in `backend_detail`. No
                `wmma::precision::tf32` fragment is instantiated anywhere.

WHY THE OUTPUT ACCUMULATOR IS NOT A WMMA FRAGMENT
-------------------------------------------------
Online softmax has to multiply the running output accumulator by a per-ROW factor
`alpha = exp(m_old - m_new)`. The mapping from `wmma::fragment::x[e]` to (row,
col) is explicitly not part of the WMMA contract, so a per-row rescale cannot be
done in an accumulator fragment without hard-coding an undocumented lane layout.
FLASH therefore keeps O in a plain `float acc[ND][NPC]` register array with an
explicit (row, col) mapping, and each PV product is computed into a *fresh*
accumulator, `store_matrix_sync`-ed into a (Br, DT) fp32 shared-memory staging
tile, and folded in with `acc = acc*alpha + staged`. The fp32 (CUDA-core) PV path
writes the same staging tile, so the accumulator layout, the rescale and the
epilogue are byte-identical between the two pdtypes and only the inner product
differs. The same staging tile is what lets the row max / row sum -- also per-row
quantities WMMA does not expose -- be taken over an explicitly indexed smem tile.

GLOBAL -> SHARED
----------------
Q/K/V arrive fp32 and the tensor cores want fp16, and `__pipeline_memcpy_async`
(the intrinsic form of cp.async, which is what this lane is allowed to use) can
only move bytes, not convert. So the fp16 paths load `float4` and store
`__floats2half2_rn` pairs instead. That is a real, reportable consequence of the
no-PTX rule combined with an fp32-in benchmark, not an oversight.
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

import common
import common2

_P1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "phase1_matmul")
if _P1 not in sys.path:
    sys.path.insert(0, _P1)
from variants import cuda_noptx_gemm as p1  # noqa: E402

SMEM_LIMIT = 101376          # sm_89 opt-in dynamic shared memory per block


# --------------------------------------------------------------- geometry ---
def flash_tile(d: int) -> dict:
    """(Br, Bc, DT, threads) for FLASH, per head dim. Fixed constants, no search.

    Br is what the (Br, d) fp32 output accumulator costs in registers:
    Br*d/threads floats per thread, so Br=64 at d=1024 would need 256 registers
    for the accumulator alone and spill. DT is pinned to 16*nwarps = 128 because
    the PV stage gives each warp exactly one 16-wide column tile.
    """
    threads = 256
    Br = 32 if d >= 1024 else 64
    return dict(Br=Br, Bc=64, DT=128, threads=threads)


def k3_tile(d: int) -> dict:
    """(BM, BN, BK, threads) for all three K3 kernels, per head dim.

    Flat across d: the per-d shrink in common2.SDPA_TILES exists only because of
    the flash output accumulator, which K3 does not carry -- its three kernels
    are ordinary GEMMs and a softmax.
    """
    return dict(BM=64, BN=64, BK=64, threads=128)


def _flash_smem(d: int, pd_fp16: bool) -> dict:
    t = flash_tile(d)
    Br, Bc, DT = t["Br"], t["Bc"], t["DT"]
    LDQ = DT + 8
    LDK = DT + 8
    LDS = Bc + 4
    LDP = Bc + 8
    LDO = DT + 4
    LDVF = DT + 4
    qk_ab = Br * LDQ * 2 + Bc * LDK * 2          # Qs + Ks
    v_bytes = Bc * LDK * 2 if pd_fp16 else Bc * LDVF * 4
    qk_region = max(qk_ab, v_bytes)
    qk_region = (qk_region + 15) // 16 * 16
    off_s = qk_region
    sz_s = Br * LDS * 4
    off_p = off_s + sz_s
    sz_p = Br * LDP * 2 if pd_fp16 else 0
    off_o = (off_p + sz_p + 15) // 16 * 16
    sz_o = Br * LDO * 4
    off_row = off_o + sz_o
    total = off_row + 3 * Br * 4
    total = (total + 15) // 16 * 16
    return dict(LDQ=LDQ, LDK=LDK, LDS=LDS, LDP=LDP, LDO=LDO, LDVF=LDVF,
                OFF_Q=0, OFF_K=Br * LDQ * 2, OFF_V=0, OFF_S=off_s,
                OFF_P=off_p, OFF_OST=off_o, OFF_ROW=off_row, SMEM=total)


# ------------------------------------------------------------ CUDA source ---
_HEADER = r"""
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define H_      %(H)d
#define S_      %(S)d
#define D_      %(D)d
#define SCALE   %(SCALE).10ef
#define SD_FP16 %(SD)d
#define PD_FP16 %(PD)d

#if SD_FP16
typedef half  sdt;
#else
typedef float sdt;
#endif
#if PD_FP16
typedef half  pdt;
#else
typedef float pdt;
#endif
"""

_FLASH = r"""
/* ===================== FLASH: one kernel, online softmax ================== */
#define BR      %(BR)d
#define BC      %(BC)d
#define DT      %(DT)d
#define THREADS %(THREADS)d
#define ND      (D_ / DT)
#define NWARPS  (THREADS / 32)
#define MF      (BR / 16)
#define NT      (BC / 16)
#define QK_TPW  ((MF * NT) / NWARPS)
#define KSTEPS  (DT / 16)
#define NKV     (S_ / BC)
#define NPC     ((BR * DT) / THREADS)
#define TPR     (THREADS / BR)
#define EPT     (BC / TPR)
#define PVTN    4
#define PVTHN   (DT / PVTN)
#define PVTHM   (THREADS / PVTHN)
#define PVTM    (BR / PVTHM)

#define LDQ     %(LDQ)d
#define LDK     %(LDK)d
#define LDS     %(LDS)d
#define LDP     %(LDP)d
#define LDO     %(LDO)d
#define LDVF    %(LDVF)d
#define OFF_Q   %(OFF_Q)d
#define OFF_K   %(OFF_K)d
#define OFF_V   %(OFF_V)d
#define OFF_S   %(OFF_S)d
#define OFF_P   %(OFF_P)d
#define OFF_OST %(OFF_OST)d
#define OFF_ROW %(OFF_ROW)d
#define FSMEM   %(SMEM)d

/* global (rows x DT) fp32 tile -> shared half tile, ld = LD.
   __pipeline_memcpy_async cannot convert fp32->fp16, so this is a vectorized
   float4 read plus __floats2half2_rn stores. */
template<int ROWS, int LD>
__device__ __forceinline__ void ld_half_tile(const float* __restrict__ g,
                                             half* __restrict__ s, int tid) {
    constexpr int VPR = DT / 4;
    constexpr int NV  = ROWS * VPR;
#pragma unroll 4
    for (int t = tid; t < NV; t += THREADS) {
        const int r = t / VPR;
        const int c = (t - r * VPR) * 4;
        const float4 x = *reinterpret_cast<const float4*>(g + (size_t)r * D_ + c);
        __half2* dst = reinterpret_cast<__half2*>(s + r * LD + c);
        dst[0] = __floats2half2_rn(x.x, x.y);
        dst[1] = __floats2half2_rn(x.z, x.w);
    }
}

/* global (rows x DT) fp32 tile -> shared fp32 tile (the pdtype=fp32 V path). */
template<int ROWS, int LD>
__device__ __forceinline__ void ld_float_tile(const float* __restrict__ g,
                                              float* __restrict__ s, int tid) {
    constexpr int VPR = DT / 4;
    constexpr int NV  = ROWS * VPR;
#pragma unroll 4
    for (int t = tid; t < NV; t += THREADS) {
        const int r = t / VPR;
        const int c = (t - r * VPR) * 4;
        *reinterpret_cast<float4*>(s + r * LD + c) =
            *reinterpret_cast<const float4*>(g + (size_t)r * D_ + c);
    }
}

__global__ __launch_bounds__(THREADS)
void flash_kernel(const float* __restrict__ Qg,
                  const float* __restrict__ Kg,
                  const float* __restrict__ Vg,
                  float* __restrict__ Og) {
    extern __shared__ __align__(16) char smem[];
    half*  Qs  = reinterpret_cast<half*>(smem + OFF_Q);
    half*  Ks  = reinterpret_cast<half*>(smem + OFF_K);
#if PD_FP16
    half*  Vs  = reinterpret_cast<half*>(smem + OFF_V);
    half*  Ps  = reinterpret_cast<half*>(smem + OFF_P);
#else
    float* Vf  = reinterpret_cast<float*>(smem + OFF_V);
#endif
    float* Ss  = reinterpret_cast<float*>(smem + OFF_S);
    float* Ost = reinterpret_cast<float*>(smem + OFF_OST);
    float* Mr  = reinterpret_cast<float*>(smem + OFF_ROW);
    float* Lr  = Mr + BR;
    float* Ar  = Lr + BR;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int bh   = blockIdx.z * H_ + blockIdx.y;
    const int rb   = blockIdx.x * BR;

    const size_t base = (size_t)bh * S_ * D_;
    const float* __restrict__ Qb = Qg + base + (size_t)rb * D_;
    const float* __restrict__ Kb = Kg + base;
    const float* __restrict__ Vb = Vg + base;
    float* __restrict__       Ob = Og + base + (size_t)rb * D_;

    /* the (Br, d) fp32 output accumulator, in registers with an explicit map:
       element n of d-chunk t belongs to (row, col) = (g/DT, g mod DT),
       with g = n*THREADS + tid */
    float acc[ND][NPC];
#pragma unroll
    for (int t = 0; t < ND; ++t)
#pragma unroll
        for (int n = 0; n < NPC; ++n) acc[t][n] = 0.0f;

    if (tid < BR) { Mr[tid] = -1e30f; Lr[tid] = 0.0f; }
    __syncthreads();

    for (int kv = 0; kv < NKV; ++kv) {
        const size_t koff = (size_t)kv * BC * D_;

        /* ---- S = Q K^T over all of d, fp16 operands, fp32 accumulator ---- */
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accs[QK_TPW];
#pragma unroll
        for (int p = 0; p < QK_TPW; ++p) wmma::fill_fragment(accs[p], 0.0f);

#pragma unroll 1
        for (int dc = 0; dc < ND; ++dc) {
            __syncthreads();
            ld_half_tile<BR, LDQ>(Qb + dc * DT, Qs, tid);
            ld_half_tile<BC, LDK>(Kb + koff + dc * DT, Ks, tid);
            __syncthreads();
#pragma unroll
            for (int p = 0; p < QK_TPW; ++p) {
                const int t  = warp + p * NWARPS;
                const int mi = t / NT, ni = t - (t / NT) * NT;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
#pragma unroll
                for (int kk = 0; kk < KSTEPS; ++kk) {
                    wmma::load_matrix_sync(af, Qs + mi * 16 * LDQ + kk * 16, LDQ);
                    /* col_major with ld=LDK reads B[k][n] = Ks[n*LDK + k] = K^T */
                    wmma::load_matrix_sync(bf, Ks + ni * 16 * LDK + kk * 16, LDK);
                    wmma::mma_sync(accs[p], af, bf, accs[p]);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < QK_TPW; ++p) {
            const int t  = warp + p * NWARPS;
            const int mi = t / NT, ni = t - (t / NT) * NT;
            wmma::store_matrix_sync(Ss + mi * 16 * LDS + ni * 16, accs[p], LDS,
                                    wmma::mem_row_major);
        }
        __syncthreads();

        /* ---- online softmax over the (BR, BC) score tile ------------------ */
        {
            const int r  = tid / TPR;
            const int c0 = (tid - r * TPR) * EPT;
            float sv[EPT];
            float mcur = -1e30f;
#pragma unroll
            for (int j = 0; j < EPT; ++j) {
                float v = Ss[r * LDS + c0 + j] * SCALE;
#if SD_FP16
                v = __half2float(__float2half_rn(v));   /* sdtype = fp16 */
#endif
                sv[j] = v;
                mcur = fmaxf(mcur, v);
            }
#pragma unroll
            for (int off = TPR / 2; off > 0; off >>= 1)
                mcur = fmaxf(mcur, __shfl_xor_sync(0xffffffffu, mcur, off));
            const float mprev = Mr[r];
            const float mnew  = fmaxf(mprev, mcur);
            const float alpha = __expf(mprev - mnew);   /* 0 on the first block */
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < EPT; ++j) {
                const float e = __expf(sv[j] - mnew);
                sv[j] = e;
                sum += e;
            }
#pragma unroll
            for (int off = TPR / 2; off > 0; off >>= 1)
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            if (c0 == 0) { Mr[r] = mnew; Lr[r] = Lr[r] * alpha + sum; Ar[r] = alpha; }
#pragma unroll
            for (int j = 0; j < EPT; ++j) {
#if PD_FP16
                Ps[r * LDP + c0 + j] = __float2half_rn(sv[j]);  /* pdtype = fp16 */
#else
                Ss[r * LDS + c0 + j] = sv[j];                   /* pdtype = fp32 */
#endif
            }
        }
        __syncthreads();

        /* ---- O = O*alpha + P V, one d-chunk at a time --------------------- */
#pragma unroll
        for (int dc = 0; dc < ND; ++dc) {
#if PD_FP16
            ld_half_tile<BC, LDK>(Vb + koff + dc * DT, Vs, tid);
            __syncthreads();
#pragma unroll
            for (int mi = 0; mi < MF; ++mi) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> o;
                wmma::fill_fragment(o, 0.0f);
#pragma unroll
                for (int kk = 0; kk < NT; ++kk) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
                    wmma::load_matrix_sync(pf, Ps + mi * 16 * LDP + kk * 16, LDP);
                    wmma::load_matrix_sync(vf, Vs + kk * 16 * LDK + warp * 16, LDK);
                    wmma::mma_sync(o, pf, vf, o);
                }
                wmma::store_matrix_sync(Ost + mi * 16 * LDO + warp * 16, o, LDO,
                                        wmma::mem_row_major);
            }
#else
            ld_float_tile<BC, LDVF>(Vb + koff + dc * DT, Vf, tid);
            __syncthreads();
            {   /* register-tiled fp32 FFMA GEMM on the CUDA cores */
                const int tn = tid & (PVTHN - 1);
                const int tm = tid / PVTHN;
                float cr[PVTM][PVTN];
#pragma unroll
                for (int i = 0; i < PVTM; ++i)
#pragma unroll
                    for (int j = 0; j < PVTN; ++j) cr[i][j] = 0.0f;
#pragma unroll 4
                for (int kk = 0; kk < BC; ++kk) {
                    float a[PVTM];
#pragma unroll
                    for (int i = 0; i < PVTM; ++i)
                        a[i] = Ss[(tm * PVTM + i) * LDS + kk];
                    const float4 bv =
                        *reinterpret_cast<const float4*>(Vf + kk * LDVF + tn * PVTN);
                    const float b[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                    for (int i = 0; i < PVTM; ++i)
#pragma unroll
                        for (int j = 0; j < PVTN; ++j) cr[i][j] += a[i] * b[j];
                }
#pragma unroll
                for (int i = 0; i < PVTM; ++i)
                    *reinterpret_cast<float4*>(Ost + (tm * PVTM + i) * LDO + tn * PVTN) =
                        make_float4(cr[i][0], cr[i][1], cr[i][2], cr[i][3]);
            }
#endif
            __syncthreads();
#pragma unroll
            for (int n = 0; n < NPC; ++n) {
                const int g = n * THREADS + tid;
                const int r = g / DT;
                const int c = g - r * DT;
                acc[dc][n] = acc[dc][n] * Ar[r] + Ost[r * LDO + c];
            }
            __syncthreads();
        }
    }

    if (tid < BR) Lr[tid] = 1.0f / Lr[tid];
    __syncthreads();
#pragma unroll
    for (int t = 0; t < ND; ++t) {
#pragma unroll
        for (int n = 0; n < NPC; ++n) {
            const int g = n * THREADS + tid;
            const int r = g / DT;
            const int c = g - r * DT;
            Ob[(size_t)r * D_ + t * DT + c] = acc[t][n] * Lr[r];
        }
    }
}
"""

_K3 = r"""
/* ======================= K3: three kernels, S materialized ================ */
#define G3M     %(BM)d
#define G3N     %(BN)d
#define G3K     %(BK)d
#define G3T     %(THREADS)d
#define G3W     (G3T / 32)
#define G3WM    2
#define G3WN    (G3W / G3WM)
#define G3TM    (G3M / G3WM)
#define G3TN    (G3N / G3WN)
#define G3FM    (G3TM / 16)
#define G3FN    (G3TN / 16)
#define LDA3    (G3K + 8)
#define LDB3    (G3K + 8)
#define LDC3    (G3N + 4)
#define QK_SMEM %(QK_SMEM)d

/* ---- kernel 1: S = Q K^T * scale, materialized in sdtype ---------------- */
__global__ __launch_bounds__(G3T)
void k3_qk_kernel(const float* __restrict__ Qg, const float* __restrict__ Kg,
                  sdt* __restrict__ Sg) {
    extern __shared__ __align__(16) char sm[];
    half* As = reinterpret_cast<half*>(sm);
    half* Bs = As + G3M * LDA3;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = warp / G3WN;
    const int wn   = warp - wm * G3WN;
    const int bh   = blockIdx.z;
    const int m0   = blockIdx.y * G3M;
    const int n0   = blockIdx.x * G3N;
    const float* __restrict__ Qb = Qg + (size_t)bh * S_ * D_ + (size_t)m0 * D_;
    const float* __restrict__ Kb = Kg + (size_t)bh * S_ * D_ + (size_t)n0 * D_;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[G3FM][G3FN];
#pragma unroll
    for (int i = 0; i < G3FM; ++i)
#pragma unroll
        for (int j = 0; j < G3FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    constexpr int VPR = G3K / 4;
    for (int k0 = 0; k0 < D_; k0 += G3K) {
        __syncthreads();
#pragma unroll 4
        for (int t = tid; t < G3M * VPR; t += G3T) {
            const int r = t / VPR, c = (t - (t / VPR) * VPR) * 4;
            const float4 x = *reinterpret_cast<const float4*>(Qb + (size_t)r * D_ + k0 + c);
            __half2* d = reinterpret_cast<__half2*>(As + r * LDA3 + c);
            d[0] = __floats2half2_rn(x.x, x.y);
            d[1] = __floats2half2_rn(x.z, x.w);
        }
#pragma unroll 4
        for (int t = tid; t < G3N * VPR; t += G3T) {
            const int r = t / VPR, c = (t - (t / VPR) * VPR) * 4;
            const float4 x = *reinterpret_cast<const float4*>(Kb + (size_t)r * D_ + k0 + c);
            __half2* d = reinterpret_cast<__half2*>(Bs + r * LDB3 + c);
            d[0] = __floats2half2_rn(x.x, x.y);
            d[1] = __floats2half2_rn(x.z, x.w);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < G3K / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[G3FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[G3FN];
#pragma unroll
            for (int i = 0; i < G3FM; ++i)
                wmma::load_matrix_sync(af[i], As + (wm * G3TM + i * 16) * LDA3 + kk * 16, LDA3);
#pragma unroll
            for (int j = 0; j < G3FN; ++j)
                wmma::load_matrix_sync(bf[j], Bs + (wn * G3TN + j * 16) * LDB3 + kk * 16, LDB3);
#pragma unroll
            for (int i = 0; i < G3FM; ++i)
#pragma unroll
                for (int j = 0; j < G3FN; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
    }

    __syncthreads();
    float* Cs = reinterpret_cast<float*>(sm);
#pragma unroll
    for (int i = 0; i < G3FM; ++i)
#pragma unroll
        for (int j = 0; j < G3FN; ++j)
            wmma::store_matrix_sync(Cs + (wm * G3TM + i * 16) * LDC3 + wn * G3TN + j * 16,
                                    acc[i][j], LDC3, wmma::mem_row_major);
    __syncthreads();
    sdt* __restrict__ Sb = Sg + (size_t)bh * S_ * S_ + (size_t)m0 * S_ + n0;
    for (int g = tid; g < G3M * G3N; g += G3T) {
        const int r = g / G3N, c = g - r * G3N;
        const float v = Cs[r * LDC3 + c] * SCALE;
#if SD_FP16
        Sb[(size_t)r * S_ + c] = __float2half_rn(v);
#else
        Sb[(size_t)r * S_ + c] = v;
#endif
    }
}

/* ---- kernel 2: P = softmax(S, dim=-1), materialized in pdtype ----------- */
#define SM3T   128
#define SM3EPT (S_ / SM3T)
#define SM3W   (SM3T / 32)

__global__ __launch_bounds__(SM3T)
void k3_softmax_kernel(const sdt* __restrict__ X, pdt* __restrict__ Y) {
    __shared__ float sm_m[SM3W];
    __shared__ float sm_s[SM3W];
    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    const size_t row = (size_t)blockIdx.x;
    const sdt* __restrict__ xr = X + row * S_;
    pdt* __restrict__ yr = Y + row * S_;

    float e[SM3EPT];
    float m = -3.402823466e+38f;
#pragma unroll
    for (int k = 0; k < SM3EPT; ++k) {
        /* torch's build adds -D__CUDA_NO_HALF_CONVERSIONS__, so the half->float
           widening has to be spelled out rather than left to the implicit
           conversion operator on __half. */
#if SD_FP16
        e[k] = __half2float(xr[tid * SM3EPT + k]);
#else
        e[k] = xr[tid * SM3EPT + k];
#endif
        m = fmaxf(m, e[k]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if (lane == 0) sm_m[wid] = m;
    __syncthreads();
#pragma unroll
    for (int s = SM3W >> 1; s > 0; s >>= 1) {
        if (tid < s) sm_m[tid] = fmaxf(sm_m[tid], sm_m[tid + s]);
        __syncthreads();
    }
    const float rmax = sm_m[0];

    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < SM3EPT; ++k) { e[k] = __expf(e[k] - rmax); sum += e[k]; }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) sm_s[wid] = sum;
    __syncthreads();
#pragma unroll
    for (int s = SM3W >> 1; s > 0; s >>= 1) {
        if (tid < s) sm_s[tid] = sm_s[tid] + sm_s[tid + s];
        __syncthreads();
    }
    const float inv = 1.0f / sm_s[0];
#pragma unroll
    for (int k = 0; k < SM3EPT; ++k) {
#if PD_FP16
        yr[tid * SM3EPT + k] = __float2half_rn(e[k] * inv);
#else
        yr[tid * SM3EPT + k] = e[k] * inv;
#endif
    }
}

/* ---- kernel 3: O = P V, fp32 out --------------------------------------- */
#define P3M   64
#define P3T   128
#if PD_FP16
#define P3N   64
#define P3K   64
#define LDPA  (P3K + 8)
#define LDPB  (P3N + 8)
#define P3WM  2
#define P3WN  ((P3T / 32) / P3WM)
#define P3TM  (P3M / P3WM)
#define P3TN  (P3N / P3WN)
#define P3FM  (P3TM / 16)
#define P3FN  (P3TN / 16)
#define PV_SMEM (P3M * LDPA * 2 + P3K * LDPB * 2)

__global__ __launch_bounds__(P3T)
void k3_pv_kernel(const half* __restrict__ Pg, const float* __restrict__ Vg,
                  float* __restrict__ Og) {
    extern __shared__ __align__(16) char sm[];
    half* As = reinterpret_cast<half*>(sm);
    half* Bs = As + P3M * LDPA;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = warp / P3WN;
    const int wn   = warp - wm * P3WN;
    const int bh   = blockIdx.z;
    const int m0   = blockIdx.y * P3M;
    const int n0   = blockIdx.x * P3N;
    const half*  __restrict__ Pb = Pg + (size_t)bh * S_ * S_ + (size_t)m0 * S_;
    const float* __restrict__ Vb = Vg + (size_t)bh * S_ * D_ + n0;
    float* __restrict__ Ob = Og + (size_t)bh * S_ * D_ + (size_t)m0 * D_ + n0;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[P3FM][P3FN];
#pragma unroll
    for (int i = 0; i < P3FM; ++i)
#pragma unroll
        for (int j = 0; j < P3FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    constexpr int APR = P3K / 8;          /* 16B (8 half) vectors per A row */
    constexpr int BPR = P3N / 4;          /* float4 per B row */
    for (int k0 = 0; k0 < S_; k0 += P3K) {
        __syncthreads();
#pragma unroll 4
        for (int t = tid; t < P3M * APR; t += P3T) {
            const int r = t / APR, c = (t - (t / APR) * APR) * 8;
            *reinterpret_cast<float4*>(As + r * LDPA + c) =
                *reinterpret_cast<const float4*>(Pb + (size_t)r * S_ + k0 + c);
        }
#pragma unroll 4
        for (int t = tid; t < P3K * BPR; t += P3T) {
            const int r = t / BPR, c = (t - (t / BPR) * BPR) * 4;
            const float4 x = *reinterpret_cast<const float4*>(Vb + (size_t)(k0 + r) * D_ + c);
            __half2* d = reinterpret_cast<__half2*>(Bs + r * LDPB + c);
            d[0] = __floats2half2_rn(x.x, x.y);
            d[1] = __floats2half2_rn(x.z, x.w);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < P3K / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[P3FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[P3FN];
#pragma unroll
            for (int i = 0; i < P3FM; ++i)
                wmma::load_matrix_sync(af[i], As + (wm * P3TM + i * 16) * LDPA + kk * 16, LDPA);
#pragma unroll
            for (int j = 0; j < P3FN; ++j)
                wmma::load_matrix_sync(bf[j], Bs + kk * 16 * LDPB + wn * P3TN + j * 16, LDPB);
#pragma unroll
            for (int i = 0; i < P3FM; ++i)
#pragma unroll
                for (int j = 0; j < P3FN; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < P3FM; ++i)
#pragma unroll
        for (int j = 0; j < P3FN; ++j)
            wmma::store_matrix_sync(Ob + (size_t)(wm * P3TM + i * 16) * D_ + wn * P3TN + j * 16,
                                    acc[i][j], D_, wmma::mem_row_major);
}

#else   /* ---- pdtype = fp32: real fp32 FFMA on the CUDA cores ------------ */
#define P3N   64
#define P3K   32
#define LDPA  (P3K + 4)
#define LDPB  (P3N + 4)
#define P3RN  4
#define P3THN (P3N / P3RN)
#define P3THM (P3T / P3THN)
#define P3RM  (P3M / P3THM)
#define PV_SMEM (P3M * LDPA * 4 + P3K * LDPB * 4)

__global__ __launch_bounds__(P3T)
void k3_pv_kernel(const float* __restrict__ Pg, const float* __restrict__ Vg,
                  float* __restrict__ Og) {
    extern __shared__ __align__(16) char sm[];
    float* As = reinterpret_cast<float*>(sm);
    float* Bs = As + P3M * LDPA;

    const int tid = threadIdx.x;
    const int tn  = tid & (P3THN - 1);
    const int tm  = tid / P3THN;
    const int bh  = blockIdx.z;
    const int m0  = blockIdx.y * P3M;
    const int n0  = blockIdx.x * P3N;
    const float* __restrict__ Pb = Pg + (size_t)bh * S_ * S_ + (size_t)m0 * S_;
    const float* __restrict__ Vb = Vg + (size_t)bh * S_ * D_ + n0;
    float* __restrict__ Ob = Og + (size_t)bh * S_ * D_ + (size_t)m0 * D_ + n0;

    float cr[P3RM][P3RN];
#pragma unroll
    for (int i = 0; i < P3RM; ++i)
#pragma unroll
        for (int j = 0; j < P3RN; ++j) cr[i][j] = 0.0f;

    constexpr int APR = P3K / 4;
    constexpr int BPR = P3N / 4;
    for (int k0 = 0; k0 < S_; k0 += P3K) {
        __syncthreads();
#pragma unroll 4
        for (int t = tid; t < P3M * APR; t += P3T) {
            const int r = t / APR, c = (t - (t / APR) * APR) * 4;
            *reinterpret_cast<float4*>(As + r * LDPA + c) =
                *reinterpret_cast<const float4*>(Pb + (size_t)r * S_ + k0 + c);
        }
#pragma unroll 4
        for (int t = tid; t < P3K * BPR; t += P3T) {
            const int r = t / BPR, c = (t - (t / BPR) * BPR) * 4;
            *reinterpret_cast<float4*>(Bs + r * LDPB + c) =
                *reinterpret_cast<const float4*>(Vb + (size_t)(k0 + r) * D_ + c);
        }
        __syncthreads();
#pragma unroll 4
        for (int kk = 0; kk < P3K; ++kk) {
            float a[P3RM];
#pragma unroll
            for (int i = 0; i < P3RM; ++i) a[i] = As[(tm * P3RM + i) * LDPA + kk];
            const float4 bv = *reinterpret_cast<const float4*>(Bs + kk * LDPB + tn * P3RN);
            const float b[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < P3RM; ++i)
#pragma unroll
                for (int j = 0; j < P3RN; ++j) cr[i][j] += a[i] * b[j];
        }
    }
#pragma unroll
    for (int i = 0; i < P3RM; ++i)
        *reinterpret_cast<float4*>(Ob + (size_t)(tm * P3RM + i) * D_ + tn * P3RN) =
            make_float4(cr[i][0], cr[i][1], cr[i][2], cr[i][3]);
}
#endif
"""

_WRAP_FLASH = r"""
torch::Tensor flash(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be fp32");
    TORCH_CHECK(q.size(1) == H_ && q.size(2) == S_ && q.size(3) == D_, "shape");
    const int Bn = (int)q.size(0);
    auto o = torch::empty_like(q);

    static bool attr_done = false;
    if (!attr_done) {
        cudaError_t e = cudaFuncSetAttribute((const void*)flash_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FSMEM);
        TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute(", FSMEM, "): ",
                    cudaGetErrorString(e));
        attr_done = true;
    }
    dim3 grid(S_ / BR, H_, Bn);
    auto stream = at::cuda::getCurrentCUDAStream();
    flash_kernel<<<grid, THREADS, FSMEM, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        o.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
"""

_WRAP_K3 = r"""
torch::Tensor k3_qk(torch::Tensor q, torch::Tensor k) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kFloat32, "q");
    TORCH_CHECK(q.size(1) == H_ && q.size(2) == S_ && q.size(3) == D_, "shape");
    const int Bn = (int)q.size(0);
    auto opts = torch::TensorOptions()
        .dtype(SD_FP16 ? torch::kHalf : torch::kFloat32).device(q.device());
    auto s = torch::empty({Bn, H_, S_, S_}, opts);
    static bool a1 = false;
    if (!a1) {
        cudaError_t e = cudaFuncSetAttribute((const void*)k3_qk_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, QK_SMEM);
        TORCH_CHECK(e == cudaSuccess, "setattr qk: ", cudaGetErrorString(e));
        a1 = true;
    }
    dim3 grid(S_ / G3N, S_ / G3M, Bn * H_);
    auto stream = at::cuda::getCurrentCUDAStream();
    k3_qk_kernel<<<grid, G3T, QK_SMEM, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(),
        reinterpret_cast<sdt*>(s.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return s;
}

torch::Tensor k3_softmax(torch::Tensor s) {
    TORCH_CHECK(s.is_cuda() && s.is_contiguous(), "s");
    const int Bn = (int)s.size(0);
    auto opts = torch::TensorOptions()
        .dtype(PD_FP16 ? torch::kHalf : torch::kFloat32).device(s.device());
    auto p = torch::empty({Bn, H_, S_, S_}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();
    k3_softmax_kernel<<<(unsigned)((size_t)Bn * H_ * S_), SM3T, 0, stream>>>(
        reinterpret_cast<const sdt*>(s.data_ptr()),
        reinterpret_cast<pdt*>(p.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return p;
}

torch::Tensor k3_pv(torch::Tensor p, torch::Tensor v) {
    TORCH_CHECK(p.is_cuda() && p.is_contiguous() && v.is_contiguous(), "p/v");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be fp32");
    const int Bn = (int)p.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(p.device());
    auto o = torch::empty({Bn, H_, S_, D_}, opts);
    static bool a2 = false;
    if (!a2) {
        cudaError_t e = cudaFuncSetAttribute((const void*)k3_pv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PV_SMEM);
        TORCH_CHECK(e == cudaSuccess, "setattr pv: ", cudaGetErrorString(e));
        a2 = true;
    }
    dim3 grid(D_ / P3N, S_ / P3M, Bn * H_);
    auto stream = at::cuda::getCurrentCUDAStream();
    k3_pv_kernel<<<grid, P3T, PV_SMEM, stream>>>(
        reinterpret_cast<const pdt*>(p.data_ptr()), v.data_ptr<float>(),
        o.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
"""

_WRAP_HEAD = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
"""

_CPP_FLASH = ("#include <torch/extension.h>\n"
              "torch::Tensor flash(torch::Tensor q, torch::Tensor k, torch::Tensor v);")
_CPP_K3 = ("#include <torch/extension.h>\n"
           "torch::Tensor k3_qk(torch::Tensor q, torch::Tensor k);\n"
           "torch::Tensor k3_softmax(torch::Tensor s);\n"
           "torch::Tensor k3_pv(torch::Tensor p, torch::Tensor v);")

_BAD_ASM = ("asm(", "asm (", "asm volatile", "__asm")


# ------------------------------------------------------------------ build ---
def _gen(cfg):
    d = int(cfg.extra["d"])
    sd = cfg.extra.get("sdtype", "fp32")
    pd = cfg.extra.get("pdtype", "fp32")
    algo = cfg.variant
    if algo not in ("K3", "FLASH"):
        raise KeyError(f"cuda_noptx sdpa lane implements K3 and FLASH; got {algo!r}")

    head = _HEADER % dict(H=common2.S_H, S=common2.S_S, D=d,
                          SCALE=1.0 / math.sqrt(d),
                          SD=int(sd == "fp16"), PD=int(pd == "fp16"))
    if algo == "FLASH":
        t = flash_tile(d)
        sm = _flash_smem(d, pd == "fp16")
        if sm["SMEM"] > SMEM_LIMIT:
            raise RuntimeError(f"FLASH smem {sm['SMEM']} > {SMEM_LIMIT}")
        body = _FLASH % dict(BR=t["Br"], BC=t["Bc"], DT=t["DT"],
                             THREADS=t["threads"], **sm)
        wrap = _WRAP_FLASH
        cpp = _CPP_FLASH
        funcs = ["flash"]
        tile = t
        info = dict(smem_bytes=sm["SMEM"], grid=[common2.S_S // t["Br"],
                                                 common2.S_H, common2.S_B],
                    block=[t["threads"], 1, 1])
    else:
        t = k3_tile(d)
        qk_smem = max(t["BM"] * (t["BK"] + 8) * 2 + t["BN"] * (t["BK"] + 8) * 2,
                      t["BM"] * (t["BN"] + 4) * 4)
        body = _K3 % dict(BM=t["BM"], BN=t["BN"], BK=t["BK"],
                          THREADS=t["threads"], QK_SMEM=qk_smem)
        wrap = _WRAP_K3
        cpp = _CPP_K3
        funcs = ["k3_qk", "k3_softmax", "k3_pv"]
        pv_smem = (64 * 72 * 2 + 64 * 72 * 2) if pd == "fp16" else (64 * 36 * 4 + 32 * 68 * 4)
        tile = t
        info = dict(smem_bytes=max(qk_smem, pv_smem),
                    grid=[common2.S_S // t["BN"], common2.S_S // t["BM"],
                          common2.S_B * common2.S_H],
                    block=[t["threads"], 1, 1])

    kernel_src = head + body
    full = kernel_src + _WRAP_HEAD + wrap
    for bad in _BAD_ASM:
        if bad in full:
            raise RuntimeError(f"inline asm {bad!r} found in cuda_noptx source")
    return dict(kernel_src=kernel_src, full=full, cpp=cpp, functions=funcs,
                tile=tile, info=info, d=d, sd=sd, pd=pd, algo=algo)


def build(cfg) -> common2.Built2:
    common.setup_cuda_env()
    gen = _gen(cfg)
    d, sd, pd, algo = gen["d"], gen["sd"], gen["pd"], gen["algo"]
    t = gen["tile"]

    if algo == "FLASH":
        name = "p2sdpa_noptx_flash_d%d_%s_%s_br%d_bc%d_dt%d_t%d" % (
            d, sd, pd, t["Br"], t["Bc"], t["DT"], t["threads"])
    else:
        name = "p2sdpa_noptx_k3_d%d_%s_%s_%dx%dx%d_t%d" % (
            d, sd, pd, t["BM"], t["BN"], t["BK"], t["threads"])

    t0 = time.perf_counter()
    mod = load_inline(
        name=name, cpp_sources=gen["cpp"], cuda_sources=gen["full"],
        functions=gen["functions"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-Xptxas=-v",
                           "-gencode=arch=compute_89,code=sm_89"],
        verbose=False)

    if algo == "FLASH":
        def run(q, k, v):
            return mod.flash(q, k, v)
    else:
        def run(q, k, v):
            return mod.k3_pv(mod.k3_softmax(mod.k3_qk(q, k)), v)

    # Force module load + first launch inside the compile window, by calling the
    # KERNELS directly rather than `run`. The batch dimension is a runtime grid
    # dimension (H, S, d are compiled in), so a B=1 warm-up exercises exactly the
    # same code with 1/32nd of the memory.
    dev = "cuda"
    wq = torch.zeros((1, common2.S_H, common2.S_S, d), dtype=torch.float32, device=dev)
    if algo == "FLASH":
        _ = mod.flash(wq, wq, wq)
        del _
    else:
        ws = mod.k3_qk(wq, wq)
        wp = mod.k3_softmax(ws)
        _ = mod.k3_pv(wp, wq)
        del ws, wp, _
    torch.cuda.synchronize()
    del wq
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    n_kernels = 1 if algo == "FLASH" else 3
    if algo == "FLASH":
        sm = _flash_smem(d, pd == "fp16")
        pvtxt = (
            "PV = wmma 16x16x16 half x half -> fp32 accumulator (fp16 tensor "
            "cores), V converted fp32->fp16 on the global->shared path"
            if pd == "fp16" else
            "PV = register-tiled FFMA GEMM on the CUDA cores in TRUE fp32 "
            "(P fp32 x V fp32, %dx%d thread tile). No tf32: no "
            "wmma::precision::tf32 fragment is instantiated, because on sm_89 "
            "tf32 tensor and fp32 FFMA share the same 91.1 TFLOP/s dense peak, "
            "so tf32 would cost 13 mantissa bits for no throughput"
            % (t["Br"] // (t["threads"] // (t["DT"] // 4)), 4))
        detail = (
            "FLASH, one kernel, online softmax (running max + running sum). "
            "tile Br=%d Bc=%d D_TILE=%d threads=%d (%d warps), %d KV blocks, "
            "%d d-chunks, smem %d B. QK^T = nvcuda::wmma 16x16x16 half x half -> "
            "fp32 accumulator, K^T via a col_major matrix_b fragment; Q/K "
            "converted fp32->fp16 on the global->shared path (float4 read + "
            "__floats2half2_rn store -- __pipeline_memcpy_async cannot convert). "
            "Score staged through an fp32 smem tile (ld=%d) because the wmma "
            "fragment element->(row,col) map is not part of the API and the row "
            "max/sum is per-row; sdtype=%s%s. %s. Output accumulator is a plain "
            "float[%d][%d] register array (Br x d = %d floats / %d threads) with "
            "an explicit (row,col) map so the per-row alpha rescale is legal; "
            "each PV product goes through a (Br,%d) fp32 smem staging tile and is "
            "folded in as acc = acc*alpha + staged, identically for both pdtypes. "
            "Zero inline PTX (asserted at build)."
            % (t["Br"], t["Bc"], t["DT"], t["threads"], t["threads"] // 32,
               common2.S_S // t["Bc"], d // t["DT"], sm["SMEM"], sm["LDS"], sd,
               (" (each score put through __float2half_rn and back; the smem "
                "container stays fp32 so smem/occupancy/tile are identical "
                "across the three dtype pairs and only the arithmetic moves)"
                if sd == "fp16" else " (fp32 accumulator used as-is)"),
               pvtxt, d // t["DT"], (t["Br"] * t["DT"]) // t["threads"],
               t["Br"] * d, t["threads"], t["DT"]))
    else:
        pvtxt = (
            "k3_pv = wmma 16x16x16 half x half -> fp32 (fp16 tensor cores), "
            "P read as fp16 from global, V converted fp32->fp16 on load, "
            "64x64x64 tile / 128 thr"
            if pd == "fp16" else
            "k3_pv = register-tiled FFMA GEMM on the CUDA cores in TRUE fp32 "
            "(P fp32 x V fp32), 64x64x32 tile / 128 thr, 8x4 thread tile. "
            "No tf32: no wmma::precision::tf32 fragment is instantiated, "
            "because on sm_89 tf32 tensor and fp32 FFMA share the same "
            "91.1 TFLOP/s dense peak")
        detail = (
            "K3, three kernels, S MATERIALIZED in global memory. "
            "k3_qk = nvcuda::wmma 16x16x16 half x half -> fp32, tile "
            "%dx%dx%d / %d thr (2x2 warp grid, 2x2 frags/warp), K^T via a "
            "col_major matrix_b fragment, Q/K converted fp32->fp16 on the "
            "global->shared path; result scaled and stored as %s "
            "((B,H,512,512) = %.2f GB). k3_softmax = one row per block, 128 thr, "
            "4 elem/thread cached in registers, __shfl_down_sync + smem tree; "
            "reads %s, writes %s. %s. Zero inline PTX (asserted at build)."
            % (t["BM"], t["BN"], t["BK"], t["threads"], sd,
               common2.S_B * common2.S_H * common2.S_S * common2.S_S
               * (2 if sd == "fp16" else 4) / 1e9,
               sd, pd, pvtxt))

    art = {
        "cuda_source": gen["full"], "kernel_source": gen["kernel_src"],
        "ext_name": name, "algo": algo, "score_dtype": sd, "prob_dtype": pd,
        "head_dim": d, "n_kernels": n_kernels,
        "shared_bytes": gen["info"]["smem_bytes"],
        "grid": gen["info"]["grid"], "block": gen["info"]["block"],
        "tile": t, "backend_detail": detail,
    }
    try:
        art.update(p1._side_compile(gen["kernel_src"], name))
    except Exception as e:  # noqa: BLE001
        art["side_compile_error"] = repr(e)

    if algo == "FLASH":
        notes = ("cuda_noptx sdpa FLASH d=%d s=%s p=%s: Br=%d Bc=%d DT=%d "
                 "threads=%d smem=%dB" % (d, sd, pd, t["Br"], t["Bc"], t["DT"],
                                          t["threads"], gen["info"]["smem_bytes"]))
    else:
        notes = ("cuda_noptx sdpa K3 d=%d s=%s p=%s: %dx%dx%d threads=%d"
                 % (d, sd, pd, t["BM"], t["BN"], t["BK"], t["threads"]))

    return common2.Built2(run=run, compile_s=compile_s, artifacts=art,
                          notes=notes, n_kernels=n_kernels,
                          x_dtype=torch.float32)
