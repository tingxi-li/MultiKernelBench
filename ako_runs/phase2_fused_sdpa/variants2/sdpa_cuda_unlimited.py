"""cuda_unlimited lane of the Phase-2 SDPA study: hand-written CUDA + inline PTX.

Everything the lane is allowed is used and nothing else: `mma.sync.aligned.
m16n8k16.row.col.f32.f16.f16.f32`, `ldmatrix.sync.aligned.m8n8.x2/x4[.trans]`,
`cvt.rn.f16x2.f32`.  No cuBLAS, no cuDNN, no torch op does any of the work --
torch only allocates and (for K3) hands out the score/probability scratch.

TWO ALGORITHMS, identical semantics (scale = 1/sqrt(d), no mask, no dropout):

  K3     three kernels.  `qk3` writes S = QK^T*scale to global memory in
         `sdtype`; `sm3` does a row softmax of the 512-wide rows and writes P in
         `pdtype` (in place when the two dtypes agree, which is exactly the
         traffic saving the sdtype factor exists to measure); `pv3`/`pv3f`
         computes O = PV.  S really is materialized -- that is the point.
  FLASH  one kernel.  Tiled online softmax over KV blocks with a running max and
         a running sum; the score tile never leaves registers/shared memory and
         the (Br, d) output accumulator is fp32 in registers for the whole KV
         loop.

THE TWO DTYPE AXES.

  sdtype  the dtype the SCORE tensor is kept in.  The QK^T *operands* are fp16
          in every cell -- that is the lane's tensor-core path and it is held
          fixed so the axis is not confounded -- and the fp32 mma accumulator is
          rounded to `sdtype` before anything else sees it.  In K3 that is also
          the global dtype of S, so `score_bytes` halves.  In FLASH the rounding
          still happens (the smem score tile is declared in `sdtype`); only the
          DRAM traffic does not exist to be saved.
  pdtype  the dtype the PROBABILITIES are in when they feed the PV matmul.
          fp16 -> P is an `ldmatrix`-fed A fragment of the same m16n8k16
          `mma.sync` used for QK^T, and V is converted fp32->fp16 on the
          global->shared path.
          fp32 -> the tensor cores are NOT used at all for PV.  This lane does
          NOT silently fall back to tf32: the PV product is genuine IEEE fp32
          `FFMA` on the CUDA cores with fp32 P and fp32 V, laid out on the same
          (row, col) accumulator mapping the mma path produces so the rest of
          the kernel is byte-identical.  Recorded explicitly in
          `backend_detail`; if a reader wants tf32 numbers they are a different
          experiment.

The accumulator -> (row, col) mapping after `mma.sync.m16n8k16` --
`gr = m0 + wm + i*16 + (lane>>2)`, `gc = n0 + wn + j*8 + ((lane&3)<<1)` -- is
what makes the FLASH lane cheap here: the online-softmax row rescale and the
final 1/l division are register operations indexed straight by `gr`, with no
staging through shared memory at all.  A lane that cannot name its own fragment
layout has to round-trip the whole (Br, d) accumulator through smem to do that.

Tiles are FIXED per (algo, d) across all three dtype pairs, so no dtype effect
can be a disguised schedule change.  What does move with pdtype is only which
*instruction* consumes P, and (for the K3 PV kernel) the BK of the fp32 CUDA-core
schedule, which has no mma-shaped analogue -- noted in `backend_detail`.
"""
from __future__ import annotations

import math
import os
import string
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
from variants import cuda_unlimited_gemm as p1  # noqa: E402

_MAX_SMEM_OPTIN = 101376          # sm_89 opt-in per-block shared memory

# ---------------------------------------------------------------------------
# FLASH geometry.  Br/Bc/threads are the tile; DKC/DV are global->shared load
# chunks along the head dimension and are NOT part of the tile (they exist only
# because a (Bc, 1024) staging buffer does not fit in 99 KB).  Both are held
# identical across the three dtype pairs anyway.
#
# The warp grid is chosen so that NACC == 64 fp32 output-accumulator registers
# per thread at every head dim -- that is the constraint d=1024 imposes and it
# is applied uniformly rather than only where it bites.
#
#   d      Br  Bc  thr  warps  PV grid   QK grid   NACC  smem (fp32,fp32)
#   128    64  64  128    4     2 x 2     2 x 2     64      52992
#   256    64  64  256    8     2 x 4     2 x 4     64      69376
#   1024   32  64  512   16     2 x 8     2 x 8     64      92544
#
# d=128/256 keep SDPA_TILES' Br=Bc=64; d=1024 keeps its Br=32, Bc=64.  Only the
# thread count is raised past SDPA_TILES' 128, because (Br, d) fp32 in registers
# over 128 threads is 128 regs/thread at d=256 and 256 at d=1024, i.e. the
# suggested thread count is unrealizable at the head dims that matter.
FLASH_CFG = {
    128:  dict(Br=64, Bc=64, threads=128, DKC=128, DV=64,
               WARPS_M=2, WARPS_N=2, SWARPS_M=2, SWARPS_N=2),
    256:  dict(Br=64, Bc=64, threads=256, DKC=128, DV=64,
               WARPS_M=2, WARPS_N=4, SWARPS_M=2, SWARPS_N=4),
    1024: dict(Br=32, Bc=64, threads=512, DKC=128, DV=64,
               WARPS_M=2, WARPS_N=8, SWARPS_M=2, SWARPS_N=8),
}

# K3 schedules.  Independent of d except through the grid.
K3_QK = dict(BM=128, BN=128, BK=32, threads=256, WM=2, WN=4)   # QK^T, mma
K3_PV = dict(BM=128, BN=128, BK=32, threads=256, WM=2, WN=4)   # PV, mma (fp16 P)
K3_PVF = dict(BM=128, BN=128, BK=16, threads=256, TM=8, TN=8)  # PV, FFMA (fp32 P)
K3_SOFT_THREADS = 128


def _esz(t):
    return 2 if t == "fp16" else 4


def _pad(t):
    """Padding in ELEMENTS that puts the row stride at 16 B mod 128 B, which is
    what makes the 8-row `ldmatrix` gather (and the fp32 column walk) hit eight
    distinct 4-bank groups."""
    return 8 if t == "fp16" else 4


_COMMON = string.Template(r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define SEQ        $SEQ
#define DH         $DH
#define SCALEV     $SCALEV
#define SD_FP16    $SD_FP16
#define PD_FP16    $PD_FP16

#if SD_FP16
typedef half ST;
#define ST_PUT(x)  __float2half_rn(x)
#define ST_GET(x)  __half2float(x)
#else
typedef float ST;
#define ST_PUT(x)  (x)
#define ST_GET(x)  (x)
#endif

#if PD_FP16
typedef half PT;
#define PT_PUT(x)  __float2half_rn(x)
#define PT_GET(x)  __half2float(x)
#else
typedef float PT;
#define PT_PUT(x)  (x)
#define PT_GET(x)  (x)
#endif

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

/* two floats -> one packed .b16x2 register, round-to-nearest-even.
   cvt.rn.f16x2.f32 d, a, b  =>  d.hi = f16(a), d.lo = f16(b)               */
__device__ __forceinline__ unsigned pack2(float lo, float hi) {
    unsigned d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

#define LDM_X4(r0,r1,r2,r3,ad) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(ad))
#define LDM_X4T(r0,r1,r2,r3,ad) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(ad))
#define LDM_X2(r0,r1,ad) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" \
                 : "=r"(r0), "=r"(r1) : "r"(ad))
#define LDM_X2T(r0,r1,ad) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" \
                 : "=r"(r0), "=r"(r1) : "r"(ad))

#define MMA(d0,d1,d2,d3,a0,a1,a2,a3,b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

/* ---- global(fp32) -> shared(fp16), converting on the way in -------------- */
template<int ROWS, int COLS, int STRIDE, int THR>
__device__ __forceinline__ void g2s_h(const float* __restrict__ src, int ld,
                                      half* dst, int tid) {
    constexpr int VPR = COLS / 4;
    constexpr int ITERS = (ROWS * COLS / 4) / THR;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        int v = tid + it * THR;
        int r = v / VPR, c = (v % VPR) * 4;
        float4 x = *(const float4*)(src + (long long)r * ld + c);
        *(uint2*)(dst + r * STRIDE + c) =
            make_uint2(pack2(x.x, x.y), pack2(x.z, x.w));
    }
}

/* ---- global(fp32) -> shared(fp32) --------------------------------------- */
template<int ROWS, int COLS, int STRIDE, int THR>
__device__ __forceinline__ void g2s_f(const float* __restrict__ src, int ld,
                                      float* dst, int tid) {
    constexpr int VPR = COLS / 4;
    constexpr int ITERS = (ROWS * COLS / 4) / THR;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        int v = tid + it * THR;
        int r = v / VPR, c = (v % VPR) * 4;
        *(float4*)(dst + r * STRIDE + c) =
            *(const float4*)(src + (long long)r * ld + c);
    }
}

/* ---- global(fp32) -> shared(fp32), TRANSPOSED (for the FFMA A operand) --- */
template<int ROWS, int COLS, int STRIDE, int THR>
__device__ __forceinline__ void g2s_fT(const float* __restrict__ src, int ld,
                                       float* dst, int tid) {
    constexpr int VPR = COLS / 4;
    constexpr int ITERS = (ROWS * COLS / 4) / THR;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        int v = tid + it * THR;
        int r = v / VPR, c = (v % VPR) * 4;
        float4 x = *(const float4*)(src + (long long)r * ld + c);
        dst[(c + 0) * STRIDE + r] = x.x;
        dst[(c + 1) * STRIDE + r] = x.y;
        dst[(c + 2) * STRIDE + r] = x.z;
        dst[(c + 3) * STRIDE + r] = x.w;
    }
}

/* ---- global(fp16) -> shared(fp16) --------------------------------------- */
template<int ROWS, int COLS, int STRIDE, int THR>
__device__ __forceinline__ void g2s_hh(const half* __restrict__ src, int ld,
                                       half* dst, int tid) {
    constexpr int VPR = COLS / 8;
    constexpr int ITERS = (ROWS * COLS / 8) / THR;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        int v = tid + it * THR;
        int r = v / VPR, c = (v % VPR) * 8;
        *(uint4*)(dst + r * STRIDE + c) =
            *(const uint4*)(src + (long long)r * ld + c);
    }
}
""")


_FLASH = string.Template(r"""
/* ======================= FLASH: one fused kernel ======================== */
#define BR         $BR
#define BC         $BC
#define THREADS    $THREADS
#define DKC        $DKC
#define DV         $DV
#define WARPS_M    $WARPS_M
#define WARPS_N    $WARPS_N
#define SWARPS_M   $SWARPS_M
#define SWARPS_N   $SWARPS_N
#define QSTRIDE    $QSTRIDE
#define KSTRIDE    $KSTRIDE
#define VSTRIDE    $VSTRIDE
#define SSTRIDE    $SSTRIDE
#define PSTRIDE    $PSTRIDE
#define QB         $QB
#define PB         $PB
#define ARENA      $ARENA
#define FSMEM      $FSMEM

#define WM       (BR/WARPS_M)          /* PV: rows per warp                  */
#define MT       (WM/16)
#define NTA      (DH/(8*WARPS_N))      /* PV: n-tiles owned per warp, total  */
#define NTV      (DV/(8*WARPS_N))      /* ... of which this V chunk supplies */
#define NCV      (DH/DV)
#define NACC     (MT*NTA*4)

#define SWM      (BR/SWARPS_M)         /* QK^T: rows per warp                */
#define SWN      (BC/SWARPS_N)
#define SMT      (SWM/16)
#define SNT      (SWN/8)
#define SNACC    (SMT*SNT*4)

#define NCK      (DH/DKC)
#define RTHR     (THREADS/BR)          /* threads cooperating on one S row   */
#define RPT      (BC/RTHR)

__global__ void __launch_bounds__(THREADS) flash_kernel(
        const float* __restrict__ Qg, const float* __restrict__ Kg,
        const float* __restrict__ Vg, float* __restrict__ Og) {
    extern __shared__ __align__(16) char sraw[];
    half*  Qs = (half*)sraw;
    char*  ar = sraw + QB;
    /* One arena, three lifetimes.  Ks dies before Ss is written; Ss dies
       before Vs is loaded; Ps spans the softmax and the PV matmul.  So the
       arena only has to hold max(K, S+P, P+V), which is what lets a (32,1024)
       fp16 Q tile and a 64-register fp32 output accumulator coexist at
       d=1024 inside 99 KB.                                                  */
    PT*    Ps = (PT*)(ar);
    ST*    Ss = (ST*)(ar + PB);
    PT*    Vs = (PT*)(ar + PB);
    half*  Ks = (half*)(ar);
    float* rowm = (float*)(sraw + QB + ARENA);
    float* rowl = rowm + BR;
    float* rowc = rowl + BR;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int pwm  = (wid / WARPS_N) * WM;     /* PV row block               */
    const int pwn  = (wid % WARPS_N);          /* PV n-tile phase (strided)  */
    const int swm  = (wid / SWARPS_N) * SWM;   /* QK^T row block             */
    const int swn  = (wid % SWARPS_N) * SWN;

    const long long base = (long long)blockIdx.y * SEQ * DH;
    const int r0 = blockIdx.x * BR;

    float acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
    if (tid < BR) { rowm[tid] = -1e30f; rowl[tid] = 0.f; }

    g2s_h<BR, DH, QSTRIDE, THREADS>(Qg + base + (long long)r0 * DH, DH, Qs, tid);
    __syncthreads();

    for (int kb = 0; kb < SEQ; kb += BC) {
        /* ---------------- S = Q K^T * scale, in registers ---------------- */
        float sacc[SNACC];
#pragma unroll
        for (int i = 0; i < SNACC; ++i) sacc[i] = 0.f;

        for (int dc = 0; dc < DH; dc += DKC) {
            __syncthreads();                     /* Ps (aliased) is dead now */
            g2s_h<BC, DKC, KSTRIDE, THREADS>(
                Kg + base + (long long)kb * DH + dc, DH, Ks, tid);
            __syncthreads();
#pragma unroll
            for (int ks = 0; ks < DKC/16; ++ks) {
                unsigned a[SMT][4], b[SNT][2];
#pragma unroll
                for (int i = 0; i < SMT; ++i) {
                    unsigned ad = smem_u32(Qs + (swm + i*16 + (lane & 15))*QSTRIDE
                                              + dc + ks*16 + ((lane >> 4) << 3));
                    LDM_X4(a[i][0], a[i][1], a[i][2], a[i][3], ad);
                }
#pragma unroll
                for (int j = 0; j < SNT; ++j) {
                    /* K is consumed in its NATIVE (seq, d) layout: with n on
                       the smem row axis the m16n8k16 B fragment is exactly what
                       a plain (non-.trans) ldmatrix.x2 delivers.             */
                    unsigned ad = smem_u32(Ks + (swn + j*8 + (lane & 7))*KSTRIDE
                                              + ks*16 + (((lane >> 3) & 1) << 3));
                    LDM_X2(b[j][0], b[j][1], ad);
                }
#pragma unroll
                for (int i = 0; i < SMT; ++i)
#pragma unroll
                    for (int j = 0; j < SNT; ++j) {
                        const int x = (i*SNT+j)*4;
                        MMA(sacc[x+0], sacc[x+1], sacc[x+2], sacc[x+3],
                            a[i][0], a[i][1], a[i][2], a[i][3],
                            b[j][0], b[j][1]);
                    }
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < SMT; ++i)
#pragma unroll
            for (int j = 0; j < SNT; ++j) {
                const int x = (i*SNT+j)*4;
                const int r = swm + i*16 + (lane >> 2);
                const int c = swn + j*8  + ((lane & 3) << 1);
                Ss[r*SSTRIDE + c]         = ST_PUT(sacc[x+0]*SCALEV);
                Ss[r*SSTRIDE + c + 1]     = ST_PUT(sacc[x+1]*SCALEV);
                Ss[(r+8)*SSTRIDE + c]     = ST_PUT(sacc[x+2]*SCALEV);
                Ss[(r+8)*SSTRIDE + c + 1] = ST_PUT(sacc[x+3]*SCALEV);
            }
        __syncthreads();

        /* ---------------- online softmax over the BC-wide strip ---------- */
        {
            const int row = tid / RTHR, lig = tid % RTHR;
            float mx = -1e30f;
#pragma unroll
            for (int t = 0; t < RPT; ++t)
                mx = fmaxf(mx, ST_GET(Ss[row*SSTRIDE + t*RTHR + lig]));
#pragma unroll
            for (int off = RTHR/2; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float mp = rowm[row];
            const float mn = fmaxf(mp, mx);
            const float corr = __expf(mp - mn);
            float ssum = 0.f;
#pragma unroll
            for (int t = 0; t < RPT; ++t) {
                const int c = t*RTHR + lig;
                const float e = __expf(ST_GET(Ss[row*SSTRIDE + c]) - mn);
                Ps[row*PSTRIDE + c] = PT_PUT(e);
                ssum += e;
            }
#pragma unroll
            for (int off = RTHR/2; off > 0; off >>= 1)
                ssum += __shfl_xor_sync(0xffffffffu, ssum, off);
            if (lig == 0) {
                rowm[row] = mn;
                rowl[row] = rowl[row]*corr + ssum;
                rowc[row] = corr;
            }
        }
        __syncthreads();

        /* ---------------- rescale the fp32 output accumulator ------------
           Pure register work: this lane knows that acc[(i*NTA+jt)*4 + {0,1}]
           lives on row pwm+i*16+(lane>>2) and {2,3} on that row + 8, so the
           per-row correction is applied in place with no smem round trip.   */
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            const float c0 = rowc[pwm + i*16 + (lane >> 2)];
            const float c1 = rowc[pwm + i*16 + (lane >> 2) + 8];
#pragma unroll
            for (int jt = 0; jt < NTA; ++jt) {
                const int x = (i*NTA+jt)*4;
                acc[x+0] *= c0; acc[x+1] *= c0;
                acc[x+2] *= c1; acc[x+3] *= c1;
            }
        }

        /* ---------------- O += P V --------------------------------------- */
#pragma unroll
        for (int dvi = 0; dvi < NCV; ++dvi) {
            __syncthreads();                       /* Ss (aliased) is dead   */
#if PD_FP16
            g2s_h<BC, DV, VSTRIDE, THREADS>(
                Vg + base + (long long)kb * DH + dvi*DV, DH, (half*)Vs, tid);
#else
            g2s_f<BC, DV, VSTRIDE, THREADS>(
                Vg + base + (long long)kb * DH + dvi*DV, DH, (float*)Vs, tid);
#endif
            __syncthreads();
#if PD_FP16
#pragma unroll
            for (int ks = 0; ks < BC/16; ++ks) {
                unsigned a[MT][4], b[NTV][2];
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    unsigned ad = smem_u32((half*)Ps
                        + (pwm + i*16 + (lane & 15))*PSTRIDE
                        + ks*16 + ((lane >> 4) << 3));
                    LDM_X4(a[i][0], a[i][1], a[i][2], a[i][3], ad);
                }
#pragma unroll
                for (int jj = 0; jj < NTV; ++jj) {
                    const int lc = (jj*WARPS_N + pwn) * 8;
                    unsigned ad = smem_u32((half*)Vs
                        + (ks*16 + (lane & 7) + (((lane >> 3) & 1) << 3))*VSTRIDE
                        + lc);
                    LDM_X2T(b[jj][0], b[jj][1], ad);
                }
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int jj = 0; jj < NTV; ++jj) {
                        const int x = (i*NTA + dvi*NTV + jj)*4;
                        MMA(acc[x+0], acc[x+1], acc[x+2], acc[x+3],
                            a[i][0], a[i][1], a[i][2], a[i][3],
                            b[jj][0], b[jj][1]);
                    }
            }
#else
            /* pdtype=fp32: NO tensor core.  Genuine IEEE fp32 FFMA on the CUDA
               cores, fp32 P x fp32 V, arranged on the mma accumulator's own
               (row, col) mapping so nothing else in the kernel changes.

               The obvious worry is that this is 4 shared loads per 4 FFMA and
               therefore LSU bound.  A k-blocked-by-4 rewrite (P read as float4
               along k, V as float2: 6 loads per 16 FFMA) was built and measured
               at all three head dims and produced no gain -- it landed inside
               the run-to-run drift of this card, and it cost registers.  The
               fp32 cell is bound by pushing an fp32 V tile through L2 once per
               KV block, which is twice the bytes of the fp16 cell, not by FFMA
               issue.  The simple form is kept.                                */
#pragma unroll
            for (int kk = 0; kk < BC; ++kk) {
                float pv[MT][2], vv[NTV][2];
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    const int r = pwm + i*16 + (lane >> 2);
                    pv[i][0] = ((float*)Ps)[r*PSTRIDE + kk];
                    pv[i][1] = ((float*)Ps)[(r+8)*PSTRIDE + kk];
                }
#pragma unroll
                for (int jj = 0; jj < NTV; ++jj) {
                    const int c = (jj*WARPS_N + pwn)*8 + ((lane & 3) << 1);
                    vv[jj][0] = ((float*)Vs)[kk*VSTRIDE + c];
                    vv[jj][1] = ((float*)Vs)[kk*VSTRIDE + c + 1];
                }
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int jj = 0; jj < NTV; ++jj) {
                        const int x = (i*NTA + dvi*NTV + jj)*4;
                        acc[x+0] += pv[i][0]*vv[jj][0];
                        acc[x+1] += pv[i][0]*vv[jj][1];
                        acc[x+2] += pv[i][1]*vv[jj][0];
                        acc[x+3] += pv[i][1]*vv[jj][1];
                    }
            }
#endif
        }
    }

    __syncthreads();
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int gr0 = pwm + i*16 + (lane >> 2);
        const float inv0 = 1.f / rowl[gr0];
        const float inv1 = 1.f / rowl[gr0 + 8];
        float* o0 = Og + base + (long long)(r0 + gr0)*DH;
        float* o1 = Og + base + (long long)(r0 + gr0 + 8)*DH;
#pragma unroll
        for (int jt = 0; jt < NTA; ++jt) {
            const int x = (i*NTA+jt)*4;
            const int gc = (jt*WARPS_N + pwn)*8 + ((lane & 3) << 1);
            *(float2*)(o0 + gc) = make_float2(acc[x+0]*inv0, acc[x+1]*inv0);
            *(float2*)(o1 + gc) = make_float2(acc[x+2]*inv1, acc[x+3]*inv1);
        }
    }
}

torch::Tensor sdpa(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be fp32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == SEQ && q.size(3) == DH, "shape mismatch");
    auto O = torch::empty({B, H, (long)SEQ, (long)DH}, q.options());
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(flash_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FSMEM);
        attr = true;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(SEQ/BR, B*H);
    flash_kernel<<<grid, THREADS, FSMEM, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        O.data_ptr<float>());
    return O;
}
""")


_K3 = string.Template(r"""
/* ==================== K3: three kernels, S materialized ================= */
#define Q_BM   $Q_BM
#define Q_BN   $Q_BN
#define Q_BK   $Q_BK
#define Q_THR  $Q_THR
#define Q_WM   $Q_WM
#define Q_WN   $Q_WN
#define Q_AS   (Q_BK+8)
#define Q_BS   (Q_BK+8)
#define Q_TM   ((Q_BM/Q_WM)/16)
#define Q_TN   ((Q_BN/Q_WN)/8)
#define Q_NACC (Q_TM*Q_TN*4)
#define Q_SMEM $Q_SMEM

#define SOFT_THR $SOFT_THR
#define SOFT_EPT (SEQ/SOFT_THR)
#define SOFT_NW  (SOFT_THR/32)

#define V_BM   $V_BM
#define V_BN   $V_BN
#define V_BK   $V_BK
#define V_THR  $V_THR
#define V_WM   $V_WM
#define V_WN   $V_WN
#define V_AS   (V_BK+8)
#define V_BS   (V_BN+8)
#define V_TM   ((V_BM/V_WM)/16)
#define V_TN   ((V_BN/V_WN)/8)
#define V_NACC (V_TM*V_TN*4)
#define V_SMEM $V_SMEM

#define F_BM   $F_BM
#define F_BN   $F_BN
#define F_BK   $F_BK
#define F_THR  $F_THR
#define F_TM   $F_TM
#define F_TN   $F_TN
#define F_NM   (F_BM/F_TM)
#define F_NN   (F_BN/F_TN)
#define F_AS   (F_BM+4)
#define F_BS   (F_BN+4)
#define F_SMEM $F_SMEM

/* ---- kernel 1: S = Q K^T * scale, written to GLOBAL memory in sdtype ---- */
__global__ void __launch_bounds__(Q_THR) qk3_kernel(
        const float* __restrict__ Qg, const float* __restrict__ Kg,
        ST* __restrict__ Sg) {
    extern __shared__ __align__(16) char sraw[];
    half* As = (half*)sraw;
    half* Bs = (half*)(sraw + Q_BM*Q_AS*2);

    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int wm = (wid / Q_WN) * (Q_BM/Q_WM);
    const int wn = (wid % Q_WN) * (Q_BN/Q_WN);
    const int m0 = blockIdx.y * Q_BM, n0 = blockIdx.x * Q_BN;
    const long long base = (long long)blockIdx.z * SEQ * DH;

    float acc[Q_NACC];
#pragma unroll
    for (int i = 0; i < Q_NACC; ++i) acc[i] = 0.f;

    for (int kb = 0; kb < DH; kb += Q_BK) {
        g2s_h<Q_BM, Q_BK, Q_AS, Q_THR>(
            Qg + base + (long long)m0*DH + kb, DH, As, tid);
        g2s_h<Q_BN, Q_BK, Q_BS, Q_THR>(
            Kg + base + (long long)n0*DH + kb, DH, Bs, tid);
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < Q_BK/16; ++ks) {
            unsigned a[Q_TM][4], b[Q_TN][2];
#pragma unroll
            for (int i = 0; i < Q_TM; ++i) {
                unsigned ad = smem_u32(As + (wm + i*16 + (lane & 15))*Q_AS
                                          + ks*16 + ((lane >> 4) << 3));
                LDM_X4(a[i][0], a[i][1], a[i][2], a[i][3], ad);
            }
#pragma unroll
            for (int j = 0; j < Q_TN; ++j) {
                unsigned ad = smem_u32(Bs + (wn + j*8 + (lane & 7))*Q_BS
                                          + ks*16 + (((lane >> 3) & 1) << 3));
                LDM_X2(b[j][0], b[j][1], ad);
            }
#pragma unroll
            for (int i = 0; i < Q_TM; ++i)
#pragma unroll
                for (int j = 0; j < Q_TN; ++j) {
                    const int x = (i*Q_TN+j)*4;
                    MMA(acc[x+0], acc[x+1], acc[x+2], acc[x+3],
                        a[i][0], a[i][1], a[i][2], a[i][3], b[j][0], b[j][1]);
                }
        }
        __syncthreads();
    }

    ST* Sp = Sg + (long long)blockIdx.z * SEQ * SEQ;
#pragma unroll
    for (int i = 0; i < Q_TM; ++i)
#pragma unroll
        for (int j = 0; j < Q_TN; ++j) {
            const int x = (i*Q_TN+j)*4;
            const int gr = m0 + wm + i*16 + (lane >> 2);
            const int gc = n0 + wn + j*8  + ((lane & 3) << 1);
#if SD_FP16
            *(__half2*)(Sp + (long long)gr*SEQ + gc) =
                __floats2half2_rn(acc[x+0]*SCALEV, acc[x+1]*SCALEV);
            *(__half2*)(Sp + (long long)(gr+8)*SEQ + gc) =
                __floats2half2_rn(acc[x+2]*SCALEV, acc[x+3]*SCALEV);
#else
            *(float2*)(Sp + (long long)gr*SEQ + gc) =
                make_float2(acc[x+0]*SCALEV, acc[x+1]*SCALEV);
            *(float2*)(Sp + (long long)(gr+8)*SEQ + gc) =
                make_float2(acc[x+2]*SCALEV, acc[x+3]*SCALEV);
#endif
        }
}

/* ---- kernel 2: row softmax, sdtype in -> pdtype out --------------------- */
/* No __restrict__ here on purpose: when sdtype == pdtype the caller passes the
   SAME buffer for both, and promising the compiler they cannot alias would let
   it sink the stores above the loads. */
__global__ void __launch_bounds__(SOFT_THR) sm3_kernel(
        const ST* Sg, PT* Pg) {
    __shared__ float rm[SOFT_NW];
    __shared__ float rs[SOFT_NW];
    const long long row = blockIdx.x;
    const ST* s = Sg + row * SEQ;
    PT* p = Pg + row * SEQ;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;

    float x[SOFT_EPT];
    float mx = -3.402823466e+38f;
#pragma unroll
    for (int e = 0; e < SOFT_EPT; ++e) {
        x[e] = ST_GET(s[tid + e*SOFT_THR]);
        mx = fmaxf(mx, x[e]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if (lane == 0) rm[wid] = mx;
    __syncthreads();
    float m = rm[0];
#pragma unroll
    for (int w = 1; w < SOFT_NW; ++w) m = fmaxf(m, rm[w]);

    float sum = 0.f;
#pragma unroll
    for (int e = 0; e < SOFT_EPT; ++e) { x[e] = __expf(x[e] - m); sum += x[e]; }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (lane == 0) rs[wid] = sum;
    __syncthreads();
    float tot = rs[0];
#pragma unroll
    for (int w = 1; w < SOFT_NW; ++w) tot += rs[w];
    const float inv = 1.f / tot;
#pragma unroll
    for (int e = 0; e < SOFT_EPT; ++e)
        p[tid + e*SOFT_THR] = PT_PUT(x[e] * inv);
}

#if PD_FP16
/* ---- kernel 3a: O = P V, fp16 P -> m16n8k16 tensor cores ---------------- */
__global__ void __launch_bounds__(V_THR) pv3_kernel(
        const half* __restrict__ Pg, const float* __restrict__ Vg,
        float* __restrict__ Og) {
    extern __shared__ __align__(16) char sraw[];
    half* As = (half*)sraw;
    half* Bs = (half*)(sraw + V_BM*V_AS*2);

    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int wm = (wid / V_WN) * (V_BM/V_WM);
    const int wn = (wid % V_WN) * (V_BN/V_WN);
    const int m0 = blockIdx.y * V_BM, n0 = blockIdx.x * V_BN;
    const long long pbase = (long long)blockIdx.z * SEQ * SEQ;
    const long long vbase = (long long)blockIdx.z * SEQ * DH;

    float acc[V_NACC];
#pragma unroll
    for (int i = 0; i < V_NACC; ++i) acc[i] = 0.f;

    for (int kb = 0; kb < SEQ; kb += V_BK) {
        g2s_hh<V_BM, V_BK, V_AS, V_THR>(
            Pg + pbase + (long long)m0*SEQ + kb, SEQ, As, tid);
        g2s_h<V_BK, V_BN, V_BS, V_THR>(
            Vg + vbase + (long long)kb*DH + n0, DH, Bs, tid);
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < V_BK/16; ++ks) {
            unsigned a[V_TM][4], b[V_TN][2];
#pragma unroll
            for (int i = 0; i < V_TM; ++i) {
                unsigned ad = smem_u32(As + (wm + i*16 + (lane & 15))*V_AS
                                          + ks*16 + ((lane >> 4) << 3));
                LDM_X4(a[i][0], a[i][1], a[i][2], a[i][3], ad);
            }
#pragma unroll
            for (int jj = 0; jj < V_TN/2; ++jj) {
                unsigned ad = smem_u32(Bs
                    + (ks*16 + (lane & 7) + (((lane >> 3) & 1) << 3))*V_BS
                    + wn + jj*16 + ((lane >> 4) << 3));
                LDM_X4T(b[2*jj][0], b[2*jj][1], b[2*jj+1][0], b[2*jj+1][1], ad);
            }
#pragma unroll
            for (int i = 0; i < V_TM; ++i)
#pragma unroll
                for (int j = 0; j < V_TN; ++j) {
                    const int x = (i*V_TN+j)*4;
                    MMA(acc[x+0], acc[x+1], acc[x+2], acc[x+3],
                        a[i][0], a[i][1], a[i][2], a[i][3], b[j][0], b[j][1]);
                }
        }
        __syncthreads();
    }

    float* Op = Og + vbase;
#pragma unroll
    for (int i = 0; i < V_TM; ++i)
#pragma unroll
        for (int j = 0; j < V_TN; ++j) {
            const int x = (i*V_TN+j)*4;
            const int gr = m0 + wm + i*16 + (lane >> 2);
            const int gc = n0 + wn + j*8  + ((lane & 3) << 1);
            *(float2*)(Op + (long long)gr*DH + gc) =
                make_float2(acc[x+0], acc[x+1]);
            *(float2*)(Op + (long long)(gr+8)*DH + gc) =
                make_float2(acc[x+2], acc[x+3]);
        }
}
#else
/* ---- kernel 3b: O = P V, fp32 P -> IEEE fp32 FFMA on the CUDA cores -----
   No tensor core is touched here.  fp32 operands into an mma would be tf32 on
   sm_89; that is a different arithmetic and is deliberately not what this cell
   measures.                                                                */
__global__ void __launch_bounds__(F_THR) pv3f_kernel(
        const float* __restrict__ Pg, const float* __restrict__ Vg,
        float* __restrict__ Og) {
    extern __shared__ __align__(16) float sm[];
    float* As = sm;                 /* [F_BK][F_AS] transposed: As[k][m] */
    float* Bs = sm + F_BK*F_AS;     /* [F_BK][F_BS]                      */

    const int tid = threadIdx.x;
    const int tx = tid % F_NN, ty = tid / F_NN;
    const int m0 = blockIdx.y * F_BM, n0 = blockIdx.x * F_BN;
    const long long pbase = (long long)blockIdx.z * SEQ * SEQ;
    const long long vbase = (long long)blockIdx.z * SEQ * DH;

    float acc[F_TM][F_TN];
#pragma unroll
    for (int i = 0; i < F_TM; ++i)
#pragma unroll
        for (int j = 0; j < F_TN; ++j) acc[i][j] = 0.f;

    for (int kb = 0; kb < SEQ; kb += F_BK) {
        g2s_fT<F_BM, F_BK, F_AS, F_THR>(
            Pg + pbase + (long long)m0*SEQ + kb, SEQ, As, tid);
        g2s_f<F_BK, F_BN, F_BS, F_THR>(
            Vg + vbase + (long long)kb*DH + n0, DH, Bs, tid);
        __syncthreads();
#pragma unroll 4
        for (int k = 0; k < F_BK; ++k) {
            float a[F_TM], b[F_TN];
#pragma unroll
            for (int i = 0; i < F_TM; ++i) a[i] = As[k*F_AS + ty*F_TM + i];
#pragma unroll
            for (int j = 0; j < F_TN; ++j) b[j] = Bs[k*F_BS + tx*F_TN + j];
#pragma unroll
            for (int i = 0; i < F_TM; ++i)
#pragma unroll
                for (int j = 0; j < F_TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    float* Op = Og + vbase;
#pragma unroll
    for (int i = 0; i < F_TM; ++i) {
        float* dst = Op + (long long)(m0 + ty*F_TM + i)*DH + n0 + tx*F_TN;
#pragma unroll
        for (int j = 0; j < F_TN; j += 4)
            *(float4*)(dst + j) = make_float4(acc[i][j], acc[i][j+1],
                                              acc[i][j+2], acc[i][j+3]);
    }
}
#endif

torch::Tensor sdpa(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be fp32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == SEQ && q.size(3) == DH, "shape mismatch");
    const long long BH = (long long)B * H;

    auto dev = q.device();
    auto sopt = torch::TensorOptions().device(dev).dtype(
        SD_FP16 ? torch::kHalf : torch::kFloat32);
    auto Sm = torch::empty({BH, (long)SEQ, (long)SEQ}, sopt);
#if SD_FP16 == PD_FP16
    torch::Tensor Pm = Sm;               /* softmax is elementwise-in-place  */
#else
    auto popt = torch::TensorOptions().device(dev).dtype(
        PD_FP16 ? torch::kHalf : torch::kFloat32);
    torch::Tensor Pm = torch::empty({BH, (long)SEQ, (long)SEQ}, popt);
#endif
    auto O = torch::empty({B, H, (long)SEQ, (long)DH}, q.options());

    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(qk3_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Q_SMEM);
#if PD_FP16
        cudaFuncSetAttribute(pv3_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, V_SMEM);
#else
        cudaFuncSetAttribute(pv3f_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, F_SMEM);
#endif
        attr = true;
    }
    auto stream = at::cuda::getCurrentCUDAStream();

    qk3_kernel<<<dim3(SEQ/Q_BN, SEQ/Q_BM, (unsigned)BH), Q_THR, Q_SMEM, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), (ST*)Sm.data_ptr());
    sm3_kernel<<<(unsigned)(BH*SEQ), SOFT_THR, 0, stream>>>(
        (const ST*)Sm.data_ptr(), (PT*)Pm.data_ptr());
#if PD_FP16
    pv3_kernel<<<dim3(DH/V_BN, SEQ/V_BM, (unsigned)BH), V_THR, V_SMEM, stream>>>(
        (const half*)Pm.data_ptr(), v.data_ptr<float>(), O.data_ptr<float>());
#else
    pv3f_kernel<<<dim3(DH/F_BN, SEQ/F_BM, (unsigned)BH), F_THR, F_SMEM, stream>>>(
        (const float*)Pm.data_ptr(), v.data_ptr<float>(), O.data_ptr<float>());
#endif
    return O;
}
""")


# --------------------------------------------------------------------------- #
def _flash_layout(d, sd, pd):
    g = dict(FLASH_CFG[d])
    Br, Bc = g["Br"], g["Bc"]
    qs = d + 8
    ks = g["DKC"] + 8
    vs = g["DV"] + _pad(pd)
    ss = Bc + _pad(sd)
    ps = Bc + _pad(pd)
    QB = Br * qs * 2
    KB = Bc * ks * 2
    VB = Bc * vs * _esz(pd)
    SB = Br * ss * _esz(sd)
    PB = Br * ps * _esz(pd)
    ARENA = max(KB, SB + PB, PB + VB)
    FSMEM = QB + ARENA + 3 * Br * 4
    g.update(QSTRIDE=qs, KSTRIDE=ks, VSTRIDE=vs, SSTRIDE=ss, PSTRIDE=ps,
             QB=QB, KB=KB, VB=VB, SB=SB, PB=PB, ARENA=ARENA, FSMEM=FSMEM)
    return g


def _check_flash(d, g, pd):
    if pd == "fp32":
        # the k-blocked FFMA PV loop reads P as float4 along k and V as float2
        assert g["Bc"] % 4 == 0 and g["PSTRIDE"] % 4 == 0, "fp32 PV float4 P"
        assert g["VSTRIDE"] % 2 == 0, "fp32 PV float2 V"
    Br, Bc, thr = g["Br"], g["Bc"], g["threads"]
    nw = thr // 32
    assert g["WARPS_M"] * g["WARPS_N"] == nw, "PV warp grid"
    assert g["SWARPS_M"] * g["SWARPS_N"] == nw, "QK warp grid"
    assert Br % (16 * g["WARPS_M"]) == 0, "PV row tiling"
    assert Br % (16 * g["SWARPS_M"]) == 0, "QK row tiling"
    assert Bc % (8 * g["SWARPS_N"]) == 0, "QK col tiling"
    assert d % (8 * g["WARPS_N"]) == 0 and g["DV"] % (8 * g["WARPS_N"]) == 0
    assert d % g["DKC"] == 0 and d % g["DV"] == 0 and g["DKC"] % 16 == 0
    assert Bc % 16 == 0
    assert thr % Br == 0 and Bc % (thr // Br) == 0, "softmax row grouping"
    assert (thr // Br) in (1, 2, 4, 8, 16, 32), "row group must fit a warp"
    assert (Br * d // 4) % thr == 0 and (Bc * g["DKC"] // 4) % thr == 0
    assert (Bc * g["DV"] // 4) % thr == 0
    if g["FSMEM"] > _MAX_SMEM_OPTIN:
        raise RuntimeError(f"FLASH d={d} needs {g['FSMEM']} B of shared memory, "
                           f"over the sm_89 cap of {_MAX_SMEM_OPTIN} B")


def _clear_stale_lock(name, max_age_s=900.0):
    """Drop a `lock` left behind by a build that was killed mid-flight.

    torch's `FileBaton` is not crash-safe: if the process that acquired the
    lock dies before `release()`, every later process that asks for the same
    extension blocks in `baton.wait()` FOREVER.  With one process per cell and
    several lanes on one host that turns a single SIGKILL into a permanently
    wedged campaign.  A build here takes ~40 s, so a lock older than 15 minutes
    cannot belong to a live compiler.
    """
    try:
        from torch.utils.cpp_extension import _get_build_directory
        d = _get_build_directory(name, False)
    except Exception:  # noqa: BLE001
        return
    lock = os.path.join(d, "lock")
    try:
        if os.path.exists(lock) and (time.time() - os.path.getmtime(lock)) > max_age_s:
            os.remove(lock)
    except OSError:
        pass


def _k3_smem():
    q = (K3_QK["BM"] * (K3_QK["BK"] + 8) + K3_QK["BN"] * (K3_QK["BK"] + 8)) * 2
    v = (K3_PV["BM"] * (K3_PV["BK"] + 8) + K3_PV["BK"] * (K3_PV["BN"] + 8)) * 2
    f = (K3_PVF["BK"] * (K3_PVF["BM"] + 4) + K3_PVF["BK"] * (K3_PVF["BN"] + 4)) * 4
    return q, v, f


def build(cfg) -> common2.Built2:
    common.setup_cuda_env()
    algo = cfg.variant
    if algo not in ("K3", "FLASH"):
        raise NotImplementedError(
            f"cuda_unlimited SDPA lane implements K3 and FLASH; got {algo!r}")
    d = int(cfg.extra["d"])
    sd = cfg.extra["sdtype"]
    pd = cfg.extra["pdtype"]
    S = common2.S_S
    scale = 1.0 / math.sqrt(d)

    common_src = _COMMON.substitute(
        SEQ=S, DH=d, SCALEV=("%.17ef" % scale),
        SD_FP16=1 if sd == "fp16" else 0,
        PD_FP16=1 if pd == "fp16" else 0)

    if algo == "FLASH":
        g = _flash_layout(d, sd, pd)
        _check_flash(d, g, pd)
        src = common_src + _FLASH.substitute(
            BR=g["Br"], BC=g["Bc"], THREADS=g["threads"],
            DKC=g["DKC"], DV=g["DV"],
            WARPS_M=g["WARPS_M"], WARPS_N=g["WARPS_N"],
            SWARPS_M=g["SWARPS_M"], SWARPS_N=g["SWARPS_N"],
            QSTRIDE=g["QSTRIDE"], KSTRIDE=g["KSTRIDE"], VSTRIDE=g["VSTRIDE"],
            SSTRIDE=g["SSTRIDE"], PSTRIDE=g["PSTRIDE"],
            QB=g["QB"], PB=g["PB"], ARENA=g["ARENA"], FSMEM=g["FSMEM"])
        n_kernels = 1
        tile_str = (f"Br={g['Br']} Bc={g['Bc']} threads={g['threads']} "
                    f"(PV warp grid {g['WARPS_M']}x{g['WARPS_N']}, QK warp grid "
                    f"{g['SWARPS_M']}x{g['SWARPS_N']}, d-load chunks "
                    f"K:{g['DKC']} V:{g['DV']})")
        smem = g["FSMEM"]
        grid = f"({S // g['Br']},B*H)"
        block = f"({g['threads']},1,1)"
    else:
        qs, vs, fs = _k3_smem()
        if d % K3_PV["BN"] or d % K3_PVF["BN"]:
            raise ValueError(f"head dim {d} must be divisible by 128")
        src = common_src + _K3.substitute(
            Q_BM=K3_QK["BM"], Q_BN=K3_QK["BN"], Q_BK=K3_QK["BK"],
            Q_THR=K3_QK["threads"], Q_WM=K3_QK["WM"], Q_WN=K3_QK["WN"],
            Q_SMEM=qs, SOFT_THR=K3_SOFT_THREADS,
            V_BM=K3_PV["BM"], V_BN=K3_PV["BN"], V_BK=K3_PV["BK"],
            V_THR=K3_PV["threads"], V_WM=K3_PV["WM"], V_WN=K3_PV["WN"],
            V_SMEM=vs,
            F_BM=K3_PVF["BM"], F_BN=K3_PVF["BN"], F_BK=K3_PVF["BK"],
            F_THR=K3_PVF["threads"], F_TM=K3_PVF["TM"], F_TN=K3_PVF["TN"],
            F_SMEM=fs)
        n_kernels = 3
        tile_str = (f"QK {K3_QK['BM']}x{K3_QK['BN']}x{K3_QK['BK']}/"
                    f"{K3_QK['threads']}thr; softmax 1 row/block x "
                    f"{K3_SOFT_THREADS}thr; PV "
                    + (f"{K3_PV['BM']}x{K3_PV['BN']}x{K3_PV['BK']}/"
                       f"{K3_PV['threads']}thr (mma)" if pd == "fp16" else
                       f"{K3_PVF['BM']}x{K3_PVF['BN']}x{K3_PVF['BK']}/"
                       f"{K3_PVF['threads']}thr TM={K3_PVF['TM']} "
                       f"TN={K3_PVF['TN']} (FFMA)"))
        smem = max(qs, vs if pd == "fp16" else fs)
        grid = f"QK({S//K3_QK['BN']},{S//K3_QK['BM']},B*H)"
        block = f"({K3_QK['threads']},1,1)"

    name = ("p2sdpa_unl_%s_d%d_s%s_p%s" % (algo.lower(), d, sd, pd))
    os.makedirs(common.ARTIFACTS_DIR, exist_ok=True)
    src_path = os.path.join(common.ARTIFACTS_DIR, name + ".cu")
    with open(src_path, "w") as fh:
        fh.write(src)

    flags = ["-O3", "-std=c++17", "-Xptxas=-v",
             "-gencode=arch=compute_89,code=sm_89"]
    cap_path = os.path.join(common.ARTIFACTS_DIR, name + ".buildlog.txt")
    ptxas_path = os.path.join(common.ARTIFACTS_DIR, name + ".ptxas.txt")

    _clear_stale_lock(name)
    t0 = time.perf_counter()
    with p1._FDCapture(cap_path) as cap:
        mod = load_inline(
            name=name,
            cpp_sources="torch::Tensor sdpa(torch::Tensor q, torch::Tensor k, "
                        "torch::Tensor v);",
            cuda_sources=src, functions=["sdpa"], with_cuda=True, verbose=True,
            extra_cuda_cflags=flags)
    info = p1._parse_ptxas(cap.text)
    # A ninja cache hit prints no ptxas banner at all, so the register and spill
    # counts would silently become null for every repeat of a cell -- i.e. for
    # almost the whole campaign, since only the first process of each cell
    # actually compiles.  Keep the last real banner in its own file and fall
    # back to it.
    if info:
        with open(ptxas_path, "w") as fh:
            fh.write("\n".join(l for l in cap.text.splitlines()
                               if ("ptxas" in l or "registers" in l
                                   or "spill" in l or "stack frame" in l)))
    elif os.path.exists(ptxas_path):
        with open(ptxas_path) as fh:
            info = p1._parse_ptxas(fh.read())

    kern = mod.sdpa

    def run(q, k, v):
        return kern(q, k, v)

    # source -> launchable INCLUDES the first launch: warm the KERNEL directly
    # (never through `run`, and never with a tensor that could prime a host-side
    # cache), at the real head dim and the real sequence length, because every
    # geometry constant is baked in and a first launch is where the lazy module
    # load and the smem opt-in actually happen.
    wq = torch.zeros((1, 1, S, d), dtype=torch.float32, device="cuda")
    _ = kern(wq, wq, wq)
    torch.cuda.synchronize()
    del wq, _
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    if pd == "fp16":
        pv_detail = ("P -> ldmatrix.x4 A fragment of "
                     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32; "
                     "V converted fp32->fp16 by cvt.rn.f16x2.f32 on the "
                     "global->shared path and read by "
                     "ldmatrix.x2.trans / ldmatrix.x4.trans")
    else:
        pv_detail = ("NO tensor core: PV is IEEE fp32 FFMA on the CUDA cores "
                     "with fp32 P and fp32 V. sm_89 would demote fp32 mma "
                     "operands to tf32, so the tensor-core path is not taken "
                     "at all rather than reported as 'fp32'")

    backend = (
        f"hand-written CUDA + inline PTX (load_inline, sm_89). algo={algo}, "
        f"head_dim={d}, tile: {tile_str}. "
        "QK^T: Q,K read fp32 from global and converted with cvt.rn.f16x2.f32 "
        "into shared memory, then mma.sync.aligned.m16n8k16.row.col."
        "f32.f16.f16.f32 with A via ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "and B via a plain (non-.trans) ldmatrix.x2 on K in its native "
        "(seq,d) layout; fp32 accumulator. "
        f"score_dtype={sd}: the fp32 mma accumulator is scaled by 1/sqrt(d) and "
        + ("rounded through __float2half_rn before the softmax reads it"
           if sd == "fp16" else "kept in fp32") + "; "
        + ("S is materialized in global memory in that dtype"
           if algo == "K3" else
           "the score tile lives only in registers and shared memory") + ". "
        f"prob_dtype={pd}: {pv_detail}. "
        "The mma accumulator is fp32 in every cell. "
        + ("online softmax: running max/sum in shared memory, per-row rescale "
           "applied straight to the fp32 accumulator REGISTERS using the "
           "mma fragment mapping gr=wm+i*16+(lane>>2), "
           "gc=wn+j*8+((lane&3)<<1) -- no smem round trip"
           if algo == "FLASH" else
           "softmax is a separate kernel over the materialized 512-wide rows "
           "(128 thr x 4 elements cached in registers, __shfl_xor_sync + a "
           "per-warp smem tree); it runs in place when sdtype==pdtype"))
    if algo == "K3" and pd == "fp32":
        backend += (". NOTE: the fp32 PV kernel's BK is 16 rather than the "
                    "mma path's 32 -- an FFMA schedule has no m16n8k16 to "
                    "match; BM=BN=128 is held identical")

    art = {
        "cuda_source": src,
        "cuda_source_path": src_path,
        "shared_bytes": smem,
        "grid": grid,
        "block": block,
        "ext_name": name,
        "algo": algo,
        "score_dtype": sd,
        "prob_dtype": pd,
        "n_kernels": n_kernels,
        "backend_detail": backend,
        "build_log_path": cap_path,
        "compile_flags": " ".join(flags),
    }
    art.update(info)
    art.setdefault("n_spills", 0)

    notes = (f"cuda_unlimited sdpa {algo} d={d} sdtype={sd} pdtype={pd}; "
             f"{tile_str}; smem={smem} B")
    return common2.Built2(run=run, compile_s=compile_s, artifacts=art,
                          notes=notes, n_kernels=n_kernels,
                          x_dtype=torch.float32)
