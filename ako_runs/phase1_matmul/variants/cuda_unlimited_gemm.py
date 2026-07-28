"""cuda_unlimited lane — hand-written CUDA C++ + inline PTX, built with
torch.utils.cpp_extension.load_inline.

Everything is on the table in this lane: inline PTX `mma.sync`, `ldmatrix`,
`cp.async`.  What is *not* on the table is anything the SPEC forbids: no cuBLAS,
no autotuning, no block swizzle, no cross-block split-K/atomics, and variant A
must contain zero tensor-core arithmetic.

Structure of the four variants (SPEC.md):

  A  fp32 register-tiled SGEMM (8xTN thread tile, plain FFMA), one accumulator
     over the whole K extent, synchronous single-buffered smem loads.
  B  fp16 operands -> `mma.sync.aligned.m16n8k16.f32.f16.f16.f32`, fp32
     accumulator registers, ONE accumulator chain over all of K, synchronous
     single-buffered smem loads (pipeline genuinely off).
  C  as B, but every `cfg.kc` elements of K the mma accumulator registers are
     added into a second fp32 register array and zeroed.
  D  as C, plus a `cfg.stages`-deep `cp.async` software pipeline on the
     global->shared loads.

Cast modes (fp16 arms only):
  precast    kernel reads fp16 global   (host hands it fp16 tensors)
  in_region  kernel reads fp16 global   (`run` calls .half() inside the timer)
  on_load    kernel reads fp32 global and converts with `cvt.rn.f16x2.f32` on
             the way into shared memory.  Because `cp.async` cannot convert
             dtype, the stages>1 pipeline for this mode is expressed as a
             register-staged prefetch (ld.global -> cvt -> st.shared) of depth
             stages-1 over an smem ring of depth stages.
"""
from __future__ import annotations

import io
import os
import re
import string
import time

import torch

import common

_MMA_SRC = string.Template(r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BM       $BM
#define BN       $BN
#define BK       $BK
#define THREADS  $THREADS
#define STAGES   $STAGES
#define KCB      $KCB          /* cfg.kc / BK ; 0 == no chunk flush        */
#define F32G     $F32G         /* 1 == cast=on_load: fp32 global pointers  */
#define APAD     $APAD
#define BPAD     $BPAD
#define SMEM_BYTES $SMEM_BYTES

#define WARPS_N  4
#define WARPS_M  (THREADS/32/WARPS_N)
#define WM       (BM/WARPS_M)
#define WN       (BN/WARPS_N)
#define MT       (WM/16)            /* m16n8k16 tiles along M per warp */
#define NT       (WN/8)             /* m16n8k16 tiles along N per warp */
#define NKS      (BK/16)            /* mma k-steps inside one BK block  */
#define NACC     (MT*NT*4)

#define ASTRIDE  (BK+APAD)
#define BSTRIDE  (BN+BPAD)
#define ASZ      (BM*ASTRIDE)       /* halves */
#define BSZ      (BK*BSTRIDE)
#define STAGE_H  (ASZ+BSZ)

#define A_VPR    (BK/8)             /* 8-half vectors per A row */
#define B_VPR    (BN/8)
#define A_ITERS  ((BM*BK/8)/THREADS)
#define B_ITERS  ((BK*BN/8)/THREADS)
#define UNR      (STAGES-1)

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

/* pack two floats into one .b16x2 register, round-to-nearest-even.
   cvt.rn.f16x2.f32 d, a, b   ->  d.hi = f16(a), d.lo = f16(b)            */
__device__ __forceinline__ unsigned pack2(float lo, float hi) {
    unsigned d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

/* ---- global -> registers (one BKxBM / BKxBN tile, already fp16-packed) --- */
__device__ __forceinline__ void g2r(const void* __restrict__ Ag,
                                    const void* __restrict__ Bg,
                                    unsigned (&ra)[A_ITERS][4],
                                    unsigned (&rb)[B_ITERS][4],
                                    int m0, int n0, int k0, int K, int N) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < A_ITERS; ++i) {
        int v = tid + i*THREADS;
        int r = v / A_VPR, c = (v % A_VPR)*8;
#if F32G
        const float* s = (const float*)Ag + (long long)(m0+r)*K + k0 + c;
        float4 x = *(const float4*)(s);
        float4 y = *(const float4*)(s+4);
        ra[i][0] = pack2(x.x, x.y); ra[i][1] = pack2(x.z, x.w);
        ra[i][2] = pack2(y.x, y.y); ra[i][3] = pack2(y.z, y.w);
#else
        const half* s = (const half*)Ag + (long long)(m0+r)*K + k0 + c;
        uint4 u = *(const uint4*)s;
        ra[i][0]=u.x; ra[i][1]=u.y; ra[i][2]=u.z; ra[i][3]=u.w;
#endif
    }
#pragma unroll
    for (int i = 0; i < B_ITERS; ++i) {
        int v = tid + i*THREADS;
        int r = v / B_VPR, c = (v % B_VPR)*8;
#if F32G
        const float* s = (const float*)Bg + (long long)(k0+r)*N + n0 + c;
        float4 x = *(const float4*)(s);
        float4 y = *(const float4*)(s+4);
        rb[i][0] = pack2(x.x, x.y); rb[i][1] = pack2(x.z, x.w);
        rb[i][2] = pack2(y.x, y.y); rb[i][3] = pack2(y.z, y.w);
#else
        const half* s = (const half*)Bg + (long long)(k0+r)*N + n0 + c;
        uint4 u = *(const uint4*)s;
        rb[i][0]=u.x; rb[i][1]=u.y; rb[i][2]=u.z; rb[i][3]=u.w;
#endif
    }
}

__device__ __forceinline__ void r2s(unsigned (&ra)[A_ITERS][4],
                                    unsigned (&rb)[B_ITERS][4],
                                    half* As, half* Bs) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < A_ITERS; ++i) {
        int v = tid + i*THREADS;
        int r = v / A_VPR, c = (v % A_VPR)*8;
        *(uint4*)(As + r*ASTRIDE + c) = make_uint4(ra[i][0],ra[i][1],ra[i][2],ra[i][3]);
    }
#pragma unroll
    for (int i = 0; i < B_ITERS; ++i) {
        int v = tid + i*THREADS;
        int r = v / B_VPR, c = (v % B_VPR)*8;
        *(uint4*)(Bs + r*BSTRIDE + c) = make_uint4(rb[i][0],rb[i][1],rb[i][2],rb[i][3]);
    }
}

/* ---- synchronous global -> shared (stages == 1) ------------------------- */
__device__ __forceinline__ void g2s_sync(const void* __restrict__ Ag,
                                         const void* __restrict__ Bg,
                                         half* As, half* Bs,
                                         int m0, int n0, int k0, int K, int N) {
    unsigned ra[A_ITERS][4], rb[B_ITERS][4];
    g2r(Ag, Bg, ra, rb, m0, n0, k0, K, N);
    r2s(ra, rb, As, Bs);
}

#if !F32G
/* ---- asynchronous global -> shared (stages > 1, fp16 global) ------------ */
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void g2s_async(const void* __restrict__ Ag,
                                          const void* __restrict__ Bg,
                                          half* As, half* Bs,
                                          int m0, int n0, int k0, int K, int N) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < A_ITERS; ++i) {
        int v = tid + i*THREADS;
        int r = v / A_VPR, c = (v % A_VPR)*8;
        cp16(smem_u32(As + r*ASTRIDE + c),
             (const half*)Ag + (long long)(m0+r)*K + k0 + c);
    }
#pragma unroll
    for (int i = 0; i < B_ITERS; ++i) {
        int v = tid + i*THREADS;
        int r = v / B_VPR, c = (v % B_VPR)*8;
        cp16(smem_u32(Bs + r*BSTRIDE + c),
             (const half*)Bg + (long long)(k0+r)*N + n0 + c);
    }
}
#endif

/* ---- one BK block of mma.sync ------------------------------------------ */
__device__ __forceinline__ void mma_tile(const half* As, const half* Bs,
                                         float (&acc)[NACC],
                                         int lane, int wm, int wn) {
#pragma unroll
    for (int ks = 0; ks < NKS; ++ks) {
        unsigned a[MT][4], b[NT][2];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            /* ldmatrix .x4 : lane l supplies row (l%8) of matrix (l/8);
               matrices are (m0..7,k0..7) (m8..15,k0..7) (m0..7,k8..15)
               (m8..15,k8..15) which is exactly the m16n8k16 A fragment.   */
            const half* p = As + (wm + i*16 + (lane & 15))*ASTRIDE
                               + ks*16 + ((lane >> 4) << 3);
            unsigned ad = smem_u32(p);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                         "{%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3])
                         : "r"(ad));
        }
#pragma unroll
        for (int jj = 0; jj < NT/2; ++jj) {
            /* .trans turns the k-major smem tile into the column-major B
               fragment; one .x4 yields two n-tiles of 8.                  */
            const half* p = Bs + (ks*16 + (lane & 7) + (((lane >> 3) & 1) << 3))*BSTRIDE
                               + wn + jj*16 + ((lane >> 4) << 3);
            unsigned ad = smem_u32(p);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                         "{%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b[2*jj][0]), "=r"(b[2*jj][1]),
                           "=r"(b[2*jj+1][0]), "=r"(b[2*jj+1][1])
                         : "r"(ad));
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[(i*NT+j)*4+0]), "+f"(acc[(i*NT+j)*4+1]),
                      "+f"(acc[(i*NT+j)*4+2]), "+f"(acc[(i*NT+j)*4+3])
                    : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]),
                      "r"(b[j][0]), "r"(b[j][1]));
            }
        }
    }
}

#if KCB > 0
#define FLUSH(kb)  do { if ((((kb)+1) % KCB) == 0) {                        \
        _Pragma("unroll") for (int _i = 0; _i < NACC; ++_i) {               \
            sacc[_i] += acc[_i]; acc[_i] = 0.f; } } } while (0)
#else
#define FLUSH(kb)  do { } while (0)
#endif

__global__ void __launch_bounds__(THREADS) mma_gemm(
        const void* __restrict__ Ag, const void* __restrict__ Bg,
        float* __restrict__ Cg, int M, int N, int K) {
    extern __shared__ __align__(16) char sraw[];
    half* sm = reinterpret_cast<half*>(sraw);

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int wm = (wid / WARPS_N) * WM;
    const int wn = (wid % WARPS_N) * WN;

    float acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
#if KCB > 0
    float sacc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) sacc[i] = 0.f;
#endif
    const int NKB = K / BK;

#if STAGES == 1
    /* ---------------- pipeline genuinely OFF ---------------------------- */
    for (int kb = 0; kb < NKB; ++kb) {
        g2s_sync(Ag, Bg, sm, sm + ASZ, m0, n0, kb*BK, K, N);
        __syncthreads();
        mma_tile(sm, sm + ASZ, acc, lane, wm, wn);
        __syncthreads();
        FLUSH(kb);
    }
#elif !F32G
    /* ---------------- cp.async ring, depth STAGES ----------------------- */
#pragma unroll
    for (int s = 0; s < STAGES-1; ++s) {
        g2s_async(Ag, Bg, sm + s*STAGE_H, sm + s*STAGE_H + ASZ, m0, n0, s*BK, K, N);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int kb = 0; kb < NKB; ++kb) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES-2));
        __syncthreads();
        {
            half* st = sm + (kb % STAGES)*STAGE_H;
            mma_tile(st, st + ASZ, acc, lane, wm, wn);
        }
        __syncthreads();
        int nk = kb + STAGES - 1;
        if (nk < NKB) {
            half* st = sm + (nk % STAGES)*STAGE_H;
            g2s_async(Ag, Bg, st, st + ASZ, m0, n0, nk*BK, K, N);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        FLUSH(kb);
    }
#else
    /* ---------------- register-staged prefetch ring, depth STAGES -------
       cp.async cannot convert fp32->fp16, so the on_load pipeline keeps the
       in-flight tiles in registers (already packed to fp16) while the smem
       ring stays STAGES deep.  The kb loop is unrolled by UNR=STAGES-1 so the
       register-ring index is a compile-time constant.                      */
    unsigned ra[UNR][A_ITERS][4], rb[UNR][B_ITERS][4];
#pragma unroll
    for (int s = 0; s < STAGES-1; ++s)
        g2s_sync(Ag, Bg, sm + s*STAGE_H, sm + s*STAGE_H + ASZ, m0, n0, s*BK, K, N);
#pragma unroll
    for (int u = 0; u < UNR; ++u)
        g2r(Ag, Bg, ra[u], rb[u], m0, n0, (STAGES-1+u)*BK, K, N);
    __syncthreads();

    for (int kb = 0; kb < NKB; kb += UNR) {
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            int k = kb + u;
            {
                half* st = sm + (k % STAGES)*STAGE_H;
                mma_tile(st, st + ASZ, acc, lane, wm, wn);
            }
            __syncthreads();
            int sk = k + STAGES - 1;
            if (sk < NKB) {
                half* st = sm + (sk % STAGES)*STAGE_H;
                r2s(ra[u], rb[u], st, st + ASZ);
            }
            __syncthreads();
            int lk = k + UNR + STAGES - 1;
            if (lk < NKB)
                g2r(Ag, Bg, ra[u], rb[u], m0, n0, lk*BK, K, N);
            FLUSH(k);
        }
    }
#endif

    /* ---------------- epilogue ------------------------------------------ */
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int x = (i*NT+j)*4;
#if KCB > 0
            float v0 = sacc[x+0] + acc[x+0], v1 = sacc[x+1] + acc[x+1];
            float v2 = sacc[x+2] + acc[x+2], v3 = sacc[x+3] + acc[x+3];
#else
            float v0 = acc[x+0], v1 = acc[x+1], v2 = acc[x+2], v3 = acc[x+3];
#endif
            int gr = m0 + wm + i*16 + (lane >> 2);
            int gc = n0 + wn + j*8  + ((lane & 3) << 1);
            *(float2*)(Cg + (long long)gr*N + gc)       = make_float2(v0, v1);
            *(float2*)(Cg + (long long)(gr+8)*N + gc)   = make_float2(v2, v3);
        }
    }
}

torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N},
                torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(mma_gemm,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        attr_set = true;
    }
    dim3 grid(N/BN, M/BM), block(THREADS);
    mma_gemm<<<grid, block, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr(), B.data_ptr(), C.data_ptr<float>(), M, N, K);
    return C;
}
""")


_SGEMM_SRC = string.Template(r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define BM       $BM
#define BN       $BN
#define BK       $BK
#define THREADS  $THREADS
#define SMEM_BYTES $SMEM_BYTES

#define TM       8
#define NTHR_M   (BM/TM)              /* 16 */
#define NTHR_N   (THREADS/NTHR_M)     /* 16 */
#define TN       (BN/NTHR_N)

#define APADF    4
#define BPADF    4
#define ASF      (BM+APADF)
#define BSF      (BN+BPADF)

#define A_ITERS  ((BM*BK/4)/THREADS)
#define B_ITERS  ((BK*BN/4)/THREADS)
#define A_VPR    (BK/4)
#define B_VPR    (BN/4)

/* Variant A: plain register-tiled fp32 SGEMM.  No wmma::, no mma.sync, no
   inline-PTX arithmetic of any kind -- the inner product is `acc += a*b`,
   which nvcc lowers to FFMA on the CUDA cores.  stages==1, so the smem stage
   is single-buffered and separated by __syncthreads().                      */
__global__ void __launch_bounds__(THREADS) sgemm(
        const float* __restrict__ Ag, const float* __restrict__ Bg,
        float* __restrict__ Cg, int M, int N, int K) {
    extern __shared__ __align__(16) float sm[];
    float* As = sm;              /* [BK][ASF]  (transposed: As[k][m]) */
    float* Bs = sm + BK*ASF;     /* [BK][BSF] */

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int tx = tid % NTHR_N;
    const int ty = tid / NTHR_N;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int NKB = K / BK;
    for (int kb = 0; kb < NKB; ++kb) {
        const int k0 = kb * BK;
#pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int v = tid + it*THREADS;
            int r = v / A_VPR, c = (v % A_VPR)*4;
            float4 x = *(const float4*)(Ag + (long long)(m0+r)*K + k0 + c);
            As[(c+0)*ASF + r] = x.x;
            As[(c+1)*ASF + r] = x.y;
            As[(c+2)*ASF + r] = x.z;
            As[(c+3)*ASF + r] = x.w;
        }
#pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int v = tid + it*THREADS;
            int r = v / B_VPR, c = (v % B_VPR)*4;
            *(float4*)(Bs + r*BSF + c) =
                *(const float4*)(Bg + (long long)(k0+r)*N + n0 + c);
        }
        __syncthreads();
#pragma unroll 4
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k*ASF + ty*TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k*BSF + tx*TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        float* dst = Cg + (long long)(m0 + ty*TM + i)*N + n0 + tx*TN;
#pragma unroll
        for (int j = 0; j < TN; j += 4)
            *(float4*)(dst + j) = make_float4(acc[i][j], acc[i][j+1],
                                              acc[i][j+2], acc[i][j+3]);
    }
}

torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N},
                torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(sgemm,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        attr_set = true;
    }
    dim3 grid(N/BN, M/BM), block(THREADS);
    sgemm<<<grid, block, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
""")


# --------------------------------------------------------------------------- #
class _FDCapture:
    """Capture the *file descriptors* 1/2 so ninja's subprocess output (which
    torch writes straight to fd 1 when verbose=True) lands in our buffer."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self._f = open(self.path, "w+")
        self._o1, self._o2 = os.dup(1), os.dup(2)
        os.dup2(self._f.fileno(), 1)
        os.dup2(self._f.fileno(), 2)
        return self

    def __exit__(self, *a):
        os.dup2(self._o1, 1); os.close(self._o1)
        os.dup2(self._o2, 2); os.close(self._o2)
        self._f.flush()
        self._f.seek(0)
        self.text = self._f.read()
        self._f.close()
        return False


def _parse_ptxas(text: str) -> dict:
    out = {}
    m = re.findall(r"Used (\d+) registers", text)
    if m:
        out["n_regs"] = max(int(x) for x in m)
    m = re.findall(r"(\d+) bytes spill stores", text)
    if m:
        out["n_spill_stores"] = max(int(x) for x in m)
    m = re.findall(r"(\d+) bytes spill loads", text)
    if m:
        out["n_spills"] = max(int(x) for x in m)
    m = re.findall(r"(\d+) bytes stack frame", text)
    if m:
        out["stack_frame_bytes"] = max(int(x) for x in m)
    return out


_MAX_SMEM_OPTIN = 101376  # sm_89: 99 KB per block


def _geom_checks(cfg: common.Config):
    if cfg.M % cfg.BM or cfg.N % cfg.BN or cfg.K % cfg.BK:
        raise ValueError(f"problem {cfg.M}x{cfg.N}x{cfg.K} not divisible by "
                         f"tile {cfg.BM}x{cfg.BN}x{cfg.BK}")
    if cfg.threads % 32 or (cfg.threads // 32) % 4:
        raise ValueError("threads must be a multiple of 128 (4 warps along N)")
    if cfg.arith == "fp32":
        # variant A: 16 thread-rows of TM=8, threads/16 thread-cols of TN
        if cfg.BM % 8 or (cfg.BM // 8) == 0 or cfg.threads % (cfg.BM // 8):
            raise ValueError("variant A needs BM%8==0 and threads%(BM/8)==0")
        tn = cfg.BN // (cfg.threads // (cfg.BM // 8))
        if tn < 4 or tn % 4 or cfg.BN % (cfg.threads // (cfg.BM // 8)):
            raise ValueError(f"variant A thread tile TN={tn} must be a positive "
                             f"multiple of 4 (float4 epilogue)")
        if (cfg.BM * cfg.BK // 4) % cfg.threads or (cfg.BK * cfg.BN // 4) % cfg.threads:
            raise ValueError("variant A: tile elements must divide 4*threads")
    else:
        warps_m = cfg.threads // 32 // 4
        if cfg.BK % 16 or cfg.BM % (16 * warps_m) or cfg.BN % 64:
            raise ValueError(f"mma path needs BK%16==0, BM%{16*warps_m}==0, BN%64==0 "
                             f"(warp grid {warps_m}x4, m16n8k16 tiles, .x4 ldmatrix "
                             f"covers 2 n-tiles)")
        if (cfg.BM * cfg.BK // 8) % cfg.threads or (cfg.BK * cfg.BN // 8) % cfg.threads:
            raise ValueError("mma path: tile halves must divide 8*threads")


def build(cfg: common.Config) -> common.Built:
    from torch.utils.cpp_extension import load_inline

    common.setup_cuda_env()
    _geom_checks(cfg)
    t0 = time.perf_counter()

    kcb = 0
    if cfg.kc:
        if cfg.kc % cfg.BK:
            raise ValueError(f"kc={cfg.kc} must be a multiple of BK={cfg.BK}")
        kcb = cfg.kc // cfg.BK

    notes = []
    if cfg.arith == "fp32":
        # ---------------- variant A: fp32 SGEMM, no tensor cores ----------
        smem = (cfg.BK * (cfg.BM + 4) + cfg.BK * (cfg.BN + 4)) * 4
        if smem > _MAX_SMEM_OPTIN:
            raise RuntimeError(f"variant A smem {smem} B > sm_89 limit "
                               f"{_MAX_SMEM_OPTIN} B at {cfg.BM}x{cfg.BN}x{cfg.BK}")
        src = _SGEMM_SRC.substitute(BM=cfg.BM, BN=cfg.BN, BK=cfg.BK,
                                    THREADS=cfg.threads, SMEM_BYTES=smem)
        name = f"cu_unl_sgemm_{cfg.BM}_{cfg.BN}_{cfg.BK}_{cfg.threads}"
        backend = ("fp32 register-tiled SGEMM, TM=8 x TN=%d, acc += a*b (FFMA); "
                   "no wmma::, no mma.sync, no inline-PTX arithmetic; "
                   "single-buffered smem, __syncthreads()-separated (stages==1)"
                   % (cfg.BN // (cfg.threads // (cfg.BM // 8))))
        f32g = 0
    else:
        # ---------------- variants B/C/D: mma.sync tensor cores -----------
        f32g = 1 if cfg.cast == "on_load" else 0
        apad, bpad = 8, 8
        def _smem(ap, bp):
            return (cfg.BM * (cfg.BK + ap) + cfg.BK * (cfg.BN + bp)) * 2 * cfg.stages
        smem = _smem(apad, bpad)
        if smem > _MAX_SMEM_OPTIN:
            apad, bpad = 0, 0
            smem = _smem(apad, bpad)
            notes.append("padding dropped to 0 to fit shared memory "
                         "(bank conflicts on ldmatrix accepted)")
        if smem > _MAX_SMEM_OPTIN:
            raise RuntimeError(
                f"shared memory {smem} B needed for BM={cfg.BM} BN={cfg.BN} "
                f"BK={cfg.BK} stages={cfg.stages} exceeds the sm_89 per-block "
                f"limit of {_MAX_SMEM_OPTIN} B even with zero padding "
                f"({(cfg.BM*cfg.BK + cfg.BK*cfg.BN)*2} B per stage). "
                f"This geometry/stage combination is not realizable on this GPU.")
        nkb = cfg.K // cfg.BK
        if f32g and cfg.stages > 1 and nkb % (cfg.stages - 1):
            raise ValueError(f"on_load pipeline unrolls by stages-1="
                             f"{cfg.stages-1}, which must divide K/BK={nkb}")
        src = _MMA_SRC.substitute(BM=cfg.BM, BN=cfg.BN, BK=cfg.BK,
                                  THREADS=cfg.threads, STAGES=cfg.stages,
                                  KCB=kcb, F32G=f32g, APAD=apad, BPAD=bpad,
                                  SMEM_BYTES=smem)
        name = (f"cu_unl_mma_{cfg.BM}_{cfg.BN}_{cfg.BK}_{cfg.threads}"
                f"_s{cfg.stages}_kcb{kcb}_g{f32g}_p{apad}{bpad}")
        pipe = ("synchronous single-buffered smem (pipeline off)" if cfg.stages == 1
                else ("cp.async.cg.shared.global.L2::128B + commit_group/"
                      f"wait_group ring, depth {cfg.stages}" if not f32g
                      else ("register-staged prefetch ring depth "
                            f"{cfg.stages-1} over an smem ring depth {cfg.stages} "
                            "(cp.async cannot convert fp32->fp16)")))
        backend = ("inline-PTX mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32; "
                   "fragments via ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16; "
                   f"warp grid {cfg.threads//32//4}x4, warp tile "
                   f"{cfg.BM//(cfg.threads//32//4)}x{cfg.BN//4}; "
                   f"{'kc flush every %d BK-blocks into a 2nd fp32 register array' % kcb if kcb else 'single fp32 accumulator chain over all of K'}; "
                   f"{pipe}; "
                   f"global operands {'fp32 -> cvt.rn.f16x2.f32 on the way into smem' if f32g else 'fp16'}")

    os.makedirs(common.ARTIFACTS_DIR, exist_ok=True)
    src_path = os.path.join(common.ARTIFACTS_DIR, f"{name}.cu")
    with open(src_path, "w") as fh:
        fh.write(src)

    flags = ["-O3", "-std=c++17", "-Xptxas=-v", "-Xptxas=-O3", "-DNDEBUG"]
    log_path = os.path.join(common.ARTIFACTS_DIR, f"{name}.ptxas.txt")
    cap_path = os.path.join(common.ARTIFACTS_DIR, f"{name}.buildlog.txt")
    with _FDCapture(cap_path) as cap:
        mod = load_inline(name=name,
                          cpp_sources="torch::Tensor gemm(torch::Tensor A, "
                                      "torch::Tensor B);",
                          cuda_sources=src,
                          functions=["gemm"], with_cuda=True, verbose=True,
                          extra_cuda_cflags=flags)
    build_text = cap.text
    info = _parse_ptxas(build_text)
    if info:
        # keep the whole ptxas block (registers AND the spill/stack-frame line,
        # which is on a continuation line that contains neither word) so a later
        # cache-hit build can still report it
        with open(log_path, "w") as fh:
            fh.write("\n".join(l for l in build_text.splitlines()
                               if ("ptxas" in l or "registers" in l
                                   or "spill" in l or "stack frame" in l)))
    elif os.path.exists(log_path):
        with open(log_path) as fh:
            info = _parse_ptxas(fh.read())

    gemm = mod.gemm
    if cfg.arith == "fp16" and cfg.cast == "in_region":
        def run(A, B):
            return gemm(A.half(), B.half())
    else:
        def run(A, B):
            return gemm(A, B)

    # force the lazy CUDA module load / any first-launch cost out of the timer
    dt = torch.float16 if cfg.input_dtype == torch.float16 else torch.float32
    kw = max(4, cfg.stages + 1)
    if kcb:
        kw = max(kw, kcb)
    tk = cfg.BK * kw
    a = torch.zeros(cfg.BM, tk, device="cuda", dtype=dt)
    b = torch.zeros(tk, cfg.BN, device="cuda", dtype=dt)
    run(a, b)
    torch.cuda.synchronize()
    del a, b
    compile_s = time.perf_counter() - t0

    smem_bytes = smem
    artifacts = {
        "cuda_source": src,
        "cuda_source_path": src_path,
        "shared_bytes": smem_bytes,
        "grid": [cfg.N // cfg.BN, cfg.M // cfg.BM],
        "block": [cfg.threads, 1, 1],
        "backend_detail": backend,
        "build_log_path": cap_path,
        "compile_flags": " ".join(flags),
    }
    artifacts.update(info)
    artifacts.setdefault("n_spills", 0)

    return common.Built(run=run, compile_s=compile_s,
                        input_dtype=cfg.input_dtype,
                        artifacts=artifacts,
                        notes="; ".join(notes))
