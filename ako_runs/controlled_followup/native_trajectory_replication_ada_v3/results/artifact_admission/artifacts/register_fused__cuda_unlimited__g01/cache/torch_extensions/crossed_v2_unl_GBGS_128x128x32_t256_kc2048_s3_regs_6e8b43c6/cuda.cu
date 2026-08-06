#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BM       128
#define BN       128
#define BK       32
#define THREADS  256
#define STAGES   3
#define KCB      64          /* cfg.kc / BK ; 0 == no chunk flush        */
#define F32G     0         /* 1 == cast=on_load: fp32 global pointers  */
#define APAD     8
#define BPAD     8
#define SMEM_BYTES 56832

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


#define HAS_BIAS 1
#define HAS_GELU 1
#define HAS_SOFTMAX 1
#define SOFT_N 8192
#define PROBLEM_M 1024
#define PROBLEM_N 8192
#define PROBLEM_K 8192
#define DYNAMIC_SMEM_BYTES 56832

__device__ __forceinline__ float gelu_exact(float v) {
    return v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));
}
__global__ void __launch_bounds__(THREADS) mma_gemm(
        const void* __restrict__ Ag, const void* __restrict__ Bg,
        const float* __restrict__ Bias,
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


    /* ---------------- Phase-2 fused epilogue, in registers --------------- */
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
#if HAS_BIAS
            const float b0 = Bias[gc], b1 = Bias[gc+1];
            v0 += b0; v1 += b1; v2 += b0; v3 += b1;
#endif
#if HAS_GELU
            v0 = gelu_exact(v0); v1 = gelu_exact(v1);
            v2 = gelu_exact(v2); v3 = gelu_exact(v3);
#endif
            *(float2*)(Cg + (long long)gr*N + gc)       = make_float2(v0, v1);
            *(float2*)(Cg + (long long)(gr+8)*N + gc)   = make_float2(v2, v3);
        }
    }
}

/* ---- Phase-2 row softmax ------------------------------------------------ */
#define SOFT_TH 256
#define SOFT_EPT (SOFT_N / SOFT_TH)
#define SOFT_NWARPS (SOFT_TH / 32)

__global__ __launch_bounds__(SOFT_TH)
void softmax_kernel(const float* __restrict__ X, float* __restrict__ Y) {
    __shared__ float sm_m[SOFT_NWARPS];
    __shared__ float sm_s[SOFT_NWARPS];
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    const float* __restrict__ xr = X + (size_t)row * SOFT_N;
    float* __restrict__ yr = Y + (size_t)row * SOFT_N;

    float lexp[SOFT_EPT];
    float m = -3.402823466e+38f;
#pragma unroll
    for (int k = 0; k < SOFT_EPT; ++k) {
        lexp[k] = xr[tid * SOFT_EPT + k];   /* reuse the buffer to hold x first */
        m = fmaxf(m, lexp[k]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if (lane == 0) sm_m[wid] = m;
    __syncthreads();
#pragma unroll
    for (int s = SOFT_NWARPS >> 1; s > 0; s >>= 1) {
        if (tid < s) sm_m[tid] = fmaxf(sm_m[tid], sm_m[tid + s]);
        __syncthreads();
    }
    const float row_max = sm_m[0];

    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < SOFT_EPT; ++k) {
        const float e = __expf(lexp[k] - row_max);
        lexp[k] = e;
        sum += e;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) sm_s[wid] = sum;
    __syncthreads();
#pragma unroll
    for (int s = SOFT_NWARPS >> 1; s > 0; s >>= 1) {
        if (tid < s) sm_s[tid] = sm_s[tid] + sm_s[tid + s];
        __syncthreads();
    }
    const float inv = 1.0f / sm_s[0];

#pragma unroll
    for (int k = 0; k < SOFT_EPT; ++k) yr[tid * SOFT_EPT + k] = lexp[k] * inv;
}

#include "checked_cuda_launch.h"
#include <ATen/cuda/CUDAContext.h>

torch::Tensor fused(torch::Tensor A, torch::Tensor B, torch::Tensor Bias) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && Bias.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.get_device() == B.get_device() && A.get_device() == Bias.get_device(),
                "same CUDA device required");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && Bias.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kHalf && B.scalar_type() == torch::kHalf,
                "operands must be fp16");
    TORCH_CHECK(Bias.scalar_type() == torch::kFloat32, "bias must be fp32");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == PROBLEM_M && A.size(1) == PROBLEM_K,
                "A shape mismatch");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == PROBLEM_K && B.size(1) == PROBLEM_N,
                "B shape mismatch");
    TORCH_CHECK(Bias.dim() == 1 && Bias.numel() == PROBLEM_N, "bias shape mismatch");

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto C = torch::empty({PROBLEM_M, PROBLEM_N}, opts);
    static bool attr_set = false;
    if (!attr_set) {
        checked_dynamic_smem((const void*)mma_gemm, DYNAMIC_SMEM_BYTES, "mma_gemm");
        attr_set = true;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(PROBLEM_N / BN, PROBLEM_M / BM), block(THREADS);
    mma_gemm<<<grid, block, DYNAMIC_SMEM_BYTES, stream>>>(
        A.data_ptr(), B.data_ptr(), Bias.data_ptr<float>(),
        C.data_ptr<float>(), PROBLEM_M, PROBLEM_N, PROBLEM_K);
    checked_kernel_launch("mma_gemm");
#if HAS_SOFTMAX
    auto Y = torch::empty({PROBLEM_M, PROBLEM_N}, opts);
    softmax_kernel<<<PROBLEM_M, SOFT_TH, 0, stream>>>(C.data_ptr<float>(),
                                                      Y.data_ptr<float>());
    checked_kernel_launch("softmax_kernel");
    return Y;
#else
    return C;
#endif
}
