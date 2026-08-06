#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define ARITH_FP32 0
#define ON_LOAD 0
#define USE_ASYNC 1
#define KC_TILES 64
#define STAGES 3
#define WARPS_M 2
#define WARPS_N 4
#define WMT 64
#define WNT 32
#define NMF 4
#define NNF 2
#define LDA 40
#define LDB 136
#define FRAG_ELEMS 8
#define M_ 1024
#define N_ 8192
#define K_ 8192
#define BM 128
#define BN 128
#define BK 32
#define THREADS 256
#define SMEM_BYTES 56832
#define GTYPE half
#define TORCH_GTYPE torch::kHalf

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
