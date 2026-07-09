import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 2 (blind redo): WMMA TF32 with cp.async double-buffered pipeline
# BM=128, BN=128, BKK=32, 8 warps, 2-stage async pipeline
# cp.async lets GPU overlap global loads with WMMA compute
# Key: sm_89 Ada has hardware support for cp.async
# Smem (2 stages): 2*(As[128][36] + Bs[32][132]) = 2*(18432+16896) = 70656B > 48KB!
# Reduce: BKK=16, 2 stages: 2*(128*20+16*132)*4 = 2*(10240+8448)*4? No... BKK=16:
#   As[128][20]=10240B, Bs[16][132]=8448B → per stage: 18688B, 2 stages: 37376B < 48KB ✓
# With BKK=32, single stage: 35328B. With double buffer BKK=16: 37376B still fits.

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <float.h>
#include <torch/extension.h>
using namespace nvcuda;

// Config for double-buffered pipeline
static constexpr int BM_DB   = 128;
static constexpr int BN_DB   = 128;
static constexpr int BKK_DB  =  16;   // smaller BK to allow double-buffering
static constexpr int STAGES  =   2;
static constexpr int WMMA_M  =  16, WMMA_N = 16, WMMA_K = 8;
static constexpr int WARPS_M =   4, WARPS_N = 2;
static constexpr int WM_DB   =   2, WN_DB = 4;
static constexpr int NTHREADS_DB = 256;
// Smem per stage: As[128][20] + Bs[16][132] = 10240+8448 = 18688 floats... wait bytes:
//   As: 128*(BKK_DB+4)*4 = 128*20*4 = 10240 bytes
//   Bs: BKK_DB*(BN_DB+4)*4 = 16*132*4 = 8448 bytes
//   Per stage: 18688 bytes. Two stages: 37376 bytes < 48KB ✓

static constexpr int SOFT_T_DB = 256;
static constexpr int EPT_DB    =  32;

__device__ __forceinline__ float gelu_ex(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// ─── WMMA TF32 GEMM with cp.async double-buffer pipeline ────────────────────
// A [M,K] row-major, WT [K,N] row-major
// C [M,N] stores raw GEMM output (bias/GELU applied by next kernel)
__global__ __launch_bounds__(NTHREADS_DB)
void wmma_gemm_db(
    const float* __restrict__ A,
    const float* __restrict__ WT,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int wid = threadIdx.x / 32;
    const int wr  = wid / WARPS_N;
    const int wc  = wid % WARPS_N;
    const int bm  = blockIdx.y * BM_DB;
    const int bn  = blockIdx.x * BN_DB;
    const int wm0 = wr * (WM_DB * WMMA_M);
    const int wn0 = wc * (WN_DB * WMMA_N);

    // Double-buffered smem: [STAGES][As+Bs]
    __shared__ float As[STAGES][BM_DB][BKK_DB + 4];  // [2][128][20] = 20480 floats = 81920B? No:
    // 2*128*20*4 = 20480 bytes
    __shared__ float Bs[STAGES][BKK_DB][BN_DB + 4];  // [2][16][132] = 4224 floats = 16896B
    // Total: 20480 + 16896 = 37376 bytes < 48KB ✓

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WM_DB][WN_DB];
    #pragma unroll
    for (int i = 0; i < WM_DB; i++)
        #pragma unroll
        for (int j = 0; j < WN_DB; j++)
            wmma::fill_fragment(acc[i][j], 0.f);

    const int tid   = threadIdx.x;
    const int nstep = (K + BKK_DB - 1) / BKK_DB;

    // Preload first stage (stage 0) using cp.async
    {
        int kb = 0;
        int s  = 0;
        #pragma unroll
        for (int e = tid; e < BM_DB * BKK_DB; e += NTHREADS_DB) {
            int m = e / BKK_DB, k = e % BKK_DB;
            int gm = bm + m, gk = kb + k;
            if (gm < M && gk < K)
                __pipeline_memcpy_async(&As[s][m][k], &A[gm * K + gk], sizeof(float));
            else
                As[s][m][k] = 0.f;
        }
        #pragma unroll
        for (int e = tid; e < BKK_DB * BN_DB; e += NTHREADS_DB) {
            int k = e / BN_DB, n = e % BN_DB;
            int gk = kb + k, gn = bn + n;
            if (gk < K && gn < N)
                __pipeline_memcpy_async(&Bs[s][k][n], &WT[gk * N + gn], sizeof(float));
            else
                Bs[s][k][n] = 0.f;
        }
        __pipeline_commit();
    }

    for (int step = 0; step < nstep; step++) {
        const int s_cur  = step % STAGES;
        const int s_next = (step + 1) % STAGES;

        // Prefetch next stage while current is in-flight
        if (step + 1 < nstep) {
            int kb = (step + 1) * BKK_DB;
            #pragma unroll
            for (int e = tid; e < BM_DB * BKK_DB; e += NTHREADS_DB) {
                int m = e / BKK_DB, k = e % BKK_DB;
                int gm = bm + m, gk = kb + k;
                if (gm < M && gk < K)
                    __pipeline_memcpy_async(&As[s_next][m][k], &A[gm * K + gk], sizeof(float));
                else
                    As[s_next][m][k] = 0.f;
            }
            #pragma unroll
            for (int e = tid; e < BKK_DB * BN_DB; e += NTHREADS_DB) {
                int k = e / BN_DB, n = e % BN_DB;
                int gk = kb + k, gn = bn + n;
                if (gk < K && gn < N)
                    __pipeline_memcpy_async(&Bs[s_next][k][n], &WT[gk * N + gn], sizeof(float));
                else
                    Bs[s_next][k][n] = 0.f;
            }
            __pipeline_commit();
        }

        // Wait for current stage data to be ready
        __pipeline_wait_prior(1);
        __syncthreads();

        // WMMA: BKK_DB/WMMA_K = 16/8 = 2 steps
        #pragma unroll
        for (int ks = 0; ks < BKK_DB / WMMA_K; ks++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> af[WM_DB];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> bf[WN_DB];
            #pragma unroll
            for (int i = 0; i < WM_DB; i++)
                wmma::load_matrix_sync(af[i],
                    &As[s_cur][wm0 + i * WMMA_M][ks * WMMA_K], BKK_DB + 4);
            #pragma unroll
            for (int j = 0; j < WN_DB; j++)
                wmma::load_matrix_sync(bf[j],
                    &Bs[s_cur][ks * WMMA_K][wn0 + j * WMMA_N], BN_DB + 4);
            #pragma unroll
            for (int i = 0; i < WM_DB; i++)
                #pragma unroll
                for (int j = 0; j < WN_DB; j++)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Epilogue: store frags to C
    #pragma unroll
    for (int i = 0; i < WM_DB; i++)
        #pragma unroll
        for (int j = 0; j < WN_DB; j++) {
            int gm = bm + wm0 + i * WMMA_M;
            int gn = bn + wn0 + j * WMMA_N;
            if (gm < M && gn < N)
                wmma::store_matrix_sync(&C[gm * N + gn], acc[i][j], N,
                                        wmma::mem_row_major);
        }
}

// ─── Fused bias + GELU + softmax (all in registers) ─────────────────────────
__global__ void bias_gelu_softmax_k(
    float* __restrict__ C,
    const float* __restrict__ b,
    int M, int N)
{
    const int row  = blockIdx.x;
    if (row >= M) return;
    float* rp = C + (long)row * N;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int nw   = SOFT_T_DB / 32;
    __shared__ float sm[8];

    float reg[EPT_DB];
    float mx = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < EPT_DB; i++) {
        int idx = threadIdx.x + i * SOFT_T_DB;
        float v = rp[idx] + b[idx];
        v = gelu_ex(v);
        reg[i] = v;
        mx = fmaxf(mx, v);
    }
    for (int d = 16; d > 0; d >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, d));
    if (!lane) sm[warp] = mx;
    __syncthreads();
    if (!warp) {
        float v = (lane < nw) ? sm[lane] : -FLT_MAX;
        for (int d = 4; d > 0; d >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, d));
        if (!lane) sm[0] = v;
    }
    __syncthreads();
    mx = sm[0];

    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < EPT_DB; i++) { reg[i] = expf(reg[i] - mx); s += reg[i]; }
    for (int d = 16; d > 0; d >>= 1) s += __shfl_xor_sync(~0u, s, d);
    if (!lane) sm[warp] = s;
    __syncthreads();
    if (!warp) {
        float v = (lane < nw) ? sm[lane] : 0.f;
        for (int d = 4; d > 0; d >>= 1) v += __shfl_xor_sync(~0u, v, d);
        if (!lane) sm[0] = v;
    }
    __syncthreads();
    float inv_s = 1.f / sm[0];

    #pragma unroll
    for (int i = 0; i < EPT_DB; i++)
        rp[threadIdx.x + i * SOFT_T_DB] = reg[i] * inv_s;
}

// ─── Host launcher ────────────────────────────────────────────────────────────
torch::Tensor fused_wmma_db_launch(
    torch::Tensor A, torch::Tensor WT, torch::Tensor bias, int M, int N, int K)
{
    auto C = torch::empty({M, N}, A.options());
    dim3 grid((N + BN_DB - 1) / BN_DB, (M + BM_DB - 1) / BM_DB);
    wmma_gemm_db<<<grid, NTHREADS_DB>>>(
        A.data_ptr<float>(), WT.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K);
    bias_gelu_softmax_k<<<M, SOFT_T_DB>>>(
        C.data_ptr<float>(), bias.data_ptr<float>(), M, N);
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor fused_wmma_db_launch(
    torch::Tensor A, torch::Tensor WT, torch::Tensor bias, int M, int N, int K);
"""

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_inline(
            name="fused_mgs_wmma_db",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["fused_wmma_db_launch"],
            extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"],
            verbose=False,
        )
    return _MODULE


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('weight_T',
            self.linear.weight.data.t().contiguous())

    def forward(self, x):
        M, K = x.shape
        N = self.linear.out_features
        return _get_module().fused_wmma_db_launch(
            x.contiguous(),
            self.weight_T,
            self.linear.bias.contiguous(),
            M, N, K
        )
