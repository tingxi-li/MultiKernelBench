import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 4 (blind redo): BM=64 BN=128 BKK=16 cp.async double-buffer, 4 warps
# Combines the BM=64 occupancy benefit from iter-9 with cp.async pipeline from iter-8
# Smem per stage: As[64][20]+Bs[16][132] = (64*20+16*132)*4 = (1280+2112)*4 = 13568B
# 2 stages: 27136B < 48KB → can fit one block, possibly 2 blocks if other state < 21KB
# Note: K loop has 8192/16 = 512 iterations (vs 256 for BKK=32) but pipeline hides latency

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <float.h>
#include <torch/extension.h>
using namespace nvcuda;

static constexpr int BM_  =  64;
static constexpr int BN_  = 128;
static constexpr int BKK_ =  16;
static constexpr int STS_ =   2;   // pipeline stages
static constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
static constexpr int WARPS_M_ = 2, WARPS_N_ = 2;
static constexpr int WM_ = 2, WN_ = 4;
static constexpr int NT_ = 128;
// Smem per stage: As[64][20]+Bs[16][132] = 13568B
// 2 stages: 27136B < 48KB ✓

static constexpr int SOFT_T = 256;
static constexpr int EPT    =  32;

__device__ __forceinline__ float gelu_ex(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__global__ __launch_bounds__(NT_)
void wmma_gemm_db64(
    const float* __restrict__ A,
    const float* __restrict__ WT,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int wid = threadIdx.x / 32;
    const int wr  = wid / WARPS_N_;
    const int wc  = wid % WARPS_N_;
    const int bm  = blockIdx.y * BM_;
    const int bn  = blockIdx.x * BN_;
    const int wm0 = wr * (WM_ * WMMA_M);   // 0, 32
    const int wn0 = wc * (WN_ * WMMA_N);   // 0, 64

    __shared__ float As[STS_][BM_][BKK_ + 4];   // 2*64*20*4 = 10240B
    __shared__ float Bs[STS_][BKK_][BN_ + 4];   // 2*16*132*4 = 16896B
    // Total: 27136B < 48KB ✓

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WM_][WN_];
    #pragma unroll
    for (int i = 0; i < WM_; i++)
        #pragma unroll
        for (int j = 0; j < WN_; j++)
            wmma::fill_fragment(acc[i][j], 0.f);

    const int tid   = threadIdx.x;
    const int nstep = (K + BKK_ - 1) / BKK_;

    // Preload stage 0
    {
        int kb = 0;
        #pragma unroll
        for (int e = tid; e < BM_ * BKK_; e += NT_) {
            int m = e / BKK_, k = e % BKK_;
            int gm = bm + m, gk = kb + k;
            if (gm < M && gk < K)
                __pipeline_memcpy_async(&As[0][m][k], &A[gm * K + gk], sizeof(float));
            else As[0][m][k] = 0.f;
        }
        #pragma unroll
        for (int e = tid; e < BKK_ * BN_; e += NT_) {
            int k = e / BN_, n = e % BN_;
            int gk = kb + k, gn = bn + n;
            if (gk < K && gn < N)
                __pipeline_memcpy_async(&Bs[0][k][n], &WT[gk * N + gn], sizeof(float));
            else Bs[0][k][n] = 0.f;
        }
        __pipeline_commit();
    }

    for (int step = 0; step < nstep; step++) {
        const int sc = step % STS_;
        const int sn = (step + 1) % STS_;

        if (step + 1 < nstep) {
            int kb = (step + 1) * BKK_;
            #pragma unroll
            for (int e = tid; e < BM_ * BKK_; e += NT_) {
                int m = e / BKK_, k = e % BKK_;
                int gm = bm + m, gk = kb + k;
                if (gm < M && gk < K)
                    __pipeline_memcpy_async(&As[sn][m][k], &A[gm * K + gk], sizeof(float));
                else As[sn][m][k] = 0.f;
            }
            #pragma unroll
            for (int e = tid; e < BKK_ * BN_; e += NT_) {
                int k = e / BN_, n = e % BN_;
                int gk = kb + k, gn = bn + n;
                if (gk < K && gn < N)
                    __pipeline_memcpy_async(&Bs[sn][k][n], &WT[gk * N + gn], sizeof(float));
                else Bs[sn][k][n] = 0.f;
            }
            __pipeline_commit();
        }

        __pipeline_wait_prior(1);
        __syncthreads();

        // WMMA: BKK_/WMMA_K = 2 steps
        #pragma unroll
        for (int ks = 0; ks < BKK_ / WMMA_K; ks++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> af[WM_];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> bf[WN_];
            #pragma unroll
            for (int i = 0; i < WM_; i++)
                wmma::load_matrix_sync(af[i],
                    &As[sc][wm0 + i * WMMA_M][ks * WMMA_K], BKK_ + 4);
            #pragma unroll
            for (int j = 0; j < WN_; j++)
                wmma::load_matrix_sync(bf[j],
                    &Bs[sc][ks * WMMA_K][wn0 + j * WMMA_N], BN_ + 4);
            #pragma unroll
            for (int i = 0; i < WM_; i++)
                #pragma unroll
                for (int j = 0; j < WN_; j++)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < WM_; i++)
        #pragma unroll
        for (int j = 0; j < WN_; j++) {
            int gm = bm + wm0 + i * WMMA_M;
            int gn = bn + wn0 + j * WMMA_N;
            if (gm < M && gn < N)
                wmma::store_matrix_sync(&C[gm * N + gn], acc[i][j], N,
                                        wmma::mem_row_major);
        }
}

// ─── Fused bias + GELU + softmax ─────────────────────────────────────────────
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
    const int nw   = SOFT_T / 32;
    __shared__ float sm[8];

    float reg[EPT];
    float mx = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        int idx = threadIdx.x + i * SOFT_T;
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
    for (int i = 0; i < EPT; i++) { reg[i] = expf(reg[i] - mx); s += reg[i]; }
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
    for (int i = 0; i < EPT; i++)
        rp[threadIdx.x + i * SOFT_T] = reg[i] * inv_s;
}

// ─── Host launcher ────────────────────────────────────────────────────────────
torch::Tensor fused_db64_launch(
    torch::Tensor A, torch::Tensor WT, torch::Tensor bias, int M, int N, int K)
{
    auto C = torch::empty({M, N}, A.options());
    dim3 grid((N + BN_ - 1) / BN_, (M + BM_ - 1) / BM_);
    wmma_gemm_db64<<<grid, NT_>>>(
        A.data_ptr<float>(), WT.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K);
    bias_gelu_softmax_k<<<M, SOFT_T>>>(
        C.data_ptr<float>(), bias.data_ptr<float>(), M, N);
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor fused_db64_launch(
    torch::Tensor A, torch::Tensor WT, torch::Tensor bias, int M, int N, int K);
"""

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_inline(
            name="fused_mgs_db64",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["fused_db64_launch"],
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
        return _get_module().fused_db64_launch(
            x.contiguous(),
            self.weight_T,
            self.linear.bias.contiguous(),
            M, N, K
        )
