import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 3 (blind redo): WMMA TF32, BM=64 BN=128 BKK=32, 4 warps (2M×2N)
# Smaller M tile → higher occupancy (2 blocks/SM instead of 1)
# Each warp: 2×4 WMMA tiles (32×64 region)
# Smem: As[64][36]+Bs[32][132] = 9216+16896 = 26112B < 48KB (fits TWO blocks → higher SM util)
# Also try: use __ldg for cache hints on A loads

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <mma.h>
#include <float.h>
#include <torch/extension.h>
using namespace nvcuda;

// Two variants: BM=64 (4-warp) and BM=128 (8-warp). Try BM=64 for occupancy.
static constexpr int BM2  =  64;
static constexpr int BN2  = 128;
static constexpr int BKK2 =  32;
static constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
static constexpr int WARPS_M2 = 2, WARPS_N2 = 2;  // 4 warps
static constexpr int WM2 = 2, WN2 = 4;
static constexpr int NT2 = 128;  // 4 warps

static constexpr int SOFT_T = 256;
static constexpr int EPT    =  32;

__device__ __forceinline__ float gelu_ex(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// ─── BM=64 variant ───────────────────────────────────────────────────────────
// Smem: As[64][36]=9216B, Bs[32][132]=16896B → 26112B
// Two blocks can co-reside per SM → better occupancy than 8-warp BM=128 version
__global__ __launch_bounds__(NT2)
void wmma_gemm_bm64(
    const float* __restrict__ A,
    const float* __restrict__ WT,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int wid = threadIdx.x / 32;
    const int wr  = wid / WARPS_N2;   // 0..1
    const int wc  = wid % WARPS_N2;   // 0..1
    const int bm  = blockIdx.y * BM2;
    const int bn  = blockIdx.x * BN2;
    const int wm0 = wr * (WM2 * WMMA_M);   // 0, 32
    const int wn0 = wc * (WN2 * WMMA_N);   // 0, 64

    __shared__ float As[BM2][BKK2 + 4];   // 9216 bytes
    __shared__ float Bs[BKK2][BN2 + 4];   // 16896 bytes

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WM2][WN2];
    #pragma unroll
    for (int i = 0; i < WM2; i++)
        #pragma unroll
        for (int j = 0; j < WN2; j++)
            wmma::fill_fragment(acc[i][j], 0.f);

    for (int kb = 0; kb < K; kb += BKK2) {
        // Load A[bm:+BM2, kb:+BKK2] → As[m][k]
        // BM2*BKK2=2048 floats, NT2=128 → 16 each
        #pragma unroll
        for (int e = threadIdx.x; e < BM2 * BKK2; e += NT2) {
            int m = e / BKK2, k = e % BKK2;
            int gm = bm + m, gk = kb + k;
            As[m][k] = (gm < M && gk < K) ? __ldg(&A[gm * K + gk]) : 0.f;
        }
        // Load WT[kb:+BKK2, bn:+BN2] → Bs[k][n]
        #pragma unroll
        for (int e = threadIdx.x; e < BKK2 * BN2; e += NT2) {
            int k = e / BN2, n = e % BN2;
            int gk = kb + k, gn = bn + n;
            Bs[k][n] = (gk < K && gn < N) ? __ldg(&WT[gk * N + gn]) : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < BKK2 / WMMA_K; ks++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> af[WM2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> bf[WN2];
            #pragma unroll
            for (int i = 0; i < WM2; i++)
                wmma::load_matrix_sync(af[i],
                    &As[wm0 + i * WMMA_M][ks * WMMA_K], BKK2 + 4);
            #pragma unroll
            for (int j = 0; j < WN2; j++)
                wmma::load_matrix_sync(bf[j],
                    &Bs[ks * WMMA_K][wn0 + j * WMMA_N], BN2 + 4);
            #pragma unroll
            for (int i = 0; i < WM2; i++)
                #pragma unroll
                for (int j = 0; j < WN2; j++)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < WM2; i++)
        #pragma unroll
        for (int j = 0; j < WN2; j++) {
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
torch::Tensor fused_bm64_launch(
    torch::Tensor A, torch::Tensor WT, torch::Tensor bias, int M, int N, int K)
{
    auto C = torch::empty({M, N}, A.options());
    dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
    wmma_gemm_bm64<<<grid, NT2>>>(
        A.data_ptr<float>(), WT.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K);
    bias_gelu_softmax_k<<<M, SOFT_T>>>(
        C.data_ptr<float>(), bias.data_ptr<float>(), M, N);
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor fused_bm64_launch(
    torch::Tensor A, torch::Tensor WT, torch::Tensor bias, int M, int N, int K);
"""

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_inline(
            name="fused_mgs_bm64",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["fused_bm64_launch"],
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
        return _get_module().fused_bm64_launch(
            x.contiguous(),
            self.weight_T,
            self.linear.bias.contiguous(),
            M, N, K
        )
