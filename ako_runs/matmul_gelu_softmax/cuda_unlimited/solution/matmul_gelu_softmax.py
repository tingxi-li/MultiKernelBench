import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 5 (blind redo): PTX mma.sync.m16n8k8 TF32, BM=64 BN=128 BKK=32
# Replace WMMA C++ API with direct PTX mma.sync instructions
# m16n8k8 has lower register footprint than m16n16k8 (8 acc vs 8, but for smaller tile)
# Warp layout: 2M×2N, each warp does WM=2, WN_ptx=4 (each WN is 8 wide, not 16)
# Use N-swizzle grid: process blocks in column-major order to improve WT L2 reuse
#
# Actually: PTX mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32:
#   A matrix: 8 registers holding 4 elements each (2×4 lanes share)
#   Operand A: 8 tf32 values per thread in m16n8k8: thread 0..15 hold A[row 0..15, col 0..3],
#              thread 16..31 hold A[row 0..15, col 4..7]
# It's more complex. Use m16n8k4 instead for BKK=32 → 8 steps.
#
# Better: stick with WMMA API but change grid traversal to L-shaped tile order
# (z-order or column-major) to improve WT reuse in L2.
#
# For a 1024×8192 GEMM with BM=64, BN=128:
#   Grid = (64, 16) = 1024 blocks
#   Each SM on RTX6000 Ada runs 2 blocks (smem-limited to ~26KB each)
#   RTX6000 Ada has 60 SMs → 120 active blocks simultaneously
#   With 1024 blocks, each SM processes ~8 wave passes
#   The default grid order is blockIdx.x fastest (row-major = N-fastest)
#   For WT reuse, we want all N-blocks for same K-range before moving to next K...
#   But K is always fully iterated for each block → can't reuse across blocks.
#
# The real key: swap blockIdx order to be N-column-major (all M-blocks for same N-tile first)
# → WT[K, n_tile] fits in 256/(N/BN) = 256/64 = 4MB per N-tile → fits in L2 (96MB) easily!
# → All 16 M-blocks for a given N-tile read the same WT slice → WT slice stays in L2!
# → Total BW: A[32MB] + WT[256MB]/reuse_factor + C[32MB]
# With 16 M-blocks per N-tile: WT loaded once per SM wave if all 16 M-blocks run on same SM.
# But with 60 SMs and 16 M-blocks per N-tile: only 1 SM processes all 16 M-blocks for a tile!
# Well, 60 SMs, 64 N-tiles: 64/60 ≈ 1 N-tile per SM in first wave. Each SM processes ~16 M-blocks.
# If N-tile slice stays in L2 across 16 M-block iterations: WT reads 256MB → 1 load per N-tile = perfect!
# But L2 is per-GPU (shared across SMs), so N-tile of WT[32*128*4=16KB per N-column] × 256 N-tiles = 256MB.
# Each N-tile: 32×128 = 4096 floats = 16KB. L2 = 96MB → can hold 6144 N-tile slices. Works!

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <mma.h>
#include <float.h>
#include <torch/extension.h>
using namespace nvcuda;

static constexpr int BMv  =  64;
static constexpr int BNv  = 128;
static constexpr int BKv  =  32;
static constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
static constexpr int WARPSm = 2, WARPSn = 2;
static constexpr int WMv = 2, WNv = 4;
static constexpr int NTv = 128;

static constexpr int SOFT_T = 256;
static constexpr int EPT    =  32;

__device__ __forceinline__ float gelu_ex(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// Same GEMM kernel as iter-9 but with different grid invocation (column-major N-first)
__global__ __launch_bounds__(NTv)
void wmma_gemm_v64(
    const float* __restrict__ A,
    const float* __restrict__ WT,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Swizzled block assignment: process all M-blocks for same N-block consecutively
    // blockIdx.x is M-block (0..M/BM-1), blockIdx.y is N-block (0..N/BN-1)
    // → WT[K, bn:bn+BN] slice stays in L2 across all M-blocks for same N-block
    const int wid = threadIdx.x / 32;
    const int wr  = wid / WARPSn;
    const int wc  = wid % WARPSn;
    // Swizzled: blockIdx.x = M-tile, blockIdx.y = N-tile
    const int bm  = blockIdx.x * BMv;
    const int bn  = blockIdx.y * BNv;
    const int wm0 = wr * (WMv * WMMA_M);
    const int wn0 = wc * (WNv * WMMA_N);

    __shared__ float As[BMv][BKv + 4];
    __shared__ float Bs[BKv][BNv + 4];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WMv][WNv];
    #pragma unroll
    for (int i = 0; i < WMv; i++)
        #pragma unroll
        for (int j = 0; j < WNv; j++)
            wmma::fill_fragment(acc[i][j], 0.f);

    for (int kb = 0; kb < K; kb += BKv) {
        #pragma unroll
        for (int e = threadIdx.x; e < BMv * BKv; e += NTv) {
            int m = e / BKv, k = e % BKv;
            int gm = bm + m, gk = kb + k;
            As[m][k] = (gm < M && gk < K) ? __ldg(&A[gm * K + gk]) : 0.f;
        }
        #pragma unroll
        for (int e = threadIdx.x; e < BKv * BNv; e += NTv) {
            int k = e / BNv, n = e % BNv;
            int gk = kb + k, gn = bn + n;
            Bs[k][n] = (gk < K && gn < N) ? __ldg(&WT[gk * N + gn]) : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < BKv / WMMA_K; ks++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> af[WMv];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> bf[WNv];
            #pragma unroll
            for (int i = 0; i < WMv; i++)
                wmma::load_matrix_sync(af[i],
                    &As[wm0 + i * WMMA_M][ks * WMMA_K], BKv + 4);
            #pragma unroll
            for (int j = 0; j < WNv; j++)
                wmma::load_matrix_sync(bf[j],
                    &Bs[ks * WMMA_K][wn0 + j * WMMA_N], BNv + 4);
            #pragma unroll
            for (int i = 0; i < WMv; i++)
                #pragma unroll
                for (int j = 0; j < WNv; j++)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < WMv; i++)
        #pragma unroll
        for (int j = 0; j < WNv; j++) {
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
// Swizzled grid: dim3((M/BM, N/BN)) so blockIdx.x is M-tile, blockIdx.y is N-tile
// → blockIdx.x cycles fastest → all M-blocks for a given N-block run consecutively
// → WT slice for that N-block stays in L2 across all M-blocks
torch::Tensor fused_v64_launch(
    torch::Tensor A, torch::Tensor WT, torch::Tensor bias, int M, int N, int K)
{
    auto C = torch::empty({M, N}, A.options());
    // Swizzled: (M-blocks, N-blocks) so M cycles fastest in SM scheduler
    dim3 grid((M + BMv - 1) / BMv, (N + BNv - 1) / BNv);
    wmma_gemm_v64<<<grid, NTv>>>(
        A.data_ptr<float>(), WT.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K);
    bias_gelu_softmax_k<<<M, SOFT_T>>>(
        C.data_ptr<float>(), bias.data_ptr<float>(), M, N);
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor fused_v64_launch(
    torch::Tensor A, torch::Tensor WT, torch::Tensor bias, int M, int N, int K);
"""

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_inline(
            name="fused_mgs_v64",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["fused_v64_launch"],
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
        return _get_module().fused_v64_launch(
            x.contiguous(),
            self.weight_T,
            self.linear.bias.contiguous(),
            M, N, K
        )
