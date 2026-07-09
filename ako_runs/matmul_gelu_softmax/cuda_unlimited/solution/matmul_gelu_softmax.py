import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# TF32 WMMA GEMM: BM=64, BN=64, BK=32
# 4 warps (WARPS_M=2, WARPS_N=2), each computes 2×2 WMMA tiles (32×32)
# Shared mem: As[64][36] + Ws[64][36] + tile_buf[4][32][36] = 9216+9216+18432 = 36864 bytes ✓

_CUDA_SRC = r"""
#include <mma.h>
#include <cuda_runtime.h>
#include <float.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8
#define BM 64
#define BN 64
#define BK 32
#define WARPS_M 2
#define WARPS_N 2
#define NWARPS 4
#define NTHREADS 128
#define WARP_TILES_M 2
#define WARP_TILES_N 2
#define PAD 4

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__global__ __launch_bounds__(NTHREADS)
void gemm_gelu_tf32(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_row = warp_id / WARPS_N;   // 0 or 1
    const int warp_col = warp_id % WARPS_N;   // 0 or 1
    const int tid = threadIdx.x;

    const int warp_m = warp_row * WARP_TILES_M * WMMA_M;  // 0 or 32
    const int warp_n = warp_col * WARP_TILES_N * WMMA_N;  // 0 or 32

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_TILES_M][WARP_TILES_N];
    for (int i = 0; i < WARP_TILES_M; i++)
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(c[i][j], 0.f);

    __shared__ float As[BM][BK + PAD];    // 64*36*4 = 9216 bytes
    __shared__ float Ws[BN][BK + PAD];    // 64*36*4 = 9216 bytes
    // Per-warp output tile: each warp stores 32×32 output
    __shared__ float tile_buf[NWARPS][WARP_TILES_M * WMMA_M][WARP_TILES_N * WMMA_N + PAD];
    // 4 * 32 * 36 * 4 = 18432 bytes
    // Total: 36864 bytes ✓

    // Load A and W tiles: BM*BK = 64*32 = 2048 elements, NTHREADS=128 → 16 each
    for (int e = tid; e < BM * BK; e += NTHREADS) {
        int r = e / BK, col = e % BK;
        int gr = block_row + r, gc = col;  // gc offset applied in loop below
        (void)gr; (void)gc;  // will be set per iteration
    }

    for (int k_base = 0; k_base < K; k_base += BK) {
        for (int e = tid; e < BM * BK; e += NTHREADS) {
            int r = e / BK, col = e % BK;
            int gr = block_row + r, gc = k_base + col;
            As[r][col] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        for (int e = tid; e < BN * BK; e += NTHREADS) {
            int r = e / BK, col = e % BK;
            int gr = block_col + r, gc = k_base + col;
            Ws[r][col] = (gr < N && gc < K) ? W[gr * K + gc] : 0.f;
        }
        __syncthreads();

        // 4 WMMA steps per BK=32
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> a_frag[WARP_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::col_major> b_frag[WARP_TILES_N];
            for (int i = 0; i < WARP_TILES_M; i++)
                wmma::load_matrix_sync(a_frag[i], &As[warp_m + i*WMMA_M][kk], BK+PAD);
            for (int j = 0; j < WARP_TILES_N; j++)
                wmma::load_matrix_sync(b_frag[j], &Ws[warp_n + j*WMMA_N][kk], BK+PAD);
            for (int i = 0; i < WARP_TILES_M; i++)
                for (int j = 0; j < WARP_TILES_N; j++)
                    wmma::mma_sync(c[i][j], a_frag[i], b_frag[j], c[i][j]);
        }
        __syncthreads();
    }

    // Store warp's tiles to private smem region
    const int WTM = WARP_TILES_M * WMMA_M;  // 32
    const int WTN = WARP_TILES_N * WMMA_N;  // 32
    for (int i = 0; i < WARP_TILES_M; i++)
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::store_matrix_sync(
                &tile_buf[warp_id][i*WMMA_M][j*WMMA_N],
                c[i][j], WTN + PAD, wmma::mem_row_major);

    // Write to global with bias + GELU (no syncthreads needed, per-warp regions)
    for (int e = lane_id; e < WTM * WTN; e += 32) {
        int r = e / WTN, col = e % WTN;
        int gm = block_row + warp_m + r;
        int gn = block_col + warp_n + col;
        if (gm < M && gn < N)
            C[gm * N + gn] = gelu_exact(tile_buf[warp_id][r][col] + bias[gn]);
    }
}

__global__ void softmax_kernel(float* __restrict__ C, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    float* rp = C + row * N;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nw = blockDim.x / 32;

    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += blockDim.x) mx = fmaxf(mx, rp[i]);
    for (int m = 16; m > 0; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, m));
    __shared__ float sm[8];
    if (!lane) sm[warp] = mx;
    __syncthreads();
    if (!warp) {
        float v = lane < nw ? sm[lane] : -FLT_MAX;
        for (int m = 4; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, m));
        if (!lane) sm[0] = v;
    }
    __syncthreads();
    float row_max = sm[0];

    float s = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) s += expf(rp[i] - row_max);
    for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(~0u, s, m);
    if (!lane) sm[warp] = s;
    __syncthreads();
    if (!warp) {
        float v = lane < nw ? sm[lane] : 0.f;
        for (int m = 4; m > 0; m >>= 1) v += __shfl_xor_sync(~0u, v, m);
        if (!lane) sm[0] = v;
    }
    __syncthreads();
    float inv_s = 1.f / sm[0];
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        rp[i] = expf(rp[i] - row_max) * inv_s;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor fused_matmul_gelu_softmax(
    torch::Tensor A, torch::Tensor W, torch::Tensor bias, int M, int N, int K);
"""

_CUDA_WRAPPER = r"""
#include <torch/extension.h>
__global__ void gemm_gelu_tf32(const float*, const float*, const float*, float*, int, int, int);
__global__ void softmax_kernel(float*, int, int);

torch::Tensor fused_matmul_gelu_softmax(
    torch::Tensor A, torch::Tensor W, torch::Tensor bias, int M, int N, int K)
{
    auto C = torch::empty({M, N}, A.options());
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    gemm_gelu_tf32<<<grid, 128>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), bias.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K);
    softmax_kernel<<<M, 256>>>(C.data_ptr<float>(), M, N);
    return C;
}
"""

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_inline(
            name="fused_mgs_tf32_v3",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC + _CUDA_WRAPPER,
            functions=["fused_matmul_gelu_softmax"],
            extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"],
            verbose=False,
        )
    return _MODULE


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        M, K = x.shape
        N = self.linear.out_features
        return _get_module().fused_matmul_gelu_softmax(
            x.contiguous(), self.linear.weight.contiguous(),
            self.linear.bias.contiguous(), M, N, K
        )
