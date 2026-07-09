import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Tiled SGEMM with shared memory and register blocking (float32, no PTX)
# BM=128, BN=128, BK=8, TM=8, TN=8, block=16x16=256 threads
_SGEMM_CUDA = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_reg_block(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tx = threadIdx.x;   // 0..15
    const int ty = threadIdx.y;   // 0..15
    const int tid = ty * 16 + tx; // 0..255

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    // Load indices
    const int As_row = tid >> 3;   // 0..31
    const int As_col = tid & 7;    // 0..7
    const int Bs_row = tid >> 7;   // 0..1
    const int Bs_col = tid & 127;  // 0..127

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_base = kt * BK;

        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            int row = As_row + s * 32;
            int gr = bm + row;
            int gk = k_base + As_col;
            As[row][As_col] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
        }

        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            int row = Bs_row + s * 2;
            int gk = k_base + row;
            int gn = bn + Bs_col;
            Bs[row][Bs_col] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < BK; ++ki) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[ty * TM + i][ki];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[ki][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gr = bm + ty * TM + i;
            int gn = bn + tx * TN + j;
            if (gr < M && gn < N)
                C[gr * N + gn] = acc[i][j];
        }
    }
}

torch::Tensor sgemm_launch(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda());
    TORCH_CHECK(A.scalar_type() == torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_reg_block<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N
    );
    return C;
}
"""

_SGEMM_CPP = r"""
#include <torch/extension.h>
torch::Tensor sgemm_launch(torch::Tensor A, torch::Tensor B);
"""

_ext = load_inline(
    name="sgemm_noptx_v2",
    cpp_sources=_SGEMM_CPP,
    cuda_sources=_SGEMM_CUDA,
    functions=["sgemm_launch"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
    verbose=False,
)


class Model(nn.Module):
    """Matrix multiplication C = A @ B via tiled SGEMM (float32, no PTX)."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        return _ext.sgemm_launch(A, B)
