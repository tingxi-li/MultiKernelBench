import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# WMMA TF32 SGEMM: tensor cores via CUDA C++ wmma::precision::tf32 (no inline PTX)
# TF32 fragment tile: 16x16x8. 8 warps (256 threads), 4x2 arrangement.
# Input: float32, accumulator: float32, precision: TF32 (same as cuBLAS default on Ada).
_WMMA_CUDA = r"""
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

// TF32 WMMA tile size: 16x16x8
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

// Warp arrangement: 4x2 in (M,N)
#define WARP_M 4
#define WARP_N 2
#define BM (WARP_M * WMMA_M)     // 64
#define BN (WARP_N * WMMA_N)     // 32
#define BK WMMA_K                 // 8
#define NUM_WARPS (WARP_M * WARP_N) // 8
#define NUM_THREADS (NUM_WARPS * 32) // 256

__global__ void wmma_tf32_sgemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / WARP_N;  // 0..3
    int warp_n = warp_id % WARP_N;  // 0..1

    int cRow = blockIdx.y * BM + warp_m * WMMA_M;
    int cCol = blockIdx.x * BN + warp_n * WMMA_N;

    // TF32 accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fill_fragment(acc, 0.0f);

    // Shared memory: BM x BK for A, BK x BN for B (float32)
    // BM*BK = 64*8=512 floats, BK*BN = 8*32=256 floats
    __shared__ float As[BM][BK + 2];  // +2 padding for alignment
    __shared__ float Bs[BK][BN + 2];

    int tid = threadIdx.x;
    int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k_base = kt * BK;

        // Load As: 64*8=512 elements, 256 threads -> 2 each
        for (int idx = tid; idx < BM * BK; idx += NUM_THREADS) {
            int r = idx / BK;
            int c = idx % BK;
            int gr = blockIdx.y * BM + r;
            int gk = k_base + c;
            As[r][c] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
        }

        // Load Bs: 8*32=256 elements, 256 threads -> 1 each
        for (int idx = tid; idx < BK * BN; idx += NUM_THREADS) {
            int r = idx / BN;
            int c = idx % BN;
            int gk = k_base + r;
            int gn = blockIdx.x * BN + c;
            Bs[r][c] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }

        __syncthreads();

        // Load warp's WMMA fragments
        const float* a_ptr = &As[warp_m * WMMA_M][0];
        const float* b_ptr = &Bs[0][warp_n * WMMA_N];

        wmma::load_matrix_sync(a_frag, a_ptr, BK + 2);
        wmma::load_matrix_sync(b_frag, b_ptr, BN + 2);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(&C[cRow * N + cCol], acc, N, wmma::mem_row_major);
    }
}

torch::Tensor wmma_matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda());
    TORCH_CHECK(A.scalar_type() == torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(NUM_THREADS);

    wmma_tf32_sgemm<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N
    );
    return C;
}
"""

_WMMA_CPP = r"""
#include <torch/extension.h>
torch::Tensor wmma_matmul(torch::Tensor A, torch::Tensor B);
"""

_ext = load_inline(
    name="wmma_tf32_v1",
    cpp_sources=_WMMA_CPP,
    cuda_sources=_WMMA_CUDA,
    functions=["wmma_matmul"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
    verbose=False,
)


class Model(nn.Module):
    """Matrix multiplication C = A @ B via WMMA TF32 tensor cores (no inline PTX)."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        return _ext.wmma_matmul(A, B)
