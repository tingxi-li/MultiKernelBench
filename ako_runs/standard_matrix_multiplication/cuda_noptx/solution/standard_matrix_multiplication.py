import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Tiled SGEMM — BM=128 BN=128 BK=16 TM=8 TN=8 256-thread block
# float4 vectorised loads, bank-conflict-free shared mem (padded)
_SGEMM_CUDA = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

// Tile sizes: each block owns a BM×BN tile of C
#define BM 128
#define BN 128
#define BK 16
// Each thread computes TM×TN = 8×8 outputs
#define TM 8
#define TN 8
// Block: (BM/TM) × (BN/TN) = 16×16 = 256 threads
#define BLOCK_ROWS (BM/TM)   // 16
#define BLOCK_COLS (BN/TN)   // 16

__global__ void sgemm_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    const int bm = blockIdx.y * BM;   // tile row base
    const int bn = blockIdx.x * BN;   // tile col base
    const int tx = threadIdx.x;        // 0..15
    const int ty = threadIdx.y;        // 0..15
    const int tid = ty * BLOCK_COLS + tx;  // 0..255

    // Accumulators
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[BM][BK + 4];   // 128×20, padded by 4
    __shared__ float Bs[BK][BN + 4];   // 16×132, padded by 4

    // ── Load A tile ──────────────────────────────────────────────────────────
    // 256 threads load BM×BK = 128×16 = 2048 floats
    // Each thread loads 8 floats (2 iterations × 4-wide = float4)
    // Layout: tid → row = tid / (BK/4), col_f4 = (tid % (BK/4)) * 4
    //   BK/4 = 4  → row stride = 256 / 4 = 64, so each of 4 loads covers 64 rows
    //   We need ceil(128 / 64) = 2 passes
    const int As_col_f4 = (tid & 3) * 4;      // 0,4,8,12
    const int As_row_base = tid >> 2;          // 0..63

    // ── Load B tile ──────────────────────────────────────────────────────────
    // 256 threads load BK×BN = 16×128 = 2048 floats
    // Each thread loads 8 floats (2 iterations)
    // Layout: row = tid / (BN/4), col_f4 = (tid % (BN/4)) * 4
    //   BN/4 = 32 → row stride = 256 / 32 = 8 → 2 rows per pass
    const int Bs_col_f4 = (tid & 31) * 4;     // 0,4,...,124
    const int Bs_row_base = tid >> 5;          // 0..7

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_base = kt * BK;

        // Load A (BM×BK) — 2 passes of 64 rows
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int row = As_row_base + s * 64;
            int gr = bm + row;
            int gk = k_base + As_col_f4;
            if (gr < M && gk + 3 < K) {
                float4 v = *reinterpret_cast<const float4*>(&A[gr * K + gk]);
                As[row][As_col_f4 + 0] = v.x;
                As[row][As_col_f4 + 1] = v.y;
                As[row][As_col_f4 + 2] = v.z;
                As[row][As_col_f4 + 3] = v.w;
            } else if (gr < M) {
                // Scalar fallback for boundary
                for (int d = 0; d < 4; d++)
                    As[row][As_col_f4 + d] = (gk + d < K) ? A[gr * K + gk + d] : 0.0f;
            } else {
                for (int d = 0; d < 4; d++)
                    As[row][As_col_f4 + d] = 0.0f;
            }
        }

        // Load B (BK×BN) — 2 passes of 8 rows
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int row = Bs_row_base + s * 8;
            int gk = k_base + row;
            int gn = bn + Bs_col_f4;
            if (gk < K && gn + 3 < N) {
                float4 v = *reinterpret_cast<const float4*>(&B[gk * N + gn]);
                Bs[row][Bs_col_f4 + 0] = v.x;
                Bs[row][Bs_col_f4 + 1] = v.y;
                Bs[row][Bs_col_f4 + 2] = v.z;
                Bs[row][Bs_col_f4 + 3] = v.w;
            } else if (gk < K) {
                for (int d = 0; d < 4; d++)
                    Bs[row][Bs_col_f4 + d] = (gn + d < N) ? B[gk * N + gn + d] : 0.0f;
            } else {
                for (int d = 0; d < 4; d++)
                    Bs[row][Bs_col_f4 + d] = 0.0f;
            }
        }

        __syncthreads();

        // Compute: each thread works on ty*TM .. ty*TM+TM-1 rows
        //                                   tx*TN .. tx*TN+TN-1 cols
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

    // Store output
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
    dim3 block(BLOCK_COLS, BLOCK_ROWS);  // (16, 16) = 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v2<<<grid, block>>>(
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
    name="sgemm_noptx_bk16_f4",
    cpp_sources=_SGEMM_CPP,
    cuda_sources=_SGEMM_CUDA,
    functions=["sgemm_launch"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
    verbose=False,
)


class Model(nn.Module):
    """Matrix multiplication C = A @ B via tiled SGEMM (float32, no PTX, BK=16 float4 loads)."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        return _ext.sgemm_launch(A, B)
