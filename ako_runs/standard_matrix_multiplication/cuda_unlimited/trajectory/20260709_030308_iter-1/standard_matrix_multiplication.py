import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Tiled register-blocked GEMM for fp32
# Block: 16×16 = 256 threads (tx: 0..15 along N, ty: 0..15 along M)
# Tile:  BM=128, BN=128, BK=8
# Per-thread: TM=8 output rows, TN=8 output cols
# smA stored transposed [BK][BM+4] for broadcast reads during compute
# smB stored [BK][BN+4]

cuda_src = r"""
#include <cuda_runtime.h>
#include <stdint.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
// threads per block: (BN/TN) * (BM/TM) = 16 * 16 = 256

// Tiled, register-blocked fp32 GEMM.
// A: [M, K] row-major  B: [K, N] row-major  C: [M, N] row-major
__global__ void sgemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    // Transposed A slab: smA[ki][row_in_block], avoids non-broadcast smA reads
    __shared__ float smA[BK][BM + 4];  // 8 * 132 * 4 = 4224 B
    __shared__ float smB[BK][BN + 4];  // 8 * 132 * 4 = 4224 B

    const int tx  = threadIdx.x;            // 0..15  (N direction)
    const int ty  = threadIdx.y;            // 0..15  (M direction)
    const int tid = ty * blockDim.x + tx;   // 0..255

    const int blockRowStart = blockIdx.y * BM;
    const int blockColStart = blockIdx.x * BN;

    // Per-thread accumulator: TM × TN = 64 registers
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (int kBase = 0; kBase < K; kBase += BK) {

        // ---- Load A tile [BM x BK] into smA[BK][BM] (transposed) ----
        // BM*BK/256 = 128*8/256 = 4 elements per thread
        #pragma unroll
        for (int i = 0; i < (BM * BK) / 256; i++) {
            int idx  = i * 256 + tid;
            int rowA = idx / BK;   // 0..127
            int colA = idx % BK;   // 0..7
            int gRow = blockRowStart + rowA;
            int gCol = kBase + colA;
            smA[colA][rowA] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }

        // ---- Load B tile [BK x BN] into smB[BK][BN] ----
        // BK*BN/256 = 8*128/256 = 4 elements per thread
        #pragma unroll
        for (int i = 0; i < (BK * BN) / 256; i++) {
            int idx  = i * 256 + tid;
            int rowB = idx / BN;   // 0..7
            int colB = idx % BN;   // 0..127
            int gRow = kBase + rowB;
            int gCol = blockColStart + colB;
            smB[rowB][colB] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }

        __syncthreads();

        // ---- Compute TM × TN outer products, accumulated over BK ----
        float regA[TM], regB[TN];
        #pragma unroll
        for (int ki = 0; ki < BK; ki++) {
            // smA[ki][ty*TM .. ty*TM+TM-1]: all threads with same ty read same bank -> broadcast
            #pragma unroll
            for (int i = 0; i < TM; i++)
                regA[i] = smA[ki][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                regB[j] = smB[ki][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    // ---- Store ----
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int row = blockRowStart + ty * TM + i;
            int col = blockColStart + tx * TN + j;
            if (row < M && col < N)
                C[row * N + col] = acc[i][j];
        }
    }
}

torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 block(BN / TN, BM / TM);                // (16, 16)
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

cpp_src = r"""
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="sgemm_tiled",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["matmul_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
            verbose=False,
        )
    return _module


class Model(nn.Module):
    """
    Matrix multiplication C = A @ B using a custom tiled register-blocked CUDA GEMM.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _get_module().matmul_forward(A.contiguous(), B.contiguous())
