import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Tiled register-blocked SGEMM with vectorized loads
# BM=BN=128, BK=16, TM=TN=8 => 16x16=256 threads per block
# Each thread accumulates an 8x8 output tile
SGEMM_SRC = r"""
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8
#define NTHREADS 256  // = (BM/TM) * (BN/TN) = 16 * 16

// Padding to avoid shared-memory bank conflicts
#define PAD 4

__global__ __launch_bounds__(NTHREADS)
void sgemm_kernel(const float* __restrict__ A,
                  const float* __restrict__ B,
                  float*       __restrict__ C,
                  int M, int K, int N)
{
    // Block tile origin in output matrix
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Thread row/col within block tile
    const int t = threadIdx.x;
    const int ty = t / (BN / TN);  // 0..15
    const int tx = t % (BN / TN);  // 0..15

    // Shared memory for A and B tiles (no double buffering for simplicity/correctness)
    __shared__ float smA[BM][BK + PAD];   // 128 x 20
    __shared__ float smB[BK][BN + PAD];   // 16  x 132

    // Per-thread register accumulators
    float regC[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            regC[i][j] = 0.f;

    // Loading patterns:
    // A tile: BM*BK = 2048 floats; 256 threads => 8 floats/thread = 2 float4s
    // B tile: BK*BN = 2048 floats; 256 threads => 8 floats/thread = 2 float4s

    const int aN = BM * BK / (4 * NTHREADS);   // = 2
    const int bN = BK * BN / (4 * NTHREADS);   // = 2

    for (int k_base = 0; k_base < K; k_base += BK) {

        // ---- Load A: each thread loads aN float4s ----
        // Linearize: thread t loads elements at flat indices [t*4*aN .. (t+1)*4*aN)
        #pragma unroll
        for (int li = 0; li < aN; li++) {
            int flat = (t + li * NTHREADS) * 4;   // flat element index in BM x BK
            int row  = flat / BK;
            int col  = flat % BK;
            int gr   = block_row + row;
            int gc   = k_base + col;
            float4 val = {0.f, 0.f, 0.f, 0.f};
            if (gr < M && gc + 3 < K) {
                val = *reinterpret_cast<const float4*>(&A[gr * K + gc]);
            } else if (gr < M) {
                // partial: load element by element
                if (gc     < K) val.x = A[gr * K + gc    ];
                if (gc + 1 < K) val.y = A[gr * K + gc + 1];
                if (gc + 2 < K) val.z = A[gr * K + gc + 2];
                if (gc + 3 < K) val.w = A[gr * K + gc + 3];
            }
            smA[row][col]   = val.x;
            smA[row][col+1] = val.y;
            smA[row][col+2] = val.z;
            smA[row][col+3] = val.w;
        }

        // ---- Load B: each thread loads bN float4s ----
        #pragma unroll
        for (int li = 0; li < bN; li++) {
            int flat = (t + li * NTHREADS) * 4;   // flat element index in BK x BN
            int row  = flat / BN;
            int col  = flat % BN;
            int gr   = k_base + row;
            int gc   = block_col + col;
            float4 val = {0.f, 0.f, 0.f, 0.f};
            if (gr < K && gc + 3 < N) {
                val = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
            } else if (gr < K) {
                if (gc     < N) val.x = B[gr * N + gc    ];
                if (gc + 1 < N) val.y = B[gr * N + gc + 1];
                if (gc + 2 < N) val.z = B[gr * N + gc + 2];
                if (gc + 3 < N) val.w = B[gr * N + gc + 3];
            }
            smB[row][col]   = val.x;
            smB[row][col+1] = val.y;
            smB[row][col+2] = val.z;
            smB[row][col+3] = val.w;
        }

        __syncthreads();

        // ---- Compute 8x8 outer product ----
        float regA[TM], regB[TN];
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                regA[tm] = smA[ty * TM + tm][kk];
            }
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                regB[tn] = smB[kk][tx * TN + tn];
            }
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    regC[tm][tn] += regA[tm] * regB[tn];
                }
            }
        }

        __syncthreads();
    }

    // ---- Store results ----
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        int row = block_row + ty * TM + tm;
        if (row < M) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn += 4) {
                int col = block_col + tx * TN + tn;
                if (col + 3 < N) {
                    float4 out;
                    out.x = regC[tm][tn];
                    out.y = regC[tm][tn+1];
                    out.z = regC[tm][tn+2];
                    out.w = regC[tm][tn+3];
                    *reinterpret_cast<float4*>(&C[row * N + col]) = out;
                } else {
                    for (int k = 0; k < 4; k++) {
                        if (col + k < N)
                            C[row * N + col + k] = regC[tm][tn + k];
                    }
                }
            }
        }
    }
}

torch::Tensor sgemm(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    dim3 block(NTHREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N);
    return C;
}
"""

_ext = load_inline(
    name="sgemm_v1",
    cpp_sources="torch::Tensor sgemm(torch::Tensor A, torch::Tensor B);",
    cuda_sources=SGEMM_SRC,
    functions=["sgemm"],
    verbose=False,
    extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"],
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _ext.sgemm(A.contiguous(), B.contiguous())
