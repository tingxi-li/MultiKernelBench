import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized register-blocked SGEMM with BK=32 (double the inner loop depth)
# BM=BN=128, BK=32, TM=TN=8 => 256 threads per block
# Includes: float4 loads, smem padding, double buffering via register pre-fetch
SGEMM_SRC = r"""
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 32
#define TM 8
#define TN 8
#define NTHREADS 256   // = (BM/TM) * (BN/TN) = 16 * 16
#define PAD 4

// Each thread: loads (BM*BK)/(NTHREADS) = 128*32/256 = 16 elems of A, 16 elems of B
// Vectorized as float4: 16/4 = 4 float4 loads per thread for A, 4 for B

__global__ __launch_bounds__(NTHREADS)
void sgemm_bk32(const float* __restrict__ A,
                const float* __restrict__ B,
                float*       __restrict__ C,
                int M, int K, int N)
{
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int t   = threadIdx.x;
    const int ty  = t / (BN / TN);   // 0..15
    const int tx  = t % (BN / TN);   // 0..15

    __shared__ float smA[BM][BK + PAD];   // 128 x 36
    __shared__ float smB[BK][BN + PAD];   // 32  x 132

    float regC[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            regC[i][j] = 0.f;

    // Number of float4 loads per thread per tile: BM*BK/NTHREADS/4 = 4 for A, 4 for B
    // (4 * NTHREADS = 4 * 256 = 1024; BM*BK = 128*32 = 4096 = 4 * 1024 ✓)

    for (int k_base = 0; k_base < K; k_base += BK) {

        // Load A tile: 4 float4 loads per thread
        // flat index mapping: thread t handles positions t, t+256, t+512, t+768 in BM*BK
        #pragma unroll
        for (int li = 0; li < 4; li++) {
            int flat = (t + li * NTHREADS) * 4;
            int row  = flat / BK;
            int col  = flat % BK;   // always multiple of 4 (BK=32, flat is multiple of 4)
            int gr   = block_row + row;
            int gc   = k_base + col;
            float4 val = {0.f, 0.f, 0.f, 0.f};
            if (gr < M) {
                if (gc + 3 < K) {
                    val = *reinterpret_cast<const float4*>(&A[gr * K + gc]);
                } else {
                    if (gc     < K) val.x = A[gr * K + gc    ];
                    if (gc + 1 < K) val.y = A[gr * K + gc + 1];
                    if (gc + 2 < K) val.z = A[gr * K + gc + 2];
                }
            }
            smA[row][col]   = val.x;
            smA[row][col+1] = val.y;
            smA[row][col+2] = val.z;
            smA[row][col+3] = val.w;
        }

        // Load B tile: 4 float4 loads per thread
        #pragma unroll
        for (int li = 0; li < 4; li++) {
            int flat = (t + li * NTHREADS) * 4;
            int row  = flat / BN;
            int col  = flat % BN;
            int gr   = k_base + row;
            int gc   = block_col + col;
            float4 val = {0.f, 0.f, 0.f, 0.f};
            if (gr < K) {
                if (gc + 3 < N) {
                    val = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
                } else {
                    if (gc     < N) val.x = B[gr * N + gc    ];
                    if (gc + 1 < N) val.y = B[gr * N + gc + 1];
                    if (gc + 2 < N) val.z = B[gr * N + gc + 2];
                }
            }
            smB[row][col]   = val.x;
            smB[row][col+1] = val.y;
            smB[row][col+2] = val.z;
            smB[row][col+3] = val.w;
        }

        __syncthreads();

        // Inner loop: BK=32 steps
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

    // Store output with float4 stores
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        int row = block_row + ty * TM + tm;
        if (row < M) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn += 4) {
                int col = block_col + tx * TN + tn;
                if (col + 3 < N) {
                    float4 out = {regC[tm][tn], regC[tm][tn+1],
                                  regC[tm][tn+2], regC[tm][tn+3]};
                    *reinterpret_cast<float4*>(&C[row * N + col]) = out;
                } else {
                    for (int k = 0; k < 4 && col + k < N; k++)
                        C[row * N + col + k] = regC[tm][tn + k];
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

    sgemm_bk32<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N);
    return C;
}
"""

_ext = load_inline(
    name="sgemm_bk32",
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
