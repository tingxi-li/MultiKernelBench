import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// C(MxN) = A(MxK) * B(KxN), all row-major fp32. 256 threads / block,
// 128x128 block tile, each thread computes an 8x8 register microtile.
__global__ __launch_bounds__(256) void sgemm_kernel(
        const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BK][BM];   // transposed for conflict-free reads
    __shared__ float Bs[BK][BN];

    const int bx = blockIdx.x;     // N-tile
    const int by = blockIdx.y;     // M-tile
    const int tid = threadIdx.x;

    const int tRow = tid / (BN / TN);   // 0..15
    const int tCol = tid % (BN / TN);   // 0..15

    const int aRow = tid / (BK / 4);    // 0..127
    const int aCol = (tid % (BK / 4)) * 4;
    const int bRow = tid / (BN / 4);    // 0..7
    const int bCol = (tid % (BN / 4)) * 4;

    const float* Ap = A + (by * BM) * K;
    const float* Bp = B + (bx * BN);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    float regA[TM], regB[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        float4 av = *reinterpret_cast<const float4*>(&Ap[aRow * K + k0 + aCol]);
        As[aCol + 0][aRow] = av.x;
        As[aCol + 1][aRow] = av.y;
        As[aCol + 2][aRow] = av.z;
        As[aCol + 3][aRow] = av.w;

        float4 bv = *reinterpret_cast<const float4*>(&Bp[(k0 + bRow) * N + bCol]);
        *reinterpret_cast<float4*>(&Bs[bRow][bCol]) = bv;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = As[kk][tRow * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[kk][tCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

    const int cRow = by * BM + tRow * TM;
    const int cCol = bx * BN + tCol * TN;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 v = make_float4(acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]);
            *reinterpret_cast<float4*>(&C[(cRow + i) * N + cCol + j]) = v;
        }
    }
}

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "cuda only");
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    auto C = torch::empty({M, N}, Ac.options());
    dim3 grid(N / BN, M / BM);
    dim3 block(256);
    sgemm_kernel<<<grid, block>>>(Ac.data_ptr<float>(), Bc.data_ptr<float>(),
                                  C.data_ptr<float>(), M, N, K);
    return C;
}
'''

_CPP = "torch::Tensor run(torch::Tensor A, torch::Tensor B);"

_ext = load_inline(
    name="sgemm_unlim_v1",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["run"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _ext.run(A, B)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []
