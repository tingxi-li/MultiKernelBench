import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// C[M,N]=A[M,K]@B[K,N] row-major fp32. WMMA tf32 (C++ API, no PTX).
// Block 128x128, 4 warps (2x2). Each warp 64x64 = 4x4 frags -> 2 mma per
// shared fragment-load (halve L1/shared pressure vs 2x2). NBANK=2 banks.
#define BM 128
#define BN 128
#define BK 32
#define WN 2
#define FM 4
#define FN 4
#define NBANK 2

__global__ void __launch_bounds__(128)
gemm_wmma(const float* __restrict__ A, const float* __restrict__ B,
          float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int warpId = threadIdx.x >> 5;
    int warpM = warpId / WN;
    int warpN = warpId % WN;
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[FM][FN][NBANK];
    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++)
            #pragma unroll
            for (int b = 0; b < NBANK; b++) wmma::fill_fragment(acc[i][j][b], 0.0f);

    int tid = threadIdx.x;
    int bank = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {          // A[128][32]=4096/128=32=8 float4
            int vec = tid + e * 128;
            int r = vec >> 3;
            int c4 = (vec & 7) << 2;
            *reinterpret_cast<float4*>(&As[r][c4]) = __ldg(
                reinterpret_cast<const float4*>(A + (long)(blockRow + r) * K + k0 + c4));
        }
        #pragma unroll
        for (int e = 0; e < 8; e++) {          // B[32][128]=4096/128=32=8 float4
            int vec = tid + e * 128;
            int r = vec >> 5;
            int c4 = (vec & 31) << 2;
            *reinterpret_cast<float4*>(&Bs[r][c4]) = __ldg(
                reinterpret_cast<const float4*>(B + (long)(k0 + r) * N + blockCol + c4));
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[FN];
            #pragma unroll
            for (int i = 0; i < FM; i++) {
                wmma::load_matrix_sync(a_frag[i], &As[warpM * 64 + i * 16][kk], BK);
                #pragma unroll
                for (int t = 0; t < a_frag[i].num_elements; t++)
                    a_frag[i].x[t] = wmma::__float_to_tf32(a_frag[i].x[t]);
            }
            #pragma unroll
            for (int j = 0; j < FN; j++) {
                wmma::load_matrix_sync(b_frag[j], &Bs[kk][warpN * 64 + j * 16], BN);
                #pragma unroll
                for (int t = 0; t < b_frag[j].num_elements; t++)
                    b_frag[j].x[t] = wmma::__float_to_tf32(b_frag[j].x[t]);
            }
            #pragma unroll
            for (int i = 0; i < FM; i++)
                #pragma unroll
                for (int j = 0; j < FN; j++)
                    wmma::mma_sync(acc[i][j][bank], a_frag[i], b_frag[j], acc[i][j][bank]);
        }
        __syncthreads();
        bank ^= 1;
    }

    #pragma unroll
    for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++) {
            #pragma unroll
            for (int t = 0; t < acc[i][j][0].num_elements; t++)
                acc[i][j][0].x[t] += acc[i][j][1].x[t];
            int row = blockRow + warpM * 64 + i * 16;
            int col = blockCol + warpN * 64 + j * 16;
            wmma::store_matrix_sync(C + (long)row * N + col, acc[i][j][0], N,
                                    wmma::mem_row_major);
        }
}

torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda());
    TORCH_CHECK(A.scalar_type() == torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 grid(N / BN, M / BM);
    gemm_wmma<<<grid, 128>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                             C.data_ptr<float>(), M, N, K);
    return C;
}
'''

_CPP = "torch::Tensor gemm(torch::Tensor A, torch::Tensor B);"

_ext = load_inline(
    name="gemm_noptx_wmma",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["gemm"],
    verbose=False,
    extra_cuda_cflags=["-O3"],
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _ext.gemm(A.contiguous(), B.contiguous())
