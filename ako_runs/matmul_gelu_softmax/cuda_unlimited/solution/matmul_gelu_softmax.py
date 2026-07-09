import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 6 (final): Exact iter-2 kernel — best performing approach
# BM=BN=128, BK=16, TM=TN=8, 256 threads
# K-major shared memory: As[BK][BM+PAD], Bs[BK][BN+PAD]
# Load pattern: k=e/BM, m=e%BM (original iter-2)
# This achieved best mean (6.26ms, 0.99x) and best min (4.21ms)

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <float.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8
#define NT 256
#define PAD 4

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__global__
void gemm_gelu_reg(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int ty = threadIdx.x / (BN / TN);
    const int tx = threadIdx.x % (BN / TN);
    const int tid = threadIdx.x;

    __shared__ float As[BK][BM + PAD];
    __shared__ float Bs[BK][BN + PAD];

    float acc[TM][TN] = {};

    for (int k_base = 0; k_base < K; k_base += BK) {
        for (int e = tid; e < BM * BK; e += NT) {
            int k = e / BM, m = e % BM;
            int gm = block_row + m, gk = k_base + k;
            As[k][m] = (gm < M && gk < K) ? A[gm * K + gk] : 0.f;
        }
        for (int e = tid; e < BN * BK; e += NT) {
            int k = e / BN, n = e % BN;
            int gn = block_col + n, gk = k_base + k;
            Bs[k][n] = (gn < N && gk < K) ? W[gn * K + gk] : 0.f;
        }
        __syncthreads();

        float a_reg[TM], b_reg[TN];
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_reg[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_reg[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = block_row + ty * TM + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = block_col + tx * TN + j;
            if (gn < N)
                C[gm * N + gn] = gelu_exact(acc[i][j] + bias[gn]);
        }
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
        for (int m=4;m>0;m>>=1) v=fmaxf(v,__shfl_xor_sync(~0u,v,m));
        if(!lane) sm[0]=v;
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
        for (int m=4;m>0;m>>=1) v+=__shfl_xor_sync(~0u,v,m);
        if(!lane) sm[0]=v;
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
__global__ void gemm_gelu_reg(const float*, const float*, const float*, float*, int, int, int);
__global__ void softmax_kernel(float*, int, int);

torch::Tensor fused_matmul_gelu_softmax(
    torch::Tensor A, torch::Tensor W, torch::Tensor bias, int M, int N, int K)
{
    auto C = torch::empty({M, N}, A.options());
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    gemm_gelu_reg<<<grid, 256>>>(
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
            name="fused_mgs_final_v1",
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
