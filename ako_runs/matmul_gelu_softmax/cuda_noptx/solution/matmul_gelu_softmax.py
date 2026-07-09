import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused bias-add + GELU + row-softmax after the GEMM.
# GEMM is done via at::mm (inherits PyTorch's cuBLAS handle/TF32 settings for
# exact fp32 correctness match). The fused kernel does 1 read pass + 1 write pass,
# keeping GELU intermediates in registers.

_cuda_src = r"""
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Fused bias-add + GELU + row-softmax
// N=8192, THREADS=256, EPT=32 (each thread processes 32 elements)
template <int THREADS, int EPT>
__global__ void fused_bias_gelu_softmax_kernel(
    const float* __restrict__ gemm_out,  // [M, N]
    const float* __restrict__ bias,       // [N]
    float* __restrict__ output,            // [M, N]
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float smem[];  // [THREADS]

    const float* in_row = gemm_out + (ptrdiff_t)row * N;
    float* out_row = output + (ptrdiff_t)row * N;

    float vals[EPT];

    // ---- Pass 1: load + bias + GELU (precise erff), track local max ----
    float lmax = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        int idx = threadIdx.x + i * THREADS;
        float v = in_row[idx] + bias[idx];
        // GELU exact: 0.5f * v * (1 + erf(v / sqrt(2)))
        float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
        vals[i] = g;
        lmax = fmaxf(lmax, g);
    }

    // Block-wide max reduction
    smem[threadIdx.x] = lmax;
    __syncthreads();
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    const float row_max = smem[0];

    // ---- Pass 2: exp(val - max), accumulate sum (still in registers) ----
    float lsum = 0.0f;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        float e = __expf(vals[i] - row_max);
        vals[i] = e;
        lsum += e;
    }

    // Block-wide sum reduction
    smem[threadIdx.x] = lsum;
    __syncthreads();
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float inv_sum = 1.0f / smem[0];

    // ---- Pass 3: write normalized output ----
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        int idx = threadIdx.x + i * THREADS;
        out_row[idx] = vals[i] * inv_sum;
    }
}
"""

_cpp_src = r"""
#include <torch/extension.h>

torch::Tensor matmul_gelu_softmax_fwd(
    torch::Tensor x,       // [M, K] float32 contiguous
    torch::Tensor weight,  // [N, K] float32 contiguous
    torch::Tensor bias     // [N]   float32 contiguous
);
"""

_cpp_impl = r"""
#include <torch/extension.h>

// forward declared in header
template <int THREADS, int EPT>
__global__ void fused_bias_gelu_softmax_kernel(
    const float* __restrict__ gemm_out,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N
);

torch::Tensor matmul_gelu_softmax_fwd(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int M = (int)x.size(0);
    const int N = (int)weight.size(0);

    // Use at::mm -> inherits PyTorch cuBLAS handle + TF32 settings for correctness match
    auto gemm_out = at::mm(x, weight.t());  // [M, N]
    auto out = torch::empty({M, N}, x.options());

    constexpr int THREADS = 256;
    constexpr int EPT = 32;
    const int smem_bytes = THREADS * sizeof(float);
    fused_bias_gelu_softmax_kernel<THREADS, EPT><<<M, THREADS, smem_bytes>>>(
        gemm_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N
    );

    return out;
}
"""

_ext = load_inline(
    name="mgf_v2",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src + "\n" + _cpp_impl,
    functions=["matmul_gelu_softmax_fwd"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return _ext.matmul_gelu_softmax_fwd(
            x,
            self.linear.weight,
            self.linear.bias,
        )
