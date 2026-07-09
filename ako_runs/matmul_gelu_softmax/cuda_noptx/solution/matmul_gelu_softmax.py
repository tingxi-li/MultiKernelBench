import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Strategy: use PyTorch's self.linear(x) for the GEMM (correct, stable, TF32-optimal),
# then fuse GELU + row-softmax in a single CUDA kernel.
# The fused kernel reads the linear output once, keeps GELU values in registers,
# and writes the softmax result — eliminating one full 32MB HBM round-trip.
# Uses float4 vectorized loads/stores (16 bytes/transaction).

_cuda_src = r"""
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Fused GELU + row-softmax
// N=8192, THREADS=256, EPT=32 (each thread covers 32 elements)
// Float4 vectorized: each thread loads/stores EPT/4=8 float4 values.
template <int THREADS, int EPT>
__global__ void fused_gelu_softmax_kernel(
    const float* __restrict__ input,   // [M, N] — linear output (GEMM+bias)
    float* __restrict__ output,         // [M, N]
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float smem[];  // [THREADS]

    const float* in_row  = input  + (ptrdiff_t)row * N;
    float* out_row = output + (ptrdiff_t)row * N;

    // Use float4 for coalesced vectorized loads (EPT/4 float4 per thread)
    const float4* in4  = reinterpret_cast<const float4*>(in_row);
    float4* out4 = reinterpret_cast<float4*>(out_row);

    float vals[EPT];

    // ---- Pass 1: load + GELU, track local max ----
    float lmax = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < EPT / 4; i++) {
        float4 v4 = in4[threadIdx.x + i * THREADS];
        float g0 = 0.5f * v4.x * (1.0f + erff(v4.x * 0.70710678118654752f));
        float g1 = 0.5f * v4.y * (1.0f + erff(v4.y * 0.70710678118654752f));
        float g2 = 0.5f * v4.z * (1.0f + erff(v4.z * 0.70710678118654752f));
        float g3 = 0.5f * v4.w * (1.0f + erff(v4.w * 0.70710678118654752f));
        vals[i*4+0] = g0; lmax = fmaxf(lmax, g0);
        vals[i*4+1] = g1; lmax = fmaxf(lmax, g1);
        vals[i*4+2] = g2; lmax = fmaxf(lmax, g2);
        vals[i*4+3] = g3; lmax = fmaxf(lmax, g3);
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

    // ---- Pass 2: exp(val - max), sum (in registers) ----
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

    // ---- Pass 3: write normalized output with float4 ----
    #pragma unroll
    for (int i = 0; i < EPT / 4; i++) {
        float4 o;
        o.x = vals[i*4+0] * inv_sum;
        o.y = vals[i*4+1] * inv_sum;
        o.z = vals[i*4+2] * inv_sum;
        o.w = vals[i*4+3] * inv_sum;
        out4[threadIdx.x + i * THREADS] = o;
    }
}

torch::Tensor gelu_softmax_fused(
    torch::Tensor input  // [M, N] float32 contiguous
) {
    const int M = (int)input.size(0);
    const int N = (int)input.size(1);
    auto out = torch::empty({M, N}, input.options());

    constexpr int THREADS = 256;
    constexpr int EPT = 32;   // 8192 / 256
    const int smem_bytes = THREADS * sizeof(float);
    fused_gelu_softmax_kernel<THREADS, EPT><<<M, THREADS, smem_bytes>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N
    );
    return out;
}
"""

_cpp_src = r"""
torch::Tensor gelu_softmax_fused(torch::Tensor input);
"""

_ext = load_inline(
    name="mgf_v3",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["gelu_softmax_fused"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return _ext.gelu_softmax_fused(x)
