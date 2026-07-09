import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 5: In-place GELU+softmax on the linear output tensor.
# Avoids allocating a 32MB output buffer alongside the 32MB linear output.
# Peak memory: 32MB (vs 64MB in iters 3/4), better L2 cache reuse.
# The kernel reads all values into registers, applies GELU, computes softmax,
# then writes the normalized result back to the same buffer.
# Warp-shuffle reductions; float4 vectorized I/O.

_cuda_src = r"""
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, mask);
    return v;
}

// In-place fused GELU + row-softmax.
// Reads the input, holds all values in registers, writes result back in-place.
// N=8192, THREADS=256 (8 warps), EPT=32, float4 vectorized.
template <int THREADS, int EPT, int WARPS>
__global__ void gelu_softmax_inplace_kernel(
    float* __restrict__ data,   // [M, N] in-place
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    __shared__ float warp_scratch[WARPS];

    float4* row4 = reinterpret_cast<float4*>(data + (ptrdiff_t)row * N);

    float vals[EPT];

    // ---- Load + GELU, find local max ----
    float lmax = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < EPT / 4; i++) {
        float4 v4 = row4[threadIdx.x + i * THREADS];
        float g0 = 0.5f * v4.x * (1.0f + erff(v4.x * 0.70710678118654752f));
        float g1 = 0.5f * v4.y * (1.0f + erff(v4.y * 0.70710678118654752f));
        float g2 = 0.5f * v4.z * (1.0f + erff(v4.z * 0.70710678118654752f));
        float g3 = 0.5f * v4.w * (1.0f + erff(v4.w * 0.70710678118654752f));
        vals[i*4+0] = g0; lmax = fmaxf(lmax, g0);
        vals[i*4+1] = g1; lmax = fmaxf(lmax, g1);
        vals[i*4+2] = g2; lmax = fmaxf(lmax, g2);
        vals[i*4+3] = g3; lmax = fmaxf(lmax, g3);
    }

    // Warp-level max
    lmax = warp_reduce_max(lmax);
    if (lane_id == 0) warp_scratch[warp_id] = lmax;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < WARPS) ? warp_scratch[lane_id] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane_id == 0) warp_scratch[0] = v;
    }
    __syncthreads();
    const float row_max = warp_scratch[0];

    // ---- exp(val - max), local sum ----
    float lsum = 0.0f;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        float e = __expf(vals[i] - row_max);
        vals[i] = e;
        lsum += e;
    }

    // Warp-level sum
    lsum = warp_reduce_sum(lsum);
    if (lane_id == 0) warp_scratch[warp_id] = lsum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < WARPS) ? warp_scratch[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) warp_scratch[0] = v;
    }
    __syncthreads();
    const float inv_sum = 1.0f / warp_scratch[0];

    // ---- Write normalized result back in-place (float4) ----
    #pragma unroll
    for (int i = 0; i < EPT / 4; i++) {
        float4 o;
        o.x = vals[i*4+0] * inv_sum;
        o.y = vals[i*4+1] * inv_sum;
        o.z = vals[i*4+2] * inv_sum;
        o.w = vals[i*4+3] * inv_sum;
        row4[threadIdx.x + i * THREADS] = o;
    }
}

void gelu_softmax_inplace(torch::Tensor data) {
    const int M = (int)data.size(0);
    const int N = (int)data.size(1);

    constexpr int THREADS = 256;
    constexpr int EPT = 32;
    constexpr int WARPS = THREADS / 32;
    gelu_softmax_inplace_kernel<THREADS, EPT, WARPS><<<M, THREADS, WARPS * sizeof(float)>>>(
        data.data_ptr<float>(), M, N
    );
}
"""

_cpp_src = r"""
void gelu_softmax_inplace(torch::Tensor data);
"""

_ext = load_inline(
    name="mgf_v5",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["gelu_softmax_inplace"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        _ext.gelu_softmax_inplace(x)
        return x
