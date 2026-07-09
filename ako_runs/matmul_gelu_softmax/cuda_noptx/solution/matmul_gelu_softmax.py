import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 4: warp-shuffle reductions replace __syncthreads() tree,
# __ldg() hints for bias/input cached reads, float4 vectorized I/O.
# THREADS=256 (8 warps), EPT=32. Two sync-free intra-warp reduce passes,
# then one small shared-mem inter-warp reduce (8 values -> 8x smaller tree).

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

// Fused GELU + row-softmax with warp-shuffle reductions.
// N=8192, THREADS=256 (8 warps), EPT=32, float4 vectorized I/O.
template <int THREADS, int EPT, int WARPS>
__global__ void fused_gelu_softmax_warp_kernel(
    const float* __restrict__ input,   // [M, N]
    float* __restrict__ output,         // [M, N]
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int warp_id  = threadIdx.x >> 5;   // threadIdx.x / 32
    const int lane_id  = threadIdx.x & 31;

    // Shared mem: WARPS floats for inter-warp reduction (max, then sum)
    __shared__ float warp_scratch[WARPS];

    const float4* in4  = reinterpret_cast<const float4*>(input  + (ptrdiff_t)row * N);
    float4* out4 = reinterpret_cast<float4*>(output + (ptrdiff_t)row * N);

    float vals[EPT];

    // ---- Load + GELU, compute local max ----
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

    // ---- Warp-level max (no syncthreads within warp) ----
    lmax = warp_reduce_max(lmax);
    if (lane_id == 0) warp_scratch[warp_id] = lmax;
    __syncthreads();
    // Inter-warp max: only warp 0 reads all warp results
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

    // ---- Warp-level sum ----
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

    // ---- Write normalized output (float4) ----
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

torch::Tensor gelu_softmax_fused(torch::Tensor input) {
    const int M = (int)input.size(0);
    const int N = (int)input.size(1);
    auto out = torch::empty({M, N}, input.options());

    constexpr int THREADS = 256;
    constexpr int EPT = 32;
    constexpr int WARPS = THREADS / 32;  // 8
    // Shared: WARPS * sizeof(float) = 32 bytes
    fused_gelu_softmax_warp_kernel<THREADS, EPT, WARPS><<<M, THREADS, WARPS * sizeof(float)>>>(
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
    name="mgf_v4",
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
