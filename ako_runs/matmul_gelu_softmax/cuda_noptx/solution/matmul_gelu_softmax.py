import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 6: Online softmax (single accumulation pass for max+sum via log-sum-exp trick)
# + 512 threads/block (EPT=16) for better erff parallelism per row.
# Online softmax: accumulate (max, scaled_sum) together, avoiding a separate
# second pass over elements. After GELU, use online update:
#   new_max = max(old_max, val)
#   new_sum = old_sum * exp(old_max - new_max) + exp(val - new_max)
# This merges the "find max" and "compute exp sum" into one pass.
# 3 warp reductions (instead of 2 separate: one for max, one for sum), but
# only 1 pass over the 8192 GELU values instead of 2.

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

// Online softmax reduction: merge two (max, sum) pairs
struct OnlineSoftmax {
    float m;   // running max
    float s;   // sum of exp(x_i - m) so far
};

__device__ __forceinline__ OnlineSoftmax merge_os(OnlineSoftmax a, OnlineSoftmax b) {
    if (a.m >= b.m) {
        return {a.m, a.s + b.s * __expf(b.m - a.m)};
    } else {
        return {b.m, b.s + a.s * __expf(a.m - b.m)};
    }
}

// Online warp reduce for (max, sum) pair
__device__ __forceinline__ OnlineSoftmax warp_reduce_os(OnlineSoftmax v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        OnlineSoftmax other;
        other.m = __shfl_xor_sync(0xffffffff, v.m, mask);
        other.s = __shfl_xor_sync(0xffffffff, v.s, mask);
        v = merge_os(v, other);
    }
    return v;
}

// Fused in-place GELU + row-softmax with online softmax algorithm.
// N=8192, THREADS=512 (16 warps), EPT=16, float4 vectorized.
// Single pass: GELU vals in registers, accumulate (max, scaled_sum) online,
// then single normalize pass.
template <int THREADS, int EPT, int WARPS>
__global__ void gelu_softmax_online_kernel(
    float* __restrict__ data,   // [M, N] in-place
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    // Shared: 2 * WARPS floats (max + sum for inter-warp online reduction)
    __shared__ float warp_m[WARPS];
    __shared__ float warp_s[WARPS];

    float4* row4 = reinterpret_cast<float4*>(data + (ptrdiff_t)row * N);

    float vals[EPT];

    // ---- Pass 1: load + GELU + online softmax accumulation ----
    OnlineSoftmax acc = {-FLT_MAX, 0.0f};
    #pragma unroll
    for (int i = 0; i < EPT / 4; i++) {
        float4 v4 = row4[threadIdx.x + i * THREADS];
        float g0 = 0.5f * v4.x * (1.0f + erff(v4.x * 0.70710678118654752f));
        float g1 = 0.5f * v4.y * (1.0f + erff(v4.y * 0.70710678118654752f));
        float g2 = 0.5f * v4.z * (1.0f + erff(v4.z * 0.70710678118654752f));
        float g3 = 0.5f * v4.w * (1.0f + erff(v4.w * 0.70710678118654752f));
        vals[i*4+0] = g0;
        vals[i*4+1] = g1;
        vals[i*4+2] = g2;
        vals[i*4+3] = g3;

        // Online update for each element
        OnlineSoftmax e0 = {g0, 1.0f};
        OnlineSoftmax e1 = {g1, 1.0f};
        OnlineSoftmax e2 = {g2, 1.0f};
        OnlineSoftmax e3 = {g3, 1.0f};
        acc = merge_os(acc, e0);
        acc = merge_os(acc, e1);
        acc = merge_os(acc, e2);
        acc = merge_os(acc, e3);
    }

    // Warp-level online softmax reduction
    acc = warp_reduce_os(acc);
    if (lane_id == 0) {
        warp_m[warp_id] = acc.m;
        warp_s[warp_id] = acc.s;
    }
    __syncthreads();

    // Inter-warp reduction by warp 0
    if (warp_id == 0) {
        OnlineSoftmax v;
        if (lane_id < WARPS) {
            v.m = warp_m[lane_id];
            v.s = warp_s[lane_id];
        } else {
            v.m = -FLT_MAX;
            v.s = 0.0f;
        }
        v = warp_reduce_os(v);
        if (lane_id == 0) {
            warp_m[0] = v.m;
            warp_s[0] = v.s;
        }
    }
    __syncthreads();

    const float row_max = warp_m[0];
    const float inv_sum = 1.0f / warp_s[0];

    // ---- Pass 2: normalize and write back (float4) ----
    #pragma unroll
    for (int i = 0; i < EPT / 4; i++) {
        float4 o;
        o.x = __expf(vals[i*4+0] - row_max) * inv_sum;
        o.y = __expf(vals[i*4+1] - row_max) * inv_sum;
        o.z = __expf(vals[i*4+2] - row_max) * inv_sum;
        o.w = __expf(vals[i*4+3] - row_max) * inv_sum;
        row4[threadIdx.x + i * THREADS] = o;
    }
}

void gelu_softmax_online(torch::Tensor data) {
    const int M = (int)data.size(0);
    const int N = (int)data.size(1);

    constexpr int THREADS = 512;
    constexpr int EPT = 16;
    constexpr int WARPS = THREADS / 32;  // 16
    gelu_softmax_online_kernel<THREADS, EPT, WARPS><<<M, THREADS, 2 * WARPS * sizeof(float)>>>(
        data.data_ptr<float>(), M, N
    );
}
"""

_cpp_src = r"""
void gelu_softmax_online(torch::Tensor data);
"""

_ext = load_inline(
    name="mgf_v6",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["gelu_softmax_online"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        _ext.gelu_softmax_online(x)
        return x
