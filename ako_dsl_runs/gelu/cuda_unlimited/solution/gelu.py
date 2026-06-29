import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA UNLIMITED — float4 (128-bit) vectorized loads + inline-PTX cache-STREAMING
# vectorized store (st.global.cs.v4.f32). Exact GELU via erf. The no-holds-barred track: 128-bit
# coalesced memory ops + a streaming store that bypasses L2 pollution for the
# write-once output (the relevant lever for a bandwidth-bound elementwise op).
# (A `ld.global.nc.v4` inline-asm *load* hangs ptxas-13.1 on transcendentals, so the
# load uses __ldg on float4* — same 128-bit non-coherent path, compiler-scheduled.)
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void stcs_v4(float* p, float4 v){
    asm volatile("st.global.cs.v4.f32 [%4], {%0,%1,%2,%3};"
                 :: "f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"l"(p));
}
__device__ __forceinline__ float actf(float a){
    return a * 0.5f * (1.0f + erff(a * 0.70710678118654752f));
}
// __launch_bounds__ caps registers -> keeps occupancy high even for erf/exp-heavy
// activations (gelu's 4x erff in a float4 spills to 0.46x without this).
__global__ void __launch_bounds__(256, 6) gelu_v4(const float4* __restrict__ x4, float* __restrict__ y, long n4){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n4; i += stride){
        float4 v = __ldg(x4 + i);
        v.x = actf(v.x); v.y = actf(v.y);
        v.z = actf(v.z); v.w = actf(v.w);
        stcs_v4(y + i*4, v);
    }
}
__global__ void gelu_tail(const float* __restrict__ x, float* __restrict__ y, long s, long n){
    long i = s + (long)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) y[i] = actf(x[i]);
}
torch::Tensor gelu_cuda(torch::Tensor x){
    auto y = torch::empty_like(x);
    long n = x.numel();
    long n4 = n / 4;
    int threads = 256;
    long want = (n4 + threads - 1) / threads;
    int blocks = (int)(want < 131072 ? want : 131072);
    if(n4 > 0) gelu_v4<<<blocks, threads>>>((const float4*)x.data_ptr<float>(), y.data_ptr<float>(), n4);
    long s = n4 * 4;
    if(s < n) gelu_tail<<<1, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), s, n);
    return y;
}
"""
_CPP = "torch::Tensor gelu_cuda(torch::Tensor x);"
_ext = load_inline(name="gelu_cuda_unlimited_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["gelu_cuda"], verbose=False, extra_cuda_cflags=["-O3"])


class Model(nn.Module):
    """Exact GELU via erf — CUDA with inline-PTX float4 vectorized memory ops."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _ext.gelu_cuda(x.contiguous())
