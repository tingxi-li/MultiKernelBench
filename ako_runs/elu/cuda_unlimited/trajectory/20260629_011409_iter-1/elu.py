import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA UNLIMITED — float4 (128-bit) vectorized loads + inline-PTX cache-STREAMING
# vectorized store (st.global.cs.v4.f32). ELU x>0?x:alpha*(exp(x)-1). The no-holds-barred track: 128-bit
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
__device__ __forceinline__ float actf(float a, float alpha){
    return (a > 0.0f ? a : alpha * (expf(a) - 1.0f));
}
__global__ void elu_v4(const float4* __restrict__ x4, float* __restrict__ y, long n4, float alpha){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n4; i += stride){
        float4 v = __ldg(x4 + i);
        v.x = actf(v.x, alpha); v.y = actf(v.y, alpha);
        v.z = actf(v.z, alpha); v.w = actf(v.w, alpha);
        stcs_v4(y + i*4, v);
    }
}
__global__ void elu_tail(const float* __restrict__ x, float* __restrict__ y, long s, long n, float alpha){
    long i = s + (long)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) y[i] = actf(x[i], alpha);
}
torch::Tensor elu_cuda(torch::Tensor x, double alpha_){
    float alpha = (float)alpha_;
    auto y = torch::empty_like(x);
    long n = x.numel();
    long n4 = n / 4;
    int threads = 256;
    long want = (n4 + threads - 1) / threads;
    int blocks = (int)(want < 131072 ? want : 131072);
    if(n4 > 0) elu_v4<<<blocks, threads>>>((const float4*)x.data_ptr<float>(), y.data_ptr<float>(), n4, alpha);
    long s = n4 * 4;
    if(s < n) elu_tail<<<1, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), s, n, alpha);
    return y;
}
"""
_CPP = "torch::Tensor elu_cuda(torch::Tensor x, double alpha);"
_ext = load_inline(name="elu_cuda_unlimited_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["elu_cuda"], verbose=False, extra_cuda_cflags=["-O3"])


class Model(nn.Module):
    """ELU x>0?x:alpha*(exp(x)-1) — CUDA with inline-PTX float4 vectorized memory ops."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return _ext.elu_cuda(x.contiguous(), self.alpha)
