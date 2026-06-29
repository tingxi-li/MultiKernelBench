import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA (no inline PTX) — plain CUDA C++ grid-stride elementwise: Exact GELU via erf.
# All compute in the kernel; forward() is allocate/launch glue only.
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float actf(float a){
    return a * 0.5f * (1.0f + erff(a * 0.70710678118654752f));
}
__global__ void gelu_k4(const float4* __restrict__ x, float4* __restrict__ y, long n4){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n4; i += stride){
        float4 v = x[i];
        v.x = actf(v.x); v.y = actf(v.y); v.z = actf(v.z); v.w = actf(v.w);
        y[i] = v;
    }
}
__global__ void gelu_tail(const float* __restrict__ x, float* __restrict__ y, long start, long n){
    long i = start + (long)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) y[i] = actf(x[i]);
}
torch::Tensor gelu_cuda(torch::Tensor x){
    auto y = torch::empty_like(x);
    long n = x.numel();
    long n4 = n / 4;
    int threads = 256;
    if(n4 > 0){
        long want = (n4 + threads - 1) / threads;
        int blocks = (int)(want < 131072 ? want : 131072);
        gelu_k4<<<blocks, threads>>>((const float4*)x.data_ptr<float>(),
                                     (float4*)y.data_ptr<float>(), n4);
    }
    long rem_start = n4 * 4;
    if(rem_start < n){
        gelu_tail<<<1, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), rem_start, n);
    }
    return y;
}
"""
_CPP = "torch::Tensor gelu_cuda(torch::Tensor x);"
_ext = load_inline(name="gelu_cuda_noptx_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["gelu_cuda"], verbose=False, extra_cuda_cflags=["-O3"])


class Model(nn.Module):
    """Exact GELU via erf — plain CUDA (no PTX) grid-stride kernel."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _ext.gelu_cuda(x.contiguous())
