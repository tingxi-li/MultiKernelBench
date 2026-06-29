import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA (no inline PTX) — plain CUDA C++ grid-stride elementwise: HardSigmoid clamp(x/6+1/2,0,1).
# All compute in the kernel; forward() is allocate/launch glue only.
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float actf(float a){
    return fminf(fmaxf(fmaf(a, 0.16666666666666666f, 0.5f), 0.0f), 1.0f);
}
__global__ void hardsigmoid_k4(const float4* __restrict__ x, float4* __restrict__ y, long n4){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n4; i += stride){
        float4 v = x[i];
        v.x = actf(v.x); v.y = actf(v.y); v.z = actf(v.z); v.w = actf(v.w);
        y[i] = v;
    }
}
__global__ void hardsigmoid_k(const float* __restrict__ x, float* __restrict__ y, long off, long n){
    long i = off + (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n; i += stride) y[i] = actf(x[i]);
}
torch::Tensor hardsigmoid_cuda(torch::Tensor x){
    auto y = torch::empty_like(x);
    long n = x.numel();
    int threads = 256;
    long n4 = n / 4;
    if(n4 > 0){
        long want = (n4 + threads - 1) / threads;
        int blocks = (int)(want < 131072 ? want : 131072);
        hardsigmoid_k4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(y.data_ptr<float>()), n4);
    }
    long rem = n4 * 4;
    if(rem < n){
        hardsigmoid_k<<<1, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), rem, n);
    }
    return y;
}
"""
_CPP = "torch::Tensor hardsigmoid_cuda(torch::Tensor x);"
_ext = load_inline(name="hardsigmoid_cuda_noptx_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["hardsigmoid_cuda"], verbose=False, extra_cuda_cflags=["-O3"])


class Model(nn.Module):
    """HardSigmoid clamp(x/6+1/2,0,1) — plain CUDA (no PTX) grid-stride kernel."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _ext.hardsigmoid_cuda(x.contiguous())
