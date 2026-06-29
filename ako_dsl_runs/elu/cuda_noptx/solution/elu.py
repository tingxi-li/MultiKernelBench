import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA (no inline PTX) — plain CUDA C++ grid-stride elementwise: ELU x>0?x:alpha*(exp(x)-1).
# All compute in the kernel; forward() is allocate/launch glue only.
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float actf(float a, float alpha){
    return (a > 0.0f ? a : alpha * (expf(a) - 1.0f));
}
__global__ void elu_k4(const float4* __restrict__ x, float4* __restrict__ y, long n4, float alpha){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n4; i += stride){
        float4 v = x[i];
        v.x = actf(v.x, alpha); v.y = actf(v.y, alpha);
        v.z = actf(v.z, alpha); v.w = actf(v.w, alpha);
        y[i] = v;
    }
}
__global__ void elu_k(const float* __restrict__ x, float* __restrict__ y, long n, long start, float alpha){
    long i = start + (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n; i += stride) y[i] = actf(x[i], alpha);
}
torch::Tensor elu_cuda(torch::Tensor x, double alpha_){
    float alpha = (float)alpha_;
    auto y = torch::empty_like(x);
    long n = x.numel();
    long n4 = n / 4;
    int threads = 256;
    if(n4 > 0){
        long want = (n4 + threads - 1) / threads;
        int blocks = (int)(want < 131072 ? want : 131072);
        elu_k4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(y.data_ptr<float>()), n4, alpha);
    }
    long tail = n4 * 4;
    if(tail < n){
        long rem = n - tail;
        int blocks = (int)((rem + threads - 1) / threads);
        elu_k<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n, tail, alpha);
    }
    return y;
}
"""
_CPP = "torch::Tensor elu_cuda(torch::Tensor x, double alpha);"
_ext = load_inline(name="elu_cuda_noptx_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["elu_cuda"], verbose=False, extra_cuda_cflags=["-O3"])


class Model(nn.Module):
    """ELU x>0?x:alpha*(exp(x)-1) — plain CUDA (no PTX) grid-stride kernel."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return _ext.elu_cuda(x.contiguous(), self.alpha)
