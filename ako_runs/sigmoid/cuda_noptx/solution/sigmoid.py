import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA (no inline PTX) — plain CUDA C++ grid-stride elementwise: Sigmoid 1/(1+exp(-x)).
# All compute in the kernel; forward() is allocate/launch glue only.
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float actf(float a){
    return 1.0f / (1.0f + expf(-a));
}
__global__ void sigmoid_k4(const float4* __restrict__ x, float4* __restrict__ y, long n4){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n4; i += stride){
        float4 v = x[i];
        v.x = actf(v.x); v.y = actf(v.y); v.z = actf(v.z); v.w = actf(v.w);
        y[i] = v;
    }
}
__global__ void sigmoid_k(const float* __restrict__ x, float* __restrict__ y, long n, long start){
    long i = start + (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n; i += stride) y[i] = actf(x[i]);
}
torch::Tensor sigmoid_cuda(torch::Tensor x){
    auto y = torch::empty_like(x);
    long n = x.numel();
    int threads = 256;
    long n4 = n / 4;
    if(n4 > 0){
        long want = (n4 + threads - 1) / threads;
        int blocks = (int)(want < 131072 ? want : 131072);
        sigmoid_k4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(y.data_ptr<float>()), n4);
    }
    long done = n4 * 4;
    if(done < n){
        sigmoid_k<<<1, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n, done);
    }
    return y;
}
"""
_CPP = "torch::Tensor sigmoid_cuda(torch::Tensor x);"
_ext = load_inline(name="sigmoid_cuda_noptx_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["sigmoid_cuda"], verbose=False, extra_cuda_cflags=["-O3"])


class Model(nn.Module):
    """Sigmoid 1/(1+exp(-x)) — plain CUDA (no PTX) grid-stride kernel."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _ext.sigmoid_cuda(x.contiguous())
