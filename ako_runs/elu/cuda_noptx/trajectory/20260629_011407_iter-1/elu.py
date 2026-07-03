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
__global__ void elu_k(const float* __restrict__ x, float* __restrict__ y, long n, float alpha){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n; i += stride) y[i] = actf(x[i], alpha);
}
torch::Tensor elu_cuda(torch::Tensor x, double alpha_){
    float alpha = (float)alpha_;
    auto y = torch::empty_like(x);
    long n = x.numel();
    int threads = 256;
    long want = (n + threads - 1) / threads;
    int blocks = (int)(want < 131072 ? want : 131072);
    elu_k<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n, alpha);
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
