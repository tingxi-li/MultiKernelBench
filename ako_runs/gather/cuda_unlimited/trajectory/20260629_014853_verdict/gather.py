import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA UNLIMITED gather: inline-PTX non-coherent loads (ld.global.nc) for both
# the int64 index and the gathered float — the gather is latency-bound, so the
# .nc cache path is the relevant lever.
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__device__ __forceinline__ long ldnc_s64(const long* p){
    long v; asm volatile("ld.global.nc.s64 %0, [%1];" : "=l"(v) : "l"(p)); return v;
}
__device__ __forceinline__ float ldnc_f32(const float* p){
    float v; asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v;
}
__global__ void gather_k(const float* __restrict__ x, const long* __restrict__ idx,
                         float* __restrict__ out, long n, long Cin, long Cout){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n; i += stride){
        long r = i / Cout;
        long col = ldnc_s64(idx + i);
        out[i] = ldnc_f32(x + r * Cin + col);
    }
}
torch::Tensor gather_cuda(torch::Tensor x, torch::Tensor idx){
    long M = x.size(0), Cin = x.size(1), Cout = idx.size(1);
    auto out = torch::empty({M, Cout}, x.options());
    long n = out.numel();
    int threads = 256;
    long want = (n + threads - 1) / threads;
    int blocks = (int)(want < 65535 ? want : 65535);
    gather_k<<<blocks, threads>>>(x.data_ptr<float>(), idx.data_ptr<long>(),
                                  out.data_ptr<float>(), n, Cin, Cout);
    return out;
}
"""
_CPP = "torch::Tensor gather_cuda(torch::Tensor x, torch::Tensor idx);"
_ext = load_inline(name="gather_cuda_unlimited_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["gather_cuda"], verbose=False, extra_cuda_cflags=["-O3"])

class Model(nn.Module):
    """gather(x, dim=1, index=idx) via CUDA with inline-PTX .nc loads."""
    def forward(self, x, idx):
        return _ext.gather_cuda(x.contiguous(), idx.contiguous())
