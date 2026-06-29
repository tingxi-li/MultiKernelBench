import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA UNLIMITED deterministic scatter: inline-PTX red.global.max.s32 (a reduction
# atomic with NO return value -> cheaper than atomicMax when the old value is
# unused) for the winner pass; a scalar grid-stride gather for the apply pass.
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__device__ __forceinline__ void red_max_s32(int* p, int v){
    asm volatile("red.global.max.s32 [%0], %1;" :: "l"(p), "r"(v));
}
__global__ void argk_k(const long* __restrict__ idx, int* __restrict__ win,
                       long n, long K, long W){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n; i += stride){
        long r = i / K;
        long slot = idx[i];
        red_max_s32(&win[r * W + slot], (int)(i % K));
    }
}
__global__ void gather_win_k(const float* __restrict__ x, const int* __restrict__ win,
                             const float* __restrict__ upd, float* __restrict__ out,
                             long n, long K, long W){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n; i += stride){
        long r = i / W;
        int wk = win[i];
        out[i] = (wk >= 0) ? upd[r * K + wk] : x[i];
    }
}
torch::Tensor scatter_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor upd){
    long Rr = x.size(0), W = x.size(1), K = idx.size(1);
    auto out = torch::empty_like(x);
    auto win = torch::full({Rr, W}, -1, x.options().dtype(torch::kInt32));
    long n1 = idx.numel(), n2 = out.numel();
    int t = 256;
    int b1 = (int)((n1 + t - 1) / t < 65535 ? (n1 + t - 1) / t : 65535);
    int b2 = (int)((n2 + t - 1) / t < 65535 ? (n2 + t - 1) / t : 65535);
    argk_k<<<b1, t>>>(idx.data_ptr<long>(), win.data_ptr<int>(), n1, K, W);
    gather_win_k<<<b2, t>>>(x.data_ptr<float>(), win.data_ptr<int>(),
                            upd.data_ptr<float>(), out.data_ptr<float>(), n2, K, W);
    return out;
}
"""
_CPP = "torch::Tensor scatter_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor upd);"
_ext = load_inline(name="scatter_cuda_unlimited_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["scatter_cuda"], verbose=False, extra_cuda_cflags=["-O3"])

class Model(nn.Module):
    """Deterministic last-wins scatter (dim=1) via CUDA inline-PTX red.global.max."""
    def forward(self, x, idx, updates):
        return _ext.scatter_cuda(x.contiguous(), idx.contiguous(), updates.contiguous())
