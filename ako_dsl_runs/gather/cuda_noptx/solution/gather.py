import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA (no PTX) gather along dim=1: out[r,c] = x[r, idx[r,c]].
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#define KPT 8
__global__ void gather_k(const float* __restrict__ x, const long* __restrict__ idx,
                         float* __restrict__ out, long Cin, long Cout){
    long r = blockIdx.y;
    const float* xr = x + r * Cin;
    long base = r * Cout + (long)blockIdx.x * blockDim.x * KPT + threadIdx.x;
    long ci[KPT];
    long col[KPT];
    #pragma unroll
    for(int k=0;k<KPT;k++) ci[k] = base + (long)k * blockDim.x;
    #pragma unroll
    for(int k=0;k<KPT;k++) if(ci[k] < (r+1)*Cout) col[k] = idx[ci[k]];
    #pragma unroll
    for(int k=0;k<KPT;k++) if(ci[k] < (r+1)*Cout) out[ci[k]] = xr[col[k]];
}
torch::Tensor gather_cuda(torch::Tensor x, torch::Tensor idx){
    long M = x.size(0), Cin = x.size(1), Cout = idx.size(1);
    auto out = torch::empty({M, Cout}, x.options());
    int threads = 256;
    long gx = (Cout + (long)threads * KPT - 1) / ((long)threads * KPT);
    dim3 blocks((unsigned)gx, (unsigned)M);
    gather_k<<<blocks, threads>>>(x.data_ptr<float>(), idx.data_ptr<long>(),
                                  out.data_ptr<float>(), Cin, Cout);
    return out;
}
"""
_CPP = "torch::Tensor gather_cuda(torch::Tensor x, torch::Tensor idx);"
_ext = load_inline(name="gather_cuda_noptx_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["gather_cuda"], verbose=False, extra_cuda_cflags=["-O3"])

class Model(nn.Module):
    """gather(x, dim=1, index=idx) via plain CUDA."""
    def forward(self, x, idx):
        return _ext.gather_cuda(x.contiguous(), idx.contiguous())
