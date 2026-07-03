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
    int r = blockIdx.y;
    int Cout_i = (int)Cout;
    const float* xr = x + (long)r * Cin;
    int base = r * Cout_i + (int)(blockIdx.x * blockDim.x * KPT + threadIdx.x);
    int lim = (r + 1) * Cout_i;
    int ci[KPT];
    int col[KPT];
    #pragma unroll
    for(int k=0;k<KPT;k++) ci[k] = base + k * (int)blockDim.x;
    #pragma unroll
    for(int k=0;k<KPT;k++) if(ci[k] < lim) col[k] = (int)idx[ci[k]];
    #pragma unroll
    for(int k=0;k<KPT;k++) if(ci[k] < lim) out[ci[k]] = xr[col[k]];
}
// Exact-tile vectorized fast path: contiguous-per-thread layout so the coalesced
// idx loads become 128-bit long2 loads and the coalesced stores become float4;
// the random x gather stays scalar. NG groups of VEC contiguous cols per thread.
#define VEC 4
#define NG 4   // KPT-equivalent ILP = VEC*NG = 16
__global__ void gather_k_exact(const float* __restrict__ x, const long* __restrict__ idx,
                               float* __restrict__ out, long Cin, long Cout){
    int r = blockIdx.y;
    const float* xr = x + (long)r * Cin;
    const long2* idx2 = (const long2*)idx;
    float4* out4 = (float4*)out;
    int block_base = r * (int)Cout + (int)(blockIdx.x * blockDim.x * VEC * NG);
    #pragma unroll
    for(int g=0; g<NG; ++g){
        int li = block_base + g * (int)(blockDim.x * VEC) + (int)threadIdx.x * VEC;
        long2 a = idx2[li >> 1];
        long2 b = idx2[(li >> 1) + 1];
        float4 v;
        v.x = xr[(int)a.x];
        v.y = xr[(int)a.y];
        v.z = xr[(int)b.x];
        v.w = xr[(int)b.y];
        out4[li >> 2] = v;
    }
}
torch::Tensor gather_cuda(torch::Tensor x, torch::Tensor idx){
    long M = x.size(0), Cin = x.size(1), Cout = idx.size(1);
    auto out = torch::empty({M, Cout}, x.options());
    int threads = 256;
    long tile = (long)threads * VEC * NG;
    if (Cout % tile == 0) {
        long gx = Cout / tile;
        dim3 blocks((unsigned)gx, (unsigned)M);
        gather_k_exact<<<blocks, threads>>>(x.data_ptr<float>(), idx.data_ptr<long>(),
                                            out.data_ptr<float>(), Cin, Cout);
    } else {
        long tile2 = (long)threads * KPT;
        long gx = (Cout + tile2 - 1) / tile2;
        dim3 blocks((unsigned)gx, (unsigned)M);
        gather_k<<<blocks, threads>>>(x.data_ptr<float>(), idx.data_ptr<long>(),
                                      out.data_ptr<float>(), Cin, Cout);
    }
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
