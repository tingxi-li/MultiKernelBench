import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#define TPB 256
__global__ void gn_stats(const float* __restrict__ x, float* __restrict__ mean,
                         float* __restrict__ rstd, long gnum, double eps){
    int ng = blockIdx.x; long base = (long)ng * gnum;
    float ls = 0.f, lss = 0.f;
    const float4* x4 = reinterpret_cast<const float4*>(x + base); long gnum4 = gnum >> 2;
    for(long k = threadIdx.x; k < gnum4; k += blockDim.x){ float4 v = x4[k];
        ls += v.x + v.y + v.z + v.w; lss += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w; }
    __shared__ float ss[TPB], sq[TPB]; int t = threadIdx.x; ss[t] = ls; sq[t] = lss; __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1){ if(t < s){ ss[t] += ss[t+s]; sq[t] += sq[t+s]; } __syncthreads(); }
    if(t == 0){ double mu = (double)ss[0] / (double)gnum, var = (double)sq[0] / (double)gnum - mu * mu;
        mean[ng] = (float)mu; rstd[ng] = (float)(1.0 / sqrt(var + eps)); }
}
__global__ void gn_apply(const float* __restrict__ x, const float* __restrict__ mean, const float* __restrict__ rstd,
                         const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y,
                         long HW, int C, int GPC, int G){
    int nc = blockIdx.x, n = nc / C, c = nc % C, g = n * G + c / GPC;
    float sc = rstd[g] * w[c], sh = b[c] - mean[g] * sc; long base = (long)nc * HW;
    const float4* x4 = reinterpret_cast<const float4*>(x + base);
    float4* y4 = reinterpret_cast<float4*>(y + base); long HW4 = HW >> 2;
    for(long k = threadIdx.x; k < HW4; k += blockDim.x){ float4 v = x4[k];
        v.x = v.x*sc + sh; v.y = v.y*sc + sh; v.z = v.z*sc + sh; v.w = v.w*sc + sh; y4[k] = v; }
}
torch::Tensor groupnorm_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, long G, double eps){
    long N = x.size(0), C = x.size(1), HW = x.numel() / (N * C), GPC = C / G, gnum = GPC * HW;
    auto mean = torch::empty({N * G}, x.options()), rstd = torch::empty({N * G}, x.options());
    auto y = torch::empty_like(x);
    gn_stats<<<(int)(N * G), TPB>>>(x.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), gnum, eps);
    gn_apply<<<(int)(N * C), TPB>>>(x.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
                                    w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), HW, (int)C, (int)GPC, (int)G);
    return y;
}
"""
_CPP = "torch::Tensor groupnorm_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, long G, double eps);"
_ext = load_inline(name="groupnorm_noptx_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["groupnorm_cuda"], verbose=False, extra_cuda_cflags=["-O3"])

class Model(nn.Module):
    """GroupNorm via plain CUDA: one block per (batch,group) reduces the group ->
    mean/rstd, then one block per (batch,channel) applies the per-channel affine.
    self.gn is a parameter container (weight/bias/eps/num_groups); never called."""
    def __init__(self, num_features, num_groups):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    def forward(self, x):
        return _ext.groupnorm_cuda(x.contiguous(), self.gn.weight.contiguous(),
                                   self.gn.bias.contiguous(), self.gn.num_groups, self.gn.eps)
