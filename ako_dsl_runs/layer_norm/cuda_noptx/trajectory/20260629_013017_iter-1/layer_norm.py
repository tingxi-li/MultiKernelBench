import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#define TPB 256
__global__ void ln_stats(const float* __restrict__ x, double* __restrict__ sum_acc,
                         double* __restrict__ sq_acc, long N, int S){
    int m = blockIdx.x / S, sc = blockIdx.x % S;
    long chunk = N / S, start = (long)m * N + (long)sc * chunk;
    float ls = 0.f, lss = 0.f;
    for(long k = threadIdx.x; k < chunk; k += blockDim.x){ float v = x[start + k]; ls += v; lss += v * v; }
    __shared__ float ss[TPB], sq[TPB];
    int t = threadIdx.x; ss[t] = ls; sq[t] = lss; __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1){ if(t < s){ ss[t] += ss[t+s]; sq[t] += sq[t+s]; } __syncthreads(); }
    if(t == 0){ atomicAdd(sum_acc + m, (double)ss[0]); atomicAdd(sq_acc + m, (double)sq[0]); }
}
__global__ void ln_final(const double* sum_acc, const double* sq_acc, float* mean, float* rstd, long N, double eps, int M){
    int m = blockIdx.x * blockDim.x + threadIdx.x; if(m >= M) return;
    double mu = sum_acc[m] / (double)N, var = sq_acc[m] / (double)N - mu * mu;
    mean[m] = (float)mu; rstd[m] = (float)(1.0 / sqrt(var + eps));
}
__global__ void ln_apply(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                         const float* __restrict__ mean, const float* __restrict__ rstd,
                         float* __restrict__ y, long N, long total){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x, stride = (long)gridDim.x * blockDim.x;
    for(; i < total; i += stride){ long m = i / N, col = i - m * N; y[i] = (x[i] - mean[m]) * rstd[m] * w[col] + b[col]; }
}
torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps){
    long N = w.numel(), M = x.numel() / N; int S = 128;
    auto od = x.options().dtype(torch::kFloat64);
    auto sum_acc = torch::zeros({M}, od), sq_acc = torch::zeros({M}, od);
    auto mean = torch::empty({M}, x.options()), rstd = torch::empty({M}, x.options());
    auto y = torch::empty_like(x);
    ln_stats<<<(int)(M * S), TPB>>>(x.data_ptr<float>(), sum_acc.data_ptr<double>(), sq_acc.data_ptr<double>(), N, S);
    ln_final<<<(int)((M + 255) / 256), 256>>>(sum_acc.data_ptr<double>(), sq_acc.data_ptr<double>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), N, eps, (int)M);
    long total = M * N; int t = 256; long wnt = (total + t - 1) / t; int blk = (int)(wnt < 131072 ? wnt : 131072);
    ln_apply<<<blk, t>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), y.data_ptr<float>(), N, total);
    return y;
}
"""
_CPP = "torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps);"
_ext = load_inline(name="layernorm_noptx_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["layernorm_cuda"], verbose=False, extra_cuda_cflags=["-O3"])

class Model(nn.Module):
    """LayerNorm over the last dims via plain CUDA: split-row reduction (double-accumulated
    partial sums) -> finalize mean/rstd -> affine apply. self.ln is a parameter
    container only (weight/bias/eps); it is never called."""
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
    def forward(self, x):
        return _ext.layernorm_cuda(x.contiguous(), self.ln.weight.contiguous(),
                                   self.ln.bias.contiguous(), self.ln.eps)
