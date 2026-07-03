import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#define TPB 256
#define KROWS 1          // rows per L2-resident block. Sweep found a sharp cliff:
                         // K=1 (16MB x + 32MB w/b = 48MB) retains in L2 -> apply reuses x;
                         // K>=2 (>=64MB) evicts before apply -> reuse lost. K=1 wins.
__device__ __forceinline__ void stcs_v4(float4* p, float4 v){
    asm volatile("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(p),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w));
}
// Split-row stats over rows [r0, r0+k): read x once (populates L2 for the apply reuse).
__global__ void ln_stats(const float* __restrict__ x, float* __restrict__ sum_acc,
                         float* __restrict__ sq_acc, long N, int S, int r0){
    int lm = blockIdx.x / S, sc = blockIdx.x % S; int m = r0 + lm;
    long chunk = N / S, start = (long)m * N + (long)sc * chunk;
    const float4* x4 = (const float4*)(x + start);
    long c4 = chunk / 4; float ls = 0.f, lss = 0.f;
    for(long k = threadIdx.x; k < c4; k += blockDim.x){ float4 v = __ldg(x4 + k);
        ls += v.x + v.y + v.z + v.w; lss += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w; }
    __shared__ float ss[TPB], sq[TPB]; int t = threadIdx.x; ss[t] = ls; sq[t] = lss; __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1){ if(t < s){ ss[t] += ss[t+s]; sq[t] += sq[t+s]; } __syncthreads(); }
    if(t == 0){ atomicAdd(sum_acc + m, ss[0]); atomicAdd(sq_acc + m, sq[0]); }
}
// Column-blocked apply over rows [r0, r0+k). Finalizes mean/rstd from the partial sums inline
// (the separate ln_final launch is folded in here -> one fewer kernel per row-block). Each
// thread owns 4 columns (float4), loads w/b ONCE and sweeps the k rows reusing them from
// registers. x for these rows is served from L2 (just filled by ln_stats), so the second x
// read is off the HBM bus. w/b (32MB) also stay L2-resident across all row-blocks. y is
// written via inline-PTX vectorized cache-STREAMING store (y is never reused).
__global__ void ln_apply(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                         const float* __restrict__ sum_acc, const float* __restrict__ sq_acc,
                         float* __restrict__ y, long N, double eps, int r0, int k){
    __shared__ float smean[KROWS], srstd[KROWS];
    if(threadIdx.x < k){ int m = r0 + threadIdx.x;
        float mu = sum_acc[m] / (float)N, var = sq_acc[m] / (float)N - mu * mu;
        smean[threadIdx.x] = mu; srstd[threadIdx.x] = rsqrtf(var + (float)eps); }
    __syncthreads();
    long N4 = N / 4;
    const float4* x4 = (const float4*)x; const float4* w4 = (const float4*)w;
    const float4* b4 = (const float4*)b; float4* y4 = (float4*)y;
    long c4 = (long)blockIdx.x * blockDim.x + threadIdx.x, cst = (long)gridDim.x * blockDim.x;
    for(; c4 < N4; c4 += cst){
        float4 wv = __ldg(w4 + c4), bv = __ldg(b4 + c4);
        for(int lm = 0; lm < k; lm++){
            long i = (long)(r0 + lm) * N4 + c4; float4 xv = __ldg(x4 + i), o;
            float mu = smean[lm], r = srstd[lm];
            o.x = (xv.x - mu) * r * wv.x + bv.x;
            o.y = (xv.y - mu) * r * wv.y + bv.y;
            o.z = (xv.z - mu) * r * wv.z + bv.z;
            o.w = (xv.w - mu) * r * wv.w + bv.w;
            stcs_v4(y4 + i, o);
        }
    }
}
torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps){
    long N = w.numel(), M = x.numel() / N; int S = 128;
    auto sum_acc = torch::zeros({M}, x.options()), sq_acc = torch::zeros({M}, x.options());
    auto y = torch::empty_like(x);
    long N4 = N / 4; int apply_blocks = (int)((N4 + TPB - 1) / TPB);
    const float* xp = x.data_ptr<float>(); const float* wp = w.data_ptr<float>(); const float* bp = b.data_ptr<float>();
    float* sap = sum_acc.data_ptr<float>(); float* qap = sq_acc.data_ptr<float>(); float* yp = y.data_ptr<float>();
    int K = KROWS;
    for(int r0 = 0; r0 < M; r0 += K){
        int k = (int)M - r0; if(k > K) k = K;
        ln_stats<<<k * S, TPB>>>(xp, sap, qap, N, S, r0);
        ln_apply<<<apply_blocks, TPB>>>(xp, wp, bp, sap, qap, yp, N, eps, r0, k);
    }
    return y;
}
"""
_CPP = "torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps);"
_ext = load_inline(name="layernorm_unlimited_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["layernorm_cuda"], verbose=False, extra_cuda_cflags=["-O3"])

class Model(nn.Module):
    """LayerNorm over the last dims via CUDA float4 + inline-PTX streaming store, processed in
    L2-resident single-row blocks so the apply pass re-reads x from L2 instead of HBM (cutting
    HBM traffic ~3GB->2GB): split-row reduction -> affine apply (mean/rstd finalized inline),
    per row. self.ln is a parameter container only (weight/bias/eps); it is never called."""
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
    def forward(self, x):
        return _ext.layernorm_cuda(x.contiguous(), self.ln.weight.contiguous(),
                                   self.ln.bias.contiguous(), self.ln.eps)
