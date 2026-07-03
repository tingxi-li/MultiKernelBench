import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA UNLIMITED — inline PTX vectorized (float4) load/store, grid-stride.
# Demonstrates the no-holds-barred track: ld.global.v4.f32 / st.global.v4.f32
# inline asm move 16 bytes/instruction. ReLU is HBM-roofline so this lands at
# the same ~1x as the plain track (documented), but the PTX path is real.
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float4 ldg_v4(const float* p){
    float4 v;
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ void st_v4(float* p, float4 v){
    asm volatile("st.global.v4.f32 [%4], {%0,%1,%2,%3};"
                 :: "f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"l"(p));
}
__device__ __forceinline__ float reluf(float a){ return fmaxf(a, 0.0f); }

__global__ void relu_v4(const float* __restrict__ x, float* __restrict__ y, long n4){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for(; i < n4; i += stride){
        float4 v = ldg_v4(x + i*4);
        v.x = reluf(v.x); v.y = reluf(v.y); v.z = reluf(v.z); v.w = reluf(v.w);
        st_v4(y + i*4, v);
    }
}
__global__ void relu_tail(const float* __restrict__ x, float* __restrict__ y, long s, long n){
    long i = s + (long)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) y[i] = fmaxf(x[i], 0.0f);
}

torch::Tensor relu_cuda(torch::Tensor x){
    auto y = torch::empty_like(x);
    long n = x.numel();
    long n4 = n / 4;
    int threads = 256;
    long want = (n4 + threads - 1) / threads;
    int blocks = (int)(want < 131072 ? want : 131072);
    if(n4 > 0) relu_v4<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n4);
    long s = n4 * 4;
    if(s < n) relu_tail<<<1, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), s, n);
    return y;
}
"""
_CPP = "torch::Tensor relu_cuda(torch::Tensor x);"
_ext = load_inline(name="relu_cuda_unlimited_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["relu_cuda"], verbose=False, extra_cuda_cflags=["-O3"])

class Model(nn.Module):
    """ReLU via CUDA with inline-PTX float4 vectorized memory ops."""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return _ext.relu_cuda(x.contiguous())
