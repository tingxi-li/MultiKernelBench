import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

// float4 variant: each thread owns 4 consecutive `inner` columns and keeps 4
// running sums. Warp reads 128 contiguous floats/iter -> fully coalesced.
__global__ void sum_reduce_vec4(const float* __restrict__ x,
                                float* __restrict__ out,
                                long outer, long R, long inner) {
    long q = inner >> 2;                              // float4 columns
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = outer * q;
    if (tid >= total) return;
    long o = tid / q;
    long c = tid - o * q;                             // float4 column index
    const float4* base = reinterpret_cast<const float4*>(x + o * R * inner) + c;
    long stride4 = q;                                 // float4 stride between rows
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll 4
    for (long r = 0; r < R; ++r) {
        float4 v;
        // inline PTX: cache-streaming (evict-first) vectorized load
        asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
                     : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                     : "l"(base + r * stride4));
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    reinterpret_cast<float4*>(out)[tid] = acc;
}

// scalar fallback for inner not divisible by 4
__global__ void sum_reduce_scalar(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  long outer, long R, long inner) {
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = outer * inner;
    if (tid >= total) return;
    long o = tid / inner;
    long i = tid - o * inner;
    const float* base = x + o * R * inner + i;
    float acc = 0.0f;
    #pragma unroll 8
    for (long r = 0; r < R; ++r) acc += base[r * inner];
    out[tid] = acc;
}

torch::Tensor run(torch::Tensor x, long dim) {
    TORCH_CHECK(x.is_cuda(), "x must be cuda");
    auto xc = x.contiguous();
    int64_t ndim = xc.dim();
    if (dim < 0) dim += ndim;
    long R = xc.size(dim);
    long outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= xc.size(d);
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= xc.size(d);

    std::vector<int64_t> osz;
    for (int64_t d = 0; d < ndim; ++d) osz.push_back(d == dim ? 1 : xc.size(d));
    auto out = torch::empty(osz, xc.options());

    int threads = 256;
    if ((inner & 3L) == 0) {
        long total = outer * (inner >> 2);
        long blocks = (total + threads - 1) / threads;
        sum_reduce_vec4<<<blocks, threads>>>(
            xc.data_ptr<float>(), out.data_ptr<float>(), outer, R, inner);
    } else {
        long total = outer * inner;
        long blocks = (total + threads - 1) / threads;
        sum_reduce_scalar<<<blocks, threads>>>(
            xc.data_ptr<float>(), out.data_ptr<float>(), outer, R, inner);
    }
    return out;
}
'''

_CPP = "torch::Tensor run(torch::Tensor x, long dim);"

_ext = load_inline(
    name="sumred_unlim_v3",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["run"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class Model(nn.Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ext.run(x, self.dim)
