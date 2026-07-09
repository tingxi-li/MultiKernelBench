import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 1: float4 vectorized streaming loads with PTX ld.cs.v4.f32
# Input: (N, M, K), reduce dim=1 (M), output: (N, 1, K)
# Strategy: 1 thread per 4 K-elements, streaming loads to avoid L2 pollution

cuda_src = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<int BLOCK>
__global__ void sum_reduce_dim1_f4(
    const float* __restrict__ x,   // [N, M, K]
    float*       __restrict__ out, // [N, K]  (will be viewed as [N,1,K])
    int N, int M, int K)
{
    // Each thread handles 4 consecutive K-elements (float4 wide)
    int tid = blockIdx.x * BLOCK + threadIdx.x;
    int k4  = tid * 4;
    int n   = blockIdx.y;

    if (k4 >= K) return;

    float4 acc = {0.f, 0.f, 0.f, 0.f};

    // base pointer to x[n, 0, k4]
    const float* base = x + (long)n * M * K + k4;

    // Unroll 8 to help the compiler pipeline loads and hide memory latency
    int i = 0;
    #pragma unroll 8
    for (; i < M; ++i) {
        float4 v;
        // ld.cs: cache streaming (evict-first policy), optimal for scan patterns
        asm volatile(
            "ld.cs.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w)
            : "l"(base + (long)i * K));
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }

    // Write result: out[n, 0, k4..k4+3]
    float4* dst = reinterpret_cast<float4*>(out + (long)n * K + k4);
    *dst = acc;
}

torch::Tensor sum_reduce_cuda(torch::Tensor x, int dim)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "need float32 CUDA tensor");
    TORCH_CHECK(dim == 1, "Only dim=1 supported in this kernel");
    TORCH_CHECK(x.dim() == 3, "Expected 3D tensor");

    int N = x.size(0), M = x.size(1), K = x.size(2);
    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for float4 loads");

    // Output: (N, 1, K) with keepdim
    auto out = torch::empty({N, 1, K}, x.options());

    const int BLOCK = 256;
    // grid_x covers K/4 outputs per batch item
    int grid_x = (K / 4 + BLOCK - 1) / BLOCK;
    dim3 grid(grid_x, N);

    sum_reduce_dim1_f4<BLOCK><<<grid, BLOCK>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, M, K);

    return out;
}
"""

cpp_src = """
torch::Tensor sum_reduce_cuda(torch::Tensor x, int dim);
"""

_mod = load_inline(
    name="sum_reduce_f4_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sum_reduce_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    """
    Optimized sum reduction over a specified dimension using float4
    vectorized streaming PTX loads.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _mod.sum_reduce_cuda(x, self.dim)
