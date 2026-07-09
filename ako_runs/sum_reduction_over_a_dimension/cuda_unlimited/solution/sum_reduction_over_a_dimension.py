import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 2: float4 + 4-way ILP via independent accumulators
# Input: (N=128, M=4096, K=4096), reduce dim=1, output: (N,1,K)
# 4 independent load-add chains expose ILP to the hardware scheduler,
# allowing 4 in-flight DRAM requests per thread and reducing stall cycles.

cuda_src = r"""
#include <cuda_runtime.h>

template<int BLOCK>
__global__ __launch_bounds__(BLOCK, 4)
void sum_reduce_dim1_f4_ilp4(
    const float* __restrict__ x,   // [N, M, K]
    float*       __restrict__ out, // [N, K]
    int N, int M, int K)
{
    // Each thread handles 4 consecutive K-elements via float4
    const int tid  = blockIdx.x * BLOCK + threadIdx.x;
    const int k4   = tid * 4;
    const int n    = blockIdx.y;

    if (k4 >= K) return;

    // 4 independent accumulator chains for ILP
    float4 acc0 = {0.f, 0.f, 0.f, 0.f};
    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
    float4 acc3 = {0.f, 0.f, 0.f, 0.f};

    const float* base = x + (long)n * M * K + k4;
    const long stride = K;

    // M must be divisible by 4 (4096 % 4 == 0)
    for (int i = 0; i < M; i += 4) {
        float4 v0, v1, v2, v3;
        // Issue 4 independent streaming loads — hardware can execute concurrently
        asm volatile("ld.cs.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v0.x),"=f"(v0.y),"=f"(v0.z),"=f"(v0.w)
            : "l"(base + (i+0)*stride));
        asm volatile("ld.cs.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v1.x),"=f"(v1.y),"=f"(v1.z),"=f"(v1.w)
            : "l"(base + (i+1)*stride));
        asm volatile("ld.cs.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v2.x),"=f"(v2.y),"=f"(v2.z),"=f"(v2.w)
            : "l"(base + (i+2)*stride));
        asm volatile("ld.cs.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v3.x),"=f"(v3.y),"=f"(v3.z),"=f"(v3.w)
            : "l"(base + (i+3)*stride));
        // Independent add chains (no dep between acc0, acc1, acc2, acc3)
        acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
        acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
        acc2.x += v2.x; acc2.y += v2.y; acc2.z += v2.z; acc2.w += v2.w;
        acc3.x += v3.x; acc3.y += v3.y; acc3.z += v3.z; acc3.w += v3.w;
    }

    // Final horizontal reduction of 4 accumulators
    float4 result;
    result.x = acc0.x + acc1.x + acc2.x + acc3.x;
    result.y = acc0.y + acc1.y + acc2.y + acc3.y;
    result.z = acc0.z + acc1.z + acc2.z + acc3.z;
    result.w = acc0.w + acc1.w + acc2.w + acc3.w;

    float4* dst = reinterpret_cast<float4*>(out + (long)n * K + k4);
    *dst = result;
}

torch::Tensor sum_reduce_cuda(torch::Tensor x, int dim)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32);
    TORCH_CHECK(dim == 1 && x.dim() == 3);

    const int N = x.size(0), M = x.size(1), K = x.size(2);
    TORCH_CHECK(K % 4 == 0 && M % 4 == 0);

    auto out = torch::empty({N, 1, K}, x.options());

    constexpr int BLOCK = 256;
    const int grid_x = (K / 4 + BLOCK - 1) / BLOCK;
    dim3 grid(grid_x, N);

    sum_reduce_dim1_f4_ilp4<BLOCK><<<grid, BLOCK>>>(
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
    name="sum_reduce_f4_ilp4_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sum_reduce_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    """
    Optimized sum reduction over dim=1 using float4 loads with 4-way ILP
    (independent accumulator chains to hide DRAM latency).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _mod.sum_reduce_cuda(x, self.dim)
