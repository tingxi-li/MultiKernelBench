import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 6: 2 float4 per thread (8 K-elements each thread) + ld.cs
# Back to basics: single accumulator per float4, but each thread handles 8 K-elements
# (2 float4 loads per row instead of 1). This halves grid size, reducing
# scheduling overhead and potentially improving L2 re-use of neighboring cache lines
# (2 consecutive float4 = 32 bytes = one full cache line).

cuda_src = r"""
#include <cuda_runtime.h>

template<int BLOCK>
__global__ __launch_bounds__(BLOCK, 8)
void sum_reduce_dim1_2f4(
    const float* __restrict__ x,   // [N, M, K]
    float*       __restrict__ out, // [N, K]
    int N, int M, int K)
{
    // Each thread handles 8 consecutive K-elements (two float4)
    const int tid = blockIdx.x * BLOCK + threadIdx.x;
    const int k8  = tid * 8;
    const int n   = blockIdx.y;

    if (k8 >= K) return;

    float4 acc0 = {0.f, 0.f, 0.f, 0.f};
    float4 acc1 = {0.f, 0.f, 0.f, 0.f};

    const float* base = x + (long)n * M * K + k8;
    const long stride = K;

    for (int i = 0; i < M; i++) {
        float4 v0, v1;
        const float* row = base + (long)i * stride;
        asm volatile("ld.cs.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v0.x),"=f"(v0.y),"=f"(v0.z),"=f"(v0.w)
            : "l"(row));
        asm volatile("ld.cs.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v1.x),"=f"(v1.y),"=f"(v1.z),"=f"(v1.w)
            : "l"(row + 4));
        acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
        acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
    }

    // Write 8 floats (two float4) to output
    float4* dst = reinterpret_cast<float4*>(out + (long)n * K + k8);
    dst[0] = acc0;
    dst[1] = acc1;
}

torch::Tensor sum_reduce_cuda(torch::Tensor x, int dim)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32);
    TORCH_CHECK(dim == 1 && x.dim() == 3);

    const int N = x.size(0), M = x.size(1), K = x.size(2);
    TORCH_CHECK(K % 8 == 0);

    auto out = torch::empty({N, 1, K}, x.options());

    constexpr int BLOCK = 256;
    const int grid_x = (K / 8 + BLOCK - 1) / BLOCK;
    dim3 grid(grid_x, N);

    sum_reduce_dim1_2f4<BLOCK><<<grid, BLOCK>>>(
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
    name="sum_reduce_2f4_v6",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sum_reduce_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    """
    Optimized sum reduction: 2 float4 per thread (8 K-elements each),
    single accumulator chain, ld.cs streaming loads.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _mod.sum_reduce_cuda(x, self.dim)
