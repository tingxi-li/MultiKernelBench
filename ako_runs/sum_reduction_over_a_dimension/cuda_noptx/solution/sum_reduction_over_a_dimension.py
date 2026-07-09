import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 5: 4 independent accumulators for the same C4 position.
# Each accumulator sums D/4=1024 rows -> 4x shorter dependency chain.
# Combined at the end. This allows better overlapping of load latency
# since the 4 dependency chains are independent.
# CRITICAL: reading positions c4_base, c4_base+0, same position 4 times.
# Different D-ranges = completely different cache lines -> no conflict.

_cuda_src = r"""
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

__global__ void sum_reduce_v5(
    const float4* __restrict__ in4,
    float4* __restrict__ out4,
    const int D,
    const int C4
) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int b  = blockIdx.y;

    if (c4 >= C4) return;

    // 4 segments of D
    const int seg = D / 4;
    const float4* base = in4 + (size_t)b * D * C4 + c4;

    // 4 independent pointers, 4 independent accumulators
    const float4* p0 = base;
    const float4* p1 = base + (size_t)(seg  ) * C4;
    const float4* p2 = base + (size_t)(seg*2) * C4;
    const float4* p3 = base + (size_t)(seg*3) * C4;

    float4 a0 = {0.f,0.f,0.f,0.f};
    float4 a1 = {0.f,0.f,0.f,0.f};
    float4 a2 = {0.f,0.f,0.f,0.f};
    float4 a3 = {0.f,0.f,0.f,0.f};

    for (int i = 0; i < seg; ++i) {
        float4 v0 = __ldg(p0); p0 += C4;
        float4 v1 = __ldg(p1); p1 += C4;
        float4 v2 = __ldg(p2); p2 += C4;
        float4 v3 = __ldg(p3); p3 += C4;
        a0.x += v0.x; a0.y += v0.y; a0.z += v0.z; a0.w += v0.w;
        a1.x += v1.x; a1.y += v1.y; a1.z += v1.z; a1.w += v1.w;
        a2.x += v2.x; a2.y += v2.y; a2.z += v2.z; a2.w += v2.w;
        a3.x += v3.x; a3.y += v3.y; a3.z += v3.z; a3.w += v3.w;
    }

    float4 acc;
    acc.x = a0.x + a1.x + a2.x + a3.x;
    acc.y = a0.y + a1.y + a2.y + a3.y;
    acc.z = a0.z + a1.z + a2.z + a3.z;
    acc.w = a0.w + a1.w + a2.w + a3.w;

    out4[(size_t)b * C4 + c4] = acc;
}

torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Expected 3D tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Expected float32");
    TORCH_CHECK(x.is_contiguous(), "Expected contiguous tensor");

    const int B = x.size(0);
    const int D = x.size(1);
    const int C = x.size(2);
    TORCH_CHECK(C % 4 == 0, "C must be divisible by 4");
    TORCH_CHECK(D % 4 == 0, "D must be divisible by 4");

    auto out = torch::empty({B, 1, C}, x.options());

    const int C4 = C / 4;
    constexpr int BLOCK_X = 256;
    dim3 block(BLOCK_X);
    dim3 grid((C4 + BLOCK_X - 1) / BLOCK_X, B);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sum_reduce_v5<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        D, C4
    );

    return out;
}
"""

_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x);
"""

_module = load_inline(
    name="sum_reduce_v5_4seg",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["sum_reduce_dim1_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.dim == 1
                and x.ndim == 3
                and x.dtype == torch.float32
                and x.is_contiguous()
                and x.size(2) % 4 == 0
                and x.size(1) % 4 == 0):
            return _module.sum_reduce_dim1_cuda(x)
        return torch.sum(x, dim=self.dim, keepdim=True)
