import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 2: Two float4 accumulators at consecutive C positions per thread.
# Each thread handles 2 float4s per D-step (8 floats = 32 bytes per D-step).
# This doubles work per thread, halves grid size, and maintains coalescing.
# 2 independent acc chains for ILP.

_cuda_src = r"""
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

__global__ void sum_reduce_dim1_v2(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int B,
    const int D,
    const int C4
) {
    // Each thread processes 2 consecutive float4 positions
    const int c4_base = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int b       = blockIdx.y;

    if (c4_base + 1 >= C4 || b >= B) return;

    const float4* __restrict__ in4 =
        reinterpret_cast<const float4*>(x + (size_t)b * D * C4 * 4);

    float4 acc0 = {0.f, 0.f, 0.f, 0.f};
    float4 acc1 = {0.f, 0.f, 0.f, 0.f};

    for (int i = 0; i < D; ++i) {
        const float4* row = in4 + (size_t)i * C4;
        float4 v0 = __ldg(row + c4_base    );
        float4 v1 = __ldg(row + c4_base + 1);
        acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
        acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
    }

    float4* out4 = reinterpret_cast<float4*>(out + (size_t)b * C4 * 4);
    out4[c4_base    ] = acc0;
    out4[c4_base + 1] = acc1;
}

torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Expected 3D tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Expected float32");
    TORCH_CHECK(x.is_contiguous(), "Expected contiguous tensor");

    const int B = x.size(0);
    const int D = x.size(1);
    const int C = x.size(2);
    TORCH_CHECK(C % 4 == 0, "C must be divisible by 4");

    auto out = torch::empty({B, 1, C}, x.options());

    const int C4 = C / 4;
    const int half_C4 = C4 / 2;  // threads per B row
    constexpr int BLOCK_X = 256;
    dim3 block(BLOCK_X);
    dim3 grid((half_C4 + BLOCK_X - 1) / BLOCK_X, B);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sum_reduce_dim1_v2<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), B, D, C4
    );

    return out;
}
"""

_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x);
"""

_module = load_inline(
    name="sum_reduce_dim1_v2c",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["sum_reduce_dim1_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class Model(nn.Module):
    """
    Sum reduction over a specified dimension, optimized for dim=1 of 3D float32.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.dim == 1
                and x.ndim == 3
                and x.dtype == torch.float32
                and x.is_contiguous()
                and x.size(2) % 4 == 0):
            return _module.sum_reduce_dim1_cuda(x)
        return torch.sum(x, dim=self.dim, keepdim=True)
