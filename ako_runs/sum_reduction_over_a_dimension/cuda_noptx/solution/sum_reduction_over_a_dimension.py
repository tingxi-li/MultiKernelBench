import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized kernel for sum reduction along dim=1 of a (B, D, C) float32 tensor.
# Strategy: float4 loads (4 floats/transaction) + 8-step unroll for ILP.
# Access pattern: each warp handles 32 consecutive float4 groups -> coalesced 512B/load.

_cuda_src = r"""
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// Reduce (B, D, C) along dim1 to (B, 1, C) using float4 + 8-unroll.
// Each thread processes 4 consecutive C-elements (one float4).
// Grid: (C/4 / BLOCK_X, B); Block: (BLOCK_X,)
__global__ void sum_reduce_dim1_f4(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int B,
    const int D,
    const int C
) {
    const int c4     = blockIdx.x * blockDim.x + threadIdx.x;
    const int b      = blockIdx.y;
    const int C4     = C >> 2;          // C / 4

    if (c4 >= C4 || b >= B) return;

    // Pointer to start of this batch element (as float4*)
    const float4* __restrict__ in4 =
        reinterpret_cast<const float4*>(x + (size_t)b * D * C);
    const int stride4 = C4;             // row stride in float4 units

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

    int i = 0;
    // 8-step unroll: issue 8 independent loads so the memory pipeline stays full
    for (; i + 7 < D; i += 8) {
        float4 v0 = __ldg(in4 + (i  ) * stride4 + c4);
        float4 v1 = __ldg(in4 + (i+1) * stride4 + c4);
        float4 v2 = __ldg(in4 + (i+2) * stride4 + c4);
        float4 v3 = __ldg(in4 + (i+3) * stride4 + c4);
        float4 v4 = __ldg(in4 + (i+4) * stride4 + c4);
        float4 v5 = __ldg(in4 + (i+5) * stride4 + c4);
        float4 v6 = __ldg(in4 + (i+6) * stride4 + c4);
        float4 v7 = __ldg(in4 + (i+7) * stride4 + c4);
        acc.x += v0.x + v1.x + v2.x + v3.x + v4.x + v5.x + v6.x + v7.x;
        acc.y += v0.y + v1.y + v2.y + v3.y + v4.y + v5.y + v6.y + v7.y;
        acc.z += v0.z + v1.z + v2.z + v3.z + v4.z + v5.z + v6.z + v7.z;
        acc.w += v0.w + v1.w + v2.w + v3.w + v4.w + v5.w + v6.w + v7.w;
    }
    // Remainder (D not a multiple of 8)
    for (; i < D; i++) {
        float4 v = __ldg(in4 + i * stride4 + c4);
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }

    // Write to out shaped (B, 1, C); memory-equivalent to (B, C) contiguous
    float4* out4 = reinterpret_cast<float4*>(out + (size_t)b * C);
    out4[c4] = acc;
}

torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Expected 3D tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Expected float32");
    TORCH_CHECK(x.is_contiguous(), "Expected contiguous tensor");

    const int B = x.size(0);
    const int D = x.size(1);
    const int C = x.size(2);
    TORCH_CHECK(C % 4 == 0, "C must be divisible by 4 for float4 loads");

    auto out = torch::empty({B, 1, C}, x.options());

    constexpr int BLOCK_X = 256;
    const int C4 = C / 4;
    dim3 block(BLOCK_X);
    dim3 grid((C4 + BLOCK_X - 1) / BLOCK_X, B);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sum_reduce_dim1_f4<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), B, D, C
    );

    return out;
}
"""

_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x);
"""

_module = load_inline(
    name="sum_reduce_dim1_v1",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["sum_reduce_dim1_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)


class Model(nn.Module):
    """
    Sum reduction over a specified dimension, optimized for dim=1 of 3D float32.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path: dim=1, 3D, float32, contiguous, C divisible by 4
        if (self.dim == 1
                and x.ndim == 3
                and x.dtype == torch.float32
                and x.is_contiguous()
                and x.size(2) % 4 == 0):
            return _module.sum_reduce_dim1_cuda(x)
        # Fallback
        return torch.sum(x, dim=self.dim, keepdim=True)
