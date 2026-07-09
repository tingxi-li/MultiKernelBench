import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 6: Best approach (from iter-4): single float4/thread, ptr-walk,
# minimal register pressure. Confirmed optimal for this bandwidth-bound op.

_cuda_src = r"""
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

__launch_bounds__(256)
__global__ void sum_reduce_final(
    const float4* __restrict__ in4,
    float4* __restrict__ out4,
    const int D,
    const int C4
) {
    const int c4 = blockIdx.x * 256 + threadIdx.x;
    const int b  = blockIdx.y;

    if (c4 >= C4) return;

    const float4* __restrict__ ptr = in4 + (size_t)b * D * C4 + c4;
    const float4* __restrict__ end = ptr + (size_t)D * C4;

    float4 acc = {0.f, 0.f, 0.f, 0.f};

    for (; ptr < end; ptr += C4) {
        float4 v = __ldg(ptr);
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }

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

    auto out = torch::empty({B, 1, C}, x.options());

    const int C4 = C / 4;
    dim3 block(256);
    dim3 grid((C4 + 255) / 256, B);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sum_reduce_final<<<grid, block, 0, stream>>>(
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
    name="sum_reduce_final_v6",
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
                and x.size(2) % 4 == 0):
            return _module.sum_reduce_dim1_cuda(x)
        return torch.sum(x, dim=self.dim, keepdim=True)
