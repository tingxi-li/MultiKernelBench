import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 3: Sweep of block sizes to maximize occupancy on RTX 6000 Ada.
# Using BLOCK_X=128 (4 warps), and 2 float4 outputs per thread (from iter-2).
# Plus 8-step loop unroll for ILP.
# Also using __ldg for read-only cache hint.

_cuda_src = r"""
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

__global__ void sum_reduce_dim1_v3(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int B,
    const int D,
    const int C4
) {
    const int c4_base = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int b       = blockIdx.y;

    if (c4_base + 1 >= C4 || b >= B) return;

    const float4* __restrict__ in4 =
        reinterpret_cast<const float4*>(x + (size_t)b * (size_t)D * C4 * 4);

    float4 acc0 = {0.f, 0.f, 0.f, 0.f};
    float4 acc1 = {0.f, 0.f, 0.f, 0.f};

    int i = 0;
    // 8-unroll: 8 rows loaded before dependent adds
    for (; i + 7 < D; i += 8) {
        float4 v00 = __ldg(in4 + (size_t)(i  ) * C4 + c4_base);
        float4 v01 = __ldg(in4 + (size_t)(i  ) * C4 + c4_base + 1);
        float4 v10 = __ldg(in4 + (size_t)(i+1) * C4 + c4_base);
        float4 v11 = __ldg(in4 + (size_t)(i+1) * C4 + c4_base + 1);
        float4 v20 = __ldg(in4 + (size_t)(i+2) * C4 + c4_base);
        float4 v21 = __ldg(in4 + (size_t)(i+2) * C4 + c4_base + 1);
        float4 v30 = __ldg(in4 + (size_t)(i+3) * C4 + c4_base);
        float4 v31 = __ldg(in4 + (size_t)(i+3) * C4 + c4_base + 1);
        float4 v40 = __ldg(in4 + (size_t)(i+4) * C4 + c4_base);
        float4 v41 = __ldg(in4 + (size_t)(i+4) * C4 + c4_base + 1);
        float4 v50 = __ldg(in4 + (size_t)(i+5) * C4 + c4_base);
        float4 v51 = __ldg(in4 + (size_t)(i+5) * C4 + c4_base + 1);
        float4 v60 = __ldg(in4 + (size_t)(i+6) * C4 + c4_base);
        float4 v61 = __ldg(in4 + (size_t)(i+6) * C4 + c4_base + 1);
        float4 v70 = __ldg(in4 + (size_t)(i+7) * C4 + c4_base);
        float4 v71 = __ldg(in4 + (size_t)(i+7) * C4 + c4_base + 1);
        acc0.x += v00.x+v10.x+v20.x+v30.x+v40.x+v50.x+v60.x+v70.x;
        acc0.y += v00.y+v10.y+v20.y+v30.y+v40.y+v50.y+v60.y+v70.y;
        acc0.z += v00.z+v10.z+v20.z+v30.z+v40.z+v50.z+v60.z+v70.z;
        acc0.w += v00.w+v10.w+v20.w+v30.w+v40.w+v50.w+v60.w+v70.w;
        acc1.x += v01.x+v11.x+v21.x+v31.x+v41.x+v51.x+v61.x+v71.x;
        acc1.y += v01.y+v11.y+v21.y+v31.y+v41.y+v51.y+v61.y+v71.y;
        acc1.z += v01.z+v11.z+v21.z+v31.z+v41.z+v51.z+v61.z+v71.z;
        acc1.w += v01.w+v11.w+v21.w+v31.w+v41.w+v51.w+v61.w+v71.w;
    }
    for (; i < D; ++i) {
        float4 v0 = __ldg(in4 + (size_t)i * C4 + c4_base);
        float4 v1 = __ldg(in4 + (size_t)i * C4 + c4_base + 1);
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
    const int half_C4 = C4 / 2;
    constexpr int BLOCK_X = 128;
    dim3 block(BLOCK_X);
    dim3 grid((half_C4 + BLOCK_X - 1) / BLOCK_X, B);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sum_reduce_dim1_v3<<<grid, block, 0, stream>>>(
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
    name="sum_reduce_dim1_v3",
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
