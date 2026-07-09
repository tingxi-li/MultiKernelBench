import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 4: ld.lu (last-use, aggressive eviction) + st.wt (write-through output)
#         + 2-accumulator ILP for moderate register pressure
# Input: (N=128, M=4096, K=4096), reduce dim=1, output: (N,1,K)
# Key idea: ld.lu frees cache lines as soon as loaded (they're never reused);
#           st.wt bypasses L2 for the output (output is write-only here),
#           preserving L2 capacity for input reads.

cuda_src = r"""
#include <cuda_runtime.h>

template<int BLOCK>
__global__ __launch_bounds__(BLOCK, 6)
void sum_reduce_dim1_lu_wt(
    const float* __restrict__ x,   // [N, M, K]
    float*       __restrict__ out, // [N, K]
    int N, int M, int K)
{
    const int tid = blockIdx.x * BLOCK + threadIdx.x;
    const int k4  = tid * 4;
    const int n   = blockIdx.y;

    if (k4 >= K) return;

    // Two independent accumulator chains for light ILP
    float4 acc0 = {0.f, 0.f, 0.f, 0.f};
    float4 acc1 = {0.f, 0.f, 0.f, 0.f};

    const float* base = x + (long)n * M * K + k4;
    const long stride = K;

    // M=4096 divisible by 2; unroll by 2 with independent accumulators
    for (int i = 0; i < M; i += 2) {
        float4 v0, v1;
        // ld.lu: "last use" hint — evict cache line immediately after load.
        // Ideal for streaming patterns with no temporal reuse.
        asm volatile("ld.lu.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v0.x),"=f"(v0.y),"=f"(v0.z),"=f"(v0.w)
            : "l"(base + (i+0)*stride));
        asm volatile("ld.lu.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v1.x),"=f"(v1.y),"=f"(v1.z),"=f"(v1.w)
            : "l"(base + (i+1)*stride));
        acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
        acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
    }

    // Merge
    acc0.x += acc1.x; acc0.y += acc1.y;
    acc0.z += acc1.z; acc0.w += acc1.w;

    // st.wt: write-through. Output is never read in this pass; skip L2 for writes.
    asm volatile("st.wt.v4.f32 [%0],{%1,%2,%3,%4};"
        :
        : "l"(out + (long)n * K + k4),
          "f"(acc0.x),"f"(acc0.y),"f"(acc0.z),"f"(acc0.w));
}

torch::Tensor sum_reduce_cuda(torch::Tensor x, int dim)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32);
    TORCH_CHECK(dim == 1 && x.dim() == 3);

    const int N = x.size(0), M = x.size(1), K = x.size(2);
    TORCH_CHECK(K % 4 == 0 && M % 2 == 0);

    auto out = torch::empty({N, 1, K}, x.options());

    constexpr int BLOCK = 256;
    const int grid_x = (K / 4 + BLOCK - 1) / BLOCK;
    dim3 grid(grid_x, N);

    sum_reduce_dim1_lu_wt<BLOCK><<<grid, BLOCK>>>(
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
    name="sum_reduce_lu_wt_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sum_reduce_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    """
    Optimized sum reduction: ld.lu (last-use eviction) + st.wt (write-through)
    to minimize L2 cache pollution from streaming reads and writes.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _mod.sum_reduce_cuda(x, self.dim)
