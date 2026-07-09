import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 3: 8-way ILP + ld.cg (cache at L2, bypass L1) + larger blocks
# Input: (N=128, M=4096, K=4096), reduce dim=1, output: (N,1,K)
# 8 independent load chains = 8 in-flight DRAM requests per thread
# ld.cg bypasses L1 (L1 hit rate is near 0% for M=4096 stride-K accesses anyway)

cuda_src = r"""
#include <cuda_runtime.h>

// 8-way ILP: accumulate 8 independent float4s in parallel
// ld.cg: cache in L2, bypass L1 (better for large-stride streaming access)
template<int BLOCK>
__global__ __launch_bounds__(BLOCK, 4)
void sum_reduce_dim1_f4_ilp8(
    const float* __restrict__ x,   // [N, M, K]
    float*       __restrict__ out, // [N, K]
    int N, int M, int K)
{
    const int tid = blockIdx.x * BLOCK + threadIdx.x;
    const int k4  = tid * 4;
    const int n   = blockIdx.y;

    if (k4 >= K) return;

    float4 acc0 = {0.f,0.f,0.f,0.f}, acc1 = {0.f,0.f,0.f,0.f};
    float4 acc2 = {0.f,0.f,0.f,0.f}, acc3 = {0.f,0.f,0.f,0.f};
    float4 acc4 = {0.f,0.f,0.f,0.f}, acc5 = {0.f,0.f,0.f,0.f};
    float4 acc6 = {0.f,0.f,0.f,0.f}, acc7 = {0.f,0.f,0.f,0.f};

    const float* base = x + (long)n * M * K + k4;
    const long stride = K;

    // M=4096 divisible by 8
    for (int i = 0; i < M; i += 8) {
        float4 v0,v1,v2,v3,v4,v5,v6,v7;
        // ld.cg: bypass L1, go directly to L2. For stride-K access (16KB apart),
        // L1 provides no benefit and wastes fill bandwidth.
        asm volatile("ld.cg.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v0.x),"=f"(v0.y),"=f"(v0.z),"=f"(v0.w)
            : "l"(base + (i+0)*stride));
        asm volatile("ld.cg.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v1.x),"=f"(v1.y),"=f"(v1.z),"=f"(v1.w)
            : "l"(base + (i+1)*stride));
        asm volatile("ld.cg.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v2.x),"=f"(v2.y),"=f"(v2.z),"=f"(v2.w)
            : "l"(base + (i+2)*stride));
        asm volatile("ld.cg.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v3.x),"=f"(v3.y),"=f"(v3.z),"=f"(v3.w)
            : "l"(base + (i+3)*stride));
        asm volatile("ld.cg.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v4.x),"=f"(v4.y),"=f"(v4.z),"=f"(v4.w)
            : "l"(base + (i+4)*stride));
        asm volatile("ld.cg.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v5.x),"=f"(v5.y),"=f"(v5.z),"=f"(v5.w)
            : "l"(base + (i+5)*stride));
        asm volatile("ld.cg.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v6.x),"=f"(v6.y),"=f"(v6.z),"=f"(v6.w)
            : "l"(base + (i+6)*stride));
        asm volatile("ld.cg.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v7.x),"=f"(v7.y),"=f"(v7.z),"=f"(v7.w)
            : "l"(base + (i+7)*stride));

        acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
        acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
        acc2.x += v2.x; acc2.y += v2.y; acc2.z += v2.z; acc2.w += v2.w;
        acc3.x += v3.x; acc3.y += v3.y; acc3.z += v3.z; acc3.w += v3.w;
        acc4.x += v4.x; acc4.y += v4.y; acc4.z += v4.z; acc4.w += v4.w;
        acc5.x += v5.x; acc5.y += v5.y; acc5.z += v5.z; acc5.w += v5.w;
        acc6.x += v6.x; acc6.y += v6.y; acc6.z += v6.z; acc6.w += v6.w;
        acc7.x += v7.x; acc7.y += v7.y; acc7.z += v7.z; acc7.w += v7.w;
    }

    // Tree-reduce the 8 accumulators
    acc0.x += acc1.x + acc2.x + acc3.x + acc4.x + acc5.x + acc6.x + acc7.x;
    acc0.y += acc1.y + acc2.y + acc3.y + acc4.y + acc5.y + acc6.y + acc7.y;
    acc0.z += acc1.z + acc2.z + acc3.z + acc4.z + acc5.z + acc6.z + acc7.z;
    acc0.w += acc1.w + acc2.w + acc3.w + acc4.w + acc5.w + acc6.w + acc7.w;

    float4* dst = reinterpret_cast<float4*>(out + (long)n * K + k4);
    *dst = acc0;
}

torch::Tensor sum_reduce_cuda(torch::Tensor x, int dim)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32);
    TORCH_CHECK(dim == 1 && x.dim() == 3);

    const int N = x.size(0), M = x.size(1), K = x.size(2);
    TORCH_CHECK(K % 4 == 0 && M % 8 == 0);

    auto out = torch::empty({N, 1, K}, x.options());

    // Use BLOCK=128 to allow more blocks per SM (smaller register file per block)
    constexpr int BLOCK = 128;
    const int grid_x = (K / 4 + BLOCK - 1) / BLOCK;
    dim3 grid(grid_x, N);

    sum_reduce_dim1_f4_ilp8<BLOCK><<<grid, BLOCK>>>(
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
    name="sum_reduce_f4_ilp8_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sum_reduce_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    """
    Optimized sum reduction over dim=1 using float4 loads with 8-way ILP
    and L1-bypassing ld.cg for maximum bandwidth utilization.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _mod.sum_reduce_cuda(x, self.dim)
