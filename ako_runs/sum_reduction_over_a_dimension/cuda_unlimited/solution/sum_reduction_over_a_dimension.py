import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 5: Warp-cooperative reduction along M dimension
# Each warp of 32 threads handles ONE float4 output (4 K-values),
# with each thread summing M/32 = 128 rows, then warp-shuffle reduces.
# This multiplies the number of outstanding loads by 32x per output element,
# significantly improving memory-level parallelism for the DRAM scheduler.

cuda_src = r"""
#include <cuda_runtime.h>

// One warp (32 threads) computes one float4 output (4 K-elements)
// Thread t sums rows [t, t+32, t+64, ..., t+4064] (= 128 rows)
// Then horizontal warp-reduce via shuffle
template<int BLOCK>
__global__ __launch_bounds__(BLOCK, 4)
void sum_reduce_warp_coop(
    const float* __restrict__ x,   // [N, M, K]
    float*       __restrict__ out, // [N, K]
    int N, int M, int K)
{
    // 1 warp = 32 threads handles 4 K-elements (one float4 output)
    const int warp_id = (blockIdx.x * BLOCK + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    const int n       = blockIdx.y;

    // Each warp maps to one float4 in the output
    const int k4 = warp_id * 4;
    if (k4 >= K) return;

    float4 acc = {0.f, 0.f, 0.f, 0.f};

    const float* base = x + (long)n * M * K + k4;
    const long stride = K;

    // Each lane sums M/32 rows with stride-32 stepping
    // M=4096, so each lane handles 128 rows
    for (int i = lane; i < M; i += 32) {
        float4 v;
        asm volatile("ld.cs.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w)
            : "l"(base + (long)i * stride));
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }

    // Warp-level reduction via shuffle down
    #pragma unroll
    for (int delta = 16; delta >= 1; delta >>= 1) {
        acc.x += __shfl_down_sync(0xffffffff, acc.x, delta);
        acc.y += __shfl_down_sync(0xffffffff, acc.y, delta);
        acc.z += __shfl_down_sync(0xffffffff, acc.z, delta);
        acc.w += __shfl_down_sync(0xffffffff, acc.w, delta);
    }

    // Lane 0 writes the result
    if (lane == 0) {
        float4* dst = reinterpret_cast<float4*>(out + (long)n * K + k4);
        *dst = acc;
    }
}

torch::Tensor sum_reduce_cuda(torch::Tensor x, int dim)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32);
    TORCH_CHECK(dim == 1 && x.dim() == 3);

    const int N = x.size(0), M = x.size(1), K = x.size(2);
    TORCH_CHECK(K % 4 == 0 && M % 32 == 0);

    auto out = torch::empty({N, 1, K}, x.options());

    // Each warp (32 threads) handles one float4 output
    // # warps needed = K/4
    // Use BLOCK=256 = 8 warps; grid_x = ceil(K/4 / 8)
    constexpr int BLOCK = 256;
    const int warps_per_block = BLOCK / 32;
    const int num_warps = K / 4;  // K=4096, so 1024 warps per n
    const int grid_x = (num_warps + warps_per_block - 1) / warps_per_block;
    dim3 grid(grid_x, N);

    sum_reduce_warp_coop<BLOCK><<<grid, BLOCK>>>(
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
    name="sum_reduce_warp_coop_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sum_reduce_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    """
    Optimized sum reduction: warp-cooperative reduction along M.
    32 threads per output float4, each handling M/32 rows; then warp shuffle.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _mod.sum_reduce_cuda(x, self.dim)
