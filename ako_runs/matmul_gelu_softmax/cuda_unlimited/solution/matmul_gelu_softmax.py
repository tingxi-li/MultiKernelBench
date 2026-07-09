import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 5: Use torch's optimized mm (via C++ at::mm) + custom fused GELU+Softmax kernel
# Strategy:
# 1. C++ extension calls at::mm(A, W.T) → cuBLAS GEMM (optimal FP32 SGEMM)
# 2. Custom CUDA kernel: reads GEMM output, adds bias+GELU, then inline softmax

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// True online softmax: 2-pass (pass 1: GELU + online max/sum, pass 2: normalize)
// Online algorithm: process elements one at a time, maintaining (m, d) where:
//   m = running max, d = sum(exp(x_i - m)) for seen elements
//   When new element x: d_new = d * exp(m - max(m, x)) + exp(x - max(m, x))
//                        m_new = max(m, x)
// This allows computing max and sum simultaneously in one pass!
// But for GPU: each thread handles N/blockDim.x elements, then reduce across threads.
// 2 passes minimum: pass 1 reads C, writes exp to output; pass 2 normalizes.
// Actually: read once, write once = 2 passes through data = minimum for softmax.

__global__ void bias_gelu_softmax_kernel(
    const float* __restrict__ C_in,  // [M, N] GEMM output (read-only)
    float* __restrict__ C_out,        // [M, N] output
    const float* __restrict__ bias,  // [N]
    int M, int N)
{
    int row = blockIdx.x;
    if (row >= M) return;
    const float* rp_in = C_in + row * N;
    float* rp_out = C_out + row * N;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nw = blockDim.x / 32;
    __shared__ float sm_max[8], sm_sum[8];

    // Online pass: compute GELU, write to output, track (max, sum) simultaneously
    float running_max = -FLT_MAX;
    float running_sum = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = gelu_exact(rp_in[i] + bias[i]);
        rp_out[i] = val;
        // Online update
        float new_max = fmaxf(running_max, val);
        running_sum = running_sum * expf(running_max - new_max) + expf(val - new_max);
        running_max = new_max;
    }

    // Warp-level online reduce
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_max = __shfl_xor_sync(~0u, running_max, mask);
        float other_sum = __shfl_xor_sync(~0u, running_sum, mask);
        float new_max = fmaxf(running_max, other_max);
        running_sum = running_sum * expf(running_max - new_max) + other_sum * expf(other_max - new_max);
        running_max = new_max;
    }
    if (!lane) { sm_max[warp] = running_max; sm_sum[warp] = running_sum; }
    __syncthreads();

    if (!warp) {
        float wm = lane < nw ? sm_max[lane] : -FLT_MAX;
        float ws = lane < nw ? sm_sum[lane] : 0.f;
        for (int mask = 4; mask > 0; mask >>= 1) {
            float om = __shfl_xor_sync(~0u, wm, mask);
            float os = __shfl_xor_sync(~0u, ws, mask);
            float nm = fmaxf(wm, om);
            ws = ws * expf(wm - nm) + os * expf(om - nm);
            wm = nm;
        }
        if (!lane) { sm_max[0] = wm; sm_sum[0] = ws; }
    }
    __syncthreads();
    float row_max = sm_max[0];
    float inv_s = 1.f / sm_sum[0];

    // Normalize pass
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        rp_out[i] = expf(rp_out[i] - row_max) * inv_s;
}

torch::Tensor fused_matmul_gelu_softmax(
    torch::Tensor A,    // [M, K]
    torch::Tensor W,    // [N, K]
    torch::Tensor bias, // [N]
    int M, int N, int K)
{
    // Use PyTorch's optimized mm: gemm = A @ W.T
    auto gemm = at::mm(A, W.t());  // [M, N], contiguous
    auto out = torch::empty({M, N}, A.options());
    // Fused bias + GELU + online softmax (2-pass, reads gemm once + writes out once)
    bias_gelu_softmax_kernel<<<M, 256>>>(
        gemm.data_ptr<float>(), out.data_ptr<float>(),
        bias.data_ptr<float>(), M, N);
    return out;
}
"""

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        _CPP_DECL = "#include <torch/extension.h>\ntorch::Tensor fused_matmul_gelu_softmax(torch::Tensor,torch::Tensor,torch::Tensor,int,int,int);"
        _MODULE = load_inline(
            name="fused_mgs_mm_v3",
            cpp_sources=_CPP_DECL,
            cuda_sources=_CUDA_SRC,
            functions=["fused_matmul_gelu_softmax"],
            extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"],
            verbose=False,
        )
    return _MODULE


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        M, K = x.shape
        N = self.linear.out_features
        return _get_module().fused_matmul_gelu_softmax(
            x.contiguous(), self.linear.weight.contiguous(),
            self.linear.bias, M, N, K
        )
