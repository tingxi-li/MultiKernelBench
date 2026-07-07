import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# layer_norm / cuda_unlimited  (CUDA via load_inline, inline PTX permitted)
#
# Shape: x = (M=64 rows, N=64*256*256 = 4,194,304 fp32).  One row = 16.78 MB;
# L2 = 96 MB.  Roofline lever: process ONE row at a time so the row stays
# L2-resident from the stats pass into the apply pass (2nd x-read hits L2, not
# HBM) AND the per-element affine w/b (16.78 MB each) stay L2-resident ACROSS
# all 64 rows (working set = x[m]+w+b = 50 MB < 96 MB).  HBM floor:
#   x read once (1.07 GB) + y write once (1.07 GB) + w+b read once (0.034 GB)
#   = ~2.18 GB   vs reference ~5.35 GB (x read x2, w/b refetched per row).
# y is written with a STREAMING store (__stcs / st.global.cs) so it bypasses L2
# and does not evict the 50 MB resident working set.
# ---------------------------------------------------------------------------

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define TPB 256
#define STATS_BLOCKS 128

__inline__ __device__ float warpReduceSum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Pass 1: per-row sum and sum-of-squares -> atomically into d_sum[m], d_sq[m].
// Reads x[m] once from HBM, populating L2 with the row.
__global__ void ln_stats_kernel(const float* __restrict__ x_row,
                                float* __restrict__ d_sum,
                                float* __restrict__ d_sq,
                                int N4) {
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x_row);
    float s = 0.f, sq = 0.f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N4; i += stride) {
        float4 v = __ldg(&x4[i]);
        s  += v.x + v.y + v.z + v.w;
        sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    s  = warpReduceSum(s);
    sq = warpReduceSum(sq);
    __shared__ float wsum[TPB / 32];
    __shared__ float wsq[TPB / 32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) { wsum[wid] = s; wsq[wid] = sq; }
    __syncthreads();
    if (wid == 0) {
        float bs  = (lane < (TPB / 32)) ? wsum[lane] : 0.f;
        float bsq = (lane < (TPB / 32)) ? wsq[lane]  : 0.f;
        bs  = warpReduceSum(bs);
        bsq = warpReduceSum(bsq);
        if (lane == 0) { atomicAdd(d_sum, bs); atomicAdd(d_sq, bsq); }
    }
}

// Pass 2: finalize mean/rstd (folded, each block recomputes from the 2 scalars)
// then y = (x-mean)*rstd*w + b.  x re-read from L2; w/b from L2 (resident across
// rows); y streamed to HBM (bypasses L2).
__global__ void ln_apply_kernel(const float* __restrict__ x_row,
                                float* __restrict__ y_row,
                                const float* __restrict__ w,
                                const float* __restrict__ b,
                                const float* __restrict__ d_sum,
                                const float* __restrict__ d_sq,
                                int N4, float invN, float eps) {
    __shared__ float s_mean, s_rstd;
    if (threadIdx.x == 0) {
        float mean = (*d_sum) * invN;
        float var  = (*d_sq) * invN - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = s_mean, rstd = s_rstd;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x_row);
    const float4* __restrict__ w4 = reinterpret_cast<const float4*>(w);
    const float4* __restrict__ b4 = reinterpret_cast<const float4*>(b);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y_row);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N4) {
        float4 xv = __ldg(&x4[i]);
        float4 wv = __ldg(&w4[i]);
        float4 bv = __ldg(&b4[i]);
        float4 yv;
        yv.x = (xv.x - mean) * rstd * wv.x + bv.x;
        yv.y = (xv.y - mean) * rstd * wv.y + bv.y;
        yv.z = (xv.z - mean) * rstd * wv.z + bv.z;
        yv.w = (xv.w - mean) * rstd * wv.w + bv.w;
        __stcs(&y4[i], yv);   // streaming store: bypass L2
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor w,
                              torch::Tensor b, double eps) {
    auto xc = x.contiguous();
    auto y = torch::empty_like(xc);
    int M = xc.size(0);
    int64_t N = xc.numel() / M;          // elements per normalized row
    int N4 = (int)(N / 4);               // N guaranteed divisible by 4 here

    auto opts = xc.options();
    auto d_sum = torch::zeros({M}, opts);
    auto d_sq  = torch::zeros({M}, opts);

    const float* xp = xc.data_ptr<float>();
    float* yp = y.data_ptr<float>();
    const float* wp = w.data_ptr<float>();
    const float* bp = b.data_ptr<float>();
    float* sp  = d_sum.data_ptr<float>();
    float* sqp = d_sq.data_ptr<float>();

    float invN = 1.0f / (float)N;
    int apply_blocks = (N4 + TPB - 1) / TPB;
    auto stream = at::cuda::getCurrentCUDAStream();

    for (int m = 0; m < M; ++m) {
        const float* xrow = xp + (int64_t)m * N;
        float* yrow = yp + (int64_t)m * N;
        ln_stats_kernel<<<STATS_BLOCKS, TPB, 0, stream>>>(xrow, sp + m, sqp + m, N4);
        ln_apply_kernel<<<apply_blocks, TPB, 0, stream>>>(
            xrow, yrow, wp, bp, sp + m, sqp + m, N4, invN, (float)eps);
    }
    return y;
}
'''

_CPP_DECL = "torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps);"

_ext = load_inline(
    name="ln_cuda_unlimited_ext",
    cpp_sources=_CPP_DECL,
    cuda_sources=_CUDA_SRC,
    functions=["layer_norm_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "-lineinfo"],
)


class Model(nn.Module):
    def __init__(self, normalized_shape):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x):
        return _ext.layer_norm_cuda(x, self.weight, self.bias, self.eps)


batch_size = 64
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [(features, dim1, dim2)]
