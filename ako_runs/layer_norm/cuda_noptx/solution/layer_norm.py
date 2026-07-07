import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================================
# layer_norm / cuda_noptx  (plain CUDA via load_inline, NO inline PTX asm)
#
# Shape: x = (M=64 rows, N=4,194,304 fp32) = 1.07 GB/tensor, 16.78 MB/row.
# L2 = 96 MB.  LEVER: one row (16.78 MB) fits in L2, so a 2-pass design
# (reduce row -> apply row) keeps that row L2-resident across stats->apply.
# The 2nd read (in apply) then hits L2 -> only 2 HBM passes (read x, write y)
# instead of the naive 3 (read x, read x, write y) -> ~2x.
#
# NATIVE METHOD: a HOST-SIDE per-row C++ loop processes ONE row at a time,
# splitting that row across `blocks` CUDA blocks so ONLY that row's 16.78 MB
# is in flight between its reduce and apply kernels (all 64 rows at once would
# spill L2 -> 3 passes).  __ldg cached loads; fp64 accumulation for accuracy.
# ============================================================================

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

__inline__ __device__ double warpReduceSum(double val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffffu, val, offset);
  return val;
}

// Reduce ONE row (pointed to by x) to (sum, sumsq) accumulated in fp64 into acc[0], acc[1].
// nv = N/4 float4 elements. acc is pre-zeroed by the host.
__global__ void ln_reduce_kernel(const float* __restrict__ x, long nv,
                                 double* __restrict__ acc) {
  double lsum = 0.0, lsq = 0.0;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  long tid    = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x  * blockDim.x;
  for (long i = tid; i < nv; i += stride) {
    float4 v = __ldg(&x4[i]);
    double a = v.x, b = v.y, c = v.z, d = v.w;
    lsum += a + b + c + d;
    lsq  += a*a + b*b + c*c + d*d;
  }
  lsum = warpReduceSum(lsum);
  lsq  = warpReduceSum(lsq);
  __shared__ double wsum[32];
  __shared__ double wsq[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  if (lane == 0) { wsum[wid] = lsum; wsq[wid] = lsq; }
  __syncthreads();
  int nwarps = (blockDim.x + 31) >> 5;
  if (wid == 0) {
    double s = (lane < nwarps) ? wsum[lane] : 0.0;
    double q = (lane < nwarps) ? wsq[lane]  : 0.0;
    s = warpReduceSum(s);
    q = warpReduceSum(q);
    if (lane == 0) {
      atomicAdd(&acc[0], s);
      atomicAdd(&acc[1], q);
    }
  }
}

// Apply LayerNorm to ONE row: y = (x - mean) * rstd * w + b.
// mean/rstd derived (fp64) from acc[0]=sum, acc[1]=sumsq.  x is expected to be
// L2-resident from the just-run reduce of this same row -> the read hits L2.
__global__ void ln_apply_kernel(const float* __restrict__ x,
                                const float* __restrict__ w,
                                const float* __restrict__ b,
                                float* __restrict__ y, long nv, long N,
                                const double* __restrict__ acc, double eps) {
  double sum   = acc[0];
  double sumsq = acc[1];
  double invN  = 1.0 / (double)N;
  double mean  = sum * invN;
  double var   = sumsq * invN - mean * mean;
  double rstd  = 1.0 / sqrt(var + eps);
  float meanf  = (float)mean;
  float rstdf  = (float)rstd;

  const float4* x4 = reinterpret_cast<const float4*>(x);
  const float4* w4 = reinterpret_cast<const float4*>(w);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  float4*       y4 = reinterpret_cast<float4*>(y);
  long tid    = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x  * blockDim.x;
  for (long i = tid; i < nv; i += stride) {
    float4 xv = __ldg(&x4[i]);
    float4 wv = __ldg(&w4[i]);
    float4 bv = __ldg(&b4[i]);
    float4 yv;
    yv.x = (xv.x - meanf) * rstdf * wv.x + bv.x;
    yv.y = (xv.y - meanf) * rstdf * wv.y + bv.y;
    yv.z = (xv.z - meanf) * rstdf * wv.z + bv.z;
    yv.w = (xv.w - meanf) * rstdf * wv.w + bv.w;
    y4[i] = yv;
  }
}

torch::Tensor ln_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                         int64_t blocks, int64_t tpb) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto bc = b.contiguous();
  long M = xc.size(0);
  long total = xc.numel();
  long N = total / M;
  long nv = N / 4;

  auto y   = torch::empty_like(xc);
  auto acc = torch::zeros({M * 2}, xc.options().dtype(torch::kFloat64));

  const float* xp = xc.data_ptr<float>();
  const float* wp = wc.data_ptr<float>();
  const float* bp = bc.data_ptr<float>();
  float*  yp = y.data_ptr<float>();
  double* ap = acc.data_ptr<double>();
  double eps = 1e-5;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // HOST-SIDE per-row loop: reduce row m, then apply row m -> row stays L2-resident.
  for (long m = 0; m < M; ++m) {
    const float* xrow = xp + m * N;
    float*       yrow = yp + m * N;
    double*      arow = ap + m * 2;
    ln_reduce_kernel<<<blocks, tpb, 0, stream>>>(xrow, nv, arow);
    ln_apply_kernel <<<blocks, tpb, 0, stream>>>(xrow, wp, bp, yrow, nv, N, arow, eps);
  }
  return y;
}
"""

CPP_SRC = "torch::Tensor ln_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, int64_t blocks, int64_t tpb);"

_ext = load_inline(
    name="ln_noptx_ext",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["ln_forward"],
    verbose=False,
)

# Per-row split; tunable at runtime (no recompile). Swept by hand per variant.
BLOCKS = 256
TPB = 256


class Model(nn.Module):
    def __init__(self, normalized_shape):
        super(Model, self).__init__()
        # Mirror nn.LayerNorm's affine params (default: weight=ones, bias=zeros).
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # glue only: reshape/allocate/launch happen inside the CUDA extension.
        return _ext.ln_forward(x, self.weight, self.bias, BLOCKS, TPB)


batch_size = 64
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [(features, dim1, dim2)]
