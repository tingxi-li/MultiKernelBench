import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include <math.h>

// Fused bias-add + GELU + row-softmax
// One block per row; THREADS=256; ELEMS_PER_THREAD=32 (for N=8192)
// Intermediate GELU values stay in registers -> only 1 global read pass + 1 write pass
template <int THREADS, int EPT>
__global__ void fused_bias_gelu_softmax_kernel(
    const float* __restrict__ gemm_out,  // [M, N]
    const float* __restrict__ bias,       // [N]
    float* __restrict__ output,            // [M, N]
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float smem[];  // [THREADS]

    const float* in_row = gemm_out + (ptrdiff_t)row * N;
    float* out_row = output + (ptrdiff_t)row * N;

    float vals[EPT];

    // ---- Pass 1: load + bias + GELU, track local max ----
    float lmax = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        int idx = threadIdx.x + i * THREADS;
        float v = in_row[idx] + bias[idx];
        // GELU exact: 0.5 * v * (1 + erf(v / sqrt(2)))
        float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
        vals[i] = g;
        lmax = fmaxf(lmax, g);
    }

    // Reduce max
    smem[threadIdx.x] = lmax;
    __syncthreads();
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    const float row_max = smem[0];

    // ---- Pass 2: exp(val - max), sum (still in registers) ----
    float lsum = 0.0f;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        float e = __expf(vals[i] - row_max);
        vals[i] = e;
        lsum += e;
    }

    // Reduce sum
    smem[threadIdx.x] = lsum;
    __syncthreads();
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float inv_sum = 1.0f / smem[0];

    // ---- Pass 3: write normalized output ----
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        int idx = threadIdx.x + i * THREADS;
        out_row[idx] = vals[i] * inv_sum;
    }
}

static cublasHandle_t g_handle = nullptr;

torch::Tensor matmul_gelu_softmax_fwd(
    torch::Tensor x,       // [M, K] float32 contiguous
    torch::Tensor weight,  // [N, K] float32 contiguous
    torch::Tensor bias     // [N]   float32 contiguous
) {
    const int M = (int)x.size(0);   // 1024
    const int K = (int)x.size(1);   // 8192
    const int N = (int)weight.size(0); // 8192

    auto gemm_out = torch::empty({M, N}, x.options());
    auto out      = torch::empty({M, N}, x.options());

    if (!g_handle) {
        cublasCreate(&g_handle);
        // Use TF32 tensor cores for fp32 GEMM (same mode as PyTorch default on Ada)
        cublasSetMathMode(g_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    // GEMM: out = x[M,K] @ weight.T[K,N]  (row-major)
    // cuBLAS is col-major: C = A * B
    // Row-major trick: C^T = weight @ x^T
    //   -> cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, 1, weight, K, x, K, 0, gemm_out, N)
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weight.data_ptr<float>(), K,
        x.data_ptr<float>(), K,
        &beta,
        gemm_out.data_ptr<float>(), N);

    // Fused bias + GELU + softmax
    // N=8192, THREADS=256, EPT=32
    constexpr int THREADS = 256;
    constexpr int EPT = 32;   // 8192 / 256
    const int smem_bytes = THREADS * sizeof(float);
    fused_bias_gelu_softmax_kernel<THREADS, EPT><<<M, THREADS, smem_bytes>>>(
        gemm_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N
    );

    return out;
}
"""

_cpp_src = r"""
torch::Tensor matmul_gelu_softmax_fwd(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

_ext = load_inline(
    name="mgf_v1",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["matmul_gelu_softmax_fwd"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_ldflags=["-lcublas"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return _ext.matmul_gelu_softmax_fwd(
            x,
            self.linear.weight,
            self.linear.bias,
        )
