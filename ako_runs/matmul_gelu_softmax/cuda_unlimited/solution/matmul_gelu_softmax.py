import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 3: Attempt split-K GEMM for better parallelism
# Observation: M=1024, N=K=8192 → only 8x64=512 blocks with BM=BN=128
# Split K into P partitions, each block computes partial sums, atomic-add to output
# This increases parallelism along the K dimension
# THEN do GELU pass, then softmax pass
# WARNING: atomic float adds may reduce accuracy

# Actually, let's try a different approach: use __ptx_isa to explicitly control
# the load/store instructions for better memory behavior.
# OR: use a transposed weight (W.T) so that the GEMM is A@W.T ≡ standard row-col
# and both A and W.T have the required memory layout for coalesced access.

# Key insight: W is [N,K], W.T is [K,N]. Standard GEMM C=A@W.T accesses:
# A[m,k] → row-major, K dimension → ok for inner loop
# W[n,k] → row-major, K dimension → ok for inner loop
# Both are accessed in K direction which IS coalesced (K is stride-1 for row-major)!
#
# So when loading A's tile (BM rows, BK cols) and W's tile (BN rows, BK cols):
# - For A: stride between consecutive k_inner = 1 byte → COALESCED ✓
#   BUT stride between consecutive m_inner = K → row-by-row, not stride-1
# - Thread access: 256 threads loading 128*16=2048 elements
#   Thread tid loads element e = tid (or tid+step*256)
#   If e → row=e%16=e%BK, col=e/BK: consecutive threads load consecutive K-positions
#   for the same M-position → that's within a single 128-byte cache line (16 floats = 64 bytes)
#   But consecutive threads load k=0,1,...,15 for same m=0 → this IS coalesced! Each warp
#   loads threads 0-15 (k=0..15, m=0) AND threads 16-31 (k=0..15, m=1)... wait no.
#   e = tid, thread 0 loads k=0,m=0; thread 1 loads k=1,m=0; ... thread 15 loads k=15,m=0
#   thread 16: k=16%16=0, m=16/16=1... BUT BK=16 so k=e%BK=e%16
#   thread 0: k=0,m=0 → A[block_row+0, k_base+0]
#   thread 1: k=1,m=0 → A[block_row+0, k_base+1]
#   ...
#   thread 15: k=15,m=0 → A[block_row+0, k_base+15]
#   thread 16: k=0,m=1 → A[block_row+1, k_base+0]
#   thread 17: k=1,m=1 → A[block_row+1, k_base+1]
# So threads 0-15 all access row block_row+0 (consecutive K): COALESCED ✓
# Threads 16-31: row block_row+1 (consecutive K): COALESCED ✓
# But threads 0 and 16 access different rows → interleaved within a warp:
#   warp: threads 0-31, 16 from row m=0 and 16 from row m=1
#   addresses: rows m=0,1 are K floats apart = 32KB apart → two separate cache lines
# This is a 2-way bank conflict in terms of cache lines, but not register bank conflict.
# Actually it's fine — 2 cache line loads for 32 threads.
#
# The LOAD pattern e → k=e%BK, m=e/BK maps BM*BK elements to (k,m) pairs.
# For a warp (threads t..t+31): loading elements [t, t+1, ..., t+31]
# These have k = t%16, t%16+1, ... (cycling), m = t/16, ...
# If t=0: k=0..15 for m=0, then k=0..15 for m=1 (within warp)
# → two consecutive rows of A, each with all BK=16 k-values
# → 2 cache line accesses (one per row, 16 floats each = 64 bytes)
# → 32 threads / 2 cache lines = 16 threads per cache line ✓ (perfect coalescing)
#
# Similarly for W. This pattern IS coalesced! The key is e%BK ordering.
# Wait, iter-2 already used this pattern (k=e/BM, m=e%BM which is different).
# Actually iter-2 used k=e/BM, m=e%BM → k=tid/128, which for tid=0..255:
#   k=0 for tid=0..127, k=1 for tid=128..255 → NOT coalesced (all tid<128 load k=0)
# Let me switch to k=e%BK, m=e/BK for better coalescing!

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <float.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8
#define NT 256
#define PAD 4

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__global__
void gemm_gelu_reg(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int ty = threadIdx.x / (BN / TN);
    const int tx = threadIdx.x % (BN / TN);
    const int tid = threadIdx.x;

    __shared__ float As[BK][BM + PAD];
    __shared__ float Bs[BK][BN + PAD];

    float acc[TM][TN] = {};

    for (int k_base = 0; k_base < K; k_base += BK) {
        // Coalesced load: e → k=e%BK, m=e/BK
        // consecutive threads: k cycles 0..BK-1, m increments
        // warp loads 2 rows of A each with BK=16 consecutive k values → 2 cache lines
        for (int e = tid; e < BM * BK; e += NT) {
            int k = e % BK, m = e / BK;
            int gm = block_row + m, gk = k_base + k;
            As[k][m] = (gm < M && gk < K) ? __ldg(&A[gm * K + gk]) : 0.f;
        }
        for (int e = tid; e < BN * BK; e += NT) {
            int k = e % BK, n = e / BK;
            int gn = block_col + n, gk = k_base + k;
            Bs[k][n] = (gn < N && gk < K) ? __ldg(&W[gn * K + gk]) : 0.f;
        }
        __syncthreads();

        float a_reg[TM], b_reg[TN];
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_reg[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_reg[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = block_row + ty * TM + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = block_col + tx * TN + j;
            if (gn < N)
                C[gm * N + gn] = gelu_exact(acc[i][j] + bias[gn]);
        }
    }
}

__global__ void softmax_kernel(float* __restrict__ C, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    float* rp = C + row * N;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nw = blockDim.x / 32;

    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += blockDim.x) mx = fmaxf(mx, rp[i]);
    for (int m = 16; m > 0; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, m));
    __shared__ float sm[8];
    if (!lane) sm[warp] = mx;
    __syncthreads();
    if (!warp) {
        float v = lane < nw ? sm[lane] : -FLT_MAX;
        for (int m=4;m>0;m>>=1) v=fmaxf(v,__shfl_xor_sync(~0u,v,m));
        if(!lane) sm[0]=v;
    }
    __syncthreads();
    float row_max = sm[0];

    float s = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) s += expf(rp[i] - row_max);
    for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(~0u, s, m);
    if (!lane) sm[warp] = s;
    __syncthreads();
    if (!warp) {
        float v = lane < nw ? sm[lane] : 0.f;
        for (int m=4;m>0;m>>=1) v+=__shfl_xor_sync(~0u,v,m);
        if(!lane) sm[0]=v;
    }
    __syncthreads();
    float inv_s = 1.f / sm[0];
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        rp[i] = expf(rp[i] - row_max) * inv_s;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor fused_matmul_gelu_softmax(
    torch::Tensor A, torch::Tensor W, torch::Tensor bias, int M, int N, int K);
"""

_CUDA_WRAPPER = r"""
#include <torch/extension.h>
__global__ void gemm_gelu_reg(const float*, const float*, const float*, float*, int, int, int);
__global__ void softmax_kernel(float*, int, int);

torch::Tensor fused_matmul_gelu_softmax(
    torch::Tensor A, torch::Tensor W, torch::Tensor bias, int M, int N, int K)
{
    auto C = torch::empty({M, N}, A.options());
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    gemm_gelu_reg<<<grid, 256>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), bias.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K);
    softmax_kernel<<<M, 256>>>(C.data_ptr<float>(), M, N);
    return C;
}
"""

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_inline(
            name="fused_mgs_reg_v6",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC + _CUDA_WRAPPER,
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
            self.linear.bias.contiguous(), M, N, K
        )
