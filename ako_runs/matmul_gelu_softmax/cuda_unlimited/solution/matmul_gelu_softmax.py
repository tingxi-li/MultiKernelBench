import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 2: High-performance FP32 register-blocking GEMM
# BM=128, BN=128, BK=16, TM=8, TN=8, 256 threads
# Float4 vectorized loads for maximum bandwidth
# Each thread: 8×8 register tile → 64 FP32 FMAs per K-step
# Arithmetic intensity: very high (256 FMAs per loaded float)

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <float.h>

// BM=128, BN=128, BK=16 (smem: 2*128*20*4 = 20480 bytes ✓ < 48KB)
// 256 threads, TM=TN=8 → 16×16 threads in (M,N)
// Global load: float4 vectorized (4 floats at once)

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

// rowA: A stored [M,K], tile into As[BM][BK]
// rowW: W stored [N,K], tile into Bs[BN][BK]
// Output: C[M,N] = gelu(A*W^T + bias)

__global__ __launch_bounds__(NT)
void gemm_gelu_reg(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int ty = threadIdx.x / (BN / TN);    // 0..15 (rows in thread grid)
    const int tx = threadIdx.x % (BN / TN);    // 0..15 (cols in thread grid)
    const int tid = threadIdx.x;

    // Shared memory: As[BK][BM+PAD], Bs[BK][BN+PAD]
    // K-major layout for column-access during compute
    __shared__ float As[BK][BM + PAD];    // 16*132*4 = 8448 bytes
    __shared__ float Bs[BK][BN + PAD];    // 16*132*4 = 8448 bytes
    // Total: 16896 bytes ✓

    // Register accumulators
    float acc[TM][TN] = {};

    // Load A tile: BM*BK = 128*16 = 2048 floats, NT=256 → 8 per thread
    // Each thread loads 2 float4 (8 floats)
    // Layout: thread tid loads As[k_inner][m_inner] where:
    // m_inner = (tid % (BM/4)) * 4 (using BM/4 = 32 different m positions per row of 4)
    // Actually: flat index e, thread tid loads e = tid + step*NT
    // e → k = e / BM, m = e % BM → As[k][m] = A[block_row+m, k_base+k]

    for (int k_base = 0; k_base < K; k_base += BK) {
        // Load As: BM*BK = 2048 elements, 8 per thread
        // Using flat index with coalescing:
        // Thread 0-31 load m=0..31, thread 32-63 load m=32..63 for k=0
        // Then thread 0-31 load m=0..31 for k=1, etc.
        // stride=NT=256, total=2048, 8 iters
        // For k = e/BM, m = e%BM: consecutive threads load consecutive m (coalesced in BM direction)
        // Global A[block_row+m, k_base+k]: each BM elements are stride K apart → NOT coalesced

        // For coalesced A load: layout As[m][k] (row-major) with consecutive threads → consecutive k
        // But we need As[k][m] for the compute phase... OR
        // Load As[m][k] (row-major), compute reads As[k][m] = As_T
        // → load As as row-major (coalesced), read As_T transposed = column-major
        // Column-major access to As_T[k][m] = As[m][k] → not friendly

        // Alternative: load A with transposed pattern
        // For coalesced global load of A[block_row:block_row+BM, k_base:k_base+BK]:
        // thread (tid) loads row = tid / BK, col = tid % BK
        // Wait: consecutive threads (0,1,...,31) load:
        //   tid=0: r=0, c=0 → A[block_row, k_base]
        //   tid=1: r=0, c=1 → A[block_row, k_base+1]
        //   ...
        //   tid=15: r=0, c=15 → A[block_row, k_base+15]  (BK=16, so last)
        //   tid=16: r=1, c=0 → A[block_row+1, k_base]
        // This is NOT coalesced (jumps of K between consecutive 16-element groups)
        //
        // Better: row = tid % BM, col = tid / BM
        //   tid=0: r=0, c=0
        //   tid=1: r=1, c=0
        //   ...
        //   tid=127: r=127, c=0
        //   tid=128: r=0, c=1
        // Also NOT coalesced

        // The truth: for A[M,K] with K=8192, the tile A[block_row:block_row+128, k_base:k_base+16]
        // is NOT contiguous. Row i is at offset (block_row+i)*K + k_base.
        // Each row is 16 floats = 64 bytes. Consecutive rows are K*4=32768 bytes apart.
        // To coalesce, we need consecutive threads to access consecutive memory.
        // If thread (128*row_in_tile + col_in_tile) loads element [col_in_tile, row_in_tile] (transposed):
        //   Threads 0-127: all load col=0 (k_base+0), rows 0-127 → stride K apart → NOT coalesced
        //   Threads 128-255: all load col=1 (k_base+1), rows 0-127 → same issue

        // The fundamental problem: A[M,K] tiles with a narrow BK=16 column slice
        // → "short" dimension is K, long dimension is M
        // → coalesced loads need consecutive threads to access A[m, k+0], A[m, k+1], ...
        // → but that means each warp loads 32 consecutive K values for the same M row
        // → BUT BK=16 < 32, so only 16 threads per warp are active for a given row!

        // SOLUTION: transpose the load. Use float4 to load 4 elements at a time.
        // For A tile (128 rows × 16 cols):
        // Arrange: 32 threads load one row of 4 elements each = 128 elements per warp
        // 2048 / 128 = 16 warps needed... but we only have 8 warps!
        // → 2 passes per warp, 4 elements each (float4)

        // Actually, let's do this more carefully with float4:
        // 2048 floats / 4 = 512 float4 loads.
        // 256 threads → 2 float4 loads per thread.
        // Thread tid loads float4 at index tid and tid+256.
        // For float4 at index i:
        //   row = i / (BK/4) = i / 4   (0..127, since BK=16 → BK/4=4 float4s per row)
        //   col4 = i % (BK/4) = i % 4  (0..3 → float4 at k=0,4,8,12)
        //   → loads A[block_row+row, k_base+col4*4 : col4*4+4]
        // thread tid: float4 at index tid (row=tid/4, col4=tid%4)
        //   tid=0: row=0, col4=0 → loads A[block_row+0, k_base:k_base+4]
        //   tid=1: row=0, col4=1 → loads A[block_row+0, k_base+4:k_base+8]
        //   tid=2: row=0, col4=2 → loads A[block_row+0, k_base+8:k_base+12]
        //   tid=3: row=0, col4=3 → loads A[block_row+0, k_base+12:k_base+16]
        //   tid=4: row=1, col4=0 → loads A[block_row+1, k_base:k_base+4]
        //   ...
        // For a warp (threads 0-31):
        //   threads 0-3: row 0, all 4 col4 positions
        //   threads 4-7: row 1, all 4 col4 positions
        //   ...
        //   threads 28-31: row 7, all 4 col4 positions
        // → thread 0 and thread 4 load different rows: A[block_row, k_base] and A[block_row+1, k_base]
        // These are K=8192*4 = 32768 bytes apart → NOT coalesced

        // FINAL DECISION: just accept non-coalesced A loads and rely on L2 cache.
        // The key bottleneck is compute, not memory (for large GEMM).
        // Use simple flat-index loads:

        for (int e = tid; e < BM * BK; e += NT) {
            int m = e / BK, k = e % BK;
            int gm = block_row + m, gk = k_base + k;
            As[k][m] = (gm < M && gk < K) ? A[gm * K + gk] : 0.f;
        }
        for (int e = tid; e < BN * BK; e += NT) {
            int n = e / BK, k = e % BK;
            int gn = block_col + n, gk = k_base + k;
            Bs[k][n] = (gn < N && gk < K) ? W[gn * K + gk] : 0.f;
        }
        __syncthreads();

        // Compute: each thread's 8×8 accumulator, unrolled over BK
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

    // Epilogue: bias + GELU, write to C
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
            name="fused_mgs_reg_v1",
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
