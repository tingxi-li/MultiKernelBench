import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 1 (blind redo): WMMA TF32 tensor-core GEMM + fused bias+GELU+softmax
# BM=128, BN=128, BK=32, 8 warps (4M×2N), each warp 2×4 WMMA m16n16k8 tiles
# WT [K,N] precomputed in __init__ for coalesced global loads (no per-call copy!)
# Separate fused bias+GELU+softmax kernel (1 block/row, registers avoid extra HBM pass)

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <mma.h>
#include <float.h>
#include <torch/extension.h>
using namespace nvcuda;

static constexpr int BM   = 128;
static constexpr int BN   = 128;
static constexpr int BKK  =  32;   // renamed to avoid macro clash
static constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
static constexpr int WARPS_M = 4, WARPS_N = 2;
static constexpr int WM = 2, WN = 4;
static constexpr int NTHREADS = 256;
static constexpr int SOFT_T = 256;
static constexpr int EPT = 32;   // N(8192)/SOFT_T

__device__ __forceinline__ float gelu_ex(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// ─── WMMA TF32 GEMM (writes raw pre-GELU values to C) ───────────────────────
// A [M,K] row-major, WT [K,N] row-major (pretransposed W)
// C [M,N] row-major output (raw GEMM result, no bias/GELU yet)
// Smem: As[BM][BKK+4]=128*36*4=18432B, Bs[BKK][BN+4]=32*132*4=16896B → 35328B < 48KB ✓
__global__ __launch_bounds__(NTHREADS)
void wmma_gemm_tf32(
    const float* __restrict__ A,
    const float* __restrict__ WT,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int wid = threadIdx.x / 32;
    const int wr  = wid / WARPS_N;
    const int wc  = wid % WARPS_N;
    const int bm  = blockIdx.y * BM;
    const int bn  = blockIdx.x * BN;
    const int wm0 = wr * (WM * WMMA_M);   // 0,32,64,96
    const int wn0 = wc * (WN * WMMA_N);   // 0,64

    __shared__ float As[BM][BKK + 4];    // 18432 bytes
    __shared__ float Bs[BKK][BN + 4];    // 16896 bytes

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WM][WN];
    #pragma unroll
    for (int i = 0; i < WM; i++)
        #pragma unroll
        for (int j = 0; j < WN; j++)
            wmma::fill_fragment(acc[i][j], 0.f);

    for (int kb = 0; kb < K; kb += BKK) {
        // Load A[bm:+BM, kb:+BKK] → As[m][k]
        // BM*BKK=4096 floats, 256 threads → 16 each
        // e → m=e/BKK, k=e%BKK: consecutive threads load consecutive k → coalesced ✓
        #pragma unroll
        for (int e = threadIdx.x; e < BM * BKK; e += NTHREADS) {
            int m = e / BKK, k = e % BKK;
            int gm = bm + m, gk = kb + k;
            As[m][k] = (gm < M && gk < K) ? A[gm * K + gk] : 0.f;
        }
        // Load WT[kb:+BKK, bn:+BN] → Bs[k][n]
        // BKK*BN=4096 floats, 256 threads → 16 each
        // e → k=e/BN, n=e%BN: consecutive threads → consecutive n (BN=128) → coalesced ✓
        #pragma unroll
        for (int e = threadIdx.x; e < BKK * BN; e += NTHREADS) {
            int k = e / BN, n = e % BN;
            int gk = kb + k, gn = bn + n;
            Bs[k][n] = (gk < K && gn < N) ? WT[gk * N + gn] : 0.f;
        }
        __syncthreads();

        // WMMA: BKK/WMMA_K = 32/8 = 4 steps
        #pragma unroll
        for (int ks = 0; ks < BKK / WMMA_K; ks++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> af[WM];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> bf[WN];
            #pragma unroll
            for (int i = 0; i < WM; i++)
                wmma::load_matrix_sync(af[i],
                    &As[wm0 + i * WMMA_M][ks * WMMA_K], BKK + 4);
            #pragma unroll
            for (int j = 0; j < WN; j++)
                wmma::load_matrix_sync(bf[j],
                    &Bs[ks * WMMA_K][wn0 + j * WMMA_N], BN + 4);
            #pragma unroll
            for (int i = 0; i < WM; i++)
                #pragma unroll
                for (int j = 0; j < WN; j++)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Store frags to C (global memory, row-major)
    #pragma unroll
    for (int i = 0; i < WM; i++)
        #pragma unroll
        for (int j = 0; j < WN; j++) {
            int gm = bm + wm0 + i * WMMA_M;
            int gn = bn + wn0 + j * WMMA_N;
            if (gm < M && gn < N)
                wmma::store_matrix_sync(&C[gm * N + gn], acc[i][j], N,
                                        wmma::mem_row_major);
        }
}

// ─── Fused bias + GELU + softmax ─────────────────────────────────────────────
// One block per row. SOFT_T=256 threads, EPT=32 elements per thread.
// Keeps all EPT=32 values in registers → 1 read + 1 write per element.
__global__ void bias_gelu_softmax_k(
    float* __restrict__ C,
    const float* __restrict__ b,
    int M, int N)
{
    const int row  = blockIdx.x;
    if (row >= M) return;
    float* rp = C + (long)row * N;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int nw   = SOFT_T / 32;
    __shared__ float sm[8];

    float reg[EPT];
    float mx = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        int idx = threadIdx.x + i * SOFT_T;
        float v = rp[idx] + b[idx];
        v = gelu_ex(v);
        reg[i] = v;
        mx = fmaxf(mx, v);
    }
    for (int d = 16; d > 0; d >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, d));
    if (!lane) sm[warp] = mx;
    __syncthreads();
    if (!warp) {
        float v = (lane < nw) ? sm[lane] : -FLT_MAX;
        for (int d = 4; d > 0; d >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, d));
        if (!lane) sm[0] = v;
    }
    __syncthreads();
    mx = sm[0];

    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < EPT; i++) { reg[i] = expf(reg[i] - mx); s += reg[i]; }
    for (int d = 16; d > 0; d >>= 1) s += __shfl_xor_sync(~0u, s, d);
    if (!lane) sm[warp] = s;
    __syncthreads();
    if (!warp) {
        float v = (lane < nw) ? sm[lane] : 0.f;
        for (int d = 4; d > 0; d >>= 1) v += __shfl_xor_sync(~0u, v, d);
        if (!lane) sm[0] = v;
    }
    __syncthreads();
    float inv_s = 1.f / sm[0];

    #pragma unroll
    for (int i = 0; i < EPT; i++)
        rp[threadIdx.x + i * SOFT_T] = reg[i] * inv_s;
}

// ─── Host launcher ────────────────────────────────────────────────────────────
torch::Tensor fused_wmma_launch(
    torch::Tensor A, torch::Tensor WT, torch::Tensor bias, int M, int N, int K)
{
    auto C = torch::empty({M, N}, A.options());
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    wmma_gemm_tf32<<<grid, NTHREADS>>>(
        A.data_ptr<float>(), WT.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K);
    bias_gelu_softmax_k<<<M, SOFT_T>>>(
        C.data_ptr<float>(), bias.data_ptr<float>(), M, N);
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor fused_wmma_launch(
    torch::Tensor A, torch::Tensor WT, torch::Tensor bias, int M, int N, int K);
"""

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_inline(
            name="fused_mgs_wmma4",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["fused_wmma_launch"],
            extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"],
            verbose=False,
        )
    return _MODULE


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Precompute transposed weight [K, N] once — avoid per-call 256MB copy
        self.register_buffer('weight_T',
            self.linear.weight.data.t().contiguous())

    def forward(self, x):
        M, K = x.shape
        N = self.linear.out_features
        return _get_module().fused_wmma_launch(
            x.contiguous(),
            self.weight_T,          # [K, N] precomputed
            self.linear.bias.contiguous(),
            M, N, K
        )
