import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BM 128
#define BN 128
#define BK 16

__device__ __forceinline__ float rtf(float x) {
    unsigned u = __float_as_uint(x);
    u = (u + 0x1000u) & 0xFFFFE000u;
    return __uint_as_float(u);
}
__device__ __forceinline__ unsigned ub(float x) { return __float_as_uint(x); }

__device__ __forceinline__ void mma_m16n8k8(float c[4],
        unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ------- batched S = scale * Q(MxK) * K_(NxK)^T  (contraction over d=K) -------
// A=Q rows [m][d]; Bmat=K rows [n][d] read in NATURAL layout (b-frag = K[n][d]).
__global__ __launch_bounds__(256) void gemm_qkt(
        const float* __restrict__ Q, const float* __restrict__ Kk,
        float* __restrict__ S, int M, int N, int K, float scale) {
    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BN * BK];   // key-major [n][k]
    const int bh = blockIdx.z;
    const float* A = Q + (long)bh * M * K;
    const float* B = Kk + (long)bh * N * K;
    float* C = S + (long)bh * M * N;

    const int tid = threadIdx.x, warpId = tid >> 5, lane = tid & 31;
    const int group = lane >> 2, tig = lane & 3;
    const int warpM = warpId >> 2, warpN = warpId & 3;
    const int mOrigin = warpM * 64, nOrigin = warpN * 32;
    const int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;

    float acc[4][4][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.0f;

    auto load_tile = [&](int k0, int buf) {
        #pragma unroll
        for (int t = tid; t < (BM * BK) / 4; t += 256) {
            int row = (t * 4) / BK, col = (t * 4) % BK;
            float4 v = *reinterpret_cast<const float4*>(&A[(blockRow + row) * K + k0 + col]);
            *reinterpret_cast<float4*>(&As[buf][row * BK + col]) =
                make_float4(rtf(v.x), rtf(v.y), rtf(v.z), rtf(v.w));
        }
        #pragma unroll
        for (int t = tid; t < (BN * BK) / 4; t += 256) {
            int row = (t * 4) / BK, col = (t * 4) % BK;   // row=key, col=d
            float4 v = *reinterpret_cast<const float4*>(&B[(blockCol + row) * K + k0 + col]);
            *reinterpret_cast<float4*>(&Bs[buf][row * BK + col]) =
                make_float4(rtf(v.x), rtf(v.y), rtf(v.z), rtf(v.w));
        }
    };
    int buf = 0;
    load_tile(0, buf);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) load_tile(k0 + BK, buf ^ 1);
        const float* Ab = As[buf];
        const float* Bb = Bs[buf];
        #pragma unroll
        for (int kk = 0; kk < BK / 8; ++kk) {
            int kBase = kk * 8;
            unsigned a[4][4], b[4][2];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r0 = mOrigin + mi * 16 + group;
                a[mi][0] = ub(Ab[r0 * BK + kBase + tig]);
                a[mi][1] = ub(Ab[(r0 + 8) * BK + kBase + tig]);
                a[mi][2] = ub(Ab[r0 * BK + kBase + tig + 4]);
                a[mi][3] = ub(Ab[(r0 + 8) * BK + kBase + tig + 4]);
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int n0 = nOrigin + ni * 8 + group;   // key index
                b[ni][0] = ub(Bb[n0 * BK + kBase + tig]);
                b[ni][1] = ub(Bb[n0 * BK + kBase + tig + 4]);
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma_m16n8k8(acc[mi][ni], a[mi][0], a[mi][1], a[mi][2], a[mi][3],
                                b[ni][0], b[ni][1]);
        }
        buf ^= 1;
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int baseRow = blockRow + mOrigin + mi * 16;
            int baseCol = blockCol + nOrigin + ni * 8;
            C[(baseRow + group) * N + baseCol + 2 * tig]         = acc[mi][ni][0] * scale;
            C[(baseRow + group) * N + baseCol + 2 * tig + 1]     = acc[mi][ni][1] * scale;
            C[(baseRow + group + 8) * N + baseCol + 2 * tig]     = acc[mi][ni][2] * scale;
            C[(baseRow + group + 8) * N + baseCol + 2 * tig + 1] = acc[mi][ni][3] * scale;
        }
}

// ------- batched O = A(MxK) * V(KxN), standard row-major (contraction over keys) -------
__global__ __launch_bounds__(256) void gemm_av(
        const float* __restrict__ Am, const float* __restrict__ Vv,
        float* __restrict__ O, int M, int N, int K) {
    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];
    const int bh = blockIdx.z;
    const float* A = Am + (long)bh * M * K;
    const float* B = Vv + (long)bh * K * N;
    float* C = O + (long)bh * M * N;

    const int tid = threadIdx.x, warpId = tid >> 5, lane = tid & 31;
    const int group = lane >> 2, tig = lane & 3;
    const int warpM = warpId >> 2, warpN = warpId & 3;
    const int mOrigin = warpM * 64, nOrigin = warpN * 32;
    const int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;

    float acc[4][4][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.0f;

    auto load_tile = [&](int k0, int buf) {
        #pragma unroll
        for (int t = tid; t < (BM * BK) / 4; t += 256) {
            int row = (t * 4) / BK, col = (t * 4) % BK;
            float4 v = *reinterpret_cast<const float4*>(&A[(blockRow + row) * K + k0 + col]);
            *reinterpret_cast<float4*>(&As[buf][row * BK + col]) =
                make_float4(rtf(v.x), rtf(v.y), rtf(v.z), rtf(v.w));
        }
        #pragma unroll
        for (int t = tid; t < (BK * BN) / 4; t += 256) {
            int row = (t * 4) / BN, col = (t * 4) % BN;
            float4 v = *reinterpret_cast<const float4*>(&B[(k0 + row) * N + blockCol + col]);
            *reinterpret_cast<float4*>(&Bs[buf][row * BN + col]) =
                make_float4(rtf(v.x), rtf(v.y), rtf(v.z), rtf(v.w));
        }
    };
    int buf = 0;
    load_tile(0, buf);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) load_tile(k0 + BK, buf ^ 1);
        const float* Ab = As[buf];
        const float* Bb = Bs[buf];
        #pragma unroll
        for (int kk = 0; kk < BK / 8; ++kk) {
            int kBase = kk * 8;
            unsigned a[4][4], b[4][2];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r0 = mOrigin + mi * 16 + group;
                a[mi][0] = ub(Ab[r0 * BK + kBase + tig]);
                a[mi][1] = ub(Ab[(r0 + 8) * BK + kBase + tig]);
                a[mi][2] = ub(Ab[r0 * BK + kBase + tig + 4]);
                a[mi][3] = ub(Ab[(r0 + 8) * BK + kBase + tig + 4]);
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int c0 = nOrigin + ni * 8 + group;
                b[ni][0] = ub(Bb[(kBase + tig) * BN + c0]);
                b[ni][1] = ub(Bb[(kBase + tig + 4) * BN + c0]);
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma_m16n8k8(acc[mi][ni], a[mi][0], a[mi][1], a[mi][2], a[mi][3],
                                b[ni][0], b[ni][1]);
        }
        buf ^= 1;
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int baseRow = blockRow + mOrigin + mi * 16;
            int baseCol = blockCol + nOrigin + ni * 8;
            C[(baseRow + group) * N + baseCol + 2 * tig]         = acc[mi][ni][0];
            C[(baseRow + group) * N + baseCol + 2 * tig + 1]     = acc[mi][ni][1];
            C[(baseRow + group + 8) * N + baseCol + 2 * tig]     = acc[mi][ni][2];
            C[(baseRow + group + 8) * N + baseCol + 2 * tig + 1] = acc[mi][ni][3];
        }
}

// row softmax over N (last dim) in place. one block per (bh*M) row.
#define ST 128
__global__ __launch_bounds__(ST) void softmax_rows(float* __restrict__ S, int N) {
    long row = blockIdx.x;
    int tid = threadIdx.x;
    float* r = S + row * N;
    float m = -1e30f;
    for (int j = tid; j < N; j += ST) m = fmaxf(m, r[j]);
    __shared__ float red[ST];
    red[tid] = m; __syncthreads();
    for (int s = ST >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    float rowMax = red[0]; __syncthreads();
    float sum = 0.f;
    for (int j = tid; j < N; j += ST) { float e = __expf(r[j] - rowMax); r[j] = e; sum += e; }
    red[tid] = sum; __syncthreads();
    for (int s = ST >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float inv = 1.f / red[0];
    for (int j = tid; j < N; j += ST) r[j] *= inv;
}

torch::Tensor run(torch::Tensor Q, torch::Tensor Kk, torch::Tensor Vv) {
    TORCH_CHECK(Q.is_cuda(), "cuda only");
    auto Qc = Q.contiguous(); auto Kc = Kk.contiguous(); auto Vc = Vv.contiguous();
    int B = Qc.size(0), H = Qc.size(1), Sq = Qc.size(2), D = Qc.size(3);
    int BH = B * H;
    float scale = 1.0f / sqrtf((float)D);
    auto opt = Qc.options();
    auto Q2 = Qc.view({BH, Sq, D});
    auto K2 = Kc.view({BH, Sq, D});
    auto V2 = Vc.view({BH, Sq, D});
    auto Sc = torch::empty({BH, Sq, Sq}, opt);        // scores

    dim3 blk(256);
    dim3 g1(Sq / BN, Sq / BM, BH);
    gemm_qkt<<<g1, blk>>>(Q2.data_ptr<float>(), K2.data_ptr<float>(),
                          Sc.data_ptr<float>(), Sq, Sq, D, scale);
    softmax_rows<<<(long)BH * Sq, ST>>>(Sc.data_ptr<float>(), Sq);

    auto O = torch::empty({BH, Sq, D}, opt);
    dim3 g2(D / BN, Sq / BM, BH);
    gemm_av<<<g2, blk>>>(Sc.data_ptr<float>(), V2.data_ptr<float>(),
                         O.data_ptr<float>(), Sq, D, Sq);
    return O.view({B, H, Sq, D});
}
'''

_CPP = "torch::Tensor run(torch::Tensor Q, torch::Tensor Kk, torch::Tensor Vv);"

_ext = load_inline(
    name="sdpa_unlim_v1",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["run"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q, K, V):
        return _ext.run(Q, K, V)


batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024

def get_inputs():
    Q = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    K = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    V = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    return [Q, K, V]

def get_init_inputs():
    return []
