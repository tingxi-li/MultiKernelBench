import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 16
#define SPLITK 4

__device__ __forceinline__ float rtf(float x) {
    unsigned u = __float_as_uint(x);
    u = (u + 0x1000u) & 0xFFFFE000u;
    return __uint_as_float(u);
}
__device__ __forceinline__ unsigned ub(float x) { return __float_as_uint(x); }

__device__ __forceinline__ void mma_m16n8k8(float c[4],
        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
        unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Y(MxN) = X(MxK) * Wt(KxN)  on TF32 tensor cores (round-nearest + split-K).
__global__ __launch_bounds__(256) void gemm_mma(
        const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];
    const int tid = threadIdx.x;
    const int warpId = tid >> 5, lane = tid & 31;
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

    const int kChunk = K / SPLITK;
    const int kStart = blockIdx.z * kChunk, kStop = kStart + kChunk;
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
    load_tile(kStart, buf);
    __syncthreads();
    for (int k0 = kStart; k0 < kStop; k0 += BK) {
        const bool has = (k0 + BK) < kStop;
        if (has) load_tile(k0 + BK, buf ^ 1);
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
            atomicAdd(&C[(baseRow + group) * N + baseCol + 2 * tig],         acc[mi][ni][0]);
            atomicAdd(&C[(baseRow + group) * N + baseCol + 2 * tig + 1],     acc[mi][ni][1]);
            atomicAdd(&C[(baseRow + group + 8) * N + baseCol + 2 * tig],     acc[mi][ni][2]);
            atomicAdd(&C[(baseRow + group + 8) * N + baseCol + 2 * tig + 1], acc[mi][ni][3]);
        }
}

// per-row: out[m][n] = softmax_n( gelu( Y[m][n] + bias[n] ) ). One block / row.
#define ROWT 256
__global__ __launch_bounds__(ROWT) void bias_gelu_softmax(
        const float* __restrict__ Y, const float* __restrict__ bias,
        float* __restrict__ Out, int M, int N) {
    int m = blockIdx.x;
    int tid = threadIdx.x;
    const float* yrow = Y + (long)m * N;
    float* orow = Out + (long)m * N;

    float vmax = -1e30f;
    for (int n = tid; n < N; n += ROWT) {
        float v = yrow[n] + bias[n];
        v = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));   // exact gelu
        orow[n] = v;                                             // stash gelu val
        vmax = fmaxf(vmax, v);
    }
    __shared__ float red[ROWT];
    red[tid] = vmax; __syncthreads();
    for (int s = ROWT >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float rowMax = red[0];
    __syncthreads();

    float lsum = 0.0f;
    for (int n = tid; n < N; n += ROWT) {
        float e = __expf(orow[n] - rowMax);
        orow[n] = e;
        lsum += e;
    }
    red[tid] = lsum; __syncthreads();
    for (int s = ROWT >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int n = tid; n < N; n += ROWT) orow[n] *= inv;
}

torch::Tensor run(torch::Tensor X, torch::Tensor Wt, torch::Tensor bias) {
    TORCH_CHECK(X.is_cuda() && Wt.is_cuda(), "cuda only");
    auto Xc = X.contiguous();
    auto Wc = Wt.contiguous();
    int M = Xc.size(0), K = Xc.size(1), N = Wc.size(1);
    auto Y = torch::zeros({M, N}, Xc.options());
    dim3 grid(N / BN, M / BM, SPLITK);
    gemm_mma<<<grid, 256>>>(Xc.data_ptr<float>(), Wc.data_ptr<float>(),
                            Y.data_ptr<float>(), M, N, K);
    auto Out = torch::empty({M, N}, Xc.options());
    bias_gelu_softmax<<<M, ROWT>>>(Y.data_ptr<float>(), bias.contiguous().data_ptr<float>(),
                                   Out.data_ptr<float>(), M, N);
    return Out;
}
'''

_CPP = "torch::Tensor run(torch::Tensor X, torch::Tensor Wt, torch::Tensor bias);"

_ext = load_inline(
    name="mgs_unlim_v1",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["run"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("Wt", self.linear.weight.detach().t().contiguous())

    def forward(self, x):
        return _ext.run(x, self.Wt, self.linear.bias)


batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
