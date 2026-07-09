import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 32

__device__ __forceinline__ unsigned rtf_bits(float x) {   // round-to-nearest tf32 bits
    unsigned u = __float_as_uint(x);
    u = (u + 0x1000u) & 0xFFFFE000u;
    return u;
}

__device__ __forceinline__ void mma_m16n8k8(float c[4],
        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
        unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// C(MxN)=A(MxK)*B(KxN) fp32 via inline-PTX mma.sync TF32 tensor cores. Inputs
// rounded to nearest tf32 in registers (exact bits fed to the mma) -> 1xTF32.
// 128 threads (2x2 warps) per 64x64 block; each warp owns 32x32 (2 m16 x 4 n8).
__global__ __launch_bounds__(128) void gemm_mma(
        const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM * BK];   // [64][32]
    __shared__ float Bs[BK * BN];   // [32][64]

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;     // 0..7
    const int tig = lane & 3;        // 0..3
    const int warpM = warpId >> 1;   // 0..1
    const int warpN = warpId & 1;    // 0..1
    const int mOrigin = warpM * 32;
    const int nOrigin = warpN * 32;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int t = tid; t < (BM * BK) / 4; t += 128) {
            int row = (t * 4) / BK, col = (t * 4) % BK;
            *reinterpret_cast<float4*>(&As[row * BK + col]) =
                *reinterpret_cast<const float4*>(&A[(blockRow + row) * K + k0 + col]);
        }
        #pragma unroll
        for (int t = tid; t < (BK * BN) / 4; t += 128) {
            int row = (t * 4) / BN, col = (t * 4) % BN;
            *reinterpret_cast<float4*>(&Bs[row * BN + col]) =
                *reinterpret_cast<const float4*>(&B[(k0 + row) * N + blockCol + col]);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK / 8; ++kk) {
            int kBase = kk * 8;
            unsigned a[2][4], b[4][2];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int r0 = mOrigin + mi * 16 + group;
                a[mi][0] = rtf_bits(As[r0 * BK + kBase + tig]);
                a[mi][1] = rtf_bits(As[(r0 + 8) * BK + kBase + tig]);
                a[mi][2] = rtf_bits(As[r0 * BK + kBase + tig + 4]);
                a[mi][3] = rtf_bits(As[(r0 + 8) * BK + kBase + tig + 4]);
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int c0 = nOrigin + ni * 8 + group;
                b[ni][0] = rtf_bits(Bs[(kBase + tig) * BN + c0]);
                b[ni][1] = rtf_bits(Bs[(kBase + tig + 4) * BN + c0]);
            }
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma_m16n8k8(acc[mi][ni], a[mi][0], a[mi][1], a[mi][2], a[mi][3],
                                b[ni][0], b[ni][1]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int baseRow = blockRow + mOrigin + mi * 16;
            int baseCol = blockCol + nOrigin + ni * 8;
            C[(baseRow + group) * N + baseCol + 2 * tig]     = acc[mi][ni][0];
            C[(baseRow + group) * N + baseCol + 2 * tig + 1] = acc[mi][ni][1];
            C[(baseRow + group + 8) * N + baseCol + 2 * tig]     = acc[mi][ni][2];
            C[(baseRow + group + 8) * N + baseCol + 2 * tig + 1] = acc[mi][ni][3];
        }
}

torch::Tensor run(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "cuda only");
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    auto C = torch::empty({M, N}, Ac.options());
    dim3 grid(N / BN, M / BM);
    dim3 block(128);
    gemm_mma<<<grid, block>>>(Ac.data_ptr<float>(), Bc.data_ptr<float>(),
                              C.data_ptr<float>(), M, N, K);
    return C;
}
'''

_CPP = "torch::Tensor run(torch::Tensor A, torch::Tensor B);"

_ext = load_inline(
    name="gemm_mma_1xtf32_v5",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["run"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _ext.run(A, B)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []
