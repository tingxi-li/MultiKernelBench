import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# QW=2 rows per warp, BM=8 warps (256 threads): each thread can use 256 regs.
# Block processes 16 Q rows. Grid: (1024, S/16) = (1024, 32).
# smem: 2*BN*HD*4 = 2*8*1024*4 = 64KB. 2 blocks can fit: 2*256=512 threads, 2*64=128KB>99KB → 1 block/SM.
# With 256 threads per block and 1024 blocks: 13 waves (76 blocks/wave).
# Per thread: 2*32 Q regs + 2*32 acc regs + ~8 scalars = 136 regs → OK (below 256 limit).

_FLASH_SRC = r"""
#include <cuda_runtime.h>
#include <float.h>

#define QW  2    // Q rows per warp
#define BM  8    // warps per block
#define BN  8    // KV rows per smem tile
#define HD  1024

// smem = 2 * BN * HD * 4 = 64 KB

__global__ __launch_bounds__(BM*32, 1)
void sdpa_qw2bm8(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__       O,
    int S,
    float scale
) {
    extern __shared__ float smem[];
    float* __restrict__ Ks = smem;
    float* __restrict__ Vs = smem + BN * HD;

    const int lane    = threadIdx.x;
    const int wid     = threadIdx.y;
    const int bh      = blockIdx.x;
    const int q_tile  = blockIdx.y;
    const int q_base  = q_tile * (QW * BM) + wid * QW;

    const long long bh_off = (long long)bh * S * HD;
    const float* Qb = Q + bh_off;
    const float* Kb = K + bh_off;
    const float* Vb = V + bh_off;
    float*       Ob = O + bh_off;

    const bool vq0 = (q_base < S);
    const bool vq1 = (q_base + 1 < S);

    float qr0[32], qr1[32];
    if (vq0) {
        const float* qp = Qb + (long long)q_base * HD;
        #pragma unroll
        for (int i = 0; i < 32; i++) qr0[i] = __ldg(qp + lane + i * 32);
    }
    if (vq1) {
        const float* qp = Qb + (long long)(q_base + 1) * HD;
        #pragma unroll
        for (int i = 0; i < 32; i++) qr1[i] = __ldg(qp + lane + i * 32);
    }

    float mi0 = -FLT_MAX, mi1 = -FLT_MAX;
    float li0 = 0.0f,     li1 = 0.0f;
    float acc0[32], acc1[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) { acc0[i] = 0.0f; acc1[i] = 0.0f; }

    const int tid  = wid * 32 + lane;
    const int nthr = BM * 32;  // 256

    for (int n = 0; n < S; n += BN) {
        const int nn = (S - n < BN) ? (S - n) : BN;

        const float4* Kt4 = (const float4*)(Kb + (long long)n * HD);
        const float4* Vt4 = (const float4*)(Vb + (long long)n * HD);
        float4* Ks4 = (float4*)Ks;
        float4* Vs4 = (float4*)Vs;
        const int n4 = nn * (HD / 4);  // 8*256 = 2048 float4s; 256 threads → 8 each
        #pragma unroll 8
        for (int k = tid; k < n4; k += nthr) {
            Ks4[k] = __ldg(Kt4 + k);
            Vs4[k] = __ldg(Vt4 + k);
        }
        __syncthreads();

        #pragma unroll 1
        for (int j = 0; j < nn; j++) {
            const float* kj = Ks + j * HD;
            const float* vj = Vs + j * HD;

            float dot0 = 0.0f, dot1 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                float kval = kj[lane + i * 32];
                if (vq0) dot0 = __fmaf_rn(qr0[i], kval, dot0);
                if (vq1) dot1 = __fmaf_rn(qr1[i], kval, dot1);
            }
            #pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                dot0 += __shfl_xor_sync(0xffffffff, dot0, off);
                dot1 += __shfl_xor_sync(0xffffffff, dot1, off);
            }
            dot0 *= scale;
            dot1 *= scale;

            if (vq0) {
                const float mn = (dot0 > mi0) ? dot0 : mi0;
                const float eO = __expf(mi0 - mn);
                const float eN = __expf(dot0 - mn);
                li0 = __fmaf_rn(li0, eO, eN);
                mi0 = mn;
                #pragma unroll
                for (int i = 0; i < 32; i++)
                    acc0[i] = __fmaf_rn(eN, vj[lane + i * 32], acc0[i] * eO);
            }
            if (vq1) {
                const float mn = (dot1 > mi1) ? dot1 : mi1;
                const float eO = __expf(mi1 - mn);
                const float eN = __expf(dot1 - mn);
                li1 = __fmaf_rn(li1, eO, eN);
                mi1 = mn;
                #pragma unroll
                for (int i = 0; i < 32; i++)
                    acc1[i] = __fmaf_rn(eN, vj[lane + i * 32], acc1[i] * eO);
            }
        }
        __syncthreads();
    }

    if (vq0) {
        const float inv = __frcp_rn(li0);
        float* op = Ob + (long long)q_base * HD;
        #pragma unroll
        for (int i = 0; i < 32; i++) op[lane + i * 32] = acc0[i] * inv;
    }
    if (vq1) {
        const float inv = __frcp_rn(li1);
        float* op = Ob + (long long)(q_base + 1) * HD;
        #pragma unroll
        for (int i = 0; i < 32; i++) op[lane + i * 32] = acc1[i] * inv;
    }
}

torch::Tensor flash_attn_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V
) {
    const int B  = Q.size(0);
    const int Hh = Q.size(1);
    const int S  = Q.size(2);
    auto O = torch::empty_like(Q);
    const float scale = 1.0f / sqrtf((float)HD);
    const int qpb = QW * BM;  // Q rows per block = 16
    dim3 grid(B * Hh, (S + qpb - 1) / qpb);
    dim3 block(32, BM);
    const size_t smem = 2ULL * BN * HD * sizeof(float);  // 64 KB
    cudaFuncSetAttribute(sdpa_qw2bm8,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    sdpa_qw2bm8<<<grid, block, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), S, scale);
    return O;
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="sdpa_qw2bm8bn8",
            cpp_sources=(
                "torch::Tensor flash_attn_cuda("
                "torch::Tensor Q, torch::Tensor K, torch::Tensor V);"
            ),
            cuda_sources=_FLASH_SRC,
            functions=["flash_attn_cuda"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
        )
    return _module


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _get_module().flash_attn_cuda(
            Q.contiguous(), K.contiguous(), V.contiguous()
        )
