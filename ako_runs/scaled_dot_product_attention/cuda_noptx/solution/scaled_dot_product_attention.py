import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Flash Attention FP32 kernel.
# BM=8 query rows per block, BN=8 KV rows per smem tile.
# Grid: (B*H, S/BM) = (1024, 64) = 65536 blocks
# Block: (32, BM) = (32, 8) = 256 threads
# smem: 2 * BN * HD * 4 = 2 * 8 * 1024 * 4 = 65536 bytes = 64 KB
# Correct but slower than PyTorch reference (fused avoids S*S DRAM, but
# re-reads K/V 64 times; net memory traffic >> reference's tensor-core path).

_FLASH_SRC = r"""
#include <cuda_runtime.h>
#include <float.h>

#define BM  8
#define BN  8
#define HD  1024

__global__ __launch_bounds__(BM*32, 4)
void sdpa_fwd(
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
    const int q       = q_tile * BM + wid;

    const long long bh_off = (long long)bh * S * HD;
    const float* Qb = Q + bh_off;
    const float* Kb = K + bh_off;
    const float* Vb = V + bh_off;
    float*       Ob = O + bh_off;

    const bool vq = (q < S);

    float qr[32];
    if (vq) {
        const float* qp = Qb + (long long)q * HD;
        #pragma unroll
        for (int i = 0; i < 32; i++)
            qr[i] = __ldg(qp + lane + i * 32);
    }

    float mi  = -FLT_MAX;
    float li  = 0.0f;
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;

    const int tid  = wid * 32 + lane;
    const int nthr = BM * 32;

    for (int n = 0; n < S; n += BN) {
        const int nn = (S - n < BN) ? (S - n) : BN;
        const float* Kt = Kb + (long long)n * HD;
        const float* Vt = Vb + (long long)n * HD;
        #pragma unroll 2
        for (int k = tid; k < nn * HD; k += nthr) {
            Ks[k] = Kt[k];
            Vs[k] = Vt[k];
        }
        __syncthreads();

        if (vq) {
            #pragma unroll 1
            for (int j = 0; j < nn; j++) {
                const float* kj = Ks + j * HD;
                const float* vj = Vs + j * HD;
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 32; i++)
                    dot = __fmaf_rn(qr[i], kj[lane + i * 32], dot);
                #pragma unroll
                for (int off = 16; off >= 1; off >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, off);
                dot *= scale;
                const float mn = (dot > mi) ? dot : mi;
                const float eO = __expf(mi - mn);
                const float eN = __expf(dot - mn);
                li = __fmaf_rn(li, eO, eN);
                mi = mn;
                #pragma unroll
                for (int i = 0; i < 32; i++)
                    acc[i] = __fmaf_rn(eN, vj[lane + i * 32], acc[i] * eO);
            }
        }
        __syncthreads();
    }

    if (vq) {
        const float inv_l = __frcp_rn(li);
        float* op = Ob + (long long)q * HD;
        #pragma unroll
        for (int i = 0; i < 32; i++)
            op[lane + i * 32] = acc[i] * inv_l;
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
    dim3 grid(B * Hh, (S + BM - 1) / BM);
    dim3 block(32, BM);
    const size_t smem = 2ULL * BN * HD * sizeof(float);
    cudaFuncSetAttribute(sdpa_fwd,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    sdpa_fwd<<<grid, block, smem>>>(
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
            name="sdpa_noptx_bm8bn8",
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
