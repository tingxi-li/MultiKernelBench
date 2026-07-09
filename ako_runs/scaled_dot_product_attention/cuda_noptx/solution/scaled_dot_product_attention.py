import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Test BM=16, BN=12 with 96KB smem (max opt-in on sm89).
# Grid: (1024, 32). Block: (32, 16) = 512 threads.
# smem = 2 * 12 * 1024 * 4 = 98304 = 96KB (< 99KB max).
# Fewer KV tiles to iterate: 512/12 = 43 tiles (vs 64 tiles for BN=8).
# float4 loads.

_FLASH_SRC = r"""
#include <cuda_runtime.h>
#include <float.h>

#define BM  16
#define BN  12
#define HD  1024

// smem = 2 * 12 * 1024 * 4 = 98304 bytes = 96 KB (< 99 KB max on sm89)

__global__ __launch_bounds__(BM*32, 1)
void sdpa_bm16bn12(
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
    const int nthr = BM * 32;  // 512

    for (int n = 0; n < S; n += BN) {
        const int nn = (S - n < BN) ? (S - n) : BN;

        // float4 load: nn * HD / 4 float4 elements, 512 threads
        const float4* Kt4 = (const float4*)(Kb + (long long)n * HD);
        const float4* Vt4 = (const float4*)(Vb + (long long)n * HD);
        float4* Ks4 = (float4*)Ks;
        float4* Vs4 = (float4*)Vs;
        const int n4 = nn * (HD / 4);
        #pragma unroll 4
        for (int k = tid; k < n4; k += nthr) {
            Ks4[k] = __ldg(Kt4 + k);
            Vs4[k] = __ldg(Vt4 + k);
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
    const size_t smem = 2ULL * BN * HD * sizeof(float);  // 98304
    cudaFuncSetAttribute(sdpa_bm16bn12,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    sdpa_bm16bn12<<<grid, block, smem>>>(
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
            name="sdpa_bm16bn12_f4",
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
