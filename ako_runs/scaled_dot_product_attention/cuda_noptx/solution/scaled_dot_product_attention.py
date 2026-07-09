import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# QW=4, BM=8 (256 threads): each thread needs 4*32+4*32=256 regs.
# With 256 threads/block: 65536/256=256 regs/thread budget.
# This is right at the limit — compiler may allocate 256 and avoid spilling.
# smem: 2*8*1024*4 = 64KB. Grid: (1024, 512/32) = (1024, 16).
# Each warp reads kval once, does 4 FMAs → 4x arithmetic intensity.
# BN=8: 128 tiles * 4 FMAs per K element = 512 FMAs per K smem read → efficient.

_FLASH_SRC = r"""
#include <cuda_runtime.h>
#include <float.h>

#define QW  4    // Q rows per warp
#define BM  8    // warps per block → 256 threads
#define BN  8    // KV rows per smem tile
#define HD  1024

// smem = 2 * BN * HD * 4 = 64 KB

__global__ __launch_bounds__(BM*32, 1)
void sdpa_qw4bm8(
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
    const bool vq1 = (q_base+1 < S);
    const bool vq2 = (q_base+2 < S);
    const bool vq3 = (q_base+3 < S);

    float qr0[32], qr1[32], qr2[32], qr3[32];
    if (vq0) {
        const float* p = Qb+(long long)q_base*HD;
        #pragma unroll
        for(int i=0;i<32;i++) qr0[i]=__ldg(p+lane+i*32);
    }
    if (vq1) {
        const float* p = Qb+(long long)(q_base+1)*HD;
        #pragma unroll
        for(int i=0;i<32;i++) qr1[i]=__ldg(p+lane+i*32);
    }
    if (vq2) {
        const float* p = Qb+(long long)(q_base+2)*HD;
        #pragma unroll
        for(int i=0;i<32;i++) qr2[i]=__ldg(p+lane+i*32);
    }
    if (vq3) {
        const float* p = Qb+(long long)(q_base+3)*HD;
        #pragma unroll
        for(int i=0;i<32;i++) qr3[i]=__ldg(p+lane+i*32);
    }

    float mi0=-FLT_MAX,mi1=-FLT_MAX,mi2=-FLT_MAX,mi3=-FLT_MAX;
    float li0=0,li1=0,li2=0,li3=0;
    float acc0[32],acc1[32],acc2[32],acc3[32];
    #pragma unroll
    for(int i=0;i<32;i++){acc0[i]=0;acc1[i]=0;acc2[i]=0;acc3[i]=0;}

    const int tid  = wid * 32 + lane;
    const int nthr = BM * 32;  // 256

    for (int n = 0; n < S; n += BN) {
        const int nn = (S - n < BN) ? (S - n) : BN;
        const float4* Kt4 = (const float4*)(Kb + (long long)n * HD);
        const float4* Vt4 = (const float4*)(Vb + (long long)n * HD);
        float4* Ks4 = (float4*)Ks;
        float4* Vs4 = (float4*)Vs;
        const int n4 = nn * (HD / 4);
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

            float d0=0,d1=0,d2=0,d3=0;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                float kv = kj[lane + i * 32];
                d0 = __fmaf_rn(qr0[i], kv, d0);
                d1 = __fmaf_rn(qr1[i], kv, d1);
                d2 = __fmaf_rn(qr2[i], kv, d2);
                d3 = __fmaf_rn(qr3[i], kv, d3);
            }
            #pragma unroll
            for (int off=16;off>=1;off>>=1) {
                d0+=__shfl_xor_sync(0xffffffff,d0,off);
                d1+=__shfl_xor_sync(0xffffffff,d1,off);
                d2+=__shfl_xor_sync(0xffffffff,d2,off);
                d3+=__shfl_xor_sync(0xffffffff,d3,off);
            }
            d0*=scale; d1*=scale; d2*=scale; d3*=scale;

            if (vq0) {
                float mn=(d0>mi0)?d0:mi0; float eO=__expf(mi0-mn); float eN=__expf(d0-mn);
                li0=__fmaf_rn(li0,eO,eN); mi0=mn;
                #pragma unroll
                for(int i=0;i<32;i++) acc0[i]=__fmaf_rn(eN,vj[lane+i*32],acc0[i]*eO);
            }
            if (vq1) {
                float mn=(d1>mi1)?d1:mi1; float eO=__expf(mi1-mn); float eN=__expf(d1-mn);
                li1=__fmaf_rn(li1,eO,eN); mi1=mn;
                #pragma unroll
                for(int i=0;i<32;i++) acc1[i]=__fmaf_rn(eN,vj[lane+i*32],acc1[i]*eO);
            }
            if (vq2) {
                float mn=(d2>mi2)?d2:mi2; float eO=__expf(mi2-mn); float eN=__expf(d2-mn);
                li2=__fmaf_rn(li2,eO,eN); mi2=mn;
                #pragma unroll
                for(int i=0;i<32;i++) acc2[i]=__fmaf_rn(eN,vj[lane+i*32],acc2[i]*eO);
            }
            if (vq3) {
                float mn=(d3>mi3)?d3:mi3; float eO=__expf(mi3-mn); float eN=__expf(d3-mn);
                li3=__fmaf_rn(li3,eO,eN); mi3=mn;
                #pragma unroll
                for(int i=0;i<32;i++) acc3[i]=__fmaf_rn(eN,vj[lane+i*32],acc3[i]*eO);
            }
        }
        __syncthreads();
    }

    auto write_out = [&](bool valid, float li_v, const float* acc_v, int qr) {
        if (valid) {
            const float inv = __frcp_rn(li_v);
            float* op = Ob + (long long)(q_base + qr) * HD;
            for (int i = 0; i < 32; i++) op[lane + i * 32] = acc_v[i] * inv;
        }
    };
    write_out(vq0, li0, acc0, 0);
    write_out(vq1, li1, acc1, 1);
    write_out(vq2, li2, acc2, 2);
    write_out(vq3, li3, acc3, 3);
}

torch::Tensor flash_attn_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V
) {
    const int B  = Q.size(0);
    const int Hh = Q.size(1);
    const int S  = Q.size(2);
    auto O = torch::empty_like(Q);
    const float scale = 1.0f / sqrtf((float)HD);
    const int qpb = QW * BM;  // 32
    dim3 grid(B * Hh, (S + qpb - 1) / qpb);
    dim3 block(32, BM);
    const size_t smem = 2ULL * BN * HD * sizeof(float);  // 64 KB
    cudaFuncSetAttribute(sdpa_qw4bm8,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    sdpa_qw4bm8<<<grid, block, smem>>>(
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
            name="sdpa_qw4bm8bn8",
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
