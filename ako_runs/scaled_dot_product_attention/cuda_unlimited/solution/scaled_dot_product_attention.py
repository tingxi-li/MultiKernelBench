import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Flash Attention 2 D=1024: O stored in smem, FULLY parallel rescale/PV update
# Br=8 Q rows/block, Bc=16 KV rows/tile
# smem: K[Bc][D]fp16 + V[Bc][D]fp16 + O[Br][D]fp32 = 64+64+128KB... too large
#
# Fix: Bc=8, Br=8: K[8][D]fp16=16KB + V[8][D]fp16=16KB + O[8][D]fp32=32KB = 64KB  OK
#
# Block: 256 threads (8 warps). Each warp handles one Q row's dot products.
# Cooperative tasks: K/V loading and O rescale/update use ALL 256 threads.

_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32

// Configurable tile sizes
// Br=8, Bc=8: smem = (8+8)*1024*2 + 8*1024*4 = 32768 + 32768 = 65536 = 64KB
// Br=8, Bc=16: smem = (16+16)*1024*2 + 8*1024*4 = 65536 + 32768 = 98304 = 96KB (fits!)
#define Br  8
#define Bc  16
#define D_  1024
#define EPL 32  // D/WARP_SIZE per lane

__global__ __launch_bounds__(256, 1)
void fa2_smemO_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int N, int D, float scale
) {
    int bh     = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid    = threadIdx.x;
    int warp   = tid / WARP_SIZE;  // 0..Br-1
    int lane   = tid % WARP_SIZE;  // 0..31

    int q_start = q_tile * Br;
    if (q_start >= N) return;
    int q_end = min(q_start + Br, N);
    int aBr = q_end - q_start;

    const float* Qbase = Q + bh * N * D;
    const float* Kbase = K + bh * N * D;
    const float* Vbase = V + bh * N * D;
    float*       Obase = O + bh * N * D;

    // smem layout (Bc=16):
    // K_smem[Bc][D] fp16  = 16*1024*2 = 32768 bytes
    // V_smem[Bc][D] fp16  = 16*1024*2 = 32768 bytes
    // O_smem[Br][D] fp32  = 8*1024*4  = 32768 bytes
    // Total: 98304 = 96KB
    extern __shared__ char smem_raw[];
    __half* Ks = (__half*)smem_raw;                  // [Bc][D]
    __half* Vs = Ks + Bc * D;                        // [Bc][D]
    float*  Os = (float*)(Vs + Bc * D);              // [Br][D]

    // Initialize O_smem to 0 (all 256 threads)
    for (int i = tid; i < Br * D; i += 256) Os[i] = 0.f;

    // Load Q row for this warp into registers (warp-local)
    float q_frag[EPL];
    {
        int q_row = q_start + warp;
        if (q_row < q_end) {
            const float* qrow = Qbase + q_row * D;
            #pragma unroll
            for (int i = 0; i < EPL; i++) q_frag[i] = qrow[lane * EPL + i];
        } else {
            #pragma unroll
            for (int i = 0; i < EPL; i++) q_frag[i] = 0.f;
        }
    }
    __syncthreads();

    // Per-warp softmax state (all in registers)
    float wmax = -FLT_MAX, wsum = 0.f;

    // KV tile loop
    for (int kv0 = 0; kv0 < N; kv0 += Bc) {
        int kv_end = min(kv0 + Bc, N);
        int aBc = kv_end - kv0;

        // Load K tile cooperatively (256 threads, Bc*D=16384 half elems, 64 per thread)
        for (int i = tid; i < Bc * D; i += 256) {
            int r = i / D, c = i % D;
            int gr = kv0 + r;
            Ks[i] = (gr < kv_end) ? __float2half(Kbase[gr * D + c]) : __float2half(0.f);
        }
        __syncthreads();

        // Each warp computes scores for its Q row
        float scores[Bc];
        {
            int q_row = q_start + warp;
            if (q_row < q_end) {
                for (int j = 0; j < aBc; j++) {
                    const __half* krow = Ks + j * D + lane * EPL;
                    float dot = 0.f;
                    #pragma unroll
                    for (int i = 0; i < EPL; i++)
                        dot += q_frag[i] * __half2float(krow[i]);
                    #pragma unroll
                    for (int off = 16; off >= 1; off >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, off);
                    scores[j] = dot * scale;
                }
            }
        }
        for (int j = aBc; j < Bc; j++) scores[j] = -FLT_MAX;

        // Online softmax (per-warp, in registers)
        float tile_max = scores[0];
        for (int j = 1; j < Bc; j++) tile_max = fmaxf(tile_max, scores[j]);
        float new_max = fmaxf(wmax, tile_max);
        float alpha = expf(wmax - new_max);

        float p[Bc];
        float tile_sum = 0.f;
        for (int j = 0; j < Bc; j++) {
            p[j] = (j < aBc) ? expf(scores[j] - new_max) : 0.f;
            tile_sum += p[j];
        }
        wsum = wsum * alpha + tile_sum;
        wmax = new_max;

        // Rescale O_smem[warp][0..D] by alpha (warp-parallel)
        // Only do if alpha != 1 (i.e., wmax changed)
        if (alpha != 1.f) {
            float* orow = Os + warp * D + lane * EPL;
            #pragma unroll
            for (int i = 0; i < EPL; i++) orow[i] *= alpha;
        }
        // No __syncthreads needed here since each warp only writes its own O rows

        // Load V tile
        for (int i = tid; i < Bc * D; i += 256) {
            int r = i / D, c = i % D;
            int gr = kv0 + r;
            Vs[i] = (gr < kv_end) ? __float2half(Vbase[gr * D + c]) : __float2half(0.f);
        }
        __syncthreads();

        // O update: o[warp][lane*EPL+i] += sum_j p[j] * V[j][lane*EPL+i]
        {
            float* orow = Os + warp * D + lane * EPL;
            for (int j = 0; j < aBc; j++) {
                if (p[j] == 0.f) continue;
                const __half* vrow = Vs + j * D + lane * EPL;
                #pragma unroll
                for (int i = 0; i < EPL; i++)
                    orow[i] += p[j] * __half2float(vrow[i]);
            }
        }
        __syncthreads();
    }

    // Normalize and write to global
    {
        int q_row = q_start + warp;
        if (q_row < q_end && wsum > 0.f) {
            float inv = 1.f / wsum;
            float* orow = Os + warp * D + lane * EPL;
            float* gout = Obase + q_row * D + lane * EPL;
            #pragma unroll
            for (int i = 0; i < EPL; i++) gout[i] = orow[i] * inv;
        }
    }
}

torch::Tensor fa2_smemO(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    TORCH_CHECK(D == 1024, "Expected D=1024");
    float scale = 1.f / sqrtf((float)D);

    auto Qr = Q.reshape({B*H, N, D}).contiguous();
    auto Kr = K.reshape({B*H, N, D}).contiguous();
    auto Vr = V.reshape({B*H, N, D}).contiguous();
    auto Or = torch::empty({B*H, N, D}, Q.options());

    // smem: K[Bc][D]fp16 + V[Bc][D]fp16 + O[Br][D]fp32
    size_t smem = (size_t)2 * Bc * D * sizeof(__half) + (size_t)Br * D * sizeof(float);
    // = 2*16*1024*2 + 8*1024*4 = 65536 + 32768 = 98304 bytes = 96KB

    dim3 block(Br * WARP_SIZE);  // 256 threads
    dim3 grid(B * H, (N + Br - 1) / Br);

    cudaFuncSetAttribute(fa2_smemO_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99328);

    fa2_smemO_kernel<<<grid, block, smem>>>(
        Qr.data_ptr<float>(), Kr.data_ptr<float>(), Vr.data_ptr<float>(),
        Or.data_ptr<float>(), N, D, scale
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA: ", cudaGetErrorString(err));

    return Or.reshape({B, H, N, D});
}
"""

_cpp = "torch::Tensor fa2_smemO(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"

_mod = load_inline(
    name="fa2_smemO_v9",
    cpp_sources=_cpp,
    cuda_sources=_src,
    functions=["fa2_smemO"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _mod.fa2_smemO(Q.contiguous(), K.contiguous(), V.contiguous())
