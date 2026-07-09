import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Flash Attention 2 for D=1024
# Strategy: 1 warp per Q row; warp-parallel D reduction for QK^T;
#           register accumulator for O (split D across lanes)
# D=1024, 32 lanes => 32 float4 (128 elements) per lane for O accumulation
# Bc: KV tile width loaded cooperatively by all warps in block

_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// D=1024, each lane owns D/WARP_SIZE = 32 floats of each Q row and O row
// Use float4 loads: 8 float4 per lane (8*4=32 floats)
// Bc=32 KV rows per tile (fits Bc*D*2 = 32*1024*2 = 64KB fp16 in smem)
// Warps per block = 8, so Br=8 (one Q row per warp)
// Grid: (B*H, ceil(N/Br))

// smem: K_smem[Bc][D] fp16 = 32*1024*2 = 64KB, V_smem[Bc][D] fp16 = 64KB
// 128KB total -- too large. Need D-tiling again.
// With DC=256: K_smem[Bc][DC] + V_smem[Bc][DC] = 2*32*256*2 = 32KB. Fine.
// But then QK^T dot product needs D/DC=4 loops

// Actually: for Q row i, K row j: dot = sum_{d=0}^{D-1} Q[i,d] * K[j,d]
// With 1 warp per Q row, 32 lanes, each handles D/32=32 D-elements
// dot partial = sum_{d=lane*32}^{(lane+1)*32-1} Q[i,d] * K[j,d]
// Then warp_reduce_sum(dot_partial) -> full dot

// This works with global memory loads if we cache Q row and K/V rows in smem
// Q row: D=1024 floats = 4KB (cache for the whole block's lifetime for that Q row)
// K/V: Bc rows each, we load Bc*D per KV tile
// smem(Q rows for block): Br * D * 4 = 8 * 1024 * 4 = 32KB
// smem(K tile): Bc * D * 4 = 32 * 1024 * 4 = 128KB -- too large

// Solution: Bc=8 (1 K row per warp per "mini-tile"), D in registers
// Or: use fp16 for K,V smem
// K_smem[Bc][D] fp16: 16 * 1024 * 2 = 32KB. Fine!
// V_smem[Bc][D] fp16: 16 * 1024 * 2 = 32KB
// Q_smem[Br][D] fp32: 8 * 1024 * 4 = 32KB
// Total: 96KB -- tight but might work (RTX 6000 Ada has 99KB configurable smem)

// Grid: (B*H, ceil(N/Br)), Block: (Br * 32) = 8 warps * 32 = 256 threads
// Br=8 Q rows per block, Bc=16 KV rows per tile (to keep smem under 99KB)

__global__ void fa2_regacc_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int N, int D, float scale
) {
    const int Br = 8, Bc = 16;
    const int WARP_SIZE = 32;
    const int num_warps = 8;  // = Br

    int bh     = blockIdx.x;
    int q_tile = blockIdx.y;
    int warp   = threadIdx.x / WARP_SIZE;  // 0..7 = which Q row in this block
    int lane   = threadIdx.x % WARP_SIZE;  // 0..31

    int q_row = q_tile * Br + warp;
    if (q_row >= N) return;

    const float* Qbase = Q + bh * N * D;
    const float* Kbase = K + bh * N * D;
    const float* Vbase = V + bh * N * D;
    float*       Obase = O + bh * N * D;

    // smem layout (sizes in elements):
    // Q_smem[Br][D] fp32   = 8 * 1024 = 8192 float  = 32768 bytes
    // K_smem[Bc][D] fp16   = 16 * 1024 = 16384 half = 32768 bytes
    // V_smem[Bc][D] fp16   = 16 * 1024 = 16384 half = 32768 bytes
    // scores[Br][Bc] float = 8 * 16 = 128 float     = 512 bytes
    // Total: 98048 bytes < 99KB
    extern __shared__ char smem_raw[];
    float*  Qs = (float*)smem_raw;              // [Br][D]
    __half* Ks = (__half*)(Qs + Br * D);        // [Bc][D]
    __half* Vs = Ks + Bc * D;                   // [Bc][D]
    float*  Ss = (float*)(Vs + Bc * D);         // [Br][Bc]

    // Load Q row for this warp into smem cooperatively
    // All 32 lanes of this warp load Q[q_row][lane..lane+D/32..]
    // D=1024, 32 lanes -> 32 floats each
    {
        int base = warp * D;
        for (int i = lane; i < D; i += WARP_SIZE) {
            Qs[base + i] = Qbase[q_row * D + i];
        }
    }
    __syncthreads();

    // Per-warp accumulators: O[q_row][lane*32..(lane+1)*32]
    // D/32 = 32 floats per lane
    const int epl = D / WARP_SIZE;  // = 32
    float o_acc[32];  // epl = 32
    for (int i = 0; i < epl; i++) o_acc[i] = 0.f;
    float wmax = -FLT_MAX, wsum = 0.f;

    // KV tile loop
    for (int kv0 = 0; kv0 < N; kv0 += Bc) {
        int kv_end = min(kv0 + Bc, N);
        int aBc = kv_end - kv0;

        // Cooperatively load K tile [kv0..+Bc, 0..D] into K_smem (fp16)
        // 256 threads load Bc*D = 16*1024 = 16384 half elements, 64 per thread
        int total_kv = Bc * D;
        for (int i = threadIdx.x; i < total_kv; i += 256) {
            int r = i / D, c = i % D;
            int gr = kv0 + r;
            Ks[i] = (gr < kv_end) ? __float2half(Kbase[gr * D + c]) : __float2half(0.f);
        }
        __syncthreads();

        // Compute scores[warp][0..aBc]: dot(Q[q_row], K[kv0+j]) for j in 0..aBc
        for (int j = 0; j < aBc; j++) {
            // dot product: D elements, 32 lanes
            float dot = 0.f;
            const float* qrow = Qs + warp * D;
            const __half* krow = Ks + j * D;
            for (int d = lane; d < D; d += WARP_SIZE) {
                dot += qrow[d] * __half2float(krow[d]);
            }
            // warp reduce
            for (int off = 16; off >= 1; off >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, off);
            if (lane == 0) Ss[warp * Bc + j] = dot * scale;
        }
        for (int j = aBc; j < Bc; j++) if (lane == 0) Ss[warp * Bc + j] = -FLT_MAX;
        __syncwarp();

        // Online softmax for this warp's Q row
        float tile_max = -FLT_MAX;
        for (int j = 0; j < aBc; j++) tile_max = fmaxf(tile_max, Ss[warp * Bc + j]);
        float new_max = fmaxf(wmax, tile_max);
        float alpha = expf(wmax - new_max);

        float p[16];  // Bc=16
        float tile_sum = 0.f;
        for (int j = 0; j < Bc; j++) {
            float pj = (j < aBc) ? expf(Ss[warp * Bc + j] - new_max) : 0.f;
            p[j] = pj;
            tile_sum += pj;
        }

        // Rescale O accumulator
        for (int i = 0; i < epl; i++) o_acc[i] *= alpha;
        wsum = wsum * alpha + tile_sum;
        wmax = new_max;

        // Load V tile (cooperatively, reuse smem)
        for (int i = threadIdx.x; i < total_kv; i += 256) {
            int r = i / D, c = i % D;
            int gr = kv0 + r;
            Vs[i] = (gr < kv_end) ? __float2half(Vbase[gr * D + c]) : __float2half(0.f);
        }
        __syncthreads();

        // O update: o_acc[lane*epl..] += sum_j p[j] * V[kv0+j][lane*epl..]
        for (int j = 0; j < aBc; j++) {
            if (p[j] == 0.f) continue;
            const __half* vrow = Vs + j * D;
            for (int i = 0; i < epl; i++) {
                o_acc[i] += p[j] * __half2float(vrow[lane * epl + i]);
            }
        }
        __syncthreads();
    }

    // Normalize and write output
    float inv = 1.f / wsum;
    const float* qbase_row = Obase + q_row * D;
    for (int i = 0; i < epl; i++) {
        Obase[q_row * D + lane * epl + i] = o_acc[i] * inv;
    }
}

torch::Tensor fa2_regacc(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    TORCH_CHECK(D == 1024, "Expected D=1024, got D=", D);
    float scale = 1.f / sqrtf((float)D);

    auto Qr = Q.reshape({B*H, N, D}).contiguous();
    auto Kr = K.reshape({B*H, N, D}).contiguous();
    auto Vr = V.reshape({B*H, N, D}).contiguous();
    auto Or = torch::empty({B*H, N, D}, Q.options());

    const int Br = 8, Bc = 16;
    // smem: Q[Br][D]*4 + K[Bc][D]*2 + V[Bc][D]*2 + S[Br][Bc]*4
    size_t smem = (size_t)Br * D * sizeof(float)
                + (size_t)Bc * D * sizeof(__half)
                + (size_t)Bc * D * sizeof(__half)
                + (size_t)Br * Bc * sizeof(float);
    // = 8*1024*4 + 16*1024*2 + 16*1024*2 + 8*16*4
    // = 32768 + 32768 + 32768 + 512 = 98816 bytes (< 99KB)

    dim3 block(Br * 32);  // 256 threads
    dim3 grid(B * H, (N + Br - 1) / Br);

    cudaFuncSetAttribute(fa2_regacc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99328);

    fa2_regacc_kernel<<<grid, block, smem>>>(
        Qr.data_ptr<float>(), Kr.data_ptr<float>(), Vr.data_ptr<float>(),
        Or.data_ptr<float>(), N, D, scale
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA: ", cudaGetErrorString(err));

    return Or.reshape({B, H, N, D});
}
"""

_cpp = "torch::Tensor fa2_regacc(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"

_mod = load_inline(
    name="fa2_regacc_v4",
    cpp_sources=_cpp,
    cuda_sources=_src,
    functions=["fa2_regacc"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89", "--ptxas-options=-v"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _mod.fa2_regacc(Q.contiguous(), K.contiguous(), V.contiguous())
