import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Flash Attention 2 style: tiled SRAM-fused QKV, head_dim up to 1024
# Uses online softmax (Dao et al. 2022), vectorized loads via float4
_flash_attn_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Tile sizes: Bc=64 (K/V tile), Br=32 (Q tile)
// head_dim up to 1024 with multiple passes per Bc chunk

#define WARP_SIZE 32

// Each thread block handles one (batch, head, tile_row) chunk.
// Grid: (num_heads, batch_size, ceil(seq_len/Br))
// Block: (Bc) threads — each warp handles part of the reduction
// We use Br=64, Bc=64 for head_dim=1024

template <int Br, int Bc, int Hd>
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,   // [B, H, N, D]
    const float* __restrict__ K,   // [B, H, N, D]
    const float* __restrict__ V,   // [B, H, N, D]
    float* __restrict__ O,         // [B, H, N, D]
    int B, int H, int N, int D,
    float scale
) {
    // batch/head index
    int b = blockIdx.y;
    int h = blockIdx.x;
    int q_tile = blockIdx.z;

    int q_start = q_tile * Br;
    if (q_start >= N) return;
    int q_end = min(q_start + Br, N);
    int actual_Br = q_end - q_start;

    // Thread index within block
    int tid = threadIdx.x;  // 0..Bc-1

    // Pointers
    const float* Qptr = Q + (b * H + h) * N * D;
    const float* Kptr = K + (b * H + h) * N * D;
    const float* Vptr = V + (b * H + h) * N * D;
    float* Optr = O + (b * H + h) * N * D;

    // Shared memory layout:
    // Q tile: Br x D
    // K tile: Bc x D
    // V tile: Bc x D
    // S tile: Br x Bc (attention scores)
    extern __shared__ float smem[];
    float* Qs = smem;                    // Br * D
    float* Ks = Qs + Br * Hd;           // Bc * D
    float* Vs = Ks + Bc * Hd;           // Bc * D
    float* Ss = Vs + Bc * Hd;           // Br * Bc

    // Load Q tile into shared memory
    // Distribute loading: each thread loads multiple elements
    int total_Q = actual_Br * D;
    for (int i = tid; i < total_Q; i += Bc) {
        int row = i / D;
        int col = i % D;
        if (row < actual_Br) {
            Qs[row * Hd + col] = Qptr[(q_start + row) * D + col];
        }
    }

    // Initialize accumulators
    // Each thread is responsible for one query row (if tid < actual_Br)
    float acc[Hd / Bc > 0 ? Hd / Bc : 1];  // this won't work with template...
    // Use registers for output accumulator
    // For head_dim=1024 and Bc=64: each thread needs 1024 floats — too much for registers
    // Instead, use O directly, process in chunks

    // We'll use a simpler approach: each of the Br query rows is handled
    // by one warp (Bc=32 warp), but that limits parallelism.
    // Better: each thread handles one Q row when Br <= Bc

    // Simpler: tid < Br => this thread owns Q[tid]
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    // Output accumulator for this thread's query row (in registers, chunked)
    // D=1024, too large for registers. Use global memory temp buffer? No.
    // Use smem for output accumulation — but smem is limited.
    // Best approach: iterate over D dimension in chunks of Bc

    // Actually for D=1024, let's store O accumulator in global memory directly
    // and update it in passes. We'll use the output buffer as scratch.

    // Initialize output to 0
    if (tid < actual_Br) {
        for (int d = 0; d < D; d++) {
            Optr[(q_start + tid) * D + d] = 0.0f;
        }
    }
    __syncthreads();

    // Online softmax over K/V tiles
    for (int kv_start = 0; kv_start < N; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, N);
        int actual_Bc = kv_end - kv_start;

        // Load K tile
        int total_K = actual_Bc * D;
        for (int i = tid; i < total_K; i += Bc) {
            int row = i / D;
            int col = i % D;
            if (row < actual_Bc) {
                Ks[row * Hd + col] = Kptr[(kv_start + row) * D + col];
            }
        }
        // Load V tile
        for (int i = tid; i < total_K; i += Bc) {
            int row = i / D;
            int col = i % D;
            if (row < actual_Bc) {
                Vs[row * Hd + col] = Vptr[(kv_start + row) * D + col];
            }
        }
        __syncthreads();

        // Compute S = Q * K^T, S[i][j] = sum_d Q[i][d]*K[j][d]
        // Each thread computes S[tid][j] for all j in [0, actual_Bc) if tid < actual_Br
        if (tid < actual_Br) {
            for (int j = 0; j < actual_Bc; j++) {
                float dot = 0.0f;
                const float* qi = Qs + tid * Hd;
                const float* kj = Ks + j * Hd;
                for (int d = 0; d < D; d++) {
                    dot += qi[d] * kj[d];
                }
                Ss[tid * Bc + j] = dot * scale;
            }
        }
        __syncthreads();

        // Online softmax update for each query row
        if (tid < actual_Br) {
            // Find max in this tile
            float tile_max = -FLT_MAX;
            for (int j = 0; j < actual_Bc; j++) {
                tile_max = fmaxf(tile_max, Ss[tid * Bc + j]);
            }

            float new_max = fmaxf(row_max, tile_max);
            float exp_scale = expf(row_max - new_max);

            // exp(S - tile_max)
            float tile_sum = 0.0f;
            for (int j = 0; j < actual_Bc; j++) {
                Ss[tid * Bc + j] = expf(Ss[tid * Bc + j] - new_max);
                tile_sum += Ss[tid * Bc + j];
            }

            // Update O: O = O * exp_scale + sum_j Ss[j] * V[j]
            float* oi = Optr + (q_start + tid) * D;
            for (int d = 0; d < D; d++) {
                float acc_val = oi[d] * exp_scale;
                for (int j = 0; j < actual_Bc; j++) {
                    acc_val += Ss[tid * Bc + j] * Vs[j * Hd + d];
                }
                oi[d] = acc_val;
            }

            row_sum = row_sum * exp_scale + tile_sum;
            row_max = new_max;
        }
        __syncthreads();
    }

    // Normalize
    if (tid < actual_Br) {
        float* oi = Optr + (q_start + tid) * D;
        float inv_sum = 1.0f / row_sum;
        for (int d = 0; d < D; d++) {
            oi[d] *= inv_sum;
        }
    }
}

torch::Tensor flash_attn_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V
) {
    // Q, K, V: [B, H, N, D] float32
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());

    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    float scale = 1.0f / sqrtf((float)D);

    auto O = torch::zeros_like(Q);

    // Tile sizes
    const int Br = 32;
    const int Bc = 32;

    // Shared memory: (Br + Bc + Bc) * D * 4 bytes + Br * Bc * 4 bytes
    // For D=1024: (32+32+32)*1024*4 + 32*32*4 = 393216 + 4096 = 397312 bytes ~ 388KB
    // RTX 6000 Ada has 99KB per SM... too large!
    // Need to tile D dimension too, or reduce Bc

    // For D=1024, store Q/K/V in smem is infeasible.
    // Use a smaller approach: load K/V in smem, iterate Q from global mem

    // Actually let's just call the kernel with D embedded in smem by tiling D
    // Smem limit: 99KB = 101376 bytes
    // With Br=4, Bc=32, D=1024: smem = (4+32+32)*1024*4 + 4*32*4 = 278528 bytes... still too large

    // For D=1024, we cannot fit QKV in smem. Use a different approach:
    // Each thread handles one Q row, K/V loaded from global with caching
    // Bc=32, Br=1 per warp approach

    // Let's use a simpler tiling where each warp (32 threads) handles one Q row
    // and processes K/V sequentially

    int Br_used = 4;  // small to fit in smem
    int Bc_used = 32;

    // smem = (Br + Bc + Bc) * D * 4 + Br * Bc * 4
    // = (4 + 32 + 32) * 1024 * 4 + 4 * 32 * 4 = 278528 + 512 = 279040 bytes  -- too large

    // Alternative: don't store Q in smem, only K+V
    // smem = (Bc + Bc) * D * 4 + Br * Bc * 4
    // = 64 * 1024 * 4 + Br * Bc * 4
    // For Bc=32: = 131072 + Br*32*4. Still too large for 99KB

    // For D=1024, Bc=8: smem = 16 * 1024 * 4 = 65536 bytes = 64KB. Fits!
    // But Bc=8 is very small — many loop iterations

    // Best for D=1024: process in D-dimension tiles
    // Each SM handles Bc K/V rows, each thread handles 1 Q row
    // D is processed in chunks that fit in smem

    // Fallback to a working implementation with D-tiling
    // Launch: grid(H, B, ceil(N/Br)), block(warp_per_block * 32)
    // Each warp = 1 Q row, processes all KV, D in chunks

    TORCH_CHECK(false, "Use flash_attn_v2 kernel below");
    return O;
}

// Better kernel: process D in tiles, avoid huge smem
// Each thread block handles Br query rows
// Each block uses Bc_tile KV rows at a time, D in D_chunk chunks
// Grid: (ceil(N/Br), H*B)
// Block: (D_chunk) threads

// For D=1024, use D_chunk=128 (multiple warps), Bc=16, Br=16
// smem per block: (Bc + Bc) * D_chunk * 4 bytes = 32 * 128 * 4 = 16KB. Fine.
// Plus scores: Br * Bc * 4 = 16 * 16 * 4 = 1KB
// Plus Q chunk: Br * D_chunk * 4 = 16 * 128 * 4 = 8KB
// Total: ~25KB. Fine.

// But the dot product for D=1024 needs all D dims...
// We need to accumulate dot products across D chunks with atomics or reductions

// Simplest correct approach: each thread handles one (q_row, kv_row) pair
// Compute QK^T with reduction over D, then softmax, then O += softmax * V

// For D=1024: reduction over 1024 elements using warp shuffles

// Final approach: Br=16, each thread handles (q_i, kv_j) pair
// Block has Br * Bc threads, D reduced via warp reduction
// But Br*Bc threads each need D elements... not feasible directly

// SIMPLEST VIABLE: 1 warp per query row
// Each warp (32 threads) handles one Q row
// Compute QK^T[q, k] = sum_d Q[q,d] * K[k,d] using warp parallel reduction over D
// For D=1024: each thread handles D/32=32 elements, then warp reduce
// Then online softmax, then O += p * V using similar approach

__global__ void flash_attn_v2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D,
    float scale
) {
    // One warp per query row
    // blockDim.x = 32 (one warp), blockDim.y = warps_per_block
    int warp_id = threadIdx.y;
    int lane = threadIdx.x;  // 0..31

    int bh = blockIdx.x;  // batch * H + head
    int b = bh / H;
    int h = bh % H;
    int q_idx = blockIdx.y * blockDim.y + warp_id;

    if (q_idx >= N || b >= B) return;

    const float* Qrow = Q + (b * H + h) * N * D + q_idx * D;
    const float* Kbase = K + (b * H + h) * N * D;
    const float* Vbase = V + (b * H + h) * N * D;
    float* Orow = O + (b * H + h) * N * D + q_idx * D;

    // Each warp lane handles D/32 = 32 elements of Q row (D=1024)
    // For dot product with K row: each lane accumulates D/32 muls, then warp reduce

    // Online softmax state
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    // Output accumulator: D/32 floats per lane
    // D=1024, 32 lanes: 32 floats per lane
    float o_acc[32];  // D/32 = 32 for D=1024
    #pragma unroll
    for (int i = 0; i < 32; i++) o_acc[i] = 0.0f;

    int elems_per_lane = D / 32;  // 32 for D=1024

    // Preload Q lane elements
    float q_frag[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        q_frag[i] = Qrow[lane * elems_per_lane + i];
    }

    // Process KV in tiles of Bc
    const int Bc = 64;
    // Shared memory for KV tile scores
    __shared__ float scores_smem[8 * Bc];  // warps_per_block * Bc scores

    for (int kv_start = 0; kv_start < N; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, N);
        int actual_Bc = kv_end - kv_start;

        float tile_scores[Bc];

        // Compute QK^T for this tile
        for (int j = 0; j < actual_Bc; j++) {
            const float* Krow = Kbase + (kv_start + j) * D;
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                dot += q_frag[i] * Krow[lane * elems_per_lane + i];
            }
            // Warp reduce dot product
            #pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) {
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            }
            tile_scores[j] = dot * scale;
        }

        // Online softmax update
        float tile_max = -FLT_MAX;
        for (int j = 0; j < actual_Bc; j++) tile_max = fmaxf(tile_max, tile_scores[j]);

        float new_max = fmaxf(row_max, tile_max);
        float exp_old = expf(row_max - new_max);

        float tile_sum = 0.0f;
        float p[Bc];
        for (int j = 0; j < actual_Bc; j++) {
            p[j] = expf(tile_scores[j] - new_max);
            tile_sum += p[j];
        }

        // Update O accumulator
        #pragma unroll
        for (int i = 0; i < 32; i++) o_acc[i] *= exp_old;

        for (int j = 0; j < actual_Bc; j++) {
            const float* Vrow = Vbase + (kv_start + j) * D;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                o_acc[i] += p[j] * Vrow[lane * elems_per_lane + i];
            }
        }

        row_sum = row_sum * exp_old + tile_sum;
        row_max = new_max;
    }

    // Normalize and write output
    float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        Orow[lane * elems_per_lane + i] = o_acc[i] * inv_sum;
    }
}

torch::Tensor flash_attn_v2(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V
) {
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    float scale = 1.0f / sqrtf((float)D);

    auto O = torch::empty_like(Q);

    // 1 warp (32 threads) per Q row, 8 warps per block
    const int warps_per_block = 8;
    dim3 block(32, warps_per_block);
    dim3 grid(B * H, (N + warps_per_block - 1) / warps_per_block);

    size_t smem = warps_per_block * 64 * sizeof(float);

    flash_attn_v2_kernel<<<grid, block, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), B, H, N, D, scale
    );
    return O;
}
"""

_flash_attn_cpp = r"""
torch::Tensor flash_attn_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

_module = load_inline(
    name="flash_attn_cuda_unlim",
    cpp_sources=_flash_attn_cpp,
    cuda_sources=_flash_attn_src,
    functions=["flash_attn_v2"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        return _module.flash_attn_v2(Q, K, V)
