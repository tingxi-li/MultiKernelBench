import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized SDPA via fp16 computation:
# 1. Cast Q, K, V to fp16
# 2. Compute A = Q @ K.T / sqrt(D) using CUDA tiled GEMM (fp16->fp32 accum)
# 3. Softmax(A) in fp32
# 4. O = softmax(A) @ V in fp16
# 5. Cast back to fp32
# This leverages fp16 memory bandwidth savings (2x) and tensor core compute

_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

// ---- Softmax kernel: in-place row softmax on [B*H, N, N] float32 ----
__global__ void softmax_rows_kernel(
    float* __restrict__ A,  // [R, N]
    int R, int N
) {
    // Each block handles one row; block has min(1024, N) threads
    int row = blockIdx.x;
    if (row >= R) return;

    float* arow = A + row * N;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Max reduction
    float lmax = -FLT_MAX;
    for (int j = tid; j < N; j += stride) lmax = fmaxf(lmax, arow[j]);
    // Warp reduce then block reduce via smem
    extern __shared__ float smem_sf[];
    for (int off = 16; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, off));
    if (tid % 32 == 0) smem_sf[tid / 32] = lmax;
    __syncthreads();
    if (tid < 32) {
        lmax = (tid < (blockDim.x / 32)) ? smem_sf[tid] : -FLT_MAX;
        for (int off = 16; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, off));
        if (tid == 0) smem_sf[0] = lmax;
    }
    __syncthreads();
    lmax = smem_sf[0];

    // Exp and sum
    float lsum = 0.f;
    for (int j = tid; j < N; j += stride) {
        float v = expf(arow[j] - lmax);
        arow[j] = v;
        lsum += v;
    }
    for (int off = 16; off >= 1; off >>= 1) lsum += __shfl_xor_sync(0xffffffff, lsum, off);
    if (tid % 32 == 0) smem_sf[tid / 32] = lsum;
    __syncthreads();
    if (tid < 32) {
        lsum = (tid < (blockDim.x / 32)) ? smem_sf[tid] : 0.f;
        for (int off = 16; off >= 1; off >>= 1) lsum += __shfl_xor_sync(0xffffffff, lsum, off);
        if (tid == 0) smem_sf[0] = lsum;
    }
    __syncthreads();
    lsum = smem_sf[0];

    float inv = 1.f / lsum;
    for (int j = tid; j < N; j += stride) arow[j] *= inv;
}

// ---- GEMM for QK^T: [BH, N, D] x [BH, D, N] -> [BH, N, N] in fp32 ----
// Use wmma m16n16k16 fp16->fp32
// Grid: (ceil(N/16), ceil(N/16), BH), Block: 32 (1 warp)
// Each warp computes one 16x16 output tile via wmma

__global__ void batched_gemm_qkt_kernel(
    const __half* __restrict__ Q,  // [BH, N, D]
    const __half* __restrict__ K,  // [BH, N, D]
    float*        __restrict__ S,  // [BH, N, N]
    int N, int D, float scale
) {
    int bh  = blockIdx.z;
    int row = blockIdx.y * 16;
    int col = blockIdx.x * 16;

    if (row >= N || col >= N) return;

    const __half* Qb = Q + bh * N * D;
    const __half* Kb = K + bh * N * D;
    float*        Sb = S + bh * N * N;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    // k-loop over D in 16-wide chunks
    for (int k = 0; k < D; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

        // A: Q[row..row+16, k..k+16], stride D, row_major
        wmma::load_matrix_sync(a_frag, Qb + row * D + k, D);
        // B: K[col..col+16, k..k+16], stride D, col_major (= K.T[k..k+16, col..col+16])
        wmma::load_matrix_sync(b_frag, Kb + col * D + k, D);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Scale and store
    for (int i = 0; i < c_frag.num_elements; i++)
        c_frag.x[i] *= scale;

    wmma::store_matrix_sync(Sb + row * N + col, c_frag, N, wmma::mem_row_major);
}

// ---- GEMM for P@V: [BH, N, N] x [BH, N, D] -> [BH, N, D] in fp32 ----
// P is fp32 (after softmax), V is fp16
// Use wmma float accumulator with fp16 inputs
// But P is fp32... need to convert to fp16 first, or use float16 for intermediate P

// Alternative: accumulate in fp32 by splitting P into fp16 chunks
// Actually: use p_frag fp16 by converting inline... wmma only supports fp16 input

// Simpler: scale P back to fp16 via a kernel, then do wmma PV
// Or just do a regular tiled GEMM in fp32

// For PV: [BH, N, N] fp32 @ [BH, N, D] fp16 -> [BH, N, D] fp32
// Each block computes a 16x16 output tile
// k-loop over N (=512 steps of 16 = 32 k-steps)

__global__ void batched_gemm_pv_kernel(
    const float*  __restrict__ P,  // [BH, N, N] fp32
    const __half* __restrict__ V,  // [BH, N, D] fp16
    float*        __restrict__ O,  // [BH, N, D] fp32
    int N, int D
) {
    int bh  = blockIdx.z;
    int row = blockIdx.y * 16;
    int col = blockIdx.x * 16;

    if (row >= N || col >= D) return;

    const float*  Pb = P + bh * N * N;
    const __half* Vb = V + bh * N * D;
    float*        Ob = O + bh * N * D;

    // P is fp32 but wmma needs fp16. Convert P tile to fp16 inline using registers.
    // k-loop over N
    // Each k-step: P[row..+16, k..+16] fp32 -> fp16, V[k..+16, col..+16] fp16

    // Use float accumulator wmma with fp16 A (P converted) and fp16 B (V)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    // Need a smem buffer to hold fp16 P chunk
    // Use registers approach: load 16x16 P chunk, convert to fp16, store to smem, wmma
    // This kernel needs smem: 16*16 fp16 = 512 bytes per block
    // Actually just declare it statically since blockDim is 32

    __shared__ __half P_half[16 * 16];  // 16x16 fp16 for P tile

    int lane = threadIdx.x;

    for (int k = 0; k < N; k += 16) {
        // Convert P[row..+16, k..+16] to fp16 in P_half
        // 32 lanes handle 16*16=256 elements, 8 per lane
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16, c = i % 16;
            int pr = row + r, pc = k + c;
            float pval = (pr < N && pc < N) ? Pb[pr * N + pc] : 0.f;
            P_half[i] = __float2half(pval);
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, P_half, 16);
        // V[k..+16, col..+16], stride D, row_major
        wmma::load_matrix_sync(b_frag, Vb + k * D + col, D);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(Ob + row * D + col, c_frag, D, wmma::mem_row_major);
}

torch::Tensor fa2_3kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    int BH = B * H;
    float scale = 1.f / sqrtf((float)D);

    auto Qr = Q.reshape({BH, N, D}).contiguous();
    auto Kr = K.reshape({BH, N, D}).contiguous();
    auto Vr = V.reshape({BH, N, D}).contiguous();

    // Convert to fp16
    auto Qh = Qr.to(torch::kHalf);
    auto Kh = Kr.to(torch::kHalf);
    auto Vh = Vr.to(torch::kHalf);

    // Allocate attention matrix S[BH, N, N]
    auto S = torch::empty({BH, N, N}, Qr.options());
    auto Or = torch::empty({BH, N, D}, Qr.options());

    const __half* Qp = reinterpret_cast<const __half*>(Qh.data_ptr<at::Half>());
    const __half* Kp = reinterpret_cast<const __half*>(Kh.data_ptr<at::Half>());
    const __half* Vp = reinterpret_cast<const __half*>(Vh.data_ptr<at::Half>());

    // Kernel 1: QK^T
    {
        dim3 block(32);
        dim3 grid((N + 15) / 16, (N + 15) / 16, BH);
        batched_gemm_qkt_kernel<<<grid, block>>>(Qp, Kp, S.data_ptr<float>(), N, D, scale);
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "QKT: ", cudaGetErrorString(err));
    }

    // Kernel 2: Row softmax
    {
        int nrows = BH * N;
        int block = min(512, (N + 31) / 32 * 32);
        int smem = (block / 32) * sizeof(float);
        softmax_rows_kernel<<<nrows, block, smem>>>(S.data_ptr<float>(), nrows, N);
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Softmax: ", cudaGetErrorString(err));
    }

    // Kernel 3: PV
    {
        dim3 block(32);
        dim3 grid((D + 15) / 16, (N + 15) / 16, BH);
        batched_gemm_pv_kernel<<<grid, block>>>(S.data_ptr<float>(), Vp, Or.data_ptr<float>(), N, D);
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "PV: ", cudaGetErrorString(err));
    }

    return Or.reshape({B, H, N, D});
}
"""

_cpp = "torch::Tensor fa2_3kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"

_mod = load_inline(
    name="fa2_3k_v10",
    cpp_sources=_cpp,
    cuda_sources=_src,
    functions=["fa2_3kernel"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _mod.fa2_3kernel(Q.contiguous(), K.contiguous(), V.contiguous())
