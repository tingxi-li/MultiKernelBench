import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Iter 6: 3-kernel SDPA with fp16 Q/K/V (like iter-5) + larger tiles (64×64 output per block)
# Key change from iter-5: 4 warps handle 64×64 output tile (each warp: 16 rows × 64 cols)
# vs iter-5 which used 4 warps for 64×16 tile (each warp: 16 rows × 16 cols)
# Larger Bc=64 means fewer KV tiles (N/64=8 vs N/16=32), reducing grid launch overhead
# Also better utilization of tensor cores per block
# Uses PTX mma.sync via WMMA API; NO fp32→fp16 conversion overhead (inputs already fp16)

_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

// QKT kernel: fp16 Q, K → fp32 S
// Block: 128 threads = 4 warps
// Tile: 64 rows (4 warps × 16) × 64 cols (each warp covers 4 × 16 col-tiles)
// smem: Q[64][64] fp16 + K[64][64] fp16 = 16 KB
__global__ __launch_bounds__(128, 3)
void qkt_h16_bc64(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    float*        __restrict__ S,
    int N, int D, float scale
) {
    const int Br = 64, Bc = 64, DC = 64;
    int bh   = blockIdx.z;
    int bcol = blockIdx.x * Bc;
    int brow = blockIdx.y * Br;
    int warp = threadIdx.x / 32;
    int warp_row = warp * 16;

    const __half* Qb = Q + (long)bh * N * D;
    const __half* Kb = K + (long)bh * N * D;
    float*        Sb = S + (long)bh * N * N;

    __shared__ __half Qs[Br * DC];
    __shared__ __half Ks[Bc * DC];

    // Each warp holds 4 accumulator tiles (16 rows × 64 cols)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[4];
    for (int t = 0; t < 4; t++) wmma::fill_fragment(c[t], 0.f);

    for (int d0 = 0; d0 < D; d0 += DC) {
        // Cooperative smem load — Q[brow..+64, d0..+64]
        for (int i = threadIdx.x; i < Br * DC; i += 128) {
            int r = i / DC, col = i % DC;
            Qs[i] = Qb[(brow + r) * D + (d0 + col)];
        }
        // K[bcol..+64, d0..+64] — stored row-major; used as col-major in wmma
        for (int i = threadIdx.x; i < Bc * DC; i += 128) {
            int r = i / DC, col = i % DC;
            Ks[i] = Kb[(bcol + r) * D + (d0 + col)];
        }
        __syncthreads();

        for (int col_tile = 0; col_tile < 4; col_tile++) {
            int kc = col_tile * 16;
            for (int kt = 0; kt < DC / 16; kt++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, Qs + warp_row * DC + kt * 16, DC);
                // Ks stored row-major [row=Bc, col=DC]; used as col-major for K^T:
                // B[k, n] = Ks[(kc+n)*DC + kt*16+k] (col_major: stride=DC)
                wmma::load_matrix_sync(b, Ks + kc * DC + kt * 16, DC);
                wmma::mma_sync(c[col_tile], a, b, c[col_tile]);
            }
        }
        __syncthreads();
    }

    for (int ct = 0; ct < 4; ct++) {
        int gc = bcol + ct * 16;
        for (int i = 0; i < c[ct].num_elements; i++) c[ct].x[i] *= scale;
        wmma::store_matrix_sync(Sb + (brow + warp_row) * N + gc, c[ct], N, wmma::mem_row_major);
    }
}

// Row softmax in-place on fp32 matrix
__global__ void softmax_f32_ker(float* __restrict__ A, int N) {
    int row = blockIdx.x;
    float* arow = A + (long)row * N;
    int tid = threadIdx.x, stride = blockDim.x;
    extern __shared__ float sm[];

    float lmax = -FLT_MAX;
    for (int j = tid; j < N; j += stride) lmax = fmaxf(lmax, arow[j]);
    for (int o = 16; o >= 1; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, o));
    if (tid % 32 == 0) sm[tid / 32] = lmax;
    __syncthreads();
    if (tid < 32) {
        lmax = (tid < stride / 32) ? sm[tid] : -FLT_MAX;
        for (int o = 16; o >= 1; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, o));
        if (tid == 0) sm[0] = lmax;
    }
    __syncthreads();
    lmax = sm[0];

    float ls = 0.f;
    for (int j = tid; j < N; j += stride) { float v = expf(arow[j] - lmax); arow[j] = v; ls += v; }
    for (int o = 16; o >= 1; o >>= 1) ls += __shfl_xor_sync(0xffffffff, ls, o);
    if (tid % 32 == 0) sm[tid / 32] = ls;
    __syncthreads();
    if (tid < 32) {
        ls = (tid < stride / 32) ? sm[tid] : 0.f;
        for (int o = 16; o >= 1; o >>= 1) ls += __shfl_xor_sync(0xffffffff, ls, o);
        if (tid == 0) sm[0] = ls;
    }
    __syncthreads();
    float inv = 1.f / sm[0];
    for (int j = tid; j < N; j += stride) arow[j] *= inv;
}

// PV: fp32 P × fp16 V → fp32 O
// P is converted to fp16 in smem; V is already fp16
// Same 64×64 tile layout as QKT
__global__ __launch_bounds__(128, 3)
void pv_h16_bc64(
    const float* __restrict__ P,
    const __half* __restrict__ V,
    float*        __restrict__ O,
    int N, int D
) {
    const int Br = 64, Bc = 64, DC = 64;
    int bh   = blockIdx.z;
    int bcol = blockIdx.x * DC;
    int brow = blockIdx.y * Br;
    int warp = threadIdx.x / 32;
    int warp_row = warp * 16;

    const float* Pb = P + (long)bh * N * N;
    const __half* Vb = V + (long)bh * N * D;
    float*        Ob = O + (long)bh * N * D;

    __shared__ __half Ps[Br * Bc];   // P fp32 → fp16 in smem
    __shared__ __half Vs[Bc * DC];   // V fp16

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[4];
    for (int t = 0; t < 4; t++) wmma::fill_fragment(c[t], 0.f);

    for (int k0 = 0; k0 < N; k0 += Bc) {
        // Load P[brow..+64, k0..+64] fp32→fp16
        for (int i = threadIdx.x; i < Br * Bc; i += 128) {
            int r = i / Bc, col = i % Bc;
            Ps[i] = __float2half(Pb[(brow + r) * N + (k0 + col)]);
        }
        // Load V[k0..+64, bcol..+64] fp16
        for (int i = threadIdx.x; i < Bc * DC; i += 128) {
            int r = i / DC, col = i % DC;
            Vs[i] = Vb[(k0 + r) * D + (bcol + col)];
        }
        __syncthreads();

        for (int col_tile = 0; col_tile < 4; col_tile++) {
            int dc = col_tile * 16;
            for (int kt = 0; kt < Bc / 16; kt++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(p_frag, Ps + warp_row * Bc + kt * 16, Bc);
                wmma::load_matrix_sync(v_frag, Vs + kt * 16 * DC + dc, DC);
                wmma::mma_sync(c[col_tile], p_frag, v_frag, c[col_tile]);
            }
        }
        __syncthreads();
    }

    for (int ct = 0; ct < 4; ct++) {
        int gc = bcol + ct * 16;
        wmma::store_matrix_sync(Ob + (brow + warp_row) * D + gc, c[ct], D, wmma::mem_row_major);
    }
}

torch::Tensor fa2_bc64(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    int BH = B * H;
    float scale = 1.f / sqrtf((float)D);

    auto Qr = Q.reshape({BH, N, D}).contiguous();
    auto Kr = K.reshape({BH, N, D}).contiguous();
    auto Vr = V.reshape({BH, N, D}).contiguous();

    // Convert Q, K, V to fp16 (saves bandwidth in QKT and PV loops)
    auto Qh = Qr.to(torch::kHalf);
    auto Kh = Kr.to(torch::kHalf);
    auto Vh = Vr.to(torch::kHalf);

    auto S  = torch::empty({BH, N, N}, Qr.options());   // fp32
    auto Or = torch::empty({BH, N, D}, Qr.options());   // fp32

    // QKT: grid=(N/64, N/64, BH), 128 threads, smem 16KB static
    {
        dim3 grid((N + 63) / 64, (N + 63) / 64, BH);
        qkt_h16_bc64<<<grid, 128>>>(
            reinterpret_cast<const __half*>(Qh.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(Kh.data_ptr<at::Half>()),
            S.data_ptr<float>(), N, D, scale);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "QKT failed");
    }

    // Softmax
    {
        int nrows = BH * N;
        int threads = min(1024, ((N + 31) / 32) * 32);
        int smem = (threads / 32) * (int)sizeof(float);
        softmax_f32_ker<<<nrows, threads, smem>>>(S.data_ptr<float>(), N);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Softmax failed");
    }

    // PV: grid=(D/64, N/64, BH), 128 threads, smem 16KB static
    {
        dim3 grid((D + 63) / 64, (N + 63) / 64, BH);
        pv_h16_bc64<<<grid, 128>>>(
            S.data_ptr<float>(),
            reinterpret_cast<const __half*>(Vh.data_ptr<at::Half>()),
            Or.data_ptr<float>(), N, D);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "PV failed");
    }

    return Or.reshape({B, H, N, D});
}
"""

_cpp = "torch::Tensor fa2_bc64(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"

_mod = load_inline(
    name="fa2_bc64_v19",
    cpp_sources=_cpp,
    cuda_sources=_src,
    functions=["fa2_bc64"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _mod.fa2_bc64(Q.contiguous(), K.contiguous(), V.contiguous())
