import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 3-kernel SDPA: wmma fp16 QKT + fp32 softmax + wmma fp16 PV
# 4 warps per block (64 rows x 16 cols tile for QKT)
# FIXED: blockIdx.z properly offsets BH dimension

_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

// QK^T: Grid(ceil(N/16), ceil(N/64), BH), Block(128) = 4 warps
__global__ __launch_bounds__(128, 2)
void qkt_kernel(
    const __half* __restrict__ Q,  // [BH, N, D]
    const __half* __restrict__ K,  // [BH, N, D]
    float*        __restrict__ S,  // [BH, N, N]
    int N, int D, float scale
) {
    int bh   = blockIdx.z;
    int bcol = blockIdx.x * 16;
    int brow = blockIdx.y * 64;
    int warp = threadIdx.x / 32;
    int row  = brow + warp * 16;

    if (row >= N || bcol >= N) return;

    const __half* Qb = Q + (long)bh * N * D;
    const __half* Kb = K + (long)bh * N * D;
    float*        Sb = S + (long)bh * N * N;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    for (int k = 0; k < D; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, Qb + row * D + k, D);
        wmma::load_matrix_sync(b_frag, Kb + bcol * D + k, D);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    for (int i = 0; i < c_frag.num_elements; i++) c_frag.x[i] *= scale;
    wmma::store_matrix_sync(Sb + row * N + bcol, c_frag, N, wmma::mem_row_major);
}

// Row softmax on fp32 S[R, N] (R = BH*N flattened)
__global__ void softmax_f32_kernel(float* __restrict__ A, int R, int N) {
    int row = blockIdx.x;
    if (row >= R) return;
    float* arow = A + row * N;
    int tid = threadIdx.x, stride = blockDim.x;
    extern __shared__ float smem[];

    float lmax = -FLT_MAX;
    for (int j = tid; j < N; j += stride) lmax = fmaxf(lmax, arow[j]);
    for (int off=16;off>=1;off>>=1) lmax=fmaxf(lmax,__shfl_xor_sync(0xffffffff,lmax,off));
    if(tid%32==0) smem[tid/32]=lmax; __syncthreads();
    if(tid<32){lmax=(tid<stride/32)?smem[tid]:-FLT_MAX; for(int off=16;off>=1;off>>=1) lmax=fmaxf(lmax,__shfl_xor_sync(0xffffffff,lmax,off)); if(tid==0) smem[0]=lmax;}
    __syncthreads(); lmax=smem[0];

    float lsum=0.f;
    for(int j=tid;j<N;j+=stride){float v=expf(arow[j]-lmax);arow[j]=v;lsum+=v;}
    for(int off=16;off>=1;off>>=1) lsum+=__shfl_xor_sync(0xffffffff,lsum,off);
    if(tid%32==0) smem[tid/32]=lsum; __syncthreads();
    if(tid<32){lsum=(tid<stride/32)?smem[tid]:0.f; for(int off=16;off>=1;off>>=1) lsum+=__shfl_xor_sync(0xffffffff,lsum,off); if(tid==0) smem[0]=lsum;}
    __syncthreads(); lsum=smem[0];
    float inv=1.f/lsum;
    for(int j=tid;j<N;j+=stride) arow[j]*=inv;
}

// PV: P[BH,N,N] fp32 × V[BH,N,D] fp16 -> O[BH,N,D] fp32
// Grid(ceil(D/16), ceil(N/64), BH), Block(128) = 4 warps
__global__ __launch_bounds__(128, 2)
void pv_kernel(
    const float*  __restrict__ P,  // [BH, N, N]
    const __half* __restrict__ V,  // [BH, N, D]
    float*        __restrict__ O,  // [BH, N, D]
    int N, int D
) {
    int bh   = blockIdx.z;
    int bcol = blockIdx.x * 16;
    int brow = blockIdx.y * 64;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row  = brow + warp * 16;

    if (row >= N || bcol >= D) return;

    const float*  Pb = P + (long)bh * N * N;
    const __half* Vb = V + (long)bh * N * D;
    float*        Ob = O + (long)bh * N * D;

    __shared__ __half Ps[4 * 16 * 16];  // 4 warps × 256 fp16 elements
    __half* wp = Ps + warp * 256;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    for (int k = 0; k < N; k += 16) {
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16, c = i % 16;
            int pr = row + r, pc = k + c;
            wp[i] = (pr < N && pc < N) ? __float2half(Pb[pr * N + pc]) : __float2half(0.f);
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
        wmma::load_matrix_sync(p_frag, wp, 16);
        wmma::load_matrix_sync(v_frag, Vb + k * D + bcol, D);
        wmma::mma_sync(c_frag, p_frag, v_frag, c_frag);
    }

    wmma::store_matrix_sync(Ob + row * D + bcol, c_frag, D, wmma::mem_row_major);
}

torch::Tensor fa2_wmma_fixed(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    int BH = B * H;
    float scale = 1.f / sqrtf((float)D);

    auto Qr = Q.reshape({BH, N, D}).contiguous();
    auto Kr = K.reshape({BH, N, D}).contiguous();
    auto Vr = V.reshape({BH, N, D}).contiguous();

    auto Qh = Qr.to(torch::kHalf);
    auto Kh = Kr.to(torch::kHalf);
    auto Vh = Vr.to(torch::kHalf);

    auto S  = torch::empty({BH, N, N}, Qr.options());
    auto Or = torch::empty({BH, N, D}, Qr.options());

    const __half* Qp = reinterpret_cast<const __half*>(Qh.data_ptr<at::Half>());
    const __half* Kp = reinterpret_cast<const __half*>(Kh.data_ptr<at::Half>());
    const __half* Vp = reinterpret_cast<const __half*>(Vh.data_ptr<at::Half>());

    // Kernel 1: QK^T
    {
        dim3 block(128);
        dim3 grid((N+15)/16, (N+63)/64, BH);
        qkt_kernel<<<grid, block>>>(Qp, Kp, S.data_ptr<float>(), N, D, scale);
        TORCH_CHECK(cudaGetLastError()==cudaSuccess, "QKT failed");
    }

    // Kernel 2: Softmax (flat [BH*N, N])
    {
        int nrows = BH * N;
        int threads = min(1024, ((N+31)/32)*32);
        int smem = (threads/32)*sizeof(float);
        softmax_f32_kernel<<<nrows, threads, smem>>>(S.data_ptr<float>(), nrows, N);
        TORCH_CHECK(cudaGetLastError()==cudaSuccess, "Softmax failed");
    }

    // Kernel 3: PV
    {
        dim3 block(128);
        dim3 grid((D+15)/16, (N+63)/64, BH);
        pv_kernel<<<grid, block>>>(S.data_ptr<float>(), Vp, Or.data_ptr<float>(), N, D);
        TORCH_CHECK(cudaGetLastError()==cudaSuccess, "PV failed");
    }

    return Or.reshape({B, H, N, D});
}
"""

_cpp = "torch::Tensor fa2_wmma_fixed(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"

_mod = load_inline(
    name="fa2_wmma_fix_v14",
    cpp_sources=_cpp,
    cuda_sources=_src,
    functions=["fa2_wmma_fixed"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _mod.fa2_wmma_fixed(Q.contiguous(), K.contiguous(), V.contiguous())
