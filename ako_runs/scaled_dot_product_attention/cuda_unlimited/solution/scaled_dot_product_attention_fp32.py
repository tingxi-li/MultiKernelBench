import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Test: pure fp32 GEMM with larger tiles BM=BN=128, BK=16, TM=TN=8
# No fp32->fp16 conversion overhead

_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define BM2 128
#define BN2 128
#define BK2 16
#define TM2 8
#define TN2 8
// Block: (BM2/TM2, BN2/TN2) = (16,16) = 256 threads
// smem: (128*16 + 16*128)*4 = 16KB per block

__global__ __launch_bounds__(256)
void gemm_qkt_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K_,
    float*       __restrict__ S,
    int M, int N, int K, float scale
) {
    int bh = blockIdx.z;
    Q  += (long)bh * M * K;
    K_ += (long)bh * N * K;
    S  += (long)bh * M * N;

    int tx = threadIdx.x % (BN2 / TN2);
    int ty = threadIdx.x / (BN2 / TN2);
    int brow = blockIdx.y * BM2;
    int bcol = blockIdx.x * BN2;
    int grow = brow + ty * TM2;
    int gcol = bcol + tx * TN2;

    __shared__ float As[BM2 * BK2];
    __shared__ float Bs[BK2 * BN2];
    float acc[TM2][TN2] = {};

    for (int k0 = 0; k0 < K; k0 += BK2) {
        for (int i = threadIdx.x; i < BM2 * BK2; i += 256) {
            int r = i / BK2, c = i % BK2;
            int gr = brow + r, gc = k0 + c;
            As[i] = (gr < M && gc < K) ? Q[gr * K + gc] : 0.f;
        }
        for (int i = threadIdx.x; i < BK2 * BN2; i += 256) {
            int r = i / BN2, c = i % BN2;
            int gr = k0 + r, gc = bcol + c;
            Bs[i] = (gr < K && gc < N) ? K_[gc * K + gr] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK2; k++) {
            float a[TM2], b[TN2];
            #pragma unroll
            for (int tm = 0; tm < TM2; tm++) a[tm] = As[(ty*TM2+tm)*BK2+k];
            #pragma unroll
            for (int tn = 0; tn < TN2; tn++) b[tn] = Bs[k*BN2+tx*TN2+tn];
            #pragma unroll
            for (int tm = 0; tm < TM2; tm++)
                #pragma unroll
                for (int tn = 0; tn < TN2; tn++)
                    acc[tm][tn] += a[tm]*b[tn];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < TM2; tm++) {
        int gr = grow+tm; if(gr>=M) continue;
        #pragma unroll
        for (int tn = 0; tn < TN2; tn++) {
            int gc = gcol+tn; if(gc<N) S[gr*N+gc]=acc[tm][tn]*scale;
        }
    }
}

__global__ __launch_bounds__(256)
void gemm_pv_f32(
    const float* __restrict__ P,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int M, int K_, int N
) {
    int bh = blockIdx.z;
    P += (long)bh * M * K_;
    V += (long)bh * K_ * N;
    O += (long)bh * M * N;

    int tx = threadIdx.x % (BN2/TN2);
    int ty = threadIdx.x / (BN2/TN2);
    int brow = blockIdx.y * BM2;
    int bcol = blockIdx.x * BN2;
    int grow = brow + ty*TM2;
    int gcol = bcol + tx*TN2;

    __shared__ float As[BM2*BK2];
    __shared__ float Bs[BK2*BN2];
    float acc[TM2][TN2] = {};

    for (int k0 = 0; k0 < K_; k0 += BK2) {
        for (int i = threadIdx.x; i < BM2*BK2; i += 256) {
            int r=i/BK2, c=i%BK2; int gr=brow+r, gc=k0+c;
            As[i] = (gr<M&&gc<K_) ? P[gr*K_+gc] : 0.f;
        }
        for (int i = threadIdx.x; i < BK2*BN2; i += 256) {
            int r=i/BN2, c=i%BN2; int gr=k0+r, gc=bcol+c;
            Bs[i] = (gr<K_&&gc<N) ? V[gr*N+gc] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK2; k++) {
            float a[TM2], b[TN2];
            #pragma unroll
            for(int tm=0;tm<TM2;tm++) a[tm]=As[(ty*TM2+tm)*BK2+k];
            #pragma unroll
            for(int tn=0;tn<TN2;tn++) b[tn]=Bs[k*BN2+tx*TN2+tn];
            #pragma unroll
            for(int tm=0;tm<TM2;tm++) for(int tn=0;tn<TN2;tn++) acc[tm][tn]+=a[tm]*b[tn];
        }
        __syncthreads();
    }
    #pragma unroll
    for(int tm=0;tm<TM2;tm++){int gr=grow+tm;if(gr>=M)continue;
        for(int tn=0;tn<TN2;tn++){int gc=gcol+tn;if(gc<N)O[gr*N+gc]=acc[tm][tn];}}
}

__global__ void softmax_f32(float* __restrict__ A, int R, int N) {
    int row=blockIdx.x; if(row>=R) return;
    float* arow=A+row*N;
    int tid=threadIdx.x,stride=blockDim.x;
    extern __shared__ float sm[];
    float lmax=-FLT_MAX;
    for(int j=tid;j<N;j+=stride) lmax=fmaxf(lmax,arow[j]);
    for(int o=16;o>=1;o>>=1) lmax=fmaxf(lmax,__shfl_xor_sync(0xffffffff,lmax,o));
    if(tid%32==0) sm[tid/32]=lmax; __syncthreads();
    if(tid<32){lmax=(tid<stride/32)?sm[tid]:-FLT_MAX; for(int o=16;o>=1;o>>=1) lmax=fmaxf(lmax,__shfl_xor_sync(0xffffffff,lmax,o)); if(tid==0) sm[0]=lmax;}
    __syncthreads(); lmax=sm[0];
    float ls=0.f;
    for(int j=tid;j<N;j+=stride){float v=expf(arow[j]-lmax);arow[j]=v;ls+=v;}
    for(int o=16;o>=1;o>>=1) ls+=__shfl_xor_sync(0xffffffff,ls,o);
    if(tid%32==0) sm[tid/32]=ls; __syncthreads();
    if(tid<32){ls=(tid<stride/32)?sm[tid]:0.f; for(int o=16;o>=1;o>>=1) ls+=__shfl_xor_sync(0xffffffff,ls,o); if(tid==0) sm[0]=ls;}
    __syncthreads(); ls=sm[0];
    float inv=1.f/ls;
    for(int j=tid;j<N;j+=stride) arow[j]*=inv;
}

torch::Tensor fa2_fp32_128(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int B=Q.size(0),H=Q.size(1),N=Q.size(2),D=Q.size(3);
    int BH=B*H;
    float scale=1.f/sqrtf((float)D);
    auto Qr=Q.reshape({BH,N,D}).contiguous();
    auto Kr=K.reshape({BH,N,D}).contiguous();
    auto Vr=V.reshape({BH,N,D}).contiguous();
    auto S=torch::empty({BH,N,N},Qr.options());
    auto Or=torch::empty({BH,N,D},Qr.options());
    {dim3 bl(256);dim3 gr((N+BN2-1)/BN2,(N+BM2-1)/BM2,BH);
    gemm_qkt_f32<<<gr,bl>>>(Qr.data_ptr<float>(),Kr.data_ptr<float>(),S.data_ptr<float>(),N,N,D,scale);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"QKT");}
    {int nr=BH*N,th=min(1024,((N+31)/32)*32),sm=(th/32)*4;
    softmax_f32<<<nr,th,sm>>>(S.data_ptr<float>(),nr,N);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"SF");}
    {dim3 bl(256);dim3 gr((D+BN2-1)/BN2,(N+BM2-1)/BM2,BH);
    gemm_pv_f32<<<gr,bl>>>(S.data_ptr<float>(),Vr.data_ptr<float>(),Or.data_ptr<float>(),N,N,D);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"PV");}
    return Or.reshape({B,H,N,D});
}
"""

_cpp = "torch::Tensor fa2_fp32_128(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"

_mod = load_inline(
    name="fa2_fp32_128_v15",
    cpp_sources=_cpp,
    cuda_sources=_src,
    functions=["fa2_fp32_128"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _mod.fa2_fp32_128(Q.contiguous(), K.contiguous(), V.contiguous())
