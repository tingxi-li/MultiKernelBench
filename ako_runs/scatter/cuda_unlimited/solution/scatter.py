import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA UNLIMITED deterministic scatter (dim=1, last-index-wins).
# FUSED single-kernel, one block per row. PACKED shared winner: each shared slot
# holds a 64-bit ((k+1)<<32 | float_bits(update)). A 64-bit shared atomicMax on
# it picks the largest k+1 (= last write) regardless of the float low bits, so
# phase 2 reads the winning value straight from shared with NO uncoalesced
# updates[] gather. Init 0 = "no write". Needs opt-in for 64KB dynamic shared.
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
__global__ void scatter_packed_k(const long* __restrict__ idx,
                                 const float* __restrict__ x,
                                 const float* __restrict__ upd,
                                 float* __restrict__ out,
                                 int K, int W){
    extern __shared__ unsigned long long s_win[];
    long r = blockIdx.x;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    // Init each shared slot to (k+1=0) in high bits, x's value in low bits.
    // Any real write packs k+1>=1 in high bits and thus wins the atomicMax;
    // an unhit slot keeps x. Phase 2 then just reads the low bits (branchless).
    const float* x_row = x + r * (long)W;
    for(int s = tid; s < W; s += nt) s_win[s] = (unsigned long long)__float_as_uint(x_row[s]);
    __syncthreads();
    const long* idx_row = idx + r * (long)K;
    const float* upd_row = upd + r * (long)K;
    for(int k = tid; k < K; k += nt){
        int slot = (int)idx_row[k];
        unsigned int vbits = __float_as_uint(upd_row[k]);
        unsigned long long packed = ((unsigned long long)(unsigned)(k + 1) << 32) | (unsigned long long)vbits;
        atomicMax(&s_win[slot], packed);
    }
    __syncthreads();
    float* out_row = out + r * (long)W;
    for(int s = tid; s < W; s += nt){
        out_row[s] = __uint_as_float((unsigned int)(s_win[s] & 0xFFFFFFFFULL));
    }
}
torch::Tensor scatter_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor upd){
    long Rr = x.size(0), W = x.size(1), K = idx.size(1);
    auto out = torch::empty_like(x);
    int t = 1024;
    size_t shmem = (size_t)W * sizeof(unsigned long long);
    static bool cfg = false;
    if(!cfg){
        cudaFuncSetAttribute(scatter_packed_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        cfg = true;
    }
    scatter_packed_k<<<(int)Rr, t, shmem>>>(idx.data_ptr<long>(), x.data_ptr<float>(),
                                            upd.data_ptr<float>(), out.data_ptr<float>(),
                                            (int)K, (int)W);
    return out;
}
"""
_CPP = "torch::Tensor scatter_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor upd);"
_ext = load_inline(name="scatter_cuda_unlimited_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["scatter_cuda"], verbose=False, extra_cuda_cflags=["-O3"])

class Model(nn.Module):
    """Deterministic last-wins scatter (dim=1) via a single fused CUDA kernel
    with 64-bit packed shared-memory atomicMax (value carried in the winner)."""
    def forward(self, x, idx, updates):
        return _ext.scatter_cuda(x.contiguous(), idx.contiguous(), updates.contiguous())
