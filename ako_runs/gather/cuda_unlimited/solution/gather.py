import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA UNLIMITED gather (dim=1). The reference input is x:(M,Cin) gathered by
# idx:(M,Cout) with random column indices per row. A naive per-element gather
# issues Cout scattered global loads per row: under bench.py's cold-L2 timing
# each 4-byte random read pays a full DRAM sector at ~50% of peak bandwidth
# (row-buffer thrashing) -> that scatter is the whole cost (measured 0.0204ms;
# a coalesced idx+store floor is 0.0072ms).
#
# Lever: one block per row stages the ENTIRE x-row into shared memory with a
# single COALESCED float4 HBM read, then performs the random gather out of
# shared memory. This converts the scattered HBM traffic into sequential HBM
# traffic (DRAM-efficient) plus cheap on-chip random shared-memory reads.
# idx loads (longlong2) and the output are coalesced; the store is a streaming
# st.global.cs.v4 (write-once, evict-first). Cin,Cout are multiples of 4.
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__device__ __forceinline__ void stcs_v4f32(float* p, float a, float b, float c, float d){
    asm volatile("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p),"f"(a),"f"(b),"f"(c),"f"(d));
}
template<int T>
__global__ void gather_shared(const float* __restrict__ x, const long* __restrict__ idx,
                              float* __restrict__ out, int Cin, int Cout){
    extern __shared__ float srow[];
    int row = blockIdx.x;
    // Phase 1: coalesced float4 load of the whole x-row into shared memory.
    const float4* xr4 = (const float4*)(x + (long)row * Cin);
    float4* s4 = (float4*)srow;
    int n4 = Cin >> 2;
    #pragma unroll 4
    for(int j = threadIdx.x; j < n4; j += T) s4[j] = __ldg(xr4 + j);
    __syncthreads();
    // Phase 2: each thread emits 4 consecutive outputs by gathering from shared.
    const long* ir = idx + (long)row * Cout;
    float* orow = out + (long)row * Cout;
    for(int base = threadIdx.x * 4; base < Cout; base += T * 4){
        longlong2 a = __ldg((const longlong2*)(ir + base));
        longlong2 b = __ldg((const longlong2*)(ir + base + 2));
        stcs_v4f32(orow + base,
                   srow[(int)a.x], srow[(int)a.y], srow[(int)b.x], srow[(int)b.y]);
    }
}
torch::Tensor gather_cuda(torch::Tensor x, torch::Tensor idx){
    long M = x.size(0), Cin = x.size(1), Cout = idx.size(1);
    auto out = torch::empty({M, Cout}, x.options());
    const int T = 256;
    int shbytes = (int)(Cin * sizeof(float));
    cudaFuncSetAttribute(gather_shared<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes);
    gather_shared<T><<<(unsigned)M, T, shbytes>>>(
        x.data_ptr<float>(), idx.data_ptr<long>(),
        out.data_ptr<float>(), (int)Cin, (int)Cout);
    return out;
}
"""
_CPP = "torch::Tensor gather_cuda(torch::Tensor x, torch::Tensor idx);"
_ext = load_inline(name="gather_cuda_unlimited_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["gather_cuda"], verbose=False, extra_cuda_cflags=["-O3"])

class Model(nn.Module):
    """gather(x, dim=1, index=idx): per-row shared-memory staging turns the
    scattered HBM gather into a coalesced row-load + on-chip random reads."""
    def forward(self, x, idx):
        return _ext.gather_cuda(x.contiguous(), idx.contiguous())
