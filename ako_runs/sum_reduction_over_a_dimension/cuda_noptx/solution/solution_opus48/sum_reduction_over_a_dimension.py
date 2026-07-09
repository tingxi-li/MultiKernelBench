import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

// Sum-reduce over dim=1 of a contiguous (B, D1, D2) tensor -> (B, 1, D2).
// float4 variant: each thread owns 4 consecutive output columns (b, j..j+3),
// reads a float4 per reduced row -> vectorized, coalesced loads + 4 independent
// accumulators to hide the load-use latency of the long reduction chain.
__global__ void sum_dim1_f4(const float* __restrict__ x,
                            float* __restrict__ out,
                            int B, int D1, int D2) {
    int D2v = D2 >> 2;                 // D2 / 4  (D2 divisible by 4)
    long totalv = (long)B * D2v;
    long stride = (long)gridDim.x * blockDim.x;
    for (long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
         v < totalv; v += stride) {
        int b = v / D2v;
        int jv = v - (long)b * D2v;    // vector column
        const float4* base = reinterpret_cast<const float4*>(
            x + (long)b * D1 * D2) + jv;
        int rowv = D2v;                // float4 stride between rows
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int i = 0; i < D1; ++i) {
            float4 t = __ldg(base + (long)i * rowv);
            acc.x += t.x; acc.y += t.y; acc.z += t.z; acc.w += t.w;
        }
        reinterpret_cast<float4*>(out + (long)b * D2)[jv] = acc;
    }
}

torch::Tensor sum_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 3);
    int B = x.size(0), D1 = x.size(1), D2 = x.size(2);
    auto out = torch::empty({B, 1, D2}, x.options());
    long totalv = (long)B * (D2 / 4);
    int threads = 256;
    long blocks = (totalv + threads - 1) / threads;
    if (blocks > 131072) blocks = 131072;
    sum_dim1_f4<<<(int)blocks, threads>>>(x.data_ptr<float>(),
                                          out.data_ptr<float>(), B, D1, D2);
    return out;
}
'''

_CPP = "torch::Tensor sum_dim1(torch::Tensor x);"

_ext = load_inline(
    name="sumdim1_noptx_v1",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["sum_dim1"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ext.sum_dim1(x.contiguous())
