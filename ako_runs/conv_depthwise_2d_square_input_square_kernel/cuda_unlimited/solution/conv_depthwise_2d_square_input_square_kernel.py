import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TH 4
#define THREADS 256

// Depthwise 3x3 conv, stride 1, pad 0, NCHW. One block does a full-width strip of
// TH output rows for one (n,c). Loads (TH+2) contiguous input rows into shared via
// float4 (perfect coalescing), then computes from shared. Input read ~once (L2
// absorbs the 2-row strip overlap); DRAM sees 1 read + 1 write.
extern "C" __global__ void dwconv_strip(
        const float* __restrict__ X, const float* __restrict__ Wt,
        float* __restrict__ Y, int C, int H, int W, int OH, int OW) {
    extern __shared__ float sh[];      // (TH+2)*W
    __shared__ float ws[9];

    const int nc = blockIdx.y;
    const int c = nc % C;
    const int ohBase = blockIdx.x * TH;
    const int tid = threadIdx.x;
    const int rows = TH + 2;

    if (tid < 9) ws[tid] = Wt[c * 9 + tid];

    const float* Xnc = X + (long)nc * H * W;
    // load rows [ohBase, ohBase+rows) x W into shared, float4 coalesced
    int total4 = (rows * W) >> 2;
    for (int i = tid; i < total4; i += THREADS) {
        int idx = i << 2;
        int r = idx / W, col = idx % W;
        int ih = ohBase + r;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (ih < H) v = *reinterpret_cast<const float4*>(&Xnc[ih * W + col]);
        *reinterpret_cast<float4*>(&sh[idx]) = v;
    }
    __syncthreads();

    const float w0 = ws[0], w1 = ws[1], w2 = ws[2];
    const float w3 = ws[3], w4 = ws[4], w5 = ws[5];
    const float w6 = ws[6], w7 = ws[7], w8 = ws[8];

    #pragma unroll
    for (int orow = 0; orow < TH; ++orow) {
        int oh = ohBase + orow;
        if (oh >= OH) break;
        const float* r0 = &sh[(orow + 0) * W];
        const float* r1 = &sh[(orow + 1) * W];
        const float* r2 = &sh[(orow + 2) * W];
        float* Yrow = &Y[((long)nc * OH + oh) * OW];
        for (int ow = tid; ow < OW; ow += THREADS) {
            float acc = r0[ow] * w0 + r0[ow + 1] * w1 + r0[ow + 2] * w2
                      + r1[ow] * w3 + r1[ow + 1] * w4 + r1[ow + 2] * w5
                      + r2[ow] * w6 + r2[ow + 1] * w7 + r2[ow + 2] * w8;
            Yrow[ow] = acc;
        }
    }
}

torch::Tensor run(torch::Tensor X, torch::Tensor Wt) {
    TORCH_CHECK(X.is_cuda(), "cuda only");
    auto Xc = X.contiguous();
    auto Wc = Wt.contiguous();
    int N = Xc.size(0), C = Xc.size(1), H = Xc.size(2), W = Xc.size(3);
    int KS = Wc.size(2);
    int OH = H - KS + 1, OW = W - KS + 1;
    auto Y = torch::empty({N, C, OH, OW}, Xc.options());
    dim3 grid((OH + TH - 1) / TH, N * C);
    size_t shmem = (size_t)(TH + 2) * W * sizeof(float);
    dwconv_strip<<<grid, THREADS, shmem>>>(Xc.data_ptr<float>(), Wc.data_ptr<float>(),
                                           Y.data_ptr<float>(), C, H, W, OH, OW);
    return Y;
}
'''

_CPP = "torch::Tensor run(torch::Tensor X, torch::Tensor Wt);"

_ext = load_inline(
    name="dwconv_unlim_v3",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["run"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size,
                                stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        return _ext.run(x, self.conv2d.weight)


batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
