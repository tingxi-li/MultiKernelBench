import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

// Depthwise conv2d, square kernel. weight [C,1,K,K]. Coalesced over OW.
// Specialized 3x3 stride-1 pad-0 fast path (the benched config); general fallback.

// Each thread computes RPT outputs down the OH axis for a fixed OW, reusing the
// 3 overlapping input rows in registers: RPT+2 row-reads (x3 cols) -> RPT outputs.
#define RPT 4
__global__ void dw3x3s1(const float* __restrict__ X, const float* __restrict__ W,
                        float* __restrict__ Y, int C, int IH, int IW,
                        int OH, int OW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (ow >= OW) return;
    int oh0 = blockIdx.y * RPT;
    int bc = blockIdx.z;
    int c = bc % C;
    const float* wc = W + (long)c * 9;
    float w0 = __ldg(wc + 0), w1 = __ldg(wc + 1), w2 = __ldg(wc + 2);
    float w3 = __ldg(wc + 3), w4 = __ldg(wc + 4), w5 = __ldg(wc + 5);
    float w6 = __ldg(wc + 6), w7 = __ldg(wc + 7), w8 = __ldg(wc + 8);

    float c0[RPT + 2], c1[RPT + 2], c2[RPT + 2];   // 3 cols of rows oh0..oh0+RPT+1
    const float* base = X + (long)bc * IH * IW + ow;
    #pragma unroll
    for (int r = 0; r < RPT + 2; r++) {
        int ih = oh0 + r;
        if (ih < IH) {
            const float* p = base + (long)ih * IW;
            c0[r] = __ldg(p); c1[r] = __ldg(p + 1); c2[r] = __ldg(p + 2);
        } else { c0[r] = c1[r] = c2[r] = 0.f; }
    }
    #pragma unroll
    for (int i = 0; i < RPT; i++) {
        int oh = oh0 + i;
        if (oh >= OH) break;
        float acc = w0 * c0[i]     + w1 * c1[i]     + w2 * c2[i]
                  + w3 * c0[i + 1] + w4 * c1[i + 1] + w5 * c2[i + 1]
                  + w6 * c0[i + 2] + w7 * c1[i + 2] + w8 * c2[i + 2];
        Y[((long)bc * OH + oh) * OW + ow] = acc;
    }
}

__global__ void dw_general(const float* __restrict__ X, const float* __restrict__ W,
                           float* __restrict__ Y, int C, int IH, int IW, int K,
                           int stride, int pad, int OH, int OW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (ow >= OW) return;
    int oh = blockIdx.y;
    int bc = blockIdx.z;
    int c = bc % C;
    const float* wc = W + (long)c * K * K;
    const float* xb = X + (long)bc * IH * IW;
    int ih0 = oh * stride - pad;
    int iw0 = ow * stride - pad;
    float acc = 0.f;
    for (int kh = 0; kh < K; kh++) {
        int ih = ih0 + kh;
        if (ih < 0 || ih >= IH) continue;
        for (int kw = 0; kw < K; kw++) {
            int iw = iw0 + kw;
            if (iw < 0 || iw >= IW) continue;
            acc += __ldg(xb + (long)ih * IW + iw) * __ldg(wc + kh * K + kw);
        }
    }
    Y[((long)bc * OH + oh) * OW + ow] = acc;
}

torch::Tensor dwconv(torch::Tensor X, torch::Tensor W, int stride, int pad) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda());
    int B = X.size(0), C = X.size(1), IH = X.size(2), IW = X.size(3);
    int K = W.size(2);
    int OH = (IH + 2 * pad - K) / stride + 1;
    int OW = (IW + 2 * pad - K) / stride + 1;
    auto Y = torch::empty({B, C, OH, OW}, X.options());
    int TPB = 256;
    if (K == 3 && stride == 1 && pad == 0) {
        dim3 grid((OW + TPB - 1) / TPB, (OH + RPT - 1) / RPT, B * C);
        dw3x3s1<<<grid, TPB>>>(X.data_ptr<float>(), W.data_ptr<float>(),
                               Y.data_ptr<float>(), C, IH, IW, OH, OW);
    } else {
        dim3 grid((OW + TPB - 1) / TPB, OH, B * C);
        dw_general<<<grid, TPB>>>(X.data_ptr<float>(), W.data_ptr<float>(),
                                  Y.data_ptr<float>(), C, IH, IW, K, stride, pad, OH, OW);
    }
    return Y;
}
'''

_CPP = "torch::Tensor dwconv(torch::Tensor X, torch::Tensor W, int stride, int pad);"

_ext = load_inline(
    name="dwconv_noptx",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["dwconv"],
    verbose=False,
    extra_cuda_cflags=["-O3"],
)


class Model(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                                padding=padding, groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ext.dwconv(x.contiguous(), self.conv2d.weight,
                           self.conv2d.stride[0], self.conv2d.padding[0])
