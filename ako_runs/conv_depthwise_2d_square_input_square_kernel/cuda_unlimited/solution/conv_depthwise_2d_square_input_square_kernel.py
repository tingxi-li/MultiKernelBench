import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Depthwise conv2d optimized CUDA kernel - Iter 2
# Wide tile approach: each thread computes 4 output columns
# Block: 32 x 8 = 256 threads
# Output tile: 128 cols x 8 rows
# Input tile (smem): 10 rows x 130 cols (no padding boundary for stride=1 pad=0 fast path)

_depthwise_conv_src = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Wide tile config
#define THX 32
#define THY 8
#define NVEC 4                    // output cols per thread
#define OUT_W  (THX * NVEC)       // 128
#define OUT_H  THY                // 8
// For 3x3 kernel, pad=0, stride=1:
//   Input rows needed: OUT_H + 2 = 10
//   Input cols needed: OUT_W + 2 = 130
#define IN_W   (OUT_W + 2)        // 130
#define IN_H   (OUT_H + 2)        // 10

// Fast path: 3x3, stride=1, pad=0
// Correct smem indexing:
//   sdata[si][sj] = inp[(oh_base + si) * W + (ow_base + sj)]
//   for si in [0..IN_H-1], sj in [0..IN_W-1]
//   Output ty, kernel row r: sdata[ty + r]
//   Output tx*NVEC + m, kernel col s: sdata[...][tx*NVEC + m + s]
__global__ void depthwise_conv2d_3x3_widetile(
    const float* __restrict__ input,   // (N, C, H, W)
    const float* __restrict__ weight,  // (C, 9) contiguous
    const float* __restrict__ bias,    // (C,) or nullptr
    float* __restrict__ output,        // (N, C, OH, OW)
    int N, int C, int H, int W,
    int OH, int OW
) {
    // Smem: IN_H rows x (IN_W+2) cols (+2 to reduce bank conflicts)
    __shared__ float sdata[IN_H][IN_W + 2];

    int tx = threadIdx.x;  // 0..31
    int ty = threadIdx.y;  // 0..7
    int ow_base = blockIdx.x * OUT_W;  // multiples of 128
    int oh_base = blockIdx.y * OUT_H;  // multiples of 8
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    // Load 9 weights into registers
    const float* wp = weight + c * 9;
    float w00 = wp[0], w01 = wp[1], w02 = wp[2];
    float w10 = wp[3], w11 = wp[4], w12 = wp[5];
    float w20 = wp[6], w21 = wp[7], w22 = wp[8];

    // Input pointer for (n, c) slice
    const float* inp = input + (n * C + c) * (H * W);

    // Load input tile into shared memory
    // Total cells: IN_H * IN_W = 10 * 130 = 1300
    // 256 threads -> ~5.1 cells each
    int flat_tid = ty * THX + tx;  // 0..255

    #pragma unroll 6
    for (int i = flat_tid; i < IN_H * IN_W; i += THX * THY) {
        int si = i / IN_W;   // row in smem [0..IN_H-1]
        int sj = i % IN_W;   // col in smem [0..IN_W-1]
        int ih = oh_base + si;
        int iw = ow_base + sj;
        float val = 0.0f;
        if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
            val = __ldg(&inp[ih * W + iw]);
        }
        sdata[si][sj] = val;
    }

    __syncthreads();

    int oh = oh_base + ty;
    int ow0 = ow_base + tx * NVEC;

    if (oh < OH) {
        int sy = ty;          // sdata row for kernel row 0
        int sx = tx * NVEC;   // sdata col for output col ow0

        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        // Kernel row 0
        float r0s0 = sdata[sy    ][sx    ];
        float r0s1 = sdata[sy    ][sx + 1];
        float r0s2 = sdata[sy    ][sx + 2];
        float r0s3 = sdata[sy    ][sx + 3];
        float r0s4 = sdata[sy    ][sx + 4];
        float r0s5 = sdata[sy    ][sx + 5];

        sum0 += w00 * r0s0 + w01 * r0s1 + w02 * r0s2;
        sum1 += w00 * r0s1 + w01 * r0s2 + w02 * r0s3;
        sum2 += w00 * r0s2 + w01 * r0s3 + w02 * r0s4;
        sum3 += w00 * r0s3 + w01 * r0s4 + w02 * r0s5;

        // Kernel row 1
        float r1s0 = sdata[sy + 1][sx    ];
        float r1s1 = sdata[sy + 1][sx + 1];
        float r1s2 = sdata[sy + 1][sx + 2];
        float r1s3 = sdata[sy + 1][sx + 3];
        float r1s4 = sdata[sy + 1][sx + 4];
        float r1s5 = sdata[sy + 1][sx + 5];

        sum0 += w10 * r1s0 + w11 * r1s1 + w12 * r1s2;
        sum1 += w10 * r1s1 + w11 * r1s2 + w12 * r1s3;
        sum2 += w10 * r1s2 + w11 * r1s3 + w12 * r1s4;
        sum3 += w10 * r1s3 + w11 * r1s4 + w12 * r1s5;

        // Kernel row 2
        float r2s0 = sdata[sy + 2][sx    ];
        float r2s1 = sdata[sy + 2][sx + 1];
        float r2s2 = sdata[sy + 2][sx + 2];
        float r2s3 = sdata[sy + 2][sx + 3];
        float r2s4 = sdata[sy + 2][sx + 4];
        float r2s5 = sdata[sy + 2][sx + 5];

        sum0 += w20 * r2s0 + w21 * r2s1 + w22 * r2s2;
        sum1 += w20 * r2s1 + w21 * r2s2 + w22 * r2s3;
        sum2 += w20 * r2s2 + w21 * r2s3 + w22 * r2s4;
        sum3 += w20 * r2s3 + w21 * r2s4 + w22 * r2s5;

        if (bias != nullptr) {
            float b = bias[c];
            sum0 += b; sum1 += b; sum2 += b; sum3 += b;
        }

        // Write 4 outputs
        float* outp = output + ((n * C + c) * OH + oh) * OW + ow0;
        int remaining = OW - ow0;
        if (remaining >= 4) {
            outp[0] = sum0; outp[1] = sum1; outp[2] = sum2; outp[3] = sum3;
        } else {
            if (remaining > 0) outp[0] = sum0;
            if (remaining > 1) outp[1] = sum1;
            if (remaining > 2) outp[2] = sum2;
        }
    }
}

// General stride/pad 3x3 with smem tiling (32x8 tile)
#define GEN_TW 32
#define GEN_TH 8
__global__ void depthwise_conv2d_3x3_general(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int OH, int OW,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    // Smem: (GEN_TH + 2) rows x (GEN_TW + 4) cols
    __shared__ float sdata_gen[GEN_TH + 2][GEN_TW + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow_base = blockIdx.x * GEN_TW;
    int oh_base = blockIdx.y * GEN_TH;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    const float* wp = weight + c * 9;
    float w00 = wp[0], w01 = wp[1], w02 = wp[2];
    float w10 = wp[3], w11 = wp[4], w12 = wp[5];
    float w20 = wp[6], w21 = wp[7], w22 = wp[8];
    const float* inp = input + (n * C + c) * (H * W);

    // Load: sdata_gen[si][sj] = inp[(oh_base*stride_h - pad_h + si) * W + (ow_base*stride_w - pad_w + sj)]
    int tile_h = GEN_TH + 2;
    int tile_w = GEN_TW + 2;
    int flat_tid = ty * GEN_TW + tx;
    for (int i = flat_tid; i < tile_h * tile_w; i += GEN_TW * GEN_TH) {
        int si = i / tile_w;
        int sj = i % tile_w;
        int ih = oh_base * stride_h - pad_h + si;
        int iw = ow_base * stride_w - pad_w + sj;
        float val = 0.0f;
        if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
            val = inp[ih * W + iw];
        }
        sdata_gen[si][sj] = val;
    }

    __syncthreads();

    int oh = oh_base + ty;
    int ow = ow_base + tx;

    if (oh < OH && ow < OW) {
        int sy = ty * stride_h;
        int sx = tx * stride_w;
        float sum = 0.0f;
        sum += w00 * sdata_gen[sy    ][sx    ];
        sum += w01 * sdata_gen[sy    ][sx + 1];
        sum += w02 * sdata_gen[sy    ][sx + 2];
        sum += w10 * sdata_gen[sy + 1][sx    ];
        sum += w11 * sdata_gen[sy + 1][sx + 1];
        sum += w12 * sdata_gen[sy + 1][sx + 2];
        sum += w20 * sdata_gen[sy + 2][sx    ];
        sum += w21 * sdata_gen[sy + 2][sx + 1];
        sum += w22 * sdata_gen[sy + 2][sx + 2];
        if (bias != nullptr) sum += bias[c];
        output[((n * C + c) * OH + oh) * OW + ow] = sum;
    }
}

// General any-kernel-size
__global__ void depthwise_conv2d_general_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int OH, int OW,
    int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    if (ow >= OW || oh >= OH) return;

    const float* wp = weight + c * KH * KW;
    const float* inp = input + (n * C + c) * H * W;

    float sum = 0.0f;
    int ih_base = oh * stride_h - pad_h;
    int iw_base = ow * stride_w - pad_w;

    for (int kh = 0; kh < KH; kh++) {
        int ih = ih_base + kh;
        if ((unsigned)ih >= (unsigned)H) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = iw_base + kw;
            if ((unsigned)iw >= (unsigned)W) continue;
            sum += wp[kh * KW + kw] * __ldg(&inp[ih * W + iw]);
        }
    }

    if (bias != nullptr) sum += bias[c];
    output[((n * C + c) * OH + oh) * OW + ow] = sum;
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int KH = weight.size(2);
    int KW = weight.size(3);
    int stride_h = stride[0], stride_w = stride[1];
    int pad_h = padding[0], pad_w = padding[1];

    int OH = (H + 2 * pad_h - KH) / stride_h + 1;
    int OW = (W + 2 * pad_w - KW) / stride_w + 1;

    auto output = torch::empty({N, C, OH, OW}, input.options());
    auto w = weight.contiguous().view({C, KH * KW});

    const float* bias_ptr = nullptr;
    torch::Tensor bias_tensor;
    if (bias.has_value() && bias.value().defined()) {
        bias_tensor = bias.value().contiguous();
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    if (KH == 3 && KW == 3 && stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0) {
        dim3 block(THX, THY);
        dim3 grid(
            (OW + OUT_W - 1) / OUT_W,
            (OH + OUT_H - 1) / OUT_H,
            N * C
        );
        depthwise_conv2d_3x3_widetile<<<grid, block>>>(
            input.data_ptr<float>(),
            w.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            N, C, H, W, OH, OW
        );
    } else if (KH == 3 && KW == 3) {
        dim3 block(GEN_TW, GEN_TH);
        dim3 grid(
            (OW + GEN_TW - 1) / GEN_TW,
            (OH + GEN_TH - 1) / GEN_TH,
            N * C
        );
        depthwise_conv2d_3x3_general<<<grid, block>>>(
            input.data_ptr<float>(),
            w.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            N, C, H, W, OH, OW,
            stride_h, stride_w, pad_h, pad_w
        );
    } else {
        dim3 block(32, 8);
        dim3 grid(
            (OW + 31) / 32,
            (OH + 7) / 8,
            N * C
        );
        depthwise_conv2d_general_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            w.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            N, C, H, W, OH, OW,
            KH, KW, stride_h, stride_w, pad_h, pad_w
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    return output;
}
"""

_depthwise_conv_decl = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
);
"""

_depthwise_ext = load_inline(
    name="depthwise_conv2d_ext_v4",
    cpp_sources=_depthwise_conv_decl,
    cuda_sources=_depthwise_conv_src,
    functions=["depthwise_conv2d_forward"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    """
    Optimized depthwise 2D convolution using custom CUDA kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride = [self.conv2d.stride[0], self.conv2d.stride[1]]
        padding = [self.conv2d.padding[0], self.conv2d.padding[1]]
        bias = self.conv2d.bias
        return _depthwise_ext.depthwise_conv2d_forward(
            x, self.conv2d.weight, bias, stride, padding
        )
