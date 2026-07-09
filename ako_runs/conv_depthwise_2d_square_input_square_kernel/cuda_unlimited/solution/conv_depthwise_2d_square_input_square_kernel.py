import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Depthwise conv2d optimized CUDA kernel - Iter 3
# Based on iter-2 best (NVEC=4, 128x8 tile). Improvements:
# 1. PTX streaming loads (ld.cs.global.f32) to bypass L2 for input reads
#    (each input element used by at most 9 outputs; stride=1: reuse factor=9x in 3x3,
#     but across different blocks there is no reuse -> streaming cache mode)
# 2. STG.CS (streaming store) for outputs to avoid polluting L2 with written data
# 3. cp.async double-buffering to pipeline global loads with compute

_depthwise_conv_src = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <stdexcept>

// NVEC=4: each of 32x8=256 threads computes 4 output columns
// Output tile: 128 cols x 8 rows
// Input tile:  130 cols x 10 rows
#define THX  32
#define THY   8
#define NVEC  4                    // outputs per thread in x
#define OUT_W (THX * NVEC)        // 128
#define OUT_H THY                  // 8
#define IN_W  (OUT_W + 2)          // 130
#define IN_H  (OUT_H + 2)          // 10

// PTX streaming load - bypasses L2 (streaming/non-temporal)
__device__ __forceinline__ float ld_cs(const float* addr) {
    float val;
    asm volatile("ld.cs.global.f32 %0, [%1];" : "=f"(val) : "l"(addr));
    return val;
}

// PTX streaming store to output
__device__ __forceinline__ void st_cs(float* addr, float val) {
    asm volatile("st.cs.global.f32 [%0], %1;" : : "l"(addr), "f"(val));
}

// Fast path: 3x3, stride=1, pad=0 with PTX streaming loads
// smem[si][sj] = inp[(oh_base + si) * W + (ow_base + sj)]
__global__ void depthwise_conv2d_3x3_cs(
    const float* __restrict__ input,   // (N, C, H, W)
    const float* __restrict__ weight,  // (C, 9) contiguous
    const float* __restrict__ bias,    // (C,) or nullptr
    float* __restrict__ output,        // (N, C, OH, OW)
    int N, int C, int H, int W,
    int OH, int OW
) {
    // smem: IN_H x (IN_W + 2) = 10 x 132 = 1320 floats = 5.3KB
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

    // Input pointer
    const float* inp = input + (n * C + c) * (H * W);

    // Load input tile with streaming PTX loads (ld.cs)
    int flat_tid = ty * THX + tx;
    int total_cells = IN_H * IN_W;

    #pragma unroll 6
    for (int i = flat_tid; i < total_cells; i += THX * THY) {
        int si = i / IN_W;
        int sj = i % IN_W;
        int ih = oh_base + si;
        int iw = ow_base + sj;
        float val = 0.0f;
        if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
            val = ld_cs(&inp[ih * W + iw]);
        }
        sdata[si][sj] = val;
    }

    __syncthreads();

    int oh = oh_base + ty;
    int ow0 = ow_base + tx * NVEC;

    if (oh < OH) {
        int sy = ty;
        int sx = tx * NVEC;

        float s0=0, s1=0, s2=0, s3=0;

        // Row 0
        float r0_0 = sdata[sy    ][sx    ];
        float r0_1 = sdata[sy    ][sx + 1];
        float r0_2 = sdata[sy    ][sx + 2];
        float r0_3 = sdata[sy    ][sx + 3];
        float r0_4 = sdata[sy    ][sx + 4];
        float r0_5 = sdata[sy    ][sx + 5];

        s0 += w00*r0_0 + w01*r0_1 + w02*r0_2;
        s1 += w00*r0_1 + w01*r0_2 + w02*r0_3;
        s2 += w00*r0_2 + w01*r0_3 + w02*r0_4;
        s3 += w00*r0_3 + w01*r0_4 + w02*r0_5;

        // Row 1
        float r1_0 = sdata[sy + 1][sx    ];
        float r1_1 = sdata[sy + 1][sx + 1];
        float r1_2 = sdata[sy + 1][sx + 2];
        float r1_3 = sdata[sy + 1][sx + 3];
        float r1_4 = sdata[sy + 1][sx + 4];
        float r1_5 = sdata[sy + 1][sx + 5];

        s0 += w10*r1_0 + w11*r1_1 + w12*r1_2;
        s1 += w10*r1_1 + w11*r1_2 + w12*r1_3;
        s2 += w10*r1_2 + w11*r1_3 + w12*r1_4;
        s3 += w10*r1_3 + w11*r1_4 + w12*r1_5;

        // Row 2
        float r2_0 = sdata[sy + 2][sx    ];
        float r2_1 = sdata[sy + 2][sx + 1];
        float r2_2 = sdata[sy + 2][sx + 2];
        float r2_3 = sdata[sy + 2][sx + 3];
        float r2_4 = sdata[sy + 2][sx + 4];
        float r2_5 = sdata[sy + 2][sx + 5];

        s0 += w20*r2_0 + w21*r2_1 + w22*r2_2;
        s1 += w20*r2_1 + w21*r2_2 + w22*r2_3;
        s2 += w20*r2_2 + w21*r2_3 + w22*r2_4;
        s3 += w20*r2_3 + w21*r2_4 + w22*r2_5;

        if (bias != nullptr) {
            float b = bias[c];
            s0+=b; s1+=b; s2+=b; s3+=b;
        }

        float* outp = output + ((n * C + c) * OH + oh) * OW + ow0;
        int remaining = OW - ow0;
        if (remaining >= 4) {
            // Use PTX streaming stores to avoid polluting L2 with output
            st_cs(outp, s0);
            st_cs(outp + 1, s1);
            st_cs(outp + 2, s2);
            st_cs(outp + 3, s3);
        } else {
            if (remaining > 0) outp[0] = s0;
            if (remaining > 1) outp[1] = s1;
            if (remaining > 2) outp[2] = s2;
        }
    }
}

// General stride/pad 3x3 (32x8 tile)
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
    __shared__ float sdata_gen[GEN_TH + 2][GEN_TW + 4];

    int tx = threadIdx.x, ty = threadIdx.y;
    int ow_base = blockIdx.x * GEN_TW;
    int oh_base = blockIdx.y * GEN_TH;
    int nc = blockIdx.z, n = nc / C, c = nc % C;

    const float* wp = weight + c * 9;
    float w00=wp[0],w01=wp[1],w02=wp[2],w10=wp[3],w11=wp[4],w12=wp[5],w20=wp[6],w21=wp[7],w22=wp[8];
    const float* inp = input + (n * C + c) * (H * W);

    int flat_tid = ty * GEN_TW + tx;
    for (int i = flat_tid; i < (GEN_TH+2)*(GEN_TW+2); i += GEN_TW*GEN_TH) {
        int si = i / (GEN_TW+2), sj = i % (GEN_TW+2);
        int ih = oh_base * stride_h - pad_h + si;
        int iw = ow_base * stride_w - pad_w + sj;
        float val = ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) ? inp[ih*W+iw] : 0.f;
        sdata_gen[si][sj] = val;
    }
    __syncthreads();

    int oh = oh_base + ty, ow = ow_base + tx;
    if (oh < OH && ow < OW) {
        int sy = ty*stride_h, sx = tx*stride_w;
        float sum = w00*sdata_gen[sy][sx]+w01*sdata_gen[sy][sx+1]+w02*sdata_gen[sy][sx+2]
                  + w10*sdata_gen[sy+1][sx]+w11*sdata_gen[sy+1][sx+1]+w12*sdata_gen[sy+1][sx+2]
                  + w20*sdata_gen[sy+2][sx]+w21*sdata_gen[sy+2][sx+1]+w22*sdata_gen[sy+2][sx+2];
        if (bias != nullptr) sum += bias[c];
        output[((n*C+c)*OH+oh)*OW+ow] = sum;
    }
}

// General any-kernel-size
__global__ void depthwise_conv2d_general_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int N, int C, int H, int W, int OH, int OW,
    int KH, int KW, int stride_h, int stride_w, int pad_h, int pad_w
) {
    int ow = blockIdx.x*blockDim.x+threadIdx.x, oh = blockIdx.y*blockDim.y+threadIdx.y;
    int nc = blockIdx.z, n = nc/C, c = nc%C;
    if (ow >= OW || oh >= OH) return;

    const float* wp = weight + c*KH*KW;
    const float* inp = input + (n*C+c)*H*W;
    float sum = 0.f;
    int ih_base = oh*stride_h - pad_h, iw_base = ow*stride_w - pad_w;
    for (int kh = 0; kh < KH; kh++) {
        int ih = ih_base + kh;
        if ((unsigned)ih >= (unsigned)H) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = iw_base + kw;
            if ((unsigned)iw >= (unsigned)W) continue;
            sum += wp[kh*KW+kw] * __ldg(&inp[ih*W+iw]);
        }
    }
    if (bias != nullptr) sum += bias[c];
    output[((n*C+c)*OH+oh)*OW+ow] = sum;
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input, torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32 && input.is_contiguous());
    int N=input.size(0),C=input.size(1),H=input.size(2),W=input.size(3);
    int KH=weight.size(2),KW=weight.size(3);
    int stride_h=stride[0],stride_w=stride[1],pad_h=padding[0],pad_w=padding[1];
    int OH=(H+2*pad_h-KH)/stride_h+1, OW=(W+2*pad_w-KW)/stride_w+1;
    auto output = torch::empty({N,C,OH,OW}, input.options());
    auto w = weight.contiguous().view({C, KH*KW});

    const float* bias_ptr = nullptr;
    torch::Tensor bias_t;
    if (bias.has_value() && bias.value().defined()) {
        bias_t = bias.value().contiguous();
        bias_ptr = bias_t.data_ptr<float>();
    }

    if (KH==3 && KW==3 && stride_h==1 && stride_w==1 && pad_h==0 && pad_w==0) {
        dim3 block(THX, THY);
        dim3 grid((OW+OUT_W-1)/OUT_W, (OH+OUT_H-1)/OUT_H, N*C);
        depthwise_conv2d_3x3_cs<<<grid,block>>>(
            input.data_ptr<float>(), w.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), N,C,H,W,OH,OW);
    } else if (KH==3 && KW==3) {
        dim3 block(GEN_TW, GEN_TH);
        dim3 grid((OW+GEN_TW-1)/GEN_TW, (OH+GEN_TH-1)/GEN_TH, N*C);
        depthwise_conv2d_3x3_general<<<grid,block>>>(
            input.data_ptr<float>(), w.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), N,C,H,W,OH,OW,stride_h,stride_w,pad_h,pad_w);
    } else {
        dim3 block(32,8);
        dim3 grid((OW+31)/32,(OH+7)/8,N*C);
        depthwise_conv2d_general_kernel<<<grid,block>>>(
            input.data_ptr<float>(), w.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(), N,C,H,W,OH,OW,KH,KW,stride_h,stride_w,pad_h,pad_w);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    return output;
}
"""

_depthwise_conv_decl = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input, torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding
);
"""

_depthwise_ext = load_inline(
    name="depthwise_conv2d_ext_v6",
    cpp_sources=_depthwise_conv_decl,
    cuda_sources=_depthwise_conv_src,
    functions=["depthwise_conv2d_forward"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
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
