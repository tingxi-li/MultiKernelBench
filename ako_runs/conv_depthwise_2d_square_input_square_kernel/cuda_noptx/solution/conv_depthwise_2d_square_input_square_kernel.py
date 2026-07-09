import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------
# CUDA kernel v6: best of all worlds — wide coalesced tiles + multi-row
#
# Best config from iter 5: TW=30, SW=32 (coalesced fill), TH=16, maxreg=40
# This iter: push further with TW=62, SW=64 (wider coalescing) and
# verify against TW=30,SW=32,TH=16 which was the winner in iter 5.
#
# Also explore: "output-stationary" approach where each thread computes
# OUT_ROWS consecutive output rows by reloading weights once per batch.
# -----------------------------------------------------------------------
_CUDA_SOURCE = r"""
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────
// Primary: TW=62, SW=64, TH=8, BW=64, block=64×8=512 threads
// SM = 64×10 = 640 floats = 2560 B
// 640/512 = 1.25 loads/thread; each warp (64-wide) covers one SM row: COALESCED
// ─────────────────────────────────────────────────────────────────
#define TW_A  62
#define TH_A   8
#define BW_A  64
#define SW_A  64   // TW_A + 2 = 64 → power of 2
#define SH_A  10   // TH_A + 2 = 10

__global__ void dw_k3s1_62x8(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W
) {
    __shared__ float sm[SH_A * SW_A];   // 640 floats

    const int bc     = blockIdx.z;
    const int b      = bc / C, c = bc % C;
    const int tile_y = blockIdx.y * TH_A;
    const int tile_x = blockIdx.x * TW_A;
    const int tx     = threadIdx.x;  // 0..63
    const int ty     = threadIdx.y;  // 0..7
    const int tid    = ty * BW_A + tx;  // 0..511

    const float* xc = x + (b * C + c) * (H * W);

    // Coalesced fill: SW_A=64 → each warp covers one SM row exactly
    #pragma unroll 2
    for (int i = tid; i < SH_A * SW_A; i += BW_A * TH_A) {
        const int sy = i >> 6;   // i / 64
        const int sx = i & 63;   // i % 64
        const int gy = tile_y + sy;
        const int gx = tile_x + sx;
        sm[i] = (gy < H && gx < W) ? __ldg(&xc[gy * W + gx]) : 0.0f;
    }
    __syncthreads();

    const int out_y = tile_y + ty;
    const int out_x = tile_x + tx;

    if (tx < TW_A && out_y < out_H && out_x < out_W) {
        const float* wc = w + c * 9;
        float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
        float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
        float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

        float s;
        s  = w00 * sm[(ty+0)*SW_A + (tx+0)];
        s += w01 * sm[(ty+0)*SW_A + (tx+1)];
        s += w02 * sm[(ty+0)*SW_A + (tx+2)];
        s += w10 * sm[(ty+1)*SW_A + (tx+0)];
        s += w11 * sm[(ty+1)*SW_A + (tx+1)];
        s += w12 * sm[(ty+1)*SW_A + (tx+2)];
        s += w20 * sm[(ty+2)*SW_A + (tx+0)];
        s += w21 * sm[(ty+2)*SW_A + (tx+1)];
        s += w22 * sm[(ty+2)*SW_A + (tx+2)];

        y[(b * C + c) * (out_H * out_W) + out_y * out_W + out_x] = s;
    }
}

// ─────────────────────────────────────────────────────────────────
// Best from iter 5: TW=30, SW=32, TH=16, block=32×16=512
// (kept for comparison — was the winner at 1.42x)
// ─────────────────────────────────────────────────────────────────
#define TW_B  30
#define TH_B  16
#define BW_B  32
#define SW_B  32
#define SH_B  18

__global__ void dw_k3s1_30x16(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W
) {
    __shared__ float sm[SH_B * SW_B];

    const int bc     = blockIdx.z;
    const int b      = bc / C, c = bc % C;
    const int tile_y = blockIdx.y * TH_B;
    const int tile_x = blockIdx.x * TW_B;
    const int tx     = threadIdx.x;
    const int ty     = threadIdx.y;
    const int tid    = ty * BW_B + tx;

    const float* xc = x + (b * C + c) * (H * W);

    #pragma unroll 2
    for (int i = tid; i < SH_B * SW_B; i += BW_B * TH_B) {
        const int sy = i >> 5;
        const int sx = i & 31;
        const int gy = tile_y + sy;
        const int gx = tile_x + sx;
        sm[i] = (gy < H && gx < W) ? __ldg(&xc[gy * W + gx]) : 0.0f;
    }
    __syncthreads();

    const int out_y = tile_y + ty;
    const int out_x = tile_x + tx;

    if (tx < TW_B && out_y < out_H && out_x < out_W) {
        const float* wc = w + c * 9;
        float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
        float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
        float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

        float s;
        s  = w00 * sm[(ty+0)*SW_B + (tx+0)];
        s += w01 * sm[(ty+0)*SW_B + (tx+1)];
        s += w02 * sm[(ty+0)*SW_B + (tx+2)];
        s += w10 * sm[(ty+1)*SW_B + (tx+0)];
        s += w11 * sm[(ty+1)*SW_B + (tx+1)];
        s += w12 * sm[(ty+1)*SW_B + (tx+2)];
        s += w20 * sm[(ty+2)*SW_B + (tx+0)];
        s += w21 * sm[(ty+2)*SW_B + (tx+1)];
        s += w22 * sm[(ty+2)*SW_B + (tx+2)];

        y[(b * C + c) * (out_H * out_W) + out_y * out_W + out_x] = s;
    }
}

// ─────────────────────────────────────────────────────────────────
// Variant C: TW=126, SW=128, TH=4, BW=128, block=128×4=512 threads
// Very wide coalescing: 128 consecutive threads fill one 128-float SM row
// = 512 bytes = 4 cache lines per SM row fill
// SM: 6×128 = 768 floats = 3072 B
// ─────────────────────────────────────────────────────────────────
#define TW_C  126
#define TH_C    4
#define BW_C  128
#define SW_C  128   // power of 2
#define SH_C    6   // TH_C + 2

__global__ void dw_k3s1_126x4(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W
) {
    __shared__ float sm[SH_C * SW_C];

    const int bc     = blockIdx.z;
    const int b      = bc / C, c = bc % C;
    const int tile_y = blockIdx.y * TH_C;
    const int tile_x = blockIdx.x * TW_C;
    const int tx     = threadIdx.x;  // 0..127
    const int ty     = threadIdx.y;  // 0..3
    const int tid    = ty * BW_C + tx;

    const float* xc = x + (b * C + c) * (H * W);

    #pragma unroll 2
    for (int i = tid; i < SH_C * SW_C; i += BW_C * TH_C) {
        const int sy = i >> 7;   // i / 128
        const int sx = i & 127;  // i % 128
        const int gy = tile_y + sy;
        const int gx = tile_x + sx;
        sm[i] = (gy < H && gx < W) ? __ldg(&xc[gy * W + gx]) : 0.0f;
    }
    __syncthreads();

    const int out_y = tile_y + ty;
    const int out_x = tile_x + tx;

    if (tx < TW_C && out_y < out_H && out_x < out_W) {
        const float* wc = w + c * 9;
        float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
        float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
        float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

        float s;
        s  = w00 * sm[(ty+0)*SW_C + (tx+0)];
        s += w01 * sm[(ty+0)*SW_C + (tx+1)];
        s += w02 * sm[(ty+0)*SW_C + (tx+2)];
        s += w10 * sm[(ty+1)*SW_C + (tx+0)];
        s += w11 * sm[(ty+1)*SW_C + (tx+1)];
        s += w12 * sm[(ty+1)*SW_C + (tx+2)];
        s += w20 * sm[(ty+2)*SW_C + (tx+0)];
        s += w21 * sm[(ty+2)*SW_C + (tx+1)];
        s += w22 * sm[(ty+2)*SW_C + (tx+2)];

        y[(b * C + c) * (out_H * out_W) + out_y * out_W + out_x] = s;
    }
}

// ─────────────────────────────────────────────────────────────────
// General fallback
// ─────────────────────────────────────────────────────────────────
__global__ void dw_conv_general(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W,
    int KH, int KW, int stride, int padding
) {
    extern __shared__ float sm[];
    const int smW = 32 + KW - 1;
    const int bc = blockIdx.z;
    const int b = bc / C, c = bc % C;
    const int tile_y = blockIdx.y * 8, tile_x = blockIdx.x * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 32 + tx;
    const float* xc = x + (b * C + c) * (H * W);
    const int in_y0 = tile_y * stride - padding, in_x0 = tile_x * stride - padding;
    const int smH = 8 + KH - 1;
    for (int i = tid; i < smH * smW; i += 256) {
        const int sy = i / smW, sx = i % smW;
        const int gy = in_y0 + sy, gx = in_x0 + sx;
        sm[i] = (gy >= 0 && gy < H && gx >= 0 && gx < W) ? __ldg(&xc[gy*W+gx]) : 0.0f;
    }
    __syncthreads();
    const int out_y = tile_y + ty, out_x = tile_x + tx;
    if (out_y < out_H && out_x < out_W) {
        const float* wc = w + c * KH * KW;
        float s = 0.0f;
        for (int ky = 0; ky < KH; ++ky)
            for (int kx = 0; kx < KW; ++kx)
                s += __ldg(&wc[ky*KW+kx]) * sm[(ty*stride+ky)*smW+(tx*stride+kx)];
        y[(b*C+c)*(out_H*out_W) + out_y*out_W + out_x] = s;
    }
}

void launch_dw_conv(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    int stride,
    int padding,
    int variant
) {
    const int B = x.size(0), C = x.size(1);
    const int H = x.size(2), W = x.size(3);
    const int KH = w.size(2), KW = w.size(3);
    const int out_H = y.size(2), out_W = y.size(3);

    if (KH == 3 && KW == 3 && stride == 1 && padding == 0) {
        if (variant == 1) {
            // TW=62, SW=64, TH=8, block=64×8=512
            const dim3 block(BW_A, TH_A);
            const dim3 grid(
                (out_W + TW_A - 1) / TW_A,
                (out_H + TH_A - 1) / TH_A,
                B * C
            );
            dw_k3s1_62x8<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W
            );
        } else if (variant == 2) {
            // TW=30, SW=32, TH=16, block=32×16=512 (iter 5 winner)
            const dim3 block(BW_B, TH_B);
            const dim3 grid(
                (out_W + TW_B - 1) / TW_B,
                (out_H + TH_B - 1) / TH_B,
                B * C
            );
            dw_k3s1_30x16<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W
            );
        } else {
            // TW=126, SW=128, TH=4, block=128×4=512
            const dim3 block(BW_C, TH_C);
            const dim3 grid(
                (out_W + TW_C - 1) / TW_C,
                (out_H + TH_C - 1) / TH_C,
                B * C
            );
            dw_k3s1_126x4<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W
            );
        }
    } else {
        const dim3 block(32, 8);
        const dim3 grid(
            (out_W + 31) / 32,
            (out_H + 7) / 8,
            B * C
        );
        const int smem = (8 + KH - 1) * (32 + KW - 1) * sizeof(float);
        dw_conv_general<<<grid, block, smem>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
            B, C, H, W, out_H, out_W, KH, KW, stride, padding
        );
    }
}
"""

_CPP_SOURCE = """
void launch_dw_conv(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    int stride,
    int padding,
    int variant
);
"""

_mod = None

def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="dw_conv2d_noptx_v6",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["launch_dw_conv"],
            extra_cuda_cflags=["-O3", "--use_fast_math",
                               "-Xptxas", "-O3,--maxrregcount=40"],
            verbose=False,
        )
    return _mod


class Model(nn.Module):
    """
    Depthwise 2-D convolution.
    Variant 1: TW=62, SW=64 (wide coalesced tiles).
    Variant 2: TW=30, SW=32 (iter-5 winner).
    Variant 3: TW=126, SW=128 (ultra-wide).
    """
    def __init__(self, in_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )
        self._variant = 1  # default: TW=62 wide

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv2d.weight.contiguous()
        pad    = self.conv2d.padding[0]
        stride = self.conv2d.stride[0]
        KH, KW = w.shape[2], w.shape[3]
        out_H  = (x.shape[2] + 2 * pad - KH) // stride + 1
        out_W  = (x.shape[3] + 2 * pad - KW) // stride + 1
        out = torch.empty(x.shape[0], x.shape[1], out_H, out_W,
                          dtype=x.dtype, device=x.device)
        _get_mod().launch_dw_conv(x.contiguous(), w, out, stride, pad, self._variant)
        return out
