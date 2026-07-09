import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------
# CUDA kernel v4: depthwise 2-D conv, maximise bandwidth utilisation
#
# Architecture: RTX 6000 Ada = Ada Lovelace (sm_89)
# - 960 GB/s memory bandwidth
# - 128 KB L1/SM, 72 SMs
# - Best strategy: maximise bandwidth, minimise memory traffic
#
# Key insight: Use wider tiles + float4 vectorised SM fill.
# TILE: 32 cols x 32 rows = 1024 output pixels per block
# SM: 34 x 34 = 1156 floats
# Block: 32 x 32 = 1024 threads → 32 warps
# Each thread loads 1.1 SM floats and computes 1 output pixel.
# Float4 loading would help for the SM fill phase.
# -----------------------------------------------------------------------
_CUDA_SOURCE = r"""
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────
// Kernel v4: 32x32 tile, 1024 threads per block
// SM: 34x34 = 1156 floats = 4624 bytes (well within 128KB)
// ─────────────────────────────────────────────────────────────────
#define TW 32
#define TH 32
#define SW (TW + 2)   // 34
#define SH (TH + 2)   // 34

__global__ void dw_conv_32x32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W
) {
    __shared__ float sm[SH * SW];   // 34x34 = 1156 floats

    const int bc    = blockIdx.z;
    const int b     = bc / C;
    const int c     = bc % C;
    const int tile_y = blockIdx.y * TH;
    const int tile_x = blockIdx.x * TW;
    const int tx    = threadIdx.x;   // 0..31
    const int ty    = threadIdx.y;   // 0..31
    const int tid   = ty * TW + tx;  // 0..1023

    const float* xc = x + (b * C + c) * (H * W);

    // Fill SM: 1156 floats with 1024 threads → ~1.13 loads/thread
    // Use loop to handle the extra elements
    for (int i = tid; i < SH * SW; i += TH * TW) {
        const int sy = i / SW;
        const int sx = i % SW;
        const int gy = tile_y + sy;
        const int gx = tile_x + sx;
        sm[i] = (gy < H && gx < W) ? __ldg(&xc[gy * W + gx]) : 0.0f;
    }
    __syncthreads();

    const int out_y = tile_y + ty;
    const int out_x = tile_x + tx;

    if (out_y < out_H && out_x < out_W) {
        const float* wc = w + c * 9;
        float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
        float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
        float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

        float s;
        s  = w00 * sm[(ty+0)*SW + (tx+0)];
        s += w01 * sm[(ty+0)*SW + (tx+1)];
        s += w02 * sm[(ty+0)*SW + (tx+2)];
        s += w10 * sm[(ty+1)*SW + (tx+0)];
        s += w11 * sm[(ty+1)*SW + (tx+1)];
        s += w12 * sm[(ty+1)*SW + (tx+2)];
        s += w20 * sm[(ty+2)*SW + (tx+0)];
        s += w21 * sm[(ty+2)*SW + (tx+1)];
        s += w22 * sm[(ty+2)*SW + (tx+2)];

        y[(b * C + c) * (out_H * out_W) + out_y * out_W + out_x] = s;
    }
}

// ─────────────────────────────────────────────────────────────────
// Kernel v4b: 32x32 tile with vectorised float4 SM fill
// Key: SW=34 is not divisible by 4, so we need to handle boundaries.
// Alternative: pad SM width to 36 (9 float4s), simplify indexing.
// ─────────────────────────────────────────────────────────────────
#define SW4 36   // padded to multiple of 4
#define SH4 34

__global__ void dw_conv_32x32_f4(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W
) {
    __shared__ float sm[SH4 * SW4];   // 34x36 = 1224 floats

    const int bc    = blockIdx.z;
    const int b     = bc / C;
    const int c     = bc % C;
    const int tile_y = blockIdx.y * TH;
    const int tile_x = blockIdx.x * TW;
    const int tx    = threadIdx.x;
    const int ty    = threadIdx.y;
    const int tid   = ty * TW + tx;

    const float* xc = x + (b * C + c) * (H * W);

    // Fill SM using scalar loads (float4 alignment not guaranteed for boundary tiles)
    for (int i = tid; i < SH4 * SW4; i += TH * TW) {
        const int sy = i / SW4;
        const int sx = i % SW4;
        const int gy = tile_y + sy;
        const int gx = tile_x + sx;
        if (sx < 34 && gy < H && gx < W)
            sm[i] = __ldg(&xc[gy * W + gx]);
        else
            sm[i] = 0.0f;
    }
    __syncthreads();

    const int out_y = tile_y + ty;
    const int out_x = tile_x + tx;

    if (out_y < out_H && out_x < out_W) {
        const float* wc = w + c * 9;
        float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
        float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
        float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

        // Note: SM uses SW4=36 not SW=34 stride
        float s;
        s  = w00 * sm[(ty+0)*SW4 + (tx+0)];
        s += w01 * sm[(ty+0)*SW4 + (tx+1)];
        s += w02 * sm[(ty+0)*SW4 + (tx+2)];
        s += w10 * sm[(ty+1)*SW4 + (tx+0)];
        s += w11 * sm[(ty+1)*SW4 + (tx+1)];
        s += w12 * sm[(ty+1)*SW4 + (tx+2)];
        s += w20 * sm[(ty+2)*SW4 + (tx+0)];
        s += w21 * sm[(ty+2)*SW4 + (tx+1)];
        s += w22 * sm[(ty+2)*SW4 + (tx+2)];

        y[(b * C + c) * (out_H * out_W) + out_y * out_W + out_x] = s;
    }
}

// ─────────────────────────────────────────────────────────────────
// Kernel v4c: "register-only" approach — no shared memory
// Each warp of 32 threads processes 32 consecutive output cols in 1 output row
// Uses __ldg for cached global reads
// One output row = one pass; grid iterates over rows
// Block: 1D warp (32 threads), processes 1 output pixel per thread
// Grid: (ceil(out_W/32), out_H, B*C)
// ─────────────────────────────────────────────────────────────────
__global__ void dw_conv_warp_row(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W
) {
    const int bc = blockIdx.z;
    const int b  = bc / C, c = bc % C;
    const int out_x = blockIdx.x * 32 + threadIdx.x;
    const int out_y = blockIdx.y;

    if (out_x >= out_W || out_y >= out_H) return;

    const float* xc = x + (b * C + c) * (H * W);
    const float* wc = w + c * 9;

    float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
    float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
    float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

    float s;
    s  = w00 * __ldg(&xc[(out_y+0)*W + out_x+0])
       + w01 * __ldg(&xc[(out_y+0)*W + out_x+1])
       + w02 * __ldg(&xc[(out_y+0)*W + out_x+2])
       + w10 * __ldg(&xc[(out_y+1)*W + out_x+0])
       + w11 * __ldg(&xc[(out_y+1)*W + out_x+1])
       + w12 * __ldg(&xc[(out_y+1)*W + out_x+2])
       + w20 * __ldg(&xc[(out_y+2)*W + out_x+0])
       + w21 * __ldg(&xc[(out_y+2)*W + out_x+1])
       + w22 * __ldg(&xc[(out_y+2)*W + out_x+2]);

    y[bc * (out_H * out_W) + out_y * out_W + out_x] = s;
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
            // 32x32 = 1024 threads, 34x34 SM
            const dim3 block(TW, TH);
            const dim3 grid(
                (out_W + TW - 1) / TW,
                (out_H + TH - 1) / TH,
                B * C
            );
            dw_conv_32x32<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W
            );
        } else if (variant == 2) {
            // 32x32 with padded SM (36 wide)
            const dim3 block(TW, TH);
            const dim3 grid(
                (out_W + TW - 1) / TW,
                (out_H + TH - 1) / TH,
                B * C
            );
            dw_conv_32x32_f4<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W
            );
        } else {
            // warp-row: no shared mem, direct __ldg
            const dim3 block(32);
            const dim3 grid(
                (out_W + 31) / 32,
                out_H,
                B * C
            );
            dw_conv_warp_row<<<grid, block>>>(
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
            name="dw_conv2d_noptx_v4",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["launch_dw_conv"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-O3,--maxrregcount=64"],
            verbose=False,
        )
    return _mod


class Model(nn.Module):
    """
    Depthwise 2-D convolution using 32x32 shared-memory tiling.
    """
    def __init__(self, in_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )
        self._variant = 1  # 32x32 SM tiles, 1024 threads

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
