import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------
# CUDA kernel v3: depthwise 2-D conv, aggressive vectorisation
# Strategy:
#  - Warp handles 32 consecutive output cols (coalesced reads/writes)
#  - Block has 8 warps = 256 threads, 8 output rows per block
#  - Shared memory: 10 x 34 per warp section, but we organise as 10 x 256 for the block
#  - Actually: let each thread handle ONE output pixel with float4 weight loads
#  - Key insight: process entire output row within one block to maximise L2 reuse
#  - 1 block per (b, c): processes entire 510x510 output in chunks of 8 rows x 256 cols
# -----------------------------------------------------------------------
_CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <float.h>

// ─────────────────────────────────────────────────────────────────
// Kernel v3: streaming per-channel kernel
// One (b,c) pair assigned per block — iterate over all output rows.
// Each thread handles one output column across all rows.
// Requires out_W <= 512 threads (we use 512-thread blocks for out_W=510).
// ─────────────────────────────────────────────────────────────────
__global__ void dw_conv_stream_1blk_per_channel(
    const float* __restrict__ x,     // [B, C, H, W]
    const float* __restrict__ w,     // [C, 1, 3, 3]
    float*       __restrict__ y,     // [B, C, out_H, out_W]
    int B, int C, int H, int W, int out_H, int out_W
) {
    // Each block handles one (b, c) pair
    // Grid: (C, B) or similar
    const int c = blockIdx.x;
    const int b = blockIdx.y;
    const int tx = threadIdx.x;  // output column index (0..out_W-1, some blocks wider)

    if (tx >= out_W) return;

    // Load 9 weights for this channel
    const float* wc = w + c * 9;
    float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
    float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
    float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

    const float* xc = x + (b * C + c) * (H * W);
    float*       yc = y + (b * C + c) * (out_H * out_W);

    // Slide down output rows; keep 3 input rows in registers (ring buffer effect)
    // For tx < out_W: read x[tx], x[tx+1], x[tx+2] for each of 3 rows
    // Then output = dot product

    for (int oy = 0; oy < out_H; oy++) {
        float s;
        // input rows: oy, oy+1, oy+2
        // input cols: tx, tx+1, tx+2
        s  = w00 * __ldg(&xc[(oy+0)*W + tx+0])
           + w01 * __ldg(&xc[(oy+0)*W + tx+1])
           + w02 * __ldg(&xc[(oy+0)*W + tx+2])
           + w10 * __ldg(&xc[(oy+1)*W + tx+0])
           + w11 * __ldg(&xc[(oy+1)*W + tx+1])
           + w12 * __ldg(&xc[(oy+1)*W + tx+2])
           + w20 * __ldg(&xc[(oy+2)*W + tx+0])
           + w21 * __ldg(&xc[(oy+2)*W + tx+1])
           + w22 * __ldg(&xc[(oy+2)*W + tx+2]);
        yc[oy * out_W + tx] = s;
    }
}

// ─────────────────────────────────────────────────────────────────
// Kernel v4: 2D shared-memory tiling, optimised for high occupancy
// TILE: 32 cols × 16 rows; SM: 34×18; 512 threads per block
// Grid: (ceil(out_W/32), ceil(out_H/16), B*C) -> fewer larger blocks
// Key: 512 threads = 16 warps → higher occupancy than 256-thread blocks
// ─────────────────────────────────────────────────────────────────
#define V4_TW 32
#define V4_TH 16
#define V4_SMW (V4_TW + 2)   // 34
#define V4_SMH (V4_TH + 2)   // 18

__global__ void dw_conv_k3s1p0_v4(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W
) {
    __shared__ float sm[V4_SMH * V4_SMW];   // 18*34 = 612 floats = 2448 B

    const int bc    = blockIdx.z;
    const int b     = bc / C;
    const int c     = bc % C;
    const int tile_y = blockIdx.y * V4_TH;
    const int tile_x = blockIdx.x * V4_TW;
    const int tx    = threadIdx.x;   // 0..31
    const int ty    = threadIdx.y;   // 0..15
    const int tid   = ty * V4_TW + tx;  // 0..511

    const float* xc = x + (b * C + c) * (H * W);

    // Fill SM: 612 floats with 512 threads → ~1.2 loads/thread
    for (int i = tid; i < V4_SMH * V4_SMW; i += V4_TH * V4_TW) {
        const int sy = i / V4_SMW;
        const int sx = i % V4_SMW;
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
        s  = w00 * sm[(ty+0)*V4_SMW + (tx+0)];
        s += w01 * sm[(ty+0)*V4_SMW + (tx+1)];
        s += w02 * sm[(ty+0)*V4_SMW + (tx+2)];
        s += w10 * sm[(ty+1)*V4_SMW + (tx+0)];
        s += w11 * sm[(ty+1)*V4_SMW + (tx+1)];
        s += w12 * sm[(ty+1)*V4_SMW + (tx+2)];
        s += w20 * sm[(ty+2)*V4_SMW + (tx+0)];
        s += w21 * sm[(ty+2)*V4_SMW + (tx+1)];
        s += w22 * sm[(ty+2)*V4_SMW + (tx+2)];

        y[(b * C + c) * (out_H * out_W) + out_y * out_W + out_x] = s;
    }
}

// ─────────────────────────────────────────────────────────────────
// Kernel v5: Multi-channel block — each block handles CH_PER_BLK channels
// for one spatial tile. This lets weight loads be shared across channels
// if they happen to be adjacent. Actually channels have independent weights,
// so benefit is reduced block count → fewer scheduler overheads.
// Strategy: 1D block of 256 threads, each thread handles one output pixel
//           for one channel. Block handles 1 spatial tile × CH_PER_BLK=4 channels.
// ─────────────────────────────────────────────────────────────────
#define V5_TW 32
#define V5_TH  8
#define V5_CH  4
// Thread layout: 256 threads = (V5_TW * V5_TH * V5_CH) / V5_CH
// Actually: threads = V5_TW * V5_TH = 256; each thread handles V5_CH output pixels
// Nope, let's do: block = (V5_TW * V5_TH) = 256 threads
// Each thread handles output (ty, tx) for one channel
// Grid: (ceil(out_W/V5_TW), ceil(out_H/V5_TH), ceil(BC / V5_CH))
#define V5_SMW (V5_TW + 2)  // 34
#define V5_SMH (V5_TH + 2)  // 10

__global__ void dw_conv_k3s1p0_v5(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W
) {
    // blockIdx.z = bc_group = (b*C + c) / V5_CH
    // Each block processes V5_CH consecutive channels
    __shared__ float sm[V5_CH][V5_SMH * V5_SMW];  // 4 * 340 = 1360 floats = 5440 B

    const int bc_group = blockIdx.z;
    const int bc0 = bc_group * V5_CH;
    const int tile_y = blockIdx.y * V5_TH;
    const int tile_x = blockIdx.x * V5_TW;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * V5_TW + tx;

    // Load SM for each channel
    for (int ci = 0; ci < V5_CH; ci++) {
        const int bc = bc0 + ci;
        if (bc >= B * C) continue;
        const int b = bc / C, c = bc % C;
        const float* xc = x + (b * C + c) * (H * W);
        for (int i = tid; i < V5_SMH * V5_SMW; i += V5_TW * V5_TH) {
            const int sy = i / V5_SMW, sx = i % V5_SMW;
            const int gy = tile_y + sy, gx = tile_x + sx;
            sm[ci][i] = (gy < H && gx < W) ? __ldg(&xc[gy*W+gx]) : 0.0f;
        }
    }
    __syncthreads();

    const int out_y = tile_y + ty, out_x = tile_x + tx;
    if (out_y >= out_H || out_x >= out_W) return;

    for (int ci = 0; ci < V5_CH; ci++) {
        const int bc = bc0 + ci;
        if (bc >= B * C) continue;
        const int c = bc % C;
        const float* wc = w + c * 9;
        float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
        float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
        float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);
        const float* sc = sm[ci];
        float s;
        s  = w00 * sc[(ty+0)*V5_SMW+(tx+0)] + w01 * sc[(ty+0)*V5_SMW+(tx+1)] + w02 * sc[(ty+0)*V5_SMW+(tx+2)];
        s += w10 * sc[(ty+1)*V5_SMW+(tx+0)] + w11 * sc[(ty+1)*V5_SMW+(tx+1)] + w12 * sc[(ty+1)*V5_SMW+(tx+2)];
        s += w20 * sc[(ty+2)*V5_SMW+(tx+0)] + w21 * sc[(ty+2)*V5_SMW+(tx+1)] + w22 * sc[(ty+2)*V5_SMW+(tx+2)];
        y[bc * (out_H * out_W) + out_y * out_W + out_x] = s;
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
        if (variant == 3) {
            // Stream kernel: one block per (b,c), out_W threads per block
            // Needs out_W <= 1024; for 510 we use 512-thread blocks
            const int blk = ((out_W + 31) / 32) * 32;  // round up to warp
            const dim3 block(blk);
            const dim3 grid(C, B);
            dw_conv_stream_1blk_per_channel<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W
            );
        } else if (variant == 4) {
            // 32x16 tile, 512 threads
            const dim3 block(V4_TW, V4_TH);
            const dim3 grid(
                (out_W + V4_TW - 1) / V4_TW,
                (out_H + V4_TH - 1) / V4_TH,
                B * C
            );
            dw_conv_k3s1p0_v4<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W
            );
        } else {
            // v5: multi-channel blocks; V5_CH=4 channels per block
            const dim3 block(V5_TW, V5_TH);
            const dim3 grid(
                (out_W + V5_TW - 1) / V5_TW,
                (out_H + V5_TH - 1) / V5_TH,
                (B * C + V5_CH - 1) / V5_CH
            );
            dw_conv_k3s1p0_v5<<<grid, block>>>(
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
            name="dw_conv2d_noptx_v3",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["launch_dw_conv"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-O3,--maxrregcount=64"],
            verbose=False,
        )
    return _mod


class Model(nn.Module):
    """
    Depthwise 2-D convolution. Uses variant=4 (32x16, 512-thread blocks)
    for the standard 3x3 s1p0 case.
    """
    def __init__(self, in_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )
        self._variant = 4   # 32x16 tiles with 512 threads

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
