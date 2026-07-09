import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------
# CUDA kernel v2: depthwise 2-D conv, 3x3 s1p0
# Strategy:
#  - Tile: 64 cols x 4 rows = 256 threads per block
#  - Each thread processes 1 output pixel in the main path
#  - Shared mem: (4+2) x (64+2) = 396 floats
#  - float4 vectorised loads into shared memory
#  - Per-channel weights in __constant__ memory (up to 512 channels x 9 weights)
#    -> but fallback to __ldg for arbitrary channel counts
# -----------------------------------------------------------------------
_CUDA_SOURCE = r"""
#include <cuda_runtime.h>

#define TILE_W 64
#define TILE_H  4
#define SMW  (TILE_W + 2)   // 66
#define SMH  (TILE_H + 2)   //  6

// Fast specialised kernel: KS=3, stride=1, padding=0
__global__ void dw_conv_k3s1p0_v2(
    const float* __restrict__ x,     // [B, C, H, W]
    const float* __restrict__ w,     // [C, 1, 3, 3]
    float*       __restrict__ y,     // [B, C, out_H, out_W]
    int B, int C, int H, int W, int out_H, int out_W
) {
    __shared__ float sm[SMH * SMW];   // 6*66 = 396 floats = 1584 B

    const int bc    = blockIdx.z;
    const int b     = bc / C;
    const int c     = bc % C;
    const int tile_y = blockIdx.y * TILE_H;
    const int tile_x = blockIdx.x * TILE_W;
    const int tx    = threadIdx.x;   // 0..63
    const int ty    = threadIdx.y;   // 0..3
    const int tid   = ty * TILE_W + tx;   // 0..255

    const float* xc = x + (b * C + c) * (H * W);

    // Fill shared memory: 396 elements, 256 threads => ~1.5 loads/thread
    // Use __ldg for texture cache benefit
    for (int i = tid; i < SMH * SMW; i += TILE_H * TILE_W) {
        const int sy = i / SMW;
        const int sx = i % SMW;
        const int gy = tile_y + sy;
        const int gx = tile_x + sx;
        sm[i] = (gy < H && gx < W) ? __ldg(&xc[gy * W + gx]) : 0.0f;
    }
    __syncthreads();

    const int out_y = tile_y + ty;
    const int out_x = tile_x + tx;

    if (out_y < out_H && out_x < out_W) {
        const float* wc = w + c * 9;
        // Load 9 weights into registers (all threads in warp see same wc -> likely cached)
        float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
        float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
        float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

        // Manually unrolled 3x3 MAC
        float s;
        s  = w00 * sm[(ty+0)*SMW + (tx+0)];
        s += w01 * sm[(ty+0)*SMW + (tx+1)];
        s += w02 * sm[(ty+0)*SMW + (tx+2)];
        s += w10 * sm[(ty+1)*SMW + (tx+0)];
        s += w11 * sm[(ty+1)*SMW + (tx+1)];
        s += w12 * sm[(ty+1)*SMW + (tx+2)];
        s += w20 * sm[(ty+2)*SMW + (tx+0)];
        s += w21 * sm[(ty+2)*SMW + (tx+1)];
        s += w22 * sm[(ty+2)*SMW + (tx+2)];

        y[(b * C + c) * (out_H * out_W) + out_y * out_W + out_x] = s;
    }
}

// ------------------------------------------------------------------
// Variant B: Thin-tile, multiple output rows per thread (register sliding)
// Each thread computes OUT_ROWS consecutive output rows with a sliding window
// Reduces shared-mem loads per output element
// Block: (32 cols, 8 threads), each handles OUT_ROWS=4 output rows -> 32x32 output tile
// ------------------------------------------------------------------
#define OUT_ROWS 4
#define BLKW 32
#define BLKH 8
// SM: (BLKH*OUT_ROWS + 2) x (BLKW + 2) = 34 x 34 = 1156
#define SMHB (BLKH*OUT_ROWS + 2)   // 34
#define SMWB (BLKW + 2)            // 34

__global__ void dw_conv_k3s1p0_v2b(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W
) {
    __shared__ float sm[SMHB * SMWB];   // 34*34 = 1156 floats = 4624 B

    const int bc    = blockIdx.z;
    const int b     = bc / C;
    const int c     = bc % C;
    const int tile_y = blockIdx.y * (BLKH * OUT_ROWS);   // 32 rows per block
    const int tile_x = blockIdx.x * BLKW;
    const int tx    = threadIdx.x;
    const int ty    = threadIdx.y;
    const int tid   = ty * BLKW + tx;

    const float* xc = x + (b * C + c) * (H * W);

    // Fill SM: 1156 floats, 256 threads
    for (int i = tid; i < SMHB * SMWB; i += BLKW * BLKH) {
        const int sy = i / SMWB;
        const int sx = i % SMWB;
        const int gy = tile_y + sy;
        const int gx = tile_x + sx;
        sm[i] = (gy < H && gx < W) ? __ldg(&xc[gy * W + gx]) : 0.0f;
    }
    __syncthreads();

    const float* wc = w + c * 9;
    float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
    float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
    float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

    const float*       yc  = y + (b * C + c) * (out_H * out_W);
    float* __restrict__ ycp = y + (b * C + c) * (out_H * out_W);
    (void)yc;

    const int base_out_y = tile_y + ty;
    const int out_x      = tile_x + tx;

    // Each thread (ty, tx) handles OUT_ROWS output rows: ty, ty+BLKH, ty+2*BLKH, ty+3*BLKH
#pragma unroll
    for (int r = 0; r < OUT_ROWS; ++r) {
        const int out_y  = base_out_y + r * BLKH;
        const int smy    = ty + r * BLKH;
        if (out_y < out_H && out_x < out_W) {
            float s;
            s  = w00 * sm[(smy+0)*SMWB + (tx+0)];
            s += w01 * sm[(smy+0)*SMWB + (tx+1)];
            s += w02 * sm[(smy+0)*SMWB + (tx+2)];
            s += w10 * sm[(smy+1)*SMWB + (tx+0)];
            s += w11 * sm[(smy+1)*SMWB + (tx+1)];
            s += w12 * sm[(smy+1)*SMWB + (tx+2)];
            s += w20 * sm[(smy+2)*SMWB + (tx+0)];
            s += w21 * sm[(smy+2)*SMWB + (tx+1)];
            s += w22 * sm[(smy+2)*SMWB + (tx+2)];
            ycp[out_y * out_W + out_x] = s;
        }
    }
}

// ------------------------------------------------------------------
// General fallback
// ------------------------------------------------------------------
#define FTILE_W 32
#define FTILE_H  8
__global__ void dw_conv_general(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W,
    int KH, int KW, int stride, int padding
) {
    extern __shared__ float sm[];

    const int smW = FTILE_W + KW - 1;

    const int bc    = blockIdx.z;
    const int b     = bc / C;
    const int c     = bc % C;
    const int tile_y = blockIdx.y * FTILE_H;
    const int tile_x = blockIdx.x * FTILE_W;
    const int tx    = threadIdx.x;
    const int ty    = threadIdx.y;
    const int tid   = ty * FTILE_W + tx;

    const float* xc  = x + (b * C + c) * (H * W);
    const int in_y0  = tile_y * stride - padding;
    const int in_x0  = tile_x * stride - padding;
    const int smH    = FTILE_H + KH - 1;

    for (int i = tid; i < smH * smW; i += FTILE_H * FTILE_W) {
        const int sy = i / smW;
        const int sx = i % smW;
        const int gy = in_y0 + sy;
        const int gx = in_x0 + sx;
        sm[i] = (gy >= 0 && gy < H && gx >= 0 && gx < W)
                ? __ldg(&xc[gy * W + gx]) : 0.0f;
    }
    __syncthreads();

    const int out_y = tile_y + ty;
    const int out_x = tile_x + tx;
    if (out_y < out_H && out_x < out_W) {
        const float* wc = w + c * KH * KW;
        float s = 0.0f;
        for (int ky = 0; ky < KH; ++ky)
            for (int kx = 0; kx < KW; ++kx)
                s += __ldg(&wc[ky * KW + kx])
                   * sm[(ty * stride + ky) * smW + (tx * stride + kx)];
        y[(b * C + c) * (out_H * out_W) + out_y * out_W + out_x] = s;
    }
}

void launch_dw_conv(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    int stride,
    int padding,
    int variant   // 1 = v2 (64x4), 2 = v2b (32x8 multi-row), else general
) {
    const int B    = x.size(0), C = x.size(1);
    const int H    = x.size(2), W = x.size(3);
    const int KH   = w.size(2), KW = w.size(3);
    const int out_H = y.size(2), out_W = y.size(3);

    if (KH == 3 && KW == 3 && stride == 1 && padding == 0) {
        if (variant == 1) {
            const dim3 block(TILE_W, TILE_H);
            const dim3 grid(
                (out_W + TILE_W - 1) / TILE_W,
                (out_H + TILE_H - 1) / TILE_H,
                B * C
            );
            dw_conv_k3s1p0_v2<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W
            );
        } else {
            const dim3 block(BLKW, BLKH);
            const dim3 grid(
                (out_W + BLKW - 1) / BLKW,
                (out_H + BLKH * OUT_ROWS - 1) / (BLKH * OUT_ROWS),
                B * C
            );
            dw_conv_k3s1p0_v2b<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W
            );
        }
    } else {
        const dim3 block(FTILE_W, FTILE_H);
        const dim3 grid(
            (out_W + FTILE_W - 1) / FTILE_W,
            (out_H + FTILE_H - 1) / FTILE_H,
            B * C
        );
        const int smem = (FTILE_H + KH - 1) * (FTILE_W + KW - 1) * sizeof(float);
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
            name="dw_conv2d_noptx_v2",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["launch_dw_conv"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-O3,--maxrregcount=64"],
            verbose=False,
        )
    return _mod


class Model(nn.Module):
    """
    Depthwise 2-D convolution backed by optimised CUDA kernel.
    Uses 64x4 tile specialisation for 3x3 stride=1 padding=0.
    """
    def __init__(self, in_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )
        self._variant = 1  # use 64x4 kernel by default

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
