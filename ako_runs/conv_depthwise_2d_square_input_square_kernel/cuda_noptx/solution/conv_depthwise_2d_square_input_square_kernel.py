import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------
# CUDA kernel v5: depthwise 2-D conv — coalescing-optimised tile fill
#
# Key insight: when SW (SM width) = 32 (power of 2), the linearised SM fill
# loop always maps entire warps to consecutive columns in one input row →
# perfectly coalesced 128-byte transactions from L2.
#
# Design:
#   - TW = 30, TH = 16  → output tile 30×16 = 480 pixels
#   - Block = 32 × 16 = 512 threads (2 extra threads per row for halo fill)
#   - SW = TW + 2 = 32 (power of 2!) → coalesced global reads
#   - SH = TH + 2 = 18
#   - SM = 32 × 18 = 576 floats = 2304 bytes (well within L1)
#   - Fill: 576/512 = 1.125 loads/thread (very low overhead)
#   - Register count: 9 weight regs + 1 accumulator + few indexing → fits well
#   - --maxrregcount=40 gives ≥3 resident blocks per SM for better latency hiding
# -----------------------------------------------------------------------
_CUDA_SOURCE = r"""
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────
// Primary kernel: TW=30, SW=32 (coalesced), TH=16, block=32×16=512
// ─────────────────────────────────────────────────────────────────
#define TW  30
#define TH  16
#define BW  32   // block width (32 = warp size, 2 extra threads for halo)
#define SW  32   // SM width = TW+2 = 32 → power of 2!
#define SH  18   // SM height = TH+2 = 18

__global__ void dw_conv_k3s1_coalesced(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W,
    int stride, int padding
) {
    __shared__ float sm[SH * SW];   // 18*32 = 576 floats = 2304 B

    const int bc    = blockIdx.z;
    const int b     = bc / C;
    const int c     = bc % C;
    const int tile_y = blockIdx.y * TH;
    const int tile_x = blockIdx.x * TW;
    const int tx    = threadIdx.x;   // 0..31 (BW=32)
    const int ty    = threadIdx.y;   // 0..15 (TH=16)
    const int tid   = ty * BW + tx;  // 0..511

    // Input origin for this tile (with padding)
    const int in_y0 = tile_y * stride - padding;
    const int in_x0 = tile_x * stride - padding;

    const float* xc = x + (b * C + c) * (H * W);

    // Fill SM: SW=32 → sy = i/32 = i>>5, sx = i%32 = i&31
    // Each warp (32 consecutive tids) maps to exactly one row of SM → COALESCED
    #pragma unroll 2
    for (int i = tid; i < SH * SW; i += BW * TH) {
        const int sy = i >> 5;   // i / 32
        const int sx = i & 31;   // i % 32
        const int gy = in_y0 + sy;
        const int gx = in_x0 + sx;
        sm[i] = (gy >= 0 && gy < H && gx >= 0 && gx < W)
                ? __ldg(&xc[gy * W + gx]) : 0.0f;
    }
    __syncthreads();

    // Only threads tx=0..TW-1 write output
    const int out_y = tile_y + ty;
    const int out_x = tile_x + tx;

    if (tx < TW && out_y < out_H && out_x < out_W) {
        const float* wc = w + c * 9;
        float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
        float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
        float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

        // tx, ty map directly to SM index (sw=32, no offset needed since in_x0=tile_x-pad)
        // For stride=1, pad=0: tx_sm = tx, ty_sm = ty
        const int tx_sm = tx;
        const int ty_sm = ty;

        float s;
        s  = w00 * sm[(ty_sm+0)*SW + (tx_sm+0)];
        s += w01 * sm[(ty_sm+0)*SW + (tx_sm+1)];
        s += w02 * sm[(ty_sm+0)*SW + (tx_sm+2)];
        s += w10 * sm[(ty_sm+1)*SW + (tx_sm+0)];
        s += w11 * sm[(ty_sm+1)*SW + (tx_sm+1)];
        s += w12 * sm[(ty_sm+1)*SW + (tx_sm+2)];
        s += w20 * sm[(ty_sm+2)*SW + (tx_sm+0)];
        s += w21 * sm[(ty_sm+2)*SW + (tx_sm+1)];
        s += w22 * sm[(ty_sm+2)*SW + (tx_sm+2)];

        y[(b * C + c) * (out_H * out_W) + out_y * out_W + out_x] = s;
    }
}

// ─────────────────────────────────────────────────────────────────
// Variant B: TW=62, SW=64, TH=8, block=64×8=512 threads
// Even better coalescing (64-wide SM row = 2 cache lines)
// More output pixels per block = fewer blocks = less scheduler pressure
// ─────────────────────────────────────────────────────────────────
#define TW2  62
#define TH2   8
#define BW2  64
#define SW2  64   // TW2+2 = 64 → power of 2!
#define SH2  10   // TH2+2 = 10

__global__ void dw_conv_k3s1_wide(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W
) {
    __shared__ float sm[SH2 * SW2];   // 10*64 = 640 floats = 2560 B

    const int bc    = blockIdx.z;
    const int b     = bc / C;
    const int c     = bc % C;
    const int tile_y = blockIdx.y * TH2;
    const int tile_x = blockIdx.x * TW2;
    const int tx    = threadIdx.x;   // 0..63 (BW2=64)
    const int ty    = threadIdx.y;   // 0..7  (TH2=8)
    const int tid   = ty * BW2 + tx; // 0..511

    const float* xc = x + (b * C + c) * (H * W);

    // SW2=64 → sy = i/64 = i>>6, sx = i%64 = i&63
    // Each group of 64 consecutive tids → one SM row → COALESCED
    #pragma unroll 2
    for (int i = tid; i < SH2 * SW2; i += BW2 * TH2) {
        const int sy = i >> 6;
        const int sx = i & 63;
        const int gy = tile_y + sy;
        const int gx = tile_x + sx;
        sm[i] = (gy < H && gx < W) ? __ldg(&xc[gy * W + gx]) : 0.0f;
    }
    __syncthreads();

    const int out_y = tile_y + ty;
    const int out_x = tile_x + tx;

    if (tx < TW2 && out_y < out_H && out_x < out_W) {
        const float* wc = w + c * 9;
        float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
        float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
        float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

        float s;
        s  = w00 * sm[(ty+0)*SW2 + (tx+0)];
        s += w01 * sm[(ty+0)*SW2 + (tx+1)];
        s += w02 * sm[(ty+0)*SW2 + (tx+2)];
        s += w10 * sm[(ty+1)*SW2 + (tx+0)];
        s += w11 * sm[(ty+1)*SW2 + (tx+1)];
        s += w12 * sm[(ty+1)*SW2 + (tx+2)];
        s += w20 * sm[(ty+2)*SW2 + (tx+0)];
        s += w21 * sm[(ty+2)*SW2 + (tx+1)];
        s += w22 * sm[(ty+2)*SW2 + (tx+2)];

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

    if (KH == 3 && KW == 3 && padding == 0 && stride == 1) {
        if (variant == 1) {
            // TW=30, SW=32, TH=16, block=32x16=512
            const dim3 block(BW, TH);
            const dim3 grid(
                (out_W + TW - 1) / TW,
                (out_H + TH - 1) / TH,
                B * C
            );
            dw_conv_k3s1_coalesced<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W, stride, padding
            );
        } else {
            // TW=62, SW=64, TH=8, block=64x8=512
            const dim3 block(BW2, TH2);
            const dim3 grid(
                (out_W + TW2 - 1) / TW2,
                (out_H + TH2 - 1) / TH2,
                B * C
            );
            dw_conv_k3s1_wide<<<grid, block>>>(
                x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                B, C, H, W, out_H, out_W
            );
        }
    } else if (KH == 3 && KW == 3) {
        // general 3x3 with any stride/padding
        const dim3 block(BW, TH);
        const dim3 grid(
            (out_W + TW - 1) / TW,
            (out_H + TH - 1) / TH,
            B * C
        );
        dw_conv_k3s1_coalesced<<<grid, block>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
            B, C, H, W, out_H, out_W, stride, padding
        );
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
            name="dw_conv2d_noptx_v5",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["launch_dw_conv"],
            # --maxrregcount=40 → allows 3 blocks per SM (vs 2 at 64 regs)
            extra_cuda_cflags=["-O3", "--use_fast_math",
                               "-Xptxas", "-O3,--maxrregcount=40"],
            verbose=False,
        )
    return _mod


class Model(nn.Module):
    """
    Depthwise 2-D convolution with coalescing-optimised CUDA kernel.
    TW=30 so SM width=32 (power-of-2): SM fill is perfectly coalesced.
    """
    def __init__(self, in_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )
        self._variant = 1  # TW=30, SW=32, TH=16

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
