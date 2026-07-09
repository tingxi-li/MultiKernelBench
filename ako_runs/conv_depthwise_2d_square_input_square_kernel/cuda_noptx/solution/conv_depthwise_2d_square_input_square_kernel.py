import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------
# CUDA kernel: depthwise 2-D conv with shared-memory tiling
# Specialised for KS=3, stride=1, padding=0; general fallback otherwise.
# -----------------------------------------------------------------------
_CUDA_SOURCE = r"""
#include <cuda_runtime.h>

// Tile dimensions for the output
#define TILE_W 32
#define TILE_H  8
// Shared-memory tile (for KS=3 specialisation)
#define SMW  (TILE_W + 2)    // 34
#define SMH  (TILE_H + 2)    // 10

// ------------------------------------------------------------------
// Specialised kernel: KS=3, stride=1, padding=0
// Grid : (ceil(out_W/TILE_W), ceil(out_H/TILE_H), B*C)
// Block: (TILE_W, TILE_H) = 256 threads
// ------------------------------------------------------------------
__global__ void dw_conv_k3s1p0(
    const float* __restrict__ x,     // [B, C, H, W]
    const float* __restrict__ w,     // [C, 1, 3, 3] contiguous
    float*       __restrict__ y,     // [B, C, out_H, out_W]
    int B, int C, int H, int W, int out_H, int out_W
) {
    __shared__ float sm[SMH * SMW];   // 10*34=340 floats = 1360 B

    const int bc    = blockIdx.z;
    const int b     = bc / C;
    const int c     = bc % C;
    const int tile_y = blockIdx.y * TILE_H;
    const int tile_x = blockIdx.x * TILE_W;
    const int tx    = threadIdx.x;
    const int ty    = threadIdx.y;
    const int tid   = ty * TILE_W + tx;   // 0..255

    // ---- fill shared memory ----------------------------------------
    // 340 elements / 256 threads: each thread loads 1 or 2 elements
    const float* xc = x + (b * C + c) * (H * W);

    for (int i = tid; i < SMH * SMW; i += TILE_H * TILE_W) {
        const int sy = i / SMW;
        const int sx = i % SMW;
        const int gy = tile_y + sy;
        const int gx = tile_x + sx;
        sm[i] = (gy < H && gx < W) ? __ldg(&xc[gy * W + gx]) : 0.0f;
    }
    __syncthreads();

    // ---- compute output --------------------------------------------
    const int out_y = tile_y + ty;
    const int out_x = tile_x + tx;

    if (out_y < out_H && out_x < out_W) {
        const float* wc = w + c * 9;    // 9 weights for this channel
        // Load weights into registers (broadcast across warp)
        const float w00 = __ldg(&wc[0]), w01 = __ldg(&wc[1]), w02 = __ldg(&wc[2]);
        const float w10 = __ldg(&wc[3]), w11 = __ldg(&wc[4]), w12 = __ldg(&wc[5]);
        const float w20 = __ldg(&wc[6]), w21 = __ldg(&wc[7]), w22 = __ldg(&wc[8]);

        // Unrolled 3x3 MAC from shared memory (no bank conflicts)
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
// General fallback: any KS, stride, padding (uses dynamic shared mem)
// ------------------------------------------------------------------
__global__ void dw_conv_general(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int B, int C, int H, int W, int out_H, int out_W,
    int KH, int KW, int stride, int padding
) {
    extern __shared__ float sm[];

    const int smW = TILE_W + KW - 1;
    const int smH = TILE_H + KH - 1;

    const int bc    = blockIdx.z;
    const int b     = bc / C;
    const int c     = bc % C;
    const int tile_y = blockIdx.y * TILE_H;
    const int tile_x = blockIdx.x * TILE_W;
    const int tx    = threadIdx.x;
    const int ty    = threadIdx.y;
    const int tid   = ty * TILE_W + tx;

    const float* xc  = x + (b * C + c) * (H * W);
    const int in_y0  = tile_y * stride - padding;
    const int in_x0  = tile_x * stride - padding;

    for (int i = tid; i < smH * smW; i += TILE_H * TILE_W) {
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

// ------------------------------------------------------------------
// Host launcher
// ------------------------------------------------------------------
void launch_dw_conv(
    torch::Tensor x,     // [B, C, H, W]  float32 contiguous
    torch::Tensor w,     // [C, 1, KH, KW] float32 contiguous
    torch::Tensor y,     // [B, C, out_H, out_W] float32
    int stride,
    int padding
) {
    const int B    = x.size(0), C = x.size(1);
    const int H    = x.size(2), W = x.size(3);
    const int KH   = w.size(2), KW = w.size(3);
    const int out_H = y.size(2), out_W = y.size(3);

    const dim3 block(TILE_W, TILE_H);
    const dim3 grid(
        (out_W + TILE_W - 1) / TILE_W,
        (out_H + TILE_H - 1) / TILE_H,
        B * C
    );

    if (KH == 3 && KW == 3 && stride == 1 && padding == 0) {
        dw_conv_k3s1p0<<<grid, block>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
            B, C, H, W, out_H, out_W
        );
    } else {
        const int smem = (TILE_H + KH - 1) * (TILE_W + KW - 1) * sizeof(float);
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
    int padding
);
"""

_mod = None

def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="dw_conv2d_noptx_v1",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["launch_dw_conv"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-O3,--maxrregcount=64"],
            verbose=False,
        )
    return _mod


class Model(nn.Module):
    """
    Depthwise 2-D convolution backed by a custom CUDA kernel with
    shared-memory tiling (specialised for 3x3, stride=1, padding=0).
    """
    def __init__(self, in_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        # Keep the same nn.Conv2d so seeded weights are identical to the ref
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv2d.weight.contiguous()   # [C, 1, KH, KW]
        pad    = self.conv2d.padding[0]
        stride = self.conv2d.stride[0]
        KH, KW = w.shape[2], w.shape[3]
        out_H  = (x.shape[2] + 2 * pad - KH) // stride + 1
        out_W  = (x.shape[3] + 2 * pad - KW) // stride + 1
        out = torch.empty(x.shape[0], x.shape[1], out_H, out_W,
                          dtype=x.dtype, device=x.device)
        _get_mod().launch_dw_conv(x.contiguous(), w, out, stride, pad)
        return out
