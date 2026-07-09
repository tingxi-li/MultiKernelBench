import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 1,  'BLOCK_W': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 2,  'BLOCK_W': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 4,  'BLOCK_W': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 4,  'BLOCK_W': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 8,  'BLOCK_W': 32},  num_warps=2, num_stages=3),
        triton.Config({'BLOCK_H': 8,  'BLOCK_W': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 8,  'BLOCK_W': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 32},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 16},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 16},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16},  num_warps=2, num_stages=3),
    ],
    key=['N', 'C', 'H', 'W', 'KH', 'KW', 'stride_h', 'stride_w', 'pad_h', 'pad_w'],
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    KH: tl.constexpr, KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Depthwise 2D convolution kernel.
    Grid: (N*C, ceil(H_out/BLOCK_H), ceil(W_out/BLOCK_W))
    """
    pid_nc = tl.program_id(0)
    pid_h  = tl.program_id(1)
    pid_w  = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C

    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W

    # Base pointers for this (n, c) slice
    # x shape: (N, C, H, W) contiguous
    # w shape: (C, 1, KH, KW) -> accessed as (C, KH, KW)
    x_base  = x_ptr  + (n * C + c) * H * W
    w_base  = w_ptr  + c * KH * KW
    out_base = out_ptr + (n * C + c) * H_out * W_out

    # Output tile indices
    oh_offs = oh_start + tl.arange(0, BLOCK_H)  # (BLOCK_H,)
    ow_offs = ow_start + tl.arange(0, BLOCK_W)  # (BLOCK_W,)

    # Output boundary masks
    out_mask_h = oh_offs < H_out  # (BLOCK_H,)
    out_mask_w = ow_offs < W_out  # (BLOCK_W,)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Unroll over kernel spatial dims
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            # Input row/col for each output tile position
            ih = oh_offs * stride_h + kh - pad_h  # (BLOCK_H,)
            iw = ow_offs * stride_w + kw - pad_w  # (BLOCK_W,)

            valid_h = (ih >= 0) & (ih < H)  # (BLOCK_H,)
            valid_w = (iw >= 0) & (iw < W)  # (BLOCK_W,)
            valid = (out_mask_h & valid_h)[:, None] & (out_mask_w & valid_w)[None, :]

            # Clamp indices so out-of-bounds loads hit a valid address (masked anyway)
            ih_c = tl.maximum(tl.minimum(ih, H - 1), 0)
            iw_c = tl.maximum(tl.minimum(iw, W - 1), 0)

            x_off = ih_c[:, None] * W + iw_c[None, :]  # (BLOCK_H, BLOCK_W)
            x_vals = tl.load(x_base + x_off, mask=valid, other=0.0)

            w_val = tl.load(w_base + kh * KW + kw)  # scalar
            acc += x_vals * w_val

    if HAS_BIAS:
        bias = tl.load(b_ptr + c)
        acc += bias

    # Write output tile
    out_mask = out_mask_h[:, None] & out_mask_w[None, :]
    out_off = oh_offs[:, None] * W_out + ow_offs[None, :]
    tl.store(out_base + out_off, acc.to(tl.float32), mask=out_mask)


class Model(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.
    Uses a custom Triton kernel for efficiency.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        # Must keep the same seeded conv2d layer for weight/bias init parity
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        stride_h = self.conv2d.stride[0]
        stride_w = self.conv2d.stride[1]
        pad_h    = self.conv2d.padding[0]
        pad_w    = self.conv2d.padding[1]
        KH, KW   = self.conv2d.kernel_size

        H_out = (H + 2 * pad_h - KH) // stride_h + 1
        W_out = (W + 2 * pad_w - KW) // stride_w + 1

        x_c    = x.contiguous()
        # weight: (C, 1, KH, KW) -> reshape to (C, KH, KW)
        weight = self.conv2d.weight.squeeze(1).contiguous()
        out    = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        has_bias = self.conv2d.bias is not None
        bias_ptr = self.conv2d.bias if has_bias else weight  # dummy ptr when no bias

        grid = (N * C, triton.cdiv(H_out, depthwise_conv2d_kernel.best_config.kwargs['BLOCK_H'])
                       if hasattr(depthwise_conv2d_kernel, 'best_config')
                       else 1,
                1)  # will be replaced by lambda below

        grid = lambda meta: (
            N * C,
            triton.cdiv(H_out, meta['BLOCK_H']),
            triton.cdiv(W_out, meta['BLOCK_W']),
        )

        depthwise_conv2d_kernel[grid](
            x_c, weight, bias_ptr, out,
            N, C, H, W,
            H_out, W_out,
            stride_h, stride_w,
            pad_h, pad_w,
            KH, KW,
            has_bias,
        )
        return out
