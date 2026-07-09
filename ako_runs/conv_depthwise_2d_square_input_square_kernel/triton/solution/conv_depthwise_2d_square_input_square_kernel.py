import torch
import torch.nn as nn
import triton
import triton.language as tl


# For 3x3 depthwise conv with stride=1, pad=0, input 512x512, output 510x510:
# Process one output row at a time (row-per-CTA approach).
# Each CTA handles (n, c, oh) → computes W_out=510 output values.
# For a 3-row input window (kh=0,1,2), we load 3 rows of 510 input values.
# This maximizes coalescing: all loads/stores are sequential in W.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=2),
    ],
    key=['NC', 'H', 'W', 'W_out', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_row_kernel(
    x_ptr,    # (N, C, H, W) — contiguous
    w_ptr,    # (C, KH, KW) — squeezed
    b_ptr,    # (C,) or dummy
    out_ptr,  # (N, C, H_out, W_out)
    NC,       # N*C
    C,
    H, W,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    KH: tl.constexpr, KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Grid: (NC, H_out, ceil(W_out/BLOCK_W))
    Each CTA: processes one (n, c, output_row) → tile of output cols.
    """
    pid_nc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_w  = tl.program_id(2)

    n  = pid_nc // C
    c  = pid_nc %  C
    oh = pid_oh
    ow0 = pid_w * BLOCK_W

    ow_offs = ow0 + tl.arange(0, BLOCK_W)  # (BLOCK_W,)
    mask_w  = ow_offs < W_out

    x_nc   = x_ptr   + (n * C + c) * H * W
    w_c    = w_ptr   + c * KH * KW
    out_nc = out_ptr + (n * C + c) * H_out * W_out

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # KH/KW unrolled — for 3x3 this is 9 iterations
    for kh in tl.static_range(KH):
        ih = oh * stride_h + kh - pad_h
        valid_h = (ih >= 0) & (ih < H)

        for kw in tl.static_range(KW):
            iw = ow_offs * stride_w + kw - pad_w  # (BLOCK_W,)
            valid_w = mask_w & (iw >= 0) & (iw < W)
            valid   = valid_h & valid_w  # (BLOCK_W,)

            iw_c = tl.maximum(0, tl.minimum(iw, W - 1))
            x_off = ih * W + iw_c
            x_val = tl.load(x_nc + x_off, mask=valid, other=0.0)

            w_val = tl.load(w_c + kh * KW + kw)
            acc  += x_val * w_val

    if HAS_BIAS:
        acc = acc + tl.load(b_ptr + c)

    out_off = oh * W_out + ow_offs
    tl.store(out_nc + out_off, acc.to(x_ptr.dtype.element_ty), mask=mask_w)


class Model(nn.Module):
    """
    Depthwise 2D convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        stride_h, stride_w = self.conv2d.stride
        pad_h,    pad_w    = self.conv2d.padding
        KH,       KW       = self.conv2d.kernel_size

        H_out = (H + 2 * pad_h - KH) // stride_h + 1
        W_out = (W + 2 * pad_w - KW) // stride_w + 1

        NC     = N * C
        x_c    = x.contiguous()
        weight = self.conv2d.weight.squeeze(1).contiguous()  # (C, KH, KW)
        out    = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        has_bias = self.conv2d.bias is not None
        bias_ptr = self.conv2d.bias if has_bias else weight

        grid = lambda meta: (
            NC,
            H_out,
            triton.cdiv(W_out, meta['BLOCK_W']),
        )

        depthwise_conv2d_row_kernel[grid](
            x_c, weight, bias_ptr, out,
            NC, C,
            H, W,
            H_out, W_out,
            stride_h, stride_w,
            pad_h, pad_w,
            KH, KW,
            has_bias,
        )
        return out
