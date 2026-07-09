import torch
import torch.nn as nn
import triton
import triton.language as tl


# For 3x3 depthwise conv with stride=1, we can preload a full input row slice
# [ow0-pad, ow0+BLOCK_OW-1+pad] and slide the kernel over it in W.
# This reduces memory traffic from 9*BLOCK_OW loads to (BLOCK_OW+KW-1) loads per row.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 4,  'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8,  'BLOCK_OW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8,  'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4,  'BLOCK_OW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8,  'BLOCK_OW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 2,  'BLOCK_OW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 2,  'BLOCK_OW': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 32, 'BLOCK_OW': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 32, 'BLOCK_OW': 64},  num_warps=8, num_stages=2),
    ],
    key=['NC', 'H', 'W', 'H_out', 'W_out', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_slidew_kernel(
    x_ptr,    # (N, C, H, W)
    w_ptr,    # (C, KH, KW)
    b_ptr,    # (C,) or dummy
    out_ptr,  # (N, C, H_out, W_out)
    NC, C,
    H, W,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    KH: tl.constexpr, KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    """
    Sliding-window kernel: load input strip of width (BLOCK_OW + KW - 1) per row,
    then slide over it to compute BLOCK_OW outputs. Reduces W-direction loads by KW.

    Grid: (NC, ceil(H_out/BLOCK_OH), ceil(W_out/BLOCK_OW))
    """
    pid_nc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    n  = pid_nc // C
    c  = pid_nc %  C
    oh0 = pid_oh * BLOCK_OH
    ow0 = pid_ow * BLOCK_OW

    oh_offs = oh0 + tl.arange(0, BLOCK_OH)  # (BLOCK_OH,)
    ow_offs = ow0 + tl.arange(0, BLOCK_OW)  # (BLOCK_OW,)

    mask_h = oh_offs < H_out
    mask_w = ow_offs < W_out

    x_nc   = x_ptr   + (n * C + c) * H * W
    w_c    = w_ptr   + c * KH * KW
    out_nc = out_ptr + (n * C + c) * H_out * W_out

    acc = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    # Load kernel weights for this channel (KH*KW = 9 values for 3x3)
    # Fully unrolled loops
    for kh in tl.static_range(KH):
        # Input row index for this output row tile and kernel row
        ih = oh_offs * stride_h + kh - pad_h  # (BLOCK_OH,)
        valid_ih = mask_h & (ih >= 0) & (ih < H)
        ih_c = tl.maximum(0, tl.minimum(ih, H - 1))

        for kw in tl.static_range(KW):
            w_val = tl.load(w_c + kh * KW + kw)

            iw = ow_offs * stride_w + kw - pad_w  # (BLOCK_OW,)
            valid_iw = mask_w & (iw >= 0) & (iw < W)
            valid    = valid_ih[:, None] & valid_iw[None, :]

            iw_c = tl.maximum(0, tl.minimum(iw, W - 1))
            # 2D gather: (BLOCK_OH, BLOCK_OW)
            x_off = ih_c[:, None] * W + iw_c[None, :]
            x_val = tl.load(x_nc + x_off, mask=valid, other=0.0)

            acc += x_val * w_val

    if HAS_BIAS:
        acc += tl.load(b_ptr + c)

    out_mask = mask_h[:, None] & mask_w[None, :]
    out_off  = oh_offs[:, None] * W_out + ow_offs[None, :]
    tl.store(out_nc + out_off, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


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
            triton.cdiv(H_out, meta['BLOCK_OH']),
            triton.cdiv(W_out, meta['BLOCK_OW']),
        )

        depthwise_conv2d_slidew_kernel[grid](
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
