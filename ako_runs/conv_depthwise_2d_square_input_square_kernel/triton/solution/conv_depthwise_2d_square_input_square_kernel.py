import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_OW is the number of output columns per CTA
        # BLOCK_NC is number of (n,c) slices per CTA — usually 1
        triton.Config({'BLOCK_OW': 64,  'BLOCK_OH': 8},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64,  'BLOCK_OH': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_OH': 4},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_OH': 8},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_OH': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 256, 'BLOCK_OH': 4},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 256, 'BLOCK_OH': 8},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 512, 'BLOCK_OH': 2},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 32,  'BLOCK_OH': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32,  'BLOCK_OH': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64,  'BLOCK_OH': 32}, num_warps=8, num_stages=2),
    ],
    key=['C', 'H_out', 'W_out', 'KH', 'KW', 'stride_h', 'stride_w', 'pad_h', 'pad_w'],
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,   # (N, C, H, W)
    w_ptr,   # (C, KH, KW)  [squeezed from (C,1,KH,KW)]
    b_ptr,   # (C,) or dummy
    out_ptr, # (N, C, H_out, W_out)
    N: tl.constexpr, C: tl.constexpr,
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
    Grid: (N*C, ceil(H_out/BLOCK_OH), ceil(W_out/BLOCK_OW))
    Each CTA handles one (n,c) slice + a tile of output rows and columns.
    """
    pid_nc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C

    oh0 = pid_oh * BLOCK_OH
    ow0 = pid_ow * BLOCK_OW

    # Offsets within tile
    oh_offs = oh0 + tl.arange(0, BLOCK_OH)  # (BLOCK_OH,)
    ow_offs = ow0 + tl.arange(0, BLOCK_OW)  # (BLOCK_OW,)

    mask_h = oh_offs < H_out
    mask_w = ow_offs < W_out

    # Accumulator
    acc = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    # base pointer for this (n, c) input channel and output channel
    x_nc  = x_ptr   + (n * C + c) * H * W
    w_c   = w_ptr   + c * KH * KW
    out_nc = out_ptr + (n * C + c) * H_out * W_out

    # Loop over kernel rows and columns
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            # input (ih, iw) for each output position
            ih = oh_offs * stride_h + kh - pad_h  # (BLOCK_OH,)
            iw = ow_offs * stride_w + kw - pad_w  # (BLOCK_OW,)

            valid_h = mask_h & (ih >= 0) & (ih < H)
            valid_w = mask_w & (iw >= 0) & (iw < W)
            valid_2d = valid_h[:, None] & valid_w[None, :]

            # Clamp to avoid OOB ptr arithmetic
            ih_c = tl.maximum(0, tl.minimum(ih, H - 1))
            iw_c = tl.maximum(0, tl.minimum(iw, W - 1))

            x_off = ih_c[:, None] * W + iw_c[None, :]  # (BLOCK_OH, BLOCK_OW)
            x_val = tl.load(x_nc + x_off, mask=valid_2d, other=0.0)

            w_val = tl.load(w_c + kh * KW + kw)
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

        x_c    = x.contiguous()
        weight = self.conv2d.weight.squeeze(1).contiguous()  # (C, KH, KW)
        out    = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        has_bias = self.conv2d.bias is not None
        bias_ptr = self.conv2d.bias if has_bias else weight  # dummy when no bias

        grid = lambda meta: (
            N * C,
            triton.cdiv(H_out, meta['BLOCK_OH']),
            triton.cdiv(W_out, meta['BLOCK_OW']),
        )

        depthwise_conv2d_kernel[grid](
            x_c, weight, bias_ptr, out,
            N, C,
            H, W,
            H_out, W_out,
            stride_h, stride_w,
            pad_h, pad_w,
            KH, KW,
            has_bias,
        )
        return out
