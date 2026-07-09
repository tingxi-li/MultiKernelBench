import torch
import torch.nn as nn
import triton
import triton.language as tl


# A focused set of high-quality configs based on iter-2 winner direction.
# BLOCK_OW should be >= 128 for good coalescing on output writes.
# BLOCK_NC batches multiple (n,c) slices per CTA to amortize launch overhead.
@triton.autotune(
    configs=[
        # Single (n,c) per block, vary spatial tiling
        triton.Config({'BLOCK_NC': 1, 'BLOCK_OH': 4,  'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_NC': 1, 'BLOCK_OH': 8,  'BLOCK_OW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_NC': 1, 'BLOCK_OH': 8,  'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_NC': 1, 'BLOCK_OH': 16, 'BLOCK_OW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_NC': 1, 'BLOCK_OH': 2,  'BLOCK_OW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_NC': 1, 'BLOCK_OH': 4,  'BLOCK_OW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_NC': 1, 'BLOCK_OH': 1,  'BLOCK_OW': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_NC': 1, 'BLOCK_OH': 2,  'BLOCK_OW': 512}, num_warps=8, num_stages=2),
        # Multiple (n,c) per block
        triton.Config({'BLOCK_NC': 2, 'BLOCK_OH': 4,  'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_NC': 2, 'BLOCK_OH': 8,  'BLOCK_OW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_NC': 4, 'BLOCK_OH': 4,  'BLOCK_OW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_NC': 4, 'BLOCK_OH': 2,  'BLOCK_OW': 128}, num_warps=4, num_stages=2),
    ],
    key=['NC', 'H', 'W', 'H_out', 'W_out', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,    # (N, C, H, W) — contiguous
    w_ptr,    # (C, KH, KW)  — squeezed, contiguous
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
    BLOCK_NC: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    """
    Grid: (ceil(NC/BLOCK_NC), ceil(H_out/BLOCK_OH), ceil(W_out/BLOCK_OW))
    Each block handles BLOCK_NC consecutive (n,c) slices and a BLOCK_OH x BLOCK_OW output tile.
    """
    pid_nc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    nc0   = pid_nc * BLOCK_NC
    oh0   = pid_oh * BLOCK_OH
    ow0   = pid_ow * BLOCK_OW

    # nc slice indices
    nc_offs = nc0 + tl.arange(0, BLOCK_NC)   # (BLOCK_NC,)
    # output spatial indices
    oh_offs = oh0 + tl.arange(0, BLOCK_OH)   # (BLOCK_OH,)
    ow_offs = ow0 + tl.arange(0, BLOCK_OW)   # (BLOCK_OW,)

    mask_nc = nc_offs < NC
    mask_h  = oh_offs < H_out
    mask_w  = ow_offs < W_out

    # We process each nc in the block sequentially.
    for i in tl.static_range(BLOCK_NC):
        nc = nc0 + i
        c = nc % C

        x_nc   = x_ptr   + nc * H * W
        w_c    = w_ptr   + c * KH * KW
        out_nc = out_ptr + nc * H_out * W_out

        # Accumulator for this (nc) slice
        acc = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.float32)

        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                w_val = tl.load(w_c + kh * KW + kw)

                ih = oh_offs * stride_h + kh - pad_h
                iw = ow_offs * stride_w + kw - pad_w

                valid_h = mask_h & (ih >= 0) & (ih < H)
                valid_w = mask_w & (iw >= 0) & (iw < W)
                valid   = valid_h[:, None] & valid_w[None, :]

                ih_c = tl.maximum(0, tl.minimum(ih, H - 1))
                iw_c = tl.maximum(0, tl.minimum(iw, W - 1))

                x_off = ih_c[:, None] * W + iw_c[None, :]
                x_val = tl.load(x_nc + x_off, mask=valid, other=0.0)

                acc += x_val * w_val

        if HAS_BIAS:
            acc = acc + tl.load(b_ptr + c)

        valid_nc = nc < NC
        out_mask = valid_nc & (mask_h[:, None] & mask_w[None, :])
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
            triton.cdiv(NC, meta['BLOCK_NC']),
            triton.cdiv(H_out, meta['BLOCK_OH']),
            triton.cdiv(W_out, meta['BLOCK_OW']),
        )

        depthwise_conv2d_kernel[grid](
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
