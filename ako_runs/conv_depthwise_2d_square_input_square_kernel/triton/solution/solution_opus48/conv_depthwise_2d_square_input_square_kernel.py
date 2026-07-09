import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 256}, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 128}, num_warps=8),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 512}, num_warps=8),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 512}, num_warps=16),
        triton.Config({'BLOCK_H': 2, 'BLOCK_W': 512}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 256}, num_warps=16),
    ],
    key=['OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def _dwconv_kernel(x_ptr, w_ptr, o_ptr, H, W, OH, OW,
                   sx_b, sx_c, sx_h, sx_w, sw_c, sw_kh, sw_kw, so_b, so_c, so_h, so_w,
                   KH: tl.constexpr, KW: tl.constexpr, STRIDE: tl.constexpr,
                   BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    b = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)
    ntw = tl.cdiv(OW, BLOCK_W)
    ti = t // ntw
    tj = t % ntw
    oh = ti * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = tj * BLOCK_W + tl.arange(0, BLOCK_W)
    oh_mask = oh < OH
    ow_mask = ow < OW
    xbc = x_ptr + b * sx_b + c * sx_c
    wc = w_ptr + c * sw_c
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for di in tl.static_range(KH):
        ih = oh * STRIDE + di
        for dj in tl.static_range(KW):
            iw = ow * STRIDE + dj
            wv = tl.load(wc + di * sw_kh + dj * sw_kw)
            xv = tl.load(xbc + ih[:, None] * sx_h + iw[None, :] * sx_w,
                         mask=(ih[:, None] < H) & (iw[None, :] < W), other=0.0)
            acc += xv * wv
    o_ptrs = o_ptr + b * so_b + c * so_c + oh[:, None] * so_h + ow[None, :] * so_w
    tl.store(o_ptrs, acc, mask=oh_mask[:, None] & ow_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                                padding=padding, groups=in_channels, bias=bias)
        self.stride = stride
        self.ksize = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        W = self.conv2d.weight  # (C, 1, KH, KW)
        B, C, H, Wd = x.shape
        KH = self.ksize
        KW = self.ksize
        OH = x.unfold(2, KH, self.stride).size(2)
        OW = x.unfold(3, KW, self.stride).size(3)
        out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
        grid = lambda meta: (
            B, C,
            torch.empty((triton.cdiv(OH, meta['BLOCK_H']), triton.cdiv(OW, meta['BLOCK_W'])),
                        device='meta').numel(),
        )
        _dwconv_kernel[grid](
            x, W, out, H, Wd, OH, OW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            W.stride(0), W.stride(2), W.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            KH=KH, KW=KW, STRIDE=self.stride,
        )
        return out


# Test code
batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
