import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def _act_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.minimum(tl.maximum(x * 0.16666666666666666 + 0.5, 0.0), 1.0)
    tl.store(y_ptr + offs, y, mask=mask)


class Model(nn.Module):
    """HardSigmoid = clamp(x/6 + 1/2, 0, 1) via Triton."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        _act_kernel[grid](x, y, n)
        return y
