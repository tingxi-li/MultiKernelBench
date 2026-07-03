import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _cumsum_rows_kernel(x_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """Inclusive prefix sum along each contiguous row.

    One program per row. The row of length N is scanned in BLOCK-sized
    chunks with a running scalar carry, so every element is read once and
    written once (HBM-roofline traffic). The loads in the loop are
    independent of the carry chain, letting Triton pipeline them.
    """
    row = tl.program_id(0)
    base = row * N
    carry = tl.zeros((), dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + offs)
        # tl.cumsum(axis=0) is INCLUSIVE (verified) — matches torch.cumsum.
        cs = tl.cumsum(x, axis=0) + carry
        tl.store(o_ptr + base + offs, cs)
        carry += tl.sum(x, axis=0)


class Model(nn.Module):
    """Cumulative sum (prefix sum) along `dim`, Triton single-pass scan."""

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Specialized fast path: 2D contiguous fp32, scan along the last dim
        # (dim == 1 here), the only shape the benchmark exercises. Each row
        # is N contiguous elements -> one Triton program per row.
        if (
            x.is_cuda
            and x.dim() == 2
            and self.dim == 1
            and x.dtype == torch.float32
        ):
            x = x.contiguous()
            M, N = x.shape
            BLOCK = 1024
            if N % BLOCK == 0:
                out = torch.empty_like(x)
                _cumsum_rows_kernel[(M,)](
                    x, out, N, BLOCK, num_warps=16, num_stages=3
                )
                return out
        # Fallback for any shape/dim/dtype the fast path doesn't cover.
        return torch.cumsum(x, dim=self.dim)
