import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _cumsum_rows_kernel(x_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """Inclusive prefix sum along each contiguous row.

    One program per row. The row of length N is scanned in BLOCK-sized
    chunks with a running scalar carry, so every element is read once and
    written once (HBM-roofline traffic). Masked loads/stores handle a final
    partial chunk, so any N is correct (no torch fallback). The loads in the
    loop are independent of the carry chain, letting Triton pipeline them.
    """
    row = tl.program_id(0)
    base = row * N
    carry = tl.zeros((), dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        # tl.cumsum(axis=0) is INCLUSIVE (verified) — matches torch.cumsum.
        cs = tl.cumsum(x, axis=0) + carry
        tl.store(o_ptr + base + offs, cs, mask=mask)
        carry += tl.sum(x, axis=0)


class Model(nn.Module):
    """Cumulative sum (prefix sum) along the last dim of a 2D tensor via a
    single-pass Triton scan. forward() only allocates and launches; the scan
    (the actual compute) lives entirely in the custom kernel."""

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.contiguous()
        M = x.shape[0]
        N = x.shape[1]
        out = torch.empty_like(x)
        _cumsum_rows_kernel[(M,)](
            x, out, N, BLOCK=1024, num_warps=16, num_stages=3
        )
        return out
