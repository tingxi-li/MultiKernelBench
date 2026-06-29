import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({'BLOCK': bs}, num_warps=w)
             for bs in (256, 512, 1024, 2048) for w in (4, 8)],
    key=['n_out'],
)
@triton.jit
def _gather_kernel(x_ptr, idx_ptr, out_ptr, n_out, ncol_in, ncol_out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    row = offs // ncol_out
    col = tl.load(idx_ptr + offs, mask=mask, other=0)          # int64 index into dim=1
    val = tl.load(x_ptr + row * ncol_in + col, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


class Model(nn.Module):
    """torch.gather(x, dim=1, index=idx) via a Triton gather kernel."""
    def forward(self, x, idx):
        x = x.contiguous()
        idx = idx.contiguous()
        M, ncol_in = x.shape
        _, ncol_out = idx.shape
        out = torch.empty((M, ncol_out), device=x.device, dtype=x.dtype)
        n_out = out.numel()
        grid = lambda meta: (triton.cdiv(n_out, meta['BLOCK']),)
        _gather_kernel[grid](x, idx, out, n_out, ncol_in, ncol_out)
        return out
