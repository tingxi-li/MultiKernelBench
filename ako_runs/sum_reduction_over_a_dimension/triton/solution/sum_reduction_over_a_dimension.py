import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_C controls output elements per program
        # We want many small programs for SM saturation
        # but BLOCK_C must be large enough for coalescing
        triton.Config({'BLOCK_C': 64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_C': 128}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_C': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 512}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 512}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_C': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 1024}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_C': 2048}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_C': 2048}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_C': 4096}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_C': 4096}, num_warps=32, num_stages=2),
    ],
    key=['B', 'R', 'C'],
)
@triton.jit
def _sum_reduce_dim1(
    x_ptr, out_ptr,
    B, R, C,
    stride_b, stride_r,
    BLOCK_C: tl.constexpr,
):
    """
    Sum-reduce x[B, R, C] over R → out[B, C].
    Uses float32 accumulation for correctness, but loads from input dtype.
    Aggressive unrolling: 4 rows per loop iteration to hide memory latency.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    col_start = pid_c * BLOCK_C
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = (col_start + offs_c) < C

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    base = x_ptr + pid_b * stride_b + col_start + offs_c

    # Unroll 4 rows per iteration to better hide memory latency
    R4 = (R // 4) * 4
    for r in tl.range(0, R4, 4):
        a0 = tl.load(base + (r + 0) * stride_r, mask=mask_c, other=0.0)
        a1 = tl.load(base + (r + 1) * stride_r, mask=mask_c, other=0.0)
        a2 = tl.load(base + (r + 2) * stride_r, mask=mask_c, other=0.0)
        a3 = tl.load(base + (r + 3) * stride_r, mask=mask_c, other=0.0)
        acc += a0 + a1 + a2 + a3

    # Tail
    for r in range(R4, R):
        acc += tl.load(base + r * stride_r, mask=mask_c, other=0.0)

    out = out_ptr + pid_b * C + col_start + offs_c
    tl.store(out, acc, mask=mask_c)


class Model(nn.Module):
    """
    Performs sum reduction over a specified dimension using a Triton kernel.
    Optimized for dim=1 reduction over 3D contiguous tensors.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1 and x.dim() == 3 and x.is_contiguous():
            B, R, C = x.shape
            out = torch.empty(B, C, dtype=x.dtype, device=x.device)

            grid = lambda meta: (B, triton.cdiv(C, meta['BLOCK_C']))
            _sum_reduce_dim1[grid](
                x, out,
                B, R, C,
                x.stride(0), x.stride(1),
            )

            return out.view(B, 1, C)

        return torch.sum(x, dim=self.dim, keepdim=True)
