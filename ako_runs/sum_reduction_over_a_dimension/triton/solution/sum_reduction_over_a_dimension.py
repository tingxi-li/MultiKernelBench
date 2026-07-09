import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sum_reduce_dim1(
    x_ptr, out_ptr,
    B, R, C,
    stride_b, stride_r,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Sum-reduce x[B, R, C] over R → out[B, C].
    Uses 2D tiles [BLOCK_R, BLOCK_C] with evict_first to maximize streaming bandwidth.
    BLOCK_R=2 improves L2 throughput by issuing 2 contiguous row loads per loop iteration,
    giving the memory controller better opportunities for coalescing.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    col_start = pid_c * BLOCK_C
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = (col_start + offs_c) < C

    offs_r = tl.arange(0, BLOCK_R)
    base = x_ptr + pid_b * stride_b + col_start + offs_c

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for r_start in tl.range(0, R, BLOCK_R):
        rows = r_start + offs_r
        mask_r = rows < R
        # Load BLOCK_R x BLOCK_C tile with streaming eviction
        tile = tl.load(
            base[None, :] + rows[:, None] * stride_r,
            mask=mask_r[:, None] & mask_c[None, :],
            other=0.0,
            eviction_policy='evict_first',
        )
        acc += tl.sum(tile, axis=0)

    tl.store(out_ptr + pid_b * C + col_start + offs_c, acc, mask=mask_c)


class Model(nn.Module):
    """
    Performs sum reduction over a specified dimension using a Triton kernel.
    Optimized for dim=1 reduction over 3D contiguous tensors.
    Config: BLOCK_C=4096, BLOCK_R=2, num_warps=16, num_stages=2 + evict_first.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1 and x.dim() == 3 and x.is_contiguous():
            B, R, C = x.shape

            BLOCK_C = 4096
            BLOCK_R = 2
            out = torch.empty(B, C, dtype=x.dtype, device=x.device)

            grid = (B, triton.cdiv(C, BLOCK_C))
            _sum_reduce_dim1[grid](
                x, out,
                B, R, C,
                x.stride(0), x.stride(1),
                BLOCK_C=BLOCK_C,
                BLOCK_R=BLOCK_R,
                num_warps=16,
                num_stages=2,
            )

            return out.view(B, 1, C)

        return torch.sum(x, dim=self.dim, keepdim=True)
