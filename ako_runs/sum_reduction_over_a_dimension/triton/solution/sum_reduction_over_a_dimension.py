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
):
    """
    Sum-reduce x[B, R, C] over R → out[B, C].
    Each program handles BLOCK_C contiguous columns for one batch element.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    col_start = pid_c * BLOCK_C
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = (col_start + offs_c) < C

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    base = x_ptr + pid_b * stride_b + col_start + offs_c

    for r in tl.range(0, R):
        acc += tl.load(base + r * stride_r, mask=mask_c, other=0.0)

    tl.store(out_ptr + pid_b * C + col_start + offs_c, acc, mask=mask_c)


class Model(nn.Module):
    """
    Performs sum reduction over a specified dimension using a Triton kernel.
    Optimized for dim=1 reduction over 3D contiguous tensors.
    Fixed to best config: BLOCK_C=4096, num_warps=16, num_stages=3.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1 and x.dim() == 3 and x.is_contiguous():
            B, R, C = x.shape

            # Optimal config from profiling: BLOCK_C=4096, nw=16, ns=3
            BLOCK_C = 4096
            out = torch.empty(B, C, dtype=x.dtype, device=x.device)

            grid = (B, triton.cdiv(C, BLOCK_C))
            _sum_reduce_dim1[grid](
                x, out,
                B, R, C,
                x.stride(0), x.stride(1),
                BLOCK_C=BLOCK_C,
                num_warps=16,
                num_stages=3,
            )

            return out.view(B, 1, C)

        return torch.sum(x, dim=self.dim, keepdim=True)
