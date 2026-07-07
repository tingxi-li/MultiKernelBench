import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def _sum_dim1_kernel(x_ptr, out_ptr, M, N,
                     sx0, sx1, sx2, so0, so2,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    base = x_ptr + pid_b * sx0 + offs_n * sx2
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        ptrs = base[None, :] + offs_m[:, None] * sx1
        tile = tl.load(ptrs, mask=(offs_m[:, None] < M) & mask_n[None, :], other=0.0)
        acc += tl.sum(tile, axis=0)
    tl.store(out_ptr + pid_b * so0 + offs_n * so2, acc, mask=mask_n)


class Model(nn.Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.dim == 1 and x.dim() == 3
        B, M, N = x.shape
        x = x.contiguous()
        out = torch.empty((B, 1, N), device=x.device, dtype=x.dtype)
        grid = lambda meta: (B, triton.cdiv(N, meta['BLOCK_N']))
        _sum_dim1_kernel[grid](
            x, out, M, N,
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(2),
        )
        return out


batch_size = 128
dim1 = 4096
dim2 = 4096
reduce_dim = 1


def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    return [reduce_dim]
