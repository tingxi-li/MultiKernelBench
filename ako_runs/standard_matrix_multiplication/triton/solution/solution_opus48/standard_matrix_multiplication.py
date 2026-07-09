import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
               sa0, sa1, sb0, sb1, sc0, sc1,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1
    b_ptrs = b_ptr + offs_k[:, None] * sb0 + offs_n[None, :] * sb1
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k0, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k0, other=0.0)
        acc += tl.dot(a, b, input_precision='ieee')
        a_ptrs += BLOCK_K * sa1
        b_ptrs += BLOCK_K * sb0
    c_ptrs = c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        Kb, N = B.shape
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        _mm_kernel[grid](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        )
        return C


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []
