import torch
import torch.nn as nn
import triton
import triton.language as tl

PREC = 'tf32'


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_gelu_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
                      sx0, sx1, sw0, sw1, sy0, sy1,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # x: (M,K) row-major ; w: nn.Linear weight (N,K) row-major, so y = x @ w^T.
    # Load w as (BLOCK_N,BLOCK_K) coalesced along K, transpose in-register for the dot.
    x_ptrs = x_ptr + offs_m[:, None] * sx0 + offs_k[None, :] * sx1
    w_ptrs = w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(x_ptrs, mask=offs_k[None, :] < K - k0, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[None, :] < K - k0, other=0.0)
        acc += tl.dot(a, tl.trans(w), input_precision=PREC)
        x_ptrs += BLOCK_K * sx1
        w_ptrs += BLOCK_K * sw1
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]
    # exact gelu: 0.5*x*(1+erf(x/sqrt(2)))
    acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))
    y_ptrs = y_ptr + offs_m[:, None] * sy0 + offs_n[None, :] * sy1
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_warps=16),
        triton.Config({'BLOCK_N': 4096}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _softmax_rows_kernel(y_ptr, o_ptr, M, N, sy0, sy1, so0, so1, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    row_y = y_ptr + row * sy0
    m = -float('inf')
    l = 0.0
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        v = tl.load(row_y + offs * sy1, mask=offs < N, other=-float('inf'))
        m_new = tl.maximum(m, tl.max(v, axis=0))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(v - m_new), axis=0)
        m = m_new
    inv = 1.0 / l
    row_o = o_ptr + row * so0
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        v = tl.load(row_y + offs * sy1, mask=offs < N, other=0.0)
        o = tl.exp(v - m) * inv
        tl.store(row_o + offs * so1, o, mask=offs < N)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous()
        W = self.linear.weight
        b = self.linear.bias
        M, K = x.shape
        N, Kw = W.shape
        Y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        _gemm_gelu_kernel[grid](
            x, W, b, Y, M, N, K,
            x.stride(0), x.stride(1), W.stride(0), W.stride(1), Y.stride(0), Y.stride(1),
            PREC=PREC,
        )
        _softmax_rows_kernel[(M,)](
            Y, out, M, N,
            Y.stride(0), Y.stride(1), out.stride(0), out.stride(1),
        )
        return out


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
