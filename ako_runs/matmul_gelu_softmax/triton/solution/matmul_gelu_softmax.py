import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_gelu_fp16_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Fused matmul (fp16 native) + bias + GELU. a,b are fp16; output fp16."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0).to(tl.float16)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0).to(tl.float16)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias (fp32) + exact GELU in fp32, then cast to fp16 for storage
    bias_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + bias_offs, mask=bias_offs < N, other=0.0)
    acc = acc + bias[None, :]
    acc_gelu = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))

    # Store as fp16 (halves softmax load bandwidth)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc_gelu.to(tl.float16), mask=c_mask)


@triton.jit
def _softmax_fp16_to_fp32_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Softmax: reads fp16 input, writes fp32 output. Arithmetic in fp32."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load fp16, convert to fp32 for arithmetic
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
    row_max = tl.max(row, axis=0)
    row_shifted = row - row_max
    exp_row = tl.exp(row_shifted)
    sum_exp = tl.sum(exp_row, axis=0)
    softmax_out = exp_row / sum_exp

    out_row_start = output_ptr + row_idx * output_row_stride
    tl.store(out_row_start + col_offsets, softmax_out, mask=mask)


def matmul_gelu_fp16_out(x_fp16, w_fp16, bias):
    """Returns fp16 output [M, N]."""
    M, K = x_fp16.shape
    N = w_fp16.shape[0]
    out = torch.empty((M, N), device=x_fp16.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    _matmul_gelu_fp16_kernel[grid](
        x_fp16, w_fp16, bias, out,
        M, N, K,
        x_fp16.stride(0), x_fp16.stride(1),
        w_fp16.stride(1), w_fp16.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


def softmax_rows_fp16_in(x_fp16):
    """Softmax over dim=1: reads fp16, writes fp32."""
    M, N = x_fp16.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty((M, N), device=x_fp16.device, dtype=torch.float32)
    _softmax_fp16_to_fp32_kernel[(M,)](
        out, x_fp16,
        x_fp16.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=max(1, BLOCK_SIZE // 512),
    )
    return out


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('weight_fp16', None)

    def forward(self, x):
        # Lazily cache fp16 weights
        if self.weight_fp16 is None or self.weight_fp16.shape != self.linear.weight.shape:
            self.weight_fp16 = self.linear.weight.to(torch.float16)
        x_fp16 = x.to(torch.float16)
        # Fused matmul+GELU → fp16 intermediate (halves softmax BW)
        gelu_fp16 = matmul_gelu_fp16_out(x_fp16, self.weight_fp16, self.linear.bias)
        # Softmax: reads fp16, writes fp32
        out = softmax_rows_fp16_in(gelu_fp16)
        return out
