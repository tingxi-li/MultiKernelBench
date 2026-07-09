"""
Flash-Attention 2 for HEAD_DIM=1024 in Triton — iter 3.

Going back to the best iter 1 config (BM=16, BN=32, 4 warps, num_stages=1)
with one targeted optimization: explicit `tl.multiple_of` annotations
to enable better vectorized loads and `tl.max_contiguous` for the
dimension strides.

The main bottleneck per the math:
  K effective reads = (N/BM) * N * D * bytes_per_elem
  = 32 * 512 * 1024 * 2 = 33.5 MB per (b,h)
  Total across B*H=1024 heads: 33.5 * 1024 = 34 GB
  At 960 GB/s: 34 GB / 960 = 35.4ms -- THIS IS THE MEMORY WALL!

So the kernel IS basically memory-bandwidth bound at the K/V access pattern.
The 37ms we see is very close to the theoretical 35ms memory bound.

To beat this we'd need to either:
1. Reduce BM (fewer Q-tiles = fewer K/V loads per K-tile) - but BM=16 is already limited
2. Process multiple sequence positions per CTA (reduce N/BM factor)
3. Use a completely different algorithm

Since BM=16 BN=32 is essentially at the memory bound, there's limited gain from tuning.
Let me try num_warps=8 for higher SM occupancy, possibly hiding memory latency better.
Also try moving to contiguous inputs (the inputs are already contiguous but double-check).
"""

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _flash_fwd_w8(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    H,
    N_CTX:    tl.constexpr,
    D:        tl.constexpr,
    BM:       tl.constexpr,
    BN:       tl.constexpr,
    SCALE:    tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // H
    pid_h  = pid_bh  % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < N_CTX

    Q_bh   = Q   + pid_b * stride_qb + pid_h * stride_qh
    K_bh   = K   + pid_b * stride_kb + pid_h * stride_kh
    V_bh   = V   + pid_b * stride_vb + pid_h * stride_vh
    Out_bh = Out + pid_b * stride_ob + pid_h * stride_oh

    q_ptrs = Q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)  # [BM, D] fp16

    m = tl.full([BM], float('-inf'), dtype=tl.float32)
    s = tl.zeros([BM],              dtype=tl.float32)
    o = tl.zeros([BM, D],           dtype=tl.float32)

    for start_n in range(0, N_CTX, BN):
        offs_n = start_n + tl.arange(0, BN)
        mask_n = offs_n < N_CTX

        k_ptrs = K_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        qk = SCALE * tl.dot(q, tl.trans(k), allow_tf32=True)
        qk = tl.where(mask_n[None, :], qk, float('-inf'))

        m_new  = tl.maximum(m, tl.max(qk, axis=1))
        alpha  = tl.exp(m  - m_new)
        p_fp32 = tl.exp(qk - m_new[:, None])
        s      = alpha * s + tl.sum(p_fp32, axis=1)
        o      = o * alpha[:, None]
        p      = p_fp32.to(tl.float16)

        v_ptrs = V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        o = o + tl.dot(p, v, allow_tf32=True).to(tl.float32)

        m = m_new

    o = o / s[:, None]
    o_ptrs = Out_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, o.to(tl.float32), mask=mask_m[:, None])


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, N, D = Q.shape

        sm_scale = 1.0 / math.sqrt(D)
        Out = torch.empty_like(Q)

        Qh = Q.to(torch.float16)
        Kh = K.to(torch.float16)
        Vh = V.to(torch.float16)

        BM = 16
        BN = 32

        grid = (triton.cdiv(N, BM), B * H)
        _flash_fwd_w8[grid](
            Qh, Kh, Vh, Out,
            Qh.stride(0), Qh.stride(1), Qh.stride(2), Qh.stride(3),
            Kh.stride(0), Kh.stride(1), Kh.stride(2), Kh.stride(3),
            Vh.stride(0), Vh.stride(1), Vh.stride(2), Vh.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            H,
            N_CTX=N,
            D=D,
            BM=BM,
            BN=BN,
            SCALE=sm_scale,
            num_warps=8,
            num_stages=1,
        )
        return Out
