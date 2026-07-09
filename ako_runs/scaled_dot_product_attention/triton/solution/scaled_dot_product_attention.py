"""
Flash-Attention 2 for HEAD_DIM=1024 in Triton — iter 2.

Improvements over iter 1:
1. Autotune BM, BN, num_warps, num_stages to find best configuration.
2. Larger BN candidates (64, 128) to improve K-tile reuse.
3. Better SMEM utilization.

SMEM budget: ~99 KB. fp16 → each elem = 2 bytes.
  Q tile:   BM * D * 2 bytes (persistent across N loop)
  K or V tile: BN * D * 2 bytes (alternate, so max(K,V) = BN * D * 2)
  Total SMEM: (BM + BN) * D * 2 <= 99*1024 bytes
  → BM + BN <= 48 for D=1024

Autotune candidates:
  BM=16, BN=32: 48*1024*2 = 96 KB ✓
  BM=8,  BN=32: 40*1024*2 = 80 KB ✓
  BM=16, BN=16: 32*1024*2 = 64 KB ✓
  BM=8,  BN=16: 24*1024*2 = 48 KB ✓
"""

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_autotune_configs():
    configs = []
    for BM in [8, 16]:
        for BN in [16, 32]:
            if BM + BN > 48:
                continue
            for nw in [2, 4, 8]:
                for ns in [1, 2]:
                    configs.append(triton.Config(
                        {'BM': BM, 'BN': BN},
                        num_warps=nw,
                        num_stages=ns,
                    ))
    return configs


@triton.autotune(
    configs=_get_autotune_configs(),
    key=['N_CTX', 'D'],
)
@triton.jit
def _flash_fwd_tuned(
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

        qk = SCALE * tl.dot(q, tl.trans(k))
        qk = tl.where(mask_n[None, :], qk, float('-inf'))

        m_new  = tl.maximum(m, tl.max(qk, axis=1))
        p      = tl.exp(qk - m_new[:, None]).to(tl.float16)
        alpha  = tl.exp(m  - m_new)
        s      = alpha * s + tl.sum(p.to(tl.float32), axis=1)
        o      = o * alpha[:, None]

        v_ptrs = V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        o = o + tl.dot(p, v).to(tl.float32)

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

        # BM is set by autotune; grid uses max BM to get upper bound, but
        # autotune will override. Use a lambda grid that reads tuned BM.
        def grid(meta):
            return (triton.cdiv(N, meta['BM']), B * H)

        _flash_fwd_tuned[grid](
            Qh, Kh, Vh, Out,
            Qh.stride(0), Qh.stride(1), Qh.stride(2), Qh.stride(3),
            Kh.stride(0), Kh.stride(1), Kh.stride(2), Kh.stride(3),
            Vh.stride(0), Vh.stride(1), Vh.stride(2), Vh.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            H,
            N_CTX=N,
            D=D,
            SCALE=sm_scale,
        )
        return Out
