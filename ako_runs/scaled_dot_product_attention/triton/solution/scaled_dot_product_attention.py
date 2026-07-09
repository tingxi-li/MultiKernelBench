"""
Flash-Attention 2 for HEAD_DIM=1024 in Triton — iter 5.

Building on iter 4 (D_TILE=256, 4 chunks, BN=32, 35.8ms, 1.72x).

Try D_TILE=512 (2 chunks instead of 4):
  SMEM: 2*(BM+BN)*D_TILE*2 = 2*(16+32)*512*2 = 96 KB ✓ (< 101 KB)
  tl.dot: K=D_TILE=512 for QKT, K=BN=32 for pV ✓

Benefits vs D_TILE=256:
  - Fewer loop iterations over D dimension: 2 vs 4 → less loop overhead
  - Larger K-dim (512) for QKT: better tensor core utilization
  - Same total data loaded

Also try D_TILE=512, BN=32, num_warps=4 (same as iter 4 except fewer tiles).
"""

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _flash_fwd_d512(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    H,
    N_CTX:    tl.constexpr,
    D:        tl.constexpr,   # 1024
    D_TILE:   tl.constexpr,   # 512
    BM:       tl.constexpr,   # 16
    BN:       tl.constexpr,   # 32
    SCALE:    tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // H
    pid_h  = pid_bh  % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < N_CTX
    offs_d = tl.arange(0, D_TILE)

    Q_bh   = Q   + pid_b * stride_qb + pid_h * stride_qh
    K_bh   = K   + pid_b * stride_kb + pid_h * stride_kh
    V_bh   = V   + pid_b * stride_vb + pid_h * stride_vh
    Out_bh = Out + pid_b * stride_ob + pid_h * stride_oh

    # Pre-load Q tiles [BM, D_TILE] x 2
    q0 = tl.load(Q_bh + offs_m[:, None] * stride_qm + (0*D_TILE + offs_d)[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)
    q1 = tl.load(Q_bh + offs_m[:, None] * stride_qm + (1*D_TILE + offs_d)[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)

    m = tl.full([BM], float('-inf'), dtype=tl.float32)
    s = tl.zeros([BM],              dtype=tl.float32)
    a0 = tl.zeros([BM, D_TILE], dtype=tl.float32)
    a1 = tl.zeros([BM, D_TILE], dtype=tl.float32)

    offs_n_base = tl.arange(0, BN)

    for start_n in range(0, N_CTX, BN):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < N_CTX

        # K D-tiles
        k0 = tl.load(K_bh + offs_n[:, None] * stride_kn + (0*D_TILE + offs_d)[None, :] * stride_kk,
                     mask=mask_n[:, None], other=0.0)
        k1 = tl.load(K_bh + offs_n[:, None] * stride_kn + (1*D_TILE + offs_d)[None, :] * stride_kk,
                     mask=mask_n[:, None], other=0.0)

        # QK^T: K-dim = D_TILE = 512
        qk = (tl.dot(q0, tl.trans(k0), allow_tf32=True) +
              tl.dot(q1, tl.trans(k1), allow_tf32=True))
        qk = SCALE * qk
        qk = tl.where(mask_n[None, :], qk, float('-inf'))

        m_new  = tl.maximum(m, tl.max(qk, axis=1))
        alpha  = tl.exp(m  - m_new)
        p_fp32 = tl.exp(qk - m_new[:, None])
        s      = alpha * s + tl.sum(p_fp32, axis=1)
        p      = p_fp32.to(tl.float16)
        a0 = a0 * alpha[:, None]
        a1 = a1 * alpha[:, None]

        # V D-tiles: K-dim = BN = 32 for pV
        v0 = tl.load(V_bh + offs_n[:, None] * stride_vn + (0*D_TILE + offs_d)[None, :] * stride_vk,
                     mask=mask_n[:, None], other=0.0)
        v1 = tl.load(V_bh + offs_n[:, None] * stride_vn + (1*D_TILE + offs_d)[None, :] * stride_vk,
                     mask=mask_n[:, None], other=0.0)

        a0 = a0 + tl.dot(p, v0, allow_tf32=True).to(tl.float32)
        a1 = a1 + tl.dot(p, v1, allow_tf32=True).to(tl.float32)

        m = m_new

    inv_s = 1.0 / s
    tl.store(Out_bh + offs_m[:, None] * stride_om + (0*D_TILE + offs_d)[None, :] * stride_ok,
             (a0 * inv_s[:, None]).to(tl.float32), mask=mask_m[:, None])
    tl.store(Out_bh + offs_m[:, None] * stride_om + (1*D_TILE + offs_d)[None, :] * stride_ok,
             (a1 * inv_s[:, None]).to(tl.float32), mask=mask_m[:, None])


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
        D_TILE = 512

        grid = (triton.cdiv(N, BM), B * H)
        _flash_fwd_d512[grid](
            Qh, Kh, Vh, Out,
            Qh.stride(0), Qh.stride(1), Qh.stride(2), Qh.stride(3),
            Kh.stride(0), Kh.stride(1), Kh.stride(2), Kh.stride(3),
            Vh.stride(0), Vh.stride(1), Vh.stride(2), Vh.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            H,
            N_CTX=N,
            D=D,
            D_TILE=D_TILE,
            BM=BM,
            BN=BN,
            SCALE=sm_scale,
            num_warps=4,
            num_stages=1,
        )
        return Out
