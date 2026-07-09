"""
Flash-Attention 2 for HEAD_DIM=1024 in Triton — iter 6 (blind run, final).

Best so far: iter 3 (1.78x, 33.9ms) — K column-major load, BM=16, BN=32,
D_TILE=256, 4 warps.

Iter 6: Try BN=16 (32 inner loop iterations, smaller per-iteration state).
With BN=16:
- QK dot: [16,256] x [256,16] -> [16,16] (less accumulation per iter)
- p matrix: [16,16] fp32 (tiny)
- pV dot: [16,16] x [16,256] -> [16,256] (smaller K-dimension)
- Half the K/V load per iteration but double the iterations

This reduces per-iteration register pressure significantly. The p matrix
shrinks from [16,32] to [16,16] fp32 = 256 elements (vs 512). The QK and pV
result matrices also shrink. This may allow better SM occupancy.

Also try K column-major load (same as iter 3 breakthrough).
"""

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _flash_fwd_v6(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    H,
    N_CTX:    tl.constexpr,
    D:        tl.constexpr,   # 1024
    D_TILE:   tl.constexpr,   # 256
    BM:       tl.constexpr,   # 16
    BN:       tl.constexpr,   # 16
    SCALE:    tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // H
    pid_h  = pid_bh  % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < N_CTX
    offs_d = tl.arange(0, D_TILE)
    offs_n_base = tl.arange(0, BN)

    Q_bh   = Q   + pid_b * stride_qb + pid_h * stride_qh
    K_bh   = K   + pid_b * stride_kb + pid_h * stride_kh
    V_bh   = V   + pid_b * stride_vb + pid_h * stride_vh
    Out_bh = Out + pid_b * stride_ob + pid_h * stride_oh

    # Pre-load Q tiles [BM, D_TILE] x 4 in fp16
    q0 = tl.load(Q_bh + offs_m[:, None] * stride_qm + (0*D_TILE + offs_d)[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)
    q1 = tl.load(Q_bh + offs_m[:, None] * stride_qm + (1*D_TILE + offs_d)[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)
    q2 = tl.load(Q_bh + offs_m[:, None] * stride_qm + (2*D_TILE + offs_d)[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)
    q3 = tl.load(Q_bh + offs_m[:, None] * stride_qm + (3*D_TILE + offs_d)[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)

    m = tl.full([BM], float('-inf'), dtype=tl.float32)
    s = tl.zeros([BM],              dtype=tl.float32)
    a0 = tl.zeros([BM, D_TILE], dtype=tl.float32)
    a1 = tl.zeros([BM, D_TILE], dtype=tl.float32)
    a2 = tl.zeros([BM, D_TILE], dtype=tl.float32)
    a3 = tl.zeros([BM, D_TILE], dtype=tl.float32)

    for start_n in range(0, N_CTX, BN):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < N_CTX

        # Load K transposed: K_T[d, n] = K[n, d]
        kt0 = tl.load(K_bh + (0*D_TILE + offs_d)[:, None] * stride_kk + offs_n[None, :] * stride_kn,
                      mask=mask_n[None, :], other=0.0)  # [D_TILE, BN]
        kt1 = tl.load(K_bh + (1*D_TILE + offs_d)[:, None] * stride_kk + offs_n[None, :] * stride_kn,
                      mask=mask_n[None, :], other=0.0)
        kt2 = tl.load(K_bh + (2*D_TILE + offs_d)[:, None] * stride_kk + offs_n[None, :] * stride_kn,
                      mask=mask_n[None, :], other=0.0)
        kt3 = tl.load(K_bh + (3*D_TILE + offs_d)[:, None] * stride_kk + offs_n[None, :] * stride_kn,
                      mask=mask_n[None, :], other=0.0)

        qk = (tl.dot(q0, kt0, allow_tf32=True) +
              tl.dot(q1, kt1, allow_tf32=True) +
              tl.dot(q2, kt2, allow_tf32=True) +
              tl.dot(q3, kt3, allow_tf32=True))
        qk = SCALE * qk
        qk = tl.where(mask_n[None, :], qk, float('-inf'))

        m_new  = tl.maximum(m, tl.max(qk, axis=1))
        alpha  = tl.exp(m  - m_new)
        p      = tl.exp(qk - m_new[:, None])   # fp32 [BM, BN]
        s      = alpha * s + tl.sum(p, axis=1)
        a0 = a0 * alpha[:, None]
        a1 = a1 * alpha[:, None]
        a2 = a2 * alpha[:, None]
        a3 = a3 * alpha[:, None]

        # Load V in fp16 [BN, D_TILE]
        v0 = tl.load(V_bh + offs_n[:, None] * stride_vn + (0*D_TILE + offs_d)[None, :] * stride_vk,
                     mask=mask_n[:, None], other=0.0)
        v1 = tl.load(V_bh + offs_n[:, None] * stride_vn + (1*D_TILE + offs_d)[None, :] * stride_vk,
                     mask=mask_n[:, None], other=0.0)
        v2 = tl.load(V_bh + offs_n[:, None] * stride_vn + (2*D_TILE + offs_d)[None, :] * stride_vk,
                     mask=mask_n[:, None], other=0.0)
        v3 = tl.load(V_bh + offs_n[:, None] * stride_vn + (3*D_TILE + offs_d)[None, :] * stride_vk,
                     mask=mask_n[:, None], other=0.0)

        p_h = p.to(tl.float16)
        a0 = a0 + tl.dot(p_h, v0, allow_tf32=True).to(tl.float32)
        a1 = a1 + tl.dot(p_h, v1, allow_tf32=True).to(tl.float32)
        a2 = a2 + tl.dot(p_h, v2, allow_tf32=True).to(tl.float32)
        a3 = a3 + tl.dot(p_h, v3, allow_tf32=True).to(tl.float32)

        m = m_new

    inv_s = 1.0 / s
    tl.store(Out_bh + offs_m[:, None] * stride_om + (0*D_TILE + offs_d)[None, :] * stride_ok,
             (a0 * inv_s[:, None]), mask=mask_m[:, None])
    tl.store(Out_bh + offs_m[:, None] * stride_om + (1*D_TILE + offs_d)[None, :] * stride_ok,
             (a1 * inv_s[:, None]), mask=mask_m[:, None])
    tl.store(Out_bh + offs_m[:, None] * stride_om + (2*D_TILE + offs_d)[None, :] * stride_ok,
             (a2 * inv_s[:, None]), mask=mask_m[:, None])
    tl.store(Out_bh + offs_m[:, None] * stride_om + (3*D_TILE + offs_d)[None, :] * stride_ok,
             (a3 * inv_s[:, None]), mask=mask_m[:, None])


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
        BN = 16
        D_TILE = 256

        grid = (triton.cdiv(N, BM), B * H)
        _flash_fwd_v6[grid](
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
