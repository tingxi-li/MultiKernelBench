"""
Flash-Attention 2 for HEAD_DIM=1024 in Triton — iter 4.

Best so far: iter 1 at 37ms (1.65x) with fp16, BM=16, BN=32, 4 warps.
Analysis: near memory-bandwidth ceiling (~35ms theoretical minimum).

Trying a different axis: tile the D dimension internally.
Instead of one tl.dot(q[16,1024], k.T[1024,32]), do 4 tl.dot calls
with D_tile=256. This reduces the register footprint of the Q tile from
16*1024 fp16 (16K elements) to 4 * 16*256 fp16 (4K per tile, loaded
in a loop). The K/V tiles are similarly tiled.

Expected: lower register pressure → higher occupancy → better latency hiding.
Trade-off: 4x more K/V load iterations (but same total bytes).

SMEM (D-tiled): (BM + BN) * D_TILE * 2 = (16+32)*256*2 = 24 KB << 101 KB.
tl.dot K-dim: D_TILE = 256 >= 16 ✓.
"""

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _flash_fwd_dtiled(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    H,
    N_CTX:    tl.constexpr,
    D:        tl.constexpr,   # 1024
    D_TILE:   tl.constexpr,   # 256 — sub-tiles of head dim
    N_DTILES: tl.constexpr,   # D // D_TILE = 4
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

    Q_bh   = Q   + pid_b * stride_qb + pid_h * stride_qh
    K_bh   = K   + pid_b * stride_kb + pid_h * stride_kh
    V_bh   = V   + pid_b * stride_vb + pid_h * stride_vh
    Out_bh = Out + pid_b * stride_ob + pid_h * stride_oh

    # Pre-load Q by D-tiles: q_tiles[d] = Q[m, d*D_TILE:(d+1)*D_TILE]
    # We keep these all in registers as 4 separate tensors.
    # Triton handles this as 4 independent register arrays.
    offs_d0 = tl.arange(0, D_TILE)
    q0 = tl.load(Q_bh + offs_m[:, None] * stride_qm + (0*D_TILE + offs_d0)[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)
    q1 = tl.load(Q_bh + offs_m[:, None] * stride_qm + (1*D_TILE + offs_d0)[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)
    q2 = tl.load(Q_bh + offs_m[:, None] * stride_qm + (2*D_TILE + offs_d0)[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)
    q3 = tl.load(Q_bh + offs_m[:, None] * stride_qm + (3*D_TILE + offs_d0)[None, :] * stride_qk,
                 mask=mask_m[:, None], other=0.0)

    m = tl.full([BM], float('-inf'), dtype=tl.float32)
    s = tl.zeros([BM],              dtype=tl.float32)
    # 4 separate D-tile accumulators to avoid one large [BM, D] tensor
    a0 = tl.zeros([BM, D_TILE], dtype=tl.float32)
    a1 = tl.zeros([BM, D_TILE], dtype=tl.float32)
    a2 = tl.zeros([BM, D_TILE], dtype=tl.float32)
    a3 = tl.zeros([BM, D_TILE], dtype=tl.float32)

    for start_n in range(0, N_CTX, BN):
        offs_n = start_n + tl.arange(0, BN)
        mask_n = offs_n < N_CTX

        # QK = sum over d-tiles of q_d @ k_d.T
        k0 = tl.load(K_bh + offs_n[:, None] * stride_kn + (0*D_TILE + offs_d0)[None, :] * stride_kk,
                     mask=mask_n[:, None], other=0.0)
        k1 = tl.load(K_bh + offs_n[:, None] * stride_kn + (1*D_TILE + offs_d0)[None, :] * stride_kk,
                     mask=mask_n[:, None], other=0.0)
        k2 = tl.load(K_bh + offs_n[:, None] * stride_kn + (2*D_TILE + offs_d0)[None, :] * stride_kk,
                     mask=mask_n[:, None], other=0.0)
        k3 = tl.load(K_bh + offs_n[:, None] * stride_kn + (3*D_TILE + offs_d0)[None, :] * stride_kk,
                     mask=mask_n[:, None], other=0.0)

        qk = (tl.dot(q0, tl.trans(k0), allow_tf32=True) +
              tl.dot(q1, tl.trans(k1), allow_tf32=True) +
              tl.dot(q2, tl.trans(k2), allow_tf32=True) +
              tl.dot(q3, tl.trans(k3), allow_tf32=True))
        qk = SCALE * qk
        qk = tl.where(mask_n[None, :], qk, float('-inf'))

        m_new  = tl.maximum(m, tl.max(qk, axis=1))
        alpha  = tl.exp(m  - m_new)
        p_fp32 = tl.exp(qk - m_new[:, None])
        s      = alpha * s + tl.sum(p_fp32, axis=1)
        p      = p_fp32.to(tl.float16)
        a0 = a0 * alpha[:, None]
        a1 = a1 * alpha[:, None]
        a2 = a2 * alpha[:, None]
        a3 = a3 * alpha[:, None]

        # V tiles
        v0 = tl.load(V_bh + offs_n[:, None] * stride_vn + (0*D_TILE + offs_d0)[None, :] * stride_vk,
                     mask=mask_n[:, None], other=0.0)
        v1 = tl.load(V_bh + offs_n[:, None] * stride_vn + (1*D_TILE + offs_d0)[None, :] * stride_vk,
                     mask=mask_n[:, None], other=0.0)
        v2 = tl.load(V_bh + offs_n[:, None] * stride_vn + (2*D_TILE + offs_d0)[None, :] * stride_vk,
                     mask=mask_n[:, None], other=0.0)
        v3 = tl.load(V_bh + offs_n[:, None] * stride_vn + (3*D_TILE + offs_d0)[None, :] * stride_vk,
                     mask=mask_n[:, None], other=0.0)

        a0 = a0 + tl.dot(p, v0, allow_tf32=True).to(tl.float32)
        a1 = a1 + tl.dot(p, v1, allow_tf32=True).to(tl.float32)
        a2 = a2 + tl.dot(p, v2, allow_tf32=True).to(tl.float32)
        a3 = a3 + tl.dot(p, v3, allow_tf32=True).to(tl.float32)

        m = m_new

    inv_s = 1.0 / s

    tl.store(Out_bh + offs_m[:, None] * stride_om + (0*D_TILE + offs_d0)[None, :] * stride_ok,
             (a0 * inv_s[:, None]).to(tl.float32), mask=mask_m[:, None])
    tl.store(Out_bh + offs_m[:, None] * stride_om + (1*D_TILE + offs_d0)[None, :] * stride_ok,
             (a1 * inv_s[:, None]).to(tl.float32), mask=mask_m[:, None])
    tl.store(Out_bh + offs_m[:, None] * stride_om + (2*D_TILE + offs_d0)[None, :] * stride_ok,
             (a2 * inv_s[:, None]).to(tl.float32), mask=mask_m[:, None])
    tl.store(Out_bh + offs_m[:, None] * stride_om + (3*D_TILE + offs_d0)[None, :] * stride_ok,
             (a3 * inv_s[:, None]).to(tl.float32), mask=mask_m[:, None])


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
        D_TILE = 256
        N_DTILES = D // D_TILE  # 4

        grid = (triton.cdiv(N, BM), B * H)
        _flash_fwd_dtiled[grid](
            Qh, Kh, Vh, Out,
            Qh.stride(0), Qh.stride(1), Qh.stride(2), Qh.stride(3),
            Kh.stride(0), Kh.stride(1), Kh.stride(2), Kh.stride(3),
            Vh.stride(0), Vh.stride(1), Vh.stride(2), Vh.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            H,
            N_CTX=N,
            D=D,
            D_TILE=D_TILE,
            N_DTILES=N_DTILES,
            BM=BM,
            BN=BN,
            SCALE=sm_scale,
            num_warps=4,
            num_stages=1,
        )
        return Out
