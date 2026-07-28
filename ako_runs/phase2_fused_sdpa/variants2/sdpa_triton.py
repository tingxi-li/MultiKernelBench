"""Triton lane of the Phase-2 SDPA study.

Two algorithms, identical semantics, differing only in whether the score matrix
is ever written to DRAM:

  K3     three kernels.  (1) S = Q@K^T * scale, materialized to global memory in
         `sdtype`.  (2) P = softmax(S, dim=-1), materialized in `pdtype`.
         (3) O = P@V, fp32 out.  The 8.6 GB of score traffic at S=512 is the
         point of the arm, so nothing here tries to avoid it.
  FLASH  one kernel.  Tiled online softmax over KV blocks with a running max and
         a running sum; S never leaves registers.  Output accumulator is fp32.

WHAT THE TWO DTYPE KNOBS ACTUALLY CONTROL -- read this before comparing lanes.

  Q, K are ALWAYS fed to the QK^T `tl.dot` as fp16, in every cell.  They are read
  from global memory in their stored fp32 layout and converted on the way into
  the dot (`tl.load(...).to(tl.float16)`), so there is no materialized fp16 copy
  of Q/K/V anywhere and no host-side cast kernel: FLASH really is one kernel.
  Holding the QK operand dtype fixed is deliberate -- `sdtype` is defined by the
  study as "the dtype the SCORE tensor is kept in", i.e. a property of the
  *result* of QK^T, so letting it also swing the operand dtype would confound
  the factor with an operand change.

  sdtype  applies to the scaled QK^T result.  fp16 -> the fp32 mma accumulator is
          rounded through fp16 before the softmax sees it (FLASH: an explicit
          `.to(fp16).to(fp32)` round trip in registers; K3: the dtype of the
          materialized S in DRAM, which is the same rounding plus the traffic).
  pdtype  applies to the probabilities as an *operand* of the PV dot, and to V,
          which must match.  fp16 -> `tl.dot` on fp16 operands, i.e.
          mma.sync.*.f32.f16.f16.f32 tensor cores.  fp32 -> see below; Triton
          gives a choice here and the choice is worth stating loudly.

  The mma accumulator is fp32 in every cell, and the softmax itself (max, exp,
  sum, reciprocal) is fp32 in every cell.

WHAT "pdtype=fp32" COMPILES TO, AND WHY IT IS NOT tf32 HERE BY DEFAULT.

  `tl.dot` on fp32 operands has three lowerings on sm_89, selected by
  `input_precision`.  Measured on this card with a 64x64x64 U(0,1) product
  against an fp64 oracle (`_triton_f32dot_probe.py`):

    input_precision   max_abs    signed mean    mean RELATIVE error
    "tf32"            1.52e-2    -1.02e-2       -6.42e-4
    "tf32x3"          1.68e-5    -8.68e-6       -5.46e-7
    "ieee"            9.61e-6    -1.72e-8       -1.22e-9
    (fp16 operands)   2.58e-3    -4.49e-5       -2.94e-6

  Triton's tf32 lowering feeds raw fp32 registers to mma.sync.*.tf32.tf32.* and
  the hardware TRUNCATES the low 13 mantissa bits -- it does not round to
  nearest.  Truncation of two operands is a one-sided error, so the bias is
  -6.4e-4 relative and does NOT average out over the 512-term PV contraction:
  every output element lands ~2.4e-4 low against a 1.5e-4 budget.  Empirically
  that misses the gate at 100% of elements at every head dim (numbers in the
  lane report), and it is *worse than the fp16 cell it is supposed to bound* --
  fp16 has the same 10-bit mantissa but rounds to nearest, so its error is
  1.3e-4 x smaller and unbiased.

  A "fp32 probabilities" cell that is less accurate than the fp16 cell is not a
  precision baseline, it is an artifact of one lowering.  So the default here is
  `input_precision="ieee"`: genuinely fp32 arithmetic, which on sm_89 means
  Triton emits an FMA dot and the PV matmul uses NO tensor cores at all -- which
  is exactly what the study means by "fp32 -> PV cannot use fp16 tensor cores",
  and whose (large) runtime cost is the thing worth measuring.

  The other two lowerings stay reachable through `cfg.extra["f32dot"]`
  ("ieee" | "tf32" | "tf32x3") so the tf32 result can be reproduced rather than
  taken on trust.  Whichever is active is named in `backend_detail`, and a
  tf32 cell is never allowed to read as "fp32" there.  Measured in-kernel at the
  shipped tiles, (fp32, fp32), max_abs / median ms:

    f32dot     FLASH d=128     FLASH d=1024    K3 d=128       K3 d=1024
    ieee       5.7e-6  4.06    6.0e-6  63.2    5.6e-6 15.5    5.7e-6 101.5
    tf32       4.1e-4  2.41    4.2e-4  47.6    4.1e-4  6.57   4.2e-4  20.5
    tf32x3     5.7e-6  3.98    5.8e-6  89.9    5.8e-6  7.08    5.9e-6  34.6

  tf32 is the only one that reaches fp16-cell speed and it is the only one that
  fails the gate, at every cell -- which is the whole point of not letting it
  ride as "fp32".

THE ONE CELL THIS LANE CANNOT PASS, AND WHY IT IS NOT THE KERNEL'S FAULT.

  (fp16, fp16) at d=1024 misses the gate in BOTH algorithms, at 1.90e-4 (FLASH)
  and 1.87e-4 (K3) against a 1.5e-4 budget, on ~1e-6 % of elements.  That is a
  property of the cell, not of this code.  `_triton_fp16score_bound.py` computes
  the implementation-free floor: take the fp64 reference and round ONLY the
  scaled scores to fp16, leaving softmax and PV in fp64, no tiling and no tensor
  cores anywhere.  It gives

    d=128   4.66e-5, 0 elements over budget
    d=256   8.67e-5, 0 elements over budget
    d=1024  1.65e-4, 19 of 5.4e8 elements over budget    <- already failing

  At d=1024 the scores sit near Q.K/sqrt(d) ~ 8.0, where fp16's ulp is 7.8e-3,
  so the rounding alone spends more than the whole budget.  No correct kernel of
  any design can pass that cell, and this lane lands within ~15% of the floor.
  The d=128 / d=256 cells match their floors almost exactly (5.9e-5 vs 4.7e-5,
  8.9e-5 vs 8.7e-5), which is the evidence that the fp16 path is not adding
  error of its own.

No `@triton.autotune`: every tile, num_warps and num_stages is a constant
written down in FLASH_TILES / K3_TILES below and recorded in `backend_detail`.
The tile is a function of (algo, d) only -- it does not change with the dtype
pair, or the dtype factor would be confounded with a schedule change.
"""
from __future__ import annotations

import math
import time

import torch
import triton
import triton.language as tl

import common2

# ---------------------------------------------------------------- schedules --
# (algo, d) -> tile.  Fixed constants, NOT search results.
#
#   BR   query rows a program owns
#   BC   KV rows per online-softmax step
#   DO   width of the output d-slice a program owns.  DO < d splits the output
#        across a third grid dimension and costs a recomputation of QK^T per
#        slice; DO == d means one pass.
#   DR   reduction tile for the QK^T contraction over d.
#
# HOW THESE WERE PICKED, since the choice interacts with the dtype factor and a
# careless pick would fake the factor's whole effect.
#
# The tile has to serve all three dtype pairs unchanged.  The pdtype=fp32 cell
# is the binding one: its PV operands are fp32, so the (BC, DO) V tile is twice
# the bytes of the fp16 cell, and Triton's IEEE fp32 dot materializes that
# operand in REGISTERS rather than handing it to an mma.  Past roughly
# BR*DO = 32768 accumulator elements the fp32 cell falls off a spill cliff --
# measured, at d=256: BR=128/BC=64 runs the fp16 cell at 5.9 ms and the fp32
# cell at 398 ms, while BR=64/BC=32 runs 7.0 ms and 41 ms.  Publishing the first
# would report "fp32 is 67x slower than fp16" when most of that number is the
# register allocator, not the arithmetic.
#
# So the rule, applied mechanically to a hand-written candidate list (see
# `_triton_tile_sweep.py`; it is an offline selection, not an autotuner, and the
# result is these frozen constants): take the tile with the fastest
# (fp32, fp16) cell AMONG those where the fp32/fp32 cell stays within 10x of it.
# Rejected-for-spill candidates are listed in the lane report with their times.
#
#   d=128   BR=128 BC=32  DO=128            3.98 / 2.32 / 2.34 ms  (fp32 1.7x)
#   d=256   BR=64  BC=32  DO=256           40.97 / 6.96 / 6.90 ms  (fp32 5.9x)
#   d=1024  BR=32  BC=16  DO=1024          62.10 / 48.04 / 48.18   (fp32 1.3x)
#
# d=1024 lands on DO == d, i.e. ONE output slice and no QK^T recomputation --
# which needs BC=16 to keep the fp32 V tile inside 101376 B of shared memory.
FLASH_TILES = {
    128:  dict(BR=128, BC=32, DO=128,  DR=128, warps=8, stages=2),
    256:  dict(BR=64,  BC=32, DO=256,  DR=128, warps=8, stages=2),
    1024: dict(BR=32,  BC=16, DO=1024, DR=128, warps=8, stages=2),
}

# K3's three kernels are three ordinary GEMM/reduction shapes:
#   BM x BN / DR      the QK^T gemm
#   BM2 x DN / BK2    the PV gemm
# The score matrix is 512x512 per (batch, head) at every d, so one tile serves
# all three head dims; DN=128 divides 128, 256 and 1024.  Chosen by the same
# rule as FLASH_TILES: at d=1024 this runs 98.9 / 17.5 / 16.9 ms across the
# three dtype pairs, so the fp32 cell is 5.6x the fp16 one -- inside the 10x
# no-spill bound, unlike the FLASH tiles it would otherwise resemble.
_K3 = dict(BM=256, BN=128, DR=64, w1=8, s1=2,
           BM2=128, DN=128, BK2=64, w3=8, s3=2)
K3_TILES = {128: dict(_K3), 256: dict(_K3), 1024: dict(_K3)}
SOFT_WARPS = 4          # the row-softmax kernel: one 512-wide row per program

# `tl.dot` lowerings for fp32 operands, as an int so it can be a constexpr the
# kernels branch on.  See the module docstring for why IEEE is the default.
F32DOT = {"ieee": 0, "tf32": 1, "tf32x3": 2}
F32DOT_DESC = {
    0: ("tl.dot on fp32 operands with input_precision='ieee' -> FMA-based dot, "
        "NO tensor cores (genuinely fp32 arithmetic)"),
    1: ("tl.dot on fp32 operands with input_precision='tf32' -> "
        "mma.sync f32.tf32.tf32.f32 TENSOR CORES: 10-bit-mantissa operands "
        "TRUNCATED from fp32, NOT fp32 arithmetic"),
    2: ("tl.dot on fp32 operands with input_precision='tf32x3' -> 3x "
        "mma.sync f32.tf32.tf32.f32 TENSOR CORES (error-compensated tf32), "
        "NOT a single fp32 multiply"),
}


@triton.jit
def _f32dot(a, b, acc, IP: tl.constexpr):
    """The pdtype=fp32 PV product, with the lowering pinned by `IP`."""
    if IP == 0:
        return tl.dot(a, b, acc, input_precision="ieee")
    elif IP == 1:
        return tl.dot(a, b, acc, input_precision="tf32")
    else:
        return tl.dot(a, b, acc, input_precision="tf32x3")


# -------------------------------------------------------------------- FLASH --
@triton.jit
def _flash_kernel(
    Q, K, V, O,
    stride_bh_q, stride_s_q,
    stride_bh_k, stride_s_k,
    stride_bh_v, stride_s_v,
    stride_bh_o, stride_s_o,
    scale,
    SEQ: tl.constexpr, D: tl.constexpr,
    BR: tl.constexpr, BC: tl.constexpr, DO: tl.constexpr, DR: tl.constexpr,
    NDR: tl.constexpr,
    SD_FP16: tl.constexpr, PD_FP16: tl.constexpr, IP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_d = tl.program_id(2)

    bh = pid_bh.to(tl.int64)
    qb = Q + bh * stride_bh_q
    kb = K + bh * stride_bh_k
    vb = V + bh * stride_bh_v
    ob = O + bh * stride_bh_o

    offs_m = pid_m * BR + tl.arange(0, BR)
    offs_c = tl.arange(0, BC)
    offs_o = pid_d * DO + tl.arange(0, DO)

    if NDR == 1:
        # d fits in one contraction tile: Q stays in registers for the whole
        # KV loop, which is the classic flash-attention schedule.
        q_all = tl.load(qb + offs_m[:, None] * stride_s_q
                        + tl.arange(0, D)[None, :]).to(tl.float16)

    m_i = tl.full([BR], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BR], dtype=tl.float32)
    acc = tl.zeros([BR, DO], dtype=tl.float32)

    for start_n in range(0, SEQ, BC):
        offs_n = start_n + offs_c

        if NDR == 1:
            k_t = tl.load(kb + offs_n[:, None] * stride_s_k
                          + tl.arange(0, D)[None, :]).to(tl.float16)
            qk = tl.dot(q_all, tl.trans(k_t))
        else:
            qk = tl.zeros([BR, BC], dtype=tl.float32)
            for i in range(NDR):
                offs_d = i * DR + tl.arange(0, DR)
                a = tl.load(qb + offs_m[:, None] * stride_s_q
                            + offs_d[None, :]).to(tl.float16)
                b = tl.load(kb + offs_n[:, None] * stride_s_k
                            + offs_d[None, :]).to(tl.float16)
                qk = tl.dot(a, tl.trans(b), qk)

        qk = qk * scale
        if SD_FP16:
            # the score factor: the fp32 mma accumulator is rounded through
            # fp16 before the softmax ever sees it.
            qk = qk.to(tl.float16).to(tl.float32)

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(vb + offs_n[:, None] * stride_s_v + offs_o[None, :])
        if PD_FP16:
            acc = tl.dot(p.to(tl.float16), v.to(tl.float16), acc)
        else:
            acc = _f32dot(p, v, acc, IP)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(ob + offs_m[:, None] * stride_s_o + offs_o[None, :], acc)


# ----------------------------------------------------------------------- K3 --
@triton.jit
def _k3_qk_kernel(
    Q, K, S,
    stride_bh_q, stride_s_q,
    stride_bh_k, stride_s_k,
    stride_bh_s, stride_r_s,
    scale,
    SEQ: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, DR: tl.constexpr, NDR: tl.constexpr,
    SD_FP16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bh = tl.program_id(2).to(tl.int64)

    qb = Q + bh * stride_bh_q
    kb = K + bh * stride_bh_k
    sb = S + bh * stride_bh_s

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for i in range(NDR):
        offs_d = i * DR + tl.arange(0, DR)
        a = tl.load(qb + offs_m[:, None] * stride_s_q
                    + offs_d[None, :]).to(tl.float16)
        b = tl.load(kb + offs_n[:, None] * stride_s_k
                    + offs_d[None, :]).to(tl.float16)
        acc = tl.dot(a, tl.trans(b), acc)
    acc = acc * scale

    sp = sb + offs_m[:, None] * stride_r_s + offs_n[None, :]
    if SD_FP16:
        tl.store(sp, acc.to(tl.float16))
    else:
        tl.store(sp, acc)


@triton.jit
def _k3_softmax_kernel(S, P, SEQ: tl.constexpr, PD_FP16: tl.constexpr):
    """One 512-wide row per program.  BN == SEQ, so the row is a single tile and
    the exponentials stay in registers between the sum and the divide."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, SEQ)
    x = tl.load(S + row * SEQ + offs).to(tl.float32)
    e = tl.exp(x - tl.max(x, axis=0))
    p = e / tl.sum(e, axis=0)
    if PD_FP16:
        tl.store(P + row * SEQ + offs, p.to(tl.float16))
    else:
        tl.store(P + row * SEQ + offs, p)


@triton.jit
def _k3_pv_kernel(
    P, V, O,
    stride_bh_p, stride_r_p,
    stride_bh_v, stride_s_v,
    stride_bh_o, stride_s_o,
    SEQ: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, DN: tl.constexpr, BK: tl.constexpr,
    PD_FP16: tl.constexpr, IP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bh = tl.program_id(2).to(tl.int64)

    pb = P + bh * stride_bh_p
    vb = V + bh * stride_bh_v
    ob = O + bh * stride_bh_o

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * DN + tl.arange(0, DN)
    offs_k = tl.arange(0, BK)

    acc = tl.zeros([BM, DN], dtype=tl.float32)
    for j in range(0, SEQ, BK):
        p = tl.load(pb + offs_m[:, None] * stride_r_p + (j + offs_k)[None, :])
        v = tl.load(vb + (j + offs_k)[:, None] * stride_s_v + offs_n[None, :])
        if PD_FP16:
            acc = tl.dot(p, v.to(tl.float16), acc)
        else:
            acc = _f32dot(p, v, acc, IP)
    tl.store(ob + offs_m[:, None] * stride_s_o + offs_n[None, :], acc)


# -------------------------------------------------------------------- build --
def _dtype_of(name):
    return torch.float16 if name == "fp16" else torch.float32


def _ip_of(cfg):
    """Which fp32 `tl.dot` lowering the pdtype=fp32 cells use.  Only meaningful
    when pdtype == 'fp32'; ignored otherwise."""
    name = str(cfg.extra.get("f32dot", "ieee"))
    if name not in F32DOT:
        raise ValueError(f"f32dot={name!r} not in {sorted(F32DOT)}")
    return F32DOT[name], name


def _build_flash(cfg):
    B, H, SEQ = common2.S_B, common2.S_H, common2.S_S
    d = int(cfg.extra["d"])
    sd = cfg.extra["sdtype"]
    pd = cfg.extra["pdtype"]
    t = FLASH_TILES[d]
    BR, BC, DO, DR = t["BR"], t["BC"], t["DO"], t["DR"]
    assert SEQ % BR == 0 and SEQ % BC == 0 and d % DO == 0 and d % DR == 0
    # NDR == 1 is the "Q stays in registers" schedule; legal only when a single
    # contraction tile covers d.
    NDR = 1 if d == DR else d // DR
    NDS = d // DO
    scale = 1.0 / math.sqrt(d)
    sd16, pd16 = sd == "fp16", pd == "fp16"
    ip, ipname = _ip_of(cfg)
    grid = (SEQ // BR, B * H, NDS)

    def launch(q, k, v, o, nbh):
        _flash_kernel[(SEQ // BR, nbh, NDS)](
            q, k, v, o,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            o.stride(0), o.stride(1),
            scale,
            SEQ=SEQ, D=d, BR=BR, BC=BC, DO=DO, DR=DR, NDR=NDR,
            SD_FP16=sd16, PD_FP16=pd16, IP=ip,
            num_warps=t["warps"], num_stages=t["stages"])

    def run(q, k, v):
        # (B, H, S, d) -> (B*H, S, d) view; no copy, the tensors are contiguous.
        nbh = B * H
        o = torch.empty_like(q)
        launch(q.view(nbh, SEQ, d), k.view(nbh, SEQ, d), v.view(nbh, SEQ, d),
               o.view(nbh, SEQ, d), nbh)
        return o

    detail = (
        f"one kernel: online-softmax flash, BR={BR} BC={BC} DO={DO} DR={DR} "
        f"({NDS} output d-slice{'s' if NDS > 1 else ''}, "
        f"{'Q hoisted to registers' if NDR == 1 else f'{NDR} QK contraction tiles'}), "
        f"num_warps={t['warps']} num_stages={t['stages']}, NO autotune; "
        f"Q/K read fp32 from global, cast in-kernel to fp16 -> tl.dot "
        f"(mma.sync f32.f16.f16.f32); score kept "
        + ("fp32 (fp32 mma accumulator, no rounding)" if not sd16
           else "fp16 (accumulator rounded .to(fp16).to(fp32) in registers)")
        + "; softmax max/exp/sum in fp32; PV "
        + ("tl.dot on fp16 operands (mma.sync f32.f16.f16.f32)" if pd16
           else F32DOT_DESC[ip])
        + "; fp32 output accumulator")
    return launch, run, grid, t["warps"], detail, 1, (ipname if not pd16 else "n/a")


def _build_k3(cfg):
    B, H, SEQ = common2.S_B, common2.S_H, common2.S_S
    d = int(cfg.extra["d"])
    sd = cfg.extra["sdtype"]
    pd = cfg.extra["pdtype"]
    t = K3_TILES[d]
    BM, BN, DR = t["BM"], t["BN"], t["DR"]
    BM2, DN, BK2 = t["BM2"], t["DN"], t["BK2"]
    assert SEQ % BM == 0 and SEQ % BN == 0 and d % DR == 0
    assert SEQ % BM2 == 0 and d % DN == 0 and SEQ % BK2 == 0
    NDR = d // DR
    scale = 1.0 / math.sqrt(d)
    sd16, pd16 = sd == "fp16", pd == "fp16"
    ip, ipname = _ip_of(cfg)
    s_dtype = _dtype_of(sd)
    p_dtype = _dtype_of(pd)
    grid1 = (SEQ // BM, SEQ // BN, B * H)
    grid3 = (SEQ // BM2, d // DN, B * H)

    def launch(q, k, v, S, P, o, nbh):
        _k3_qk_kernel[(SEQ // BM, SEQ // BN, nbh)](
            q, k, S,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            S.stride(0), S.stride(1),
            scale,
            SEQ=SEQ, D=d, BM=BM, BN=BN, DR=DR, NDR=NDR, SD_FP16=sd16,
            num_warps=t["w1"], num_stages=t["s1"])
        _k3_softmax_kernel[(nbh * SEQ,)](
            S, P, SEQ=SEQ, PD_FP16=pd16,
            num_warps=SOFT_WARPS, num_stages=1)
        _k3_pv_kernel[(SEQ // BM2, d // DN, nbh)](
            P, v, o,
            P.stride(0), P.stride(1),
            v.stride(0), v.stride(1),
            o.stride(0), o.stride(1),
            SEQ=SEQ, D=d, BM=BM2, DN=DN, BK=BK2, PD_FP16=pd16, IP=ip,
            num_warps=t["w3"], num_stages=t["s3"])

    def run(q, k, v):
        nbh = B * H
        # S IS materialized -- that is the whole point of the arm.
        S = torch.empty((nbh, SEQ, SEQ), dtype=s_dtype, device=q.device)
        P = torch.empty((nbh, SEQ, SEQ), dtype=p_dtype, device=q.device)
        o = torch.empty_like(q)
        launch(q.view(nbh, SEQ, d), k.view(nbh, SEQ, d), v.view(nbh, SEQ, d),
               S, P, o.view(nbh, SEQ, d), nbh)
        return o

    detail = (
        f"three kernels: (1) QK^T gemm {BM}x{BN}, {NDR} contraction tile(s) of "
        f"{DR}, num_warps={t['w1']} num_stages={t['s1']} -> S MATERIALIZED to "
        f"global in {sd} ({SEQ}x{SEQ} per bh, "
        f"{B*H*SEQ*SEQ*(2 if sd16 else 4)/1e9:.2f} GB); "
        f"(2) row softmax, one {SEQ}-wide row per program, BN=SEQ so exp is "
        f"computed once in registers, fp32 max/exp/sum, num_warps={SOFT_WARPS} "
        f"-> P materialized in {pd}; "
        f"(3) PV gemm {BM2}x{DN} k={BK2}, num_warps={t['w3']} "
        f"num_stages={t['s3']}, fp32 out.  NO autotune.  "
        f"Q/K read fp32 from global, cast in-kernel to fp16 -> tl.dot "
        f"(mma.sync f32.f16.f16.f32); PV "
        + ("tl.dot on fp16 operands (mma.sync f32.f16.f16.f32)" if pd16
           else F32DOT_DESC[ip]))
    return launch, run, grid1, t["w1"], detail, 3, (ipname if not pd16 else "n/a")


def build(cfg) -> common2.Built2:
    algo = cfg.variant
    if algo not in ("K3", "FLASH"):
        raise KeyError(f"triton sdpa lane implements K3 and FLASH, not {algo!r}")
    d = int(cfg.extra["d"])
    sd, pd = cfg.extra["sdtype"], cfg.extra["pdtype"]
    B, H, SEQ = common2.S_B, common2.S_H, common2.S_S

    t0 = time.perf_counter()
    if algo == "FLASH":
        launch, run, grid, warps, detail, nk, ipname = _build_flash(cfg)
        tile = FLASH_TILES[d]
    else:
        launch, run, grid, warps, detail, nk, ipname = _build_k3(cfg)
        tile = K3_TILES[d]

    # Warm the KERNELS directly -- never through `run` -- so no host-side state
    # can be primed with a dummy.  One (batch, head) worth of the real per-head
    # geometry: the strides and every constexpr are identical to the timed
    # launch, so this compiles and first-launches exactly the module that will
    # be timed, inside the compile_s window.
    qw = torch.zeros((1, SEQ, d), dtype=torch.float32, device="cuda")
    kw = torch.zeros_like(qw)
    vw = torch.zeros_like(qw)
    ow = torch.empty_like(qw)
    if algo == "FLASH":
        launch(qw, kw, vw, ow, 1)
    else:
        Sw = torch.empty((1, SEQ, SEQ), dtype=_dtype_of(sd), device="cuda")
        Pw = torch.empty((1, SEQ, SEQ), dtype=_dtype_of(pd), device="cuda")
        launch(qw, kw, vw, Sw, Pw, ow, 1)
        del Sw, Pw
    torch.cuda.synchronize()
    del qw, kw, vw, ow
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    artifacts = {
        "grid": list(grid), "block": [warps * 32, 1, 1],
        "triton_version": triton.__version__,
        "algo": algo, "score_dtype": sd, "prob_dtype": pd,
        "n_kernels": nk,
        "tile": dict(tile), "f32dot": ipname,
        "backend_detail": detail,
    }
    notes = (f"triton {triton.__version__} sdpa {algo} d={d} "
             f"sdtype={sd} pdtype={pd} f32dot={ipname} tile={tile}")
    return common2.Built2(run=run, compile_s=compile_s, artifacts=artifacts,
                          notes=notes, n_kernels=nk, x_dtype=torch.float32)
