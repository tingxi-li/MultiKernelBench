#!/usr/bin/env python3
"""Phase-2 shared definitions: the fused GEMM+bias+GELU+softmax op and SDPA.

Deliberately thin. The measurement machinery -- fixed-time warmup, median of
per-process medians, the gate, the fp64 truth comparison -- is imported from
`phase1_matmul/common.py` rather than reimplemented, so a Phase-2 number and a
Phase-1 number mean the same thing. Phase 1 established that protocol against a
card that thermally soaks and has no steady state; re-deriving it here would only
create a second, subtly different one.

Two problems, taken verbatim from the benchmark references:

  FUSED  reference/fuse/matmul_gelu_softmax.py
         nn.Linear(8192, 8192) on (1024, 8192), then exact GELU, then
         softmax(dim=1) over the 8192-wide rows.
         NOTE the GELU is `F.gelu(x)` with approximate='none' -- the erf form.
         The tanh approximation is a different function and using it is a
         precision shortcut, not an implementation detail.

  SDPA   reference/attention/scaled_dot_product_attention.py
         B=32, H=32, S=512, D=1024, no mask, no dropout, no scale override.
         D=1024 is far past FlashAttention's 256 cap -- see
         `sdpa_reference_audit.py` for what the reference actually runs.
"""
from __future__ import annotations

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE1 = os.path.join(os.path.dirname(HERE), "phase1_matmul")
if PHASE1 not in sys.path:
    sys.path.insert(0, PHASE1)

# the frozen Phase-1 protocol, reused wholesale
from common import (  # noqa: E402,F401
    GATE_ATOL, GATE_RTOL, RAND_RMS, DISTRIBUTIONS,
    time_kernel, gate_stats, setup_cuda_env, write_json,
    SM89_MAX_SMEM_PER_BLOCK,
)

RESULTS_DIR = os.path.join(HERE, "results")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
INPUTS_DIR = os.path.join(HERE, "inputs")
for _d in (RESULTS_DIR, ARTIFACTS_DIR, INPUTS_DIR):
    os.makedirs(_d, exist_ok=True)

DSLS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")

# ------------------------------------------------------------------ FUSED ---
F_M, F_K, F_N = 1024, 8192, 8192           # (batch, in_features, out_features)
F_FLOPS_GEMM = 2.0 * F_M * F_N * F_K        # 137.44 GFLOP, same as Phase 1's GEMM

# The incremental ladder the study specifies: add exactly one feature per step.
#
#   G      the matched Phase-1 GEMM, nothing else            -> transfer check
#   GB     + bias                                            -> bias cost
#   GBG    + fused exact (erf) GELU in the epilogue           -> fusion of a
#                                                               transcendental
#   GBGS   + softmax over dim=1                              -> the full op
#
# `wcache` is an orthogonal two-way factor, not a step on the ladder: it decides
# whether the fp16 copy of the (8192, 8192) weight is built once and reused, or
# rebuilt every forward. The incumbent solution caches. That is legitimate for
# inference -- weights do not change -- but the benchmark times steady-state
# forward calls, so a one-time ~400 MB conversion is amortized to exactly zero.
# Reporting the factor is the only way the reader can tell how much of the
# published number depends on it.
FUSED_ARMS = {
    "G":    dict(bias=False, gelu=False, softmax=False, label="GEMM only"),
    "GB":   dict(bias=True,  gelu=False, softmax=False, label="+ bias"),
    "GBG":  dict(bias=True,  gelu=True,  softmax=False, label="+ fused exact GELU"),
    "GBGS": dict(bias=True,  gelu=True,  softmax=True,  label="+ softmax (full op)"),
}

# The weight factor. The study asks for two levels; a third is carried because
# without it the "uncached" level is a straw man.
#
#   cached     fp16 (K, N) copy of W built once, reused forever. What the
#              incumbent does.
#   uncached   `W.t().contiguous().half()` on every forward. What a literal
#              un-caching of the incumbent costs -- a real transpose kernel plus
#              a real cast kernel, ~400 MB of traffic per call.
#   native     no host-side op at all: the kernel consumes W in its stored
#              (N, K) layout and converts fp32->fp16 on the global->shared path.
#              This is what an implementation that never wanted the cache in the
#              first place would do, and it is the honest floor for "uncached".
#              Costs 2x the global bytes for W, but no materialization.
#
# Reporting only cached-vs-uncached would credit the cache with work that a
# competent uncached kernel simply never does.
WCACHE_MODES = ("cached", "uncached", "native")
WCACHE = ("cached", "uncached")          # the two-way factor the study specifies

# TileLang-only softmax-abstraction arms. They all compute the full op, so their
# reference is arm GBGS; only the reduction's abstraction level differs. Kept in
# a separate namespace so nothing in the cross-DSL ladder can accidentally
# collect them.
FUSED_ABS_ARMS = ("F1", "F2", "F3", "F4", "F2c", "F3c", "F4c")


def reference_arm(variant: str) -> str:
    """Which ladder arm's ground truth a variant should be checked against."""
    return "GBGS" if variant in FUSED_ABS_ARMS else variant


# ---- matched configuration -------------------------------------------------
# Not a new tuning point. This is Phase 1's matched variant D at the primary
# geometry, transplanted verbatim onto the fused shape, which is the whole point
# of the ladder: "use the matched GEMM and add one feature at a time".
#
# Phase 1 primary geometry: BM=128 BN=128 BK=32 threads=256; variant D:
# arith=fp16, kc=2048, stages=3. Shared memory (128*32 + 32*128)*2*3 = 49152 B,
# comfortably inside sm_89's 101376 B.
#
# The fused shape has the same arithmetic volume as Phase 1's GEMM
# (2*1024*8192*8192 == 2*2048*8192*4096 == 137.44 GFLOP), so arm G is directly
# comparable to Phase 1's variant D. That equality is the transfer check.
FUSED_GEOM = dict(BM=128, BN=128, BK=32, threads=256)
FUSED_SPEC = dict(arith="fp16", kc=2048, stages=3, cast="precast")

# Softmax is a second kernel: the row is 8192 wide and one GEMM block owns only
# BN=128 of it, so no amount of epilogue fusion can produce a row-normalized
# result without cross-block communication. 256 threads x 32 elements = one row
# per block, which is the incumbent's shape.
SOFT_THREADS = 256


def make_fused_config(dsl: str, arm: str, **over):
    """A Phase-1 Config carrying the fused shape, the matched schedule, and the
    weight-cache factor in `extra`."""
    from common import Config
    if arm not in FUSED_ARMS and arm not in FUSED_ABS_ARMS:
        raise KeyError(f"unknown fused arm {arm!r}; known: "
                       f"{sorted(FUSED_ARMS) + sorted(FUSED_ABS_ARMS)}")
    cfg = Config(dsl=dsl, variant=arm, M=F_M, N=F_N, K=F_K,
                 **FUSED_GEOM, **FUSED_SPEC)
    cfg.extra.setdefault("wcache", "cached")
    for k, v in over.items():
        if k == "extra":
            cfg.extra.update(v)
        else:
            setattr(cfg, k, v)
    if cfg.extra["wcache"] not in WCACHE_MODES:
        raise ValueError(f"wcache={cfg.extra['wcache']!r} not in {WCACHE_MODES}")
    return cfg


class Built2:
    """What a Phase-2 fused module returns.

    Differs from Phase 1's `Built` only in the arity of `run`: the fused op takes
    (x, W, b) where W is fp32 (N, K) exactly as `nn.Linear` stores it and b is
    fp32 (N,). Whatever a given `wcache` mode needs to do to W is `run`'s job and
    happens inside the timed region unless the mode is `cached`.

    `x_dtype` is the dtype the *activation* must arrive in, handled exactly as
    Phase 1 handled `cast`: with cfg.cast="precast" the fp16 copy of x is made
    outside the timed region (so arm G is Phase-1 variant D transplanted, byte
    for byte), and with cfg.cast="in_region" `run` casts it itself and pays for
    it. x is 32 MB, two orders below the weight, so this is a small control --
    but Phase 1 found the cast worth 0.21 ms on a 1.05 ms kernel, so it is not
    a control that can be skipped.
    """

    __slots__ = ("run", "compile_s", "artifacts", "notes", "n_kernels", "x_dtype")

    def __init__(self, run, compile_s, artifacts=None, notes="", n_kernels=1,
                 x_dtype=torch.float16):
        self.run = run
        self.compile_s = compile_s
        self.artifacts = artifacts or {}
        self.notes = notes
        self.n_kernels = n_kernels
        self.x_dtype = x_dtype


def weight_fn(mode: str):
    """The weight-cache factor, implemented ONCE and shared by every DSL.

    Returns `f(W_fp32_NK) -> operand`. Whether this runs inside the timed region
    is decided by the caller placing it inside `run` -- every lane places it in
    the same spot, so the factor cannot become a per-DSL implementation
    difference.

    `native` is the identity: the kernel takes W in its stored (N, K) fp32 layout
    and does the conversion on the way into shared memory.
    """
    if mode == "cached":
        box = {}

        def f(W):
            # Keyed on the source tensor's address, not merely "have I run
            # before". A cache that ignores which weight it was built from will
            # happily serve a warm-up dummy for the rest of the campaign and
            # produce an all-zero result that still times beautifully.
            #   `!=`, NOT `is not`: data_ptr() returns a large Python int and
            #   CPython does not intern those, so an identity test is always
            #   true and the "cached" arm silently becomes the uncached one --
            #   which is the exact difference this factor exists to measure.
            if box.get("src") != W.data_ptr():
                box["src"] = W.data_ptr()
                box["wt"] = W.t().contiguous().half()
            return box["wt"]
    elif mode == "uncached":
        def f(W):
            return W.t().contiguous().half()
    elif mode == "native":
        def f(W):
            return W
    else:
        raise ValueError(mode)
    return f


def weight_kernel_spec(mode: str) -> dict:
    """What the *kernel* has to look like for a given weight mode.

    cached/uncached both hand the kernel a fp16 (K, N) operand, so they compile
    to the identical kernel and differ only in host-side work -- which is
    exactly what makes them a clean two-way factor. `native` needs a different
    kernel body (transposed B, fp32 global), so it is reported apart from the
    two-way comparison rather than inside it.
    """
    if mode == "native":
        return {"b_dtype": "float32", "b_shape": "NK", "transpose_b": True}
    return {"b_dtype": "float16", "b_shape": "KN", "transpose_b": False}


def fused_reference(x, W, b, arm="GBGS", dtype=torch.float32):
    """Ground truth for an arm, in `dtype`. W is (N, K) as nn.Linear stores it."""
    spec = FUSED_ARMS[arm]
    y = x.to(dtype) @ W.to(dtype).t()
    if spec["bias"]:
        y = y + b.to(dtype)
    if spec["gelu"]:
        # exact erf GELU, matching F.gelu(approximate='none')
        y = y * 0.5 * (1.0 + torch.erf(y / math.sqrt(2.0)))
    if spec["softmax"]:
        y = torch.softmax(y, dim=1)
    return y


def fused_inputs(seed=0, dist="rand", device="cuda"):
    """x, W, b exactly as the reference builds them.

    `nn.Linear` initializes W and b from U(-1/sqrt(K), 1/sqrt(K)); the reference
    passes x ~ U(0,1). Both are reproduced rather than approximated, because
    Phase 1 showed the *input distribution* is what decides whether a reduced
    precision path passes the gate at all."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    bound = 1.0 / math.sqrt(F_K)
    if dist == "rand":
        x = torch.rand(F_M, F_K, generator=g)
    elif dist == "randn":
        x = torch.randn(F_M, F_K, generator=g) * RAND_RMS
    else:
        raise ValueError(dist)
    W = (torch.rand(F_N, F_K, generator=g) * 2 - 1) * bound
    b = (torch.rand(F_N, generator=g) * 2 - 1) * bound
    return x.to(device), W.to(device), b.to(device)


# ------------------------------------------------------------------- SDPA ---
S_B, S_H, S_S = 32, 32, 512
S_HEAD_DIMS = (128, 256, 1024)
S_D_BENCH = 1024                            # what the benchmark actually uses


def sdpa_flops(d):
    """QK^T is (S,d)x(d,S) and PV is (S,S)x(S,d), per (batch, head)."""
    return 2.0 * S_B * S_H * (S_S * S_S * d + S_S * d * S_S)


def score_bytes(d, score_dtype_size=4):
    """DRAM traffic for materializing S = QK^T once, written then read.

    This is the number that separates the two algorithms: the three-kernel
    design pays it and the flash design does not. At D=1024 it is
    32*32*512*512*4 = 8.6 GB written plus the same read again."""
    return 2.0 * S_B * S_H * S_S * S_S * score_dtype_size


# score / probability dtype pairs, fixed independently as specified
SDPA_DTYPES = (
    ("fp32", "fp32"),
    ("fp32", "fp16"),
    ("fp16", "fp16"),
)
SDPA_ALGOS = {
    "K3":    "three kernels: QK^T -> fp32 softmax -> PV (materializes S)",
    "FLASH": "one kernel: tiled online softmax, S never leaves registers/smem",
    "K2":    "two kernels: fused QK^T+softmax, then PV",
}


SDPA_TILES = {
    # (Br, Bc, threads) per head_dim. Br x d is the output fragment a block
    # carries; at d=1024 that is already 64 KB of fp32 accumulator for Br=16,
    # so the tile has to shrink with d or nothing fits.
    128:  dict(Br=64, Bc=64, threads=128),
    256:  dict(Br=64, Bc=64, threads=128),
    1024: dict(Br=32, Bc=64, threads=128),
}


def make_sdpa_config(dsl: str, algo: str, **over):
    """A Phase-1 Config carrying the SDPA problem in `extra`.

    `variant` is the algorithm (K3/K2/FLASH or an S3-* abstraction arm); the
    head dimension and the two dtypes are separate factors and live in `extra`
    so that every cell is identified by (algo, d, score_dtype, prob_dtype).
    """
    from common import Config
    extra = dict(over.pop("extra", {}))
    d = int(extra.get("d", S_D_BENCH))
    sd = extra.get("sdtype", "fp32")
    pd = extra.get("pdtype", "fp32")
    if (sd, pd) not in SDPA_DTYPES:
        raise ValueError(f"score/prob dtype pair {(sd, pd)} not in {SDPA_DTYPES}")
    if pd == "fp32" and sd == "fp16":
        raise ValueError("fp16 scores with fp32 probabilities is not a studied cell")
    tile = dict(SDPA_TILES[d])
    extra.update(d=d, sdtype=sd, pdtype=pd)
    cfg = Config(dsl=dsl, variant=algo, M=S_S, N=S_S, K=d,
                 BM=tile["Br"], BN=tile["Bc"], BK=64, threads=tile["threads"],
                 kc=0, stages=1, arith="fp16", cast="precast")
    cfg.extra.update(extra)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def sdpa_inputs(d, seed=0, dist="rand", device="cuda", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def mk():
        if dist == "rand":
            t = torch.rand(S_B, S_H, S_S, d, generator=g)
        else:
            t = torch.randn(S_B, S_H, S_S, d, generator=g) * RAND_RMS
        return t.to(device=device, dtype=dtype)
    return mk(), mk(), mk()


def sdpa_reference(q, k, v, dtype=torch.float32, chunk=4):
    """Math-form reference in an explicit dtype, so the comparison is against a
    known arithmetic rather than whatever backend torch selects that day.

    Chunked over the batch: at d=1024 the full score tensor is
    32*32*512*512*4 = 8.6 GB and a whole-tensor reference would not coexist
    with the kernel under test on a 48 GB card. Chunking changes no arithmetic
    -- each (batch, head) row is independent -- it only bounds the peak.
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    outs = []
    for i in range(0, q.shape[0], chunk):
        qq = q[i:i + chunk].to(dtype)
        kk = k[i:i + chunk].to(dtype)
        vv = v[i:i + chunk].to(dtype)
        s = (qq @ kk.transpose(-2, -1)) * scale
        p = torch.softmax(s, dim=-1)
        outs.append(p @ vv)
        del qq, kk, vv, s, p
    return torch.cat(outs, dim=0)


TORCH_DEFAULT = "torch.nn.functional.scaled_dot_product_attention (default backend)"
