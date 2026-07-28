"""TileLang lane of the Phase-2 SDPA study.

Three algorithms, identical semantics, at three head dims and three (score dtype,
probability dtype) pairs:

  K3     three kernels.  (1) S = Q@K^T * 1/sqrt(d), MATERIALIZED to global in
         `sdtype`.  (2) P = softmax(S, dim=-1), materialized in `pdtype`.
         (3) O = P@V into fp32.  The round trip through DRAM is the point of the
         arm -- at d=1024 it is 2 x 32*32*512*512 elements written and read
         twice, which `common2.score_bytes` prices.
  K2     two kernels.  (1) QK^T and the softmax FUSED: the block owns a whole
         (BM, S) row block, keeps the scaled score in S//BN fp32 mma
         accumulators, and writes only P to global in `pdtype`.  (2) the SAME
         PV kernel K3 uses, byte for byte -- `_src_pv` is called with the same
         tile from the same table, so K2-vs-K3 differs in exactly one thing:
         whether S is written to and read back from global memory.
  FLASH  one kernel.  Tiled online softmax over KV blocks with a running max and
         a running sum; S never leaves registers/shared memory.  The output
         accumulator is fp32 (Br, D_TILE), one per output d-tile.

OPERANDS ARE NOT CAST ON THE HOST.  `run` is kernel launches plus the output
allocation and nothing else: Q, K and V are handed to the kernel in the fp32 the
benchmark stores them in, `.view`ed (not `.reshape`d, so a hidden copy would
raise rather than be paid for silently) to (B*H, S, d), and the fp16 operands
`T.gemm` needs are produced by `T.copy` on the global->shared path.  An earlier
version of this file did `q.reshape(BH, S, d).half()` per call, which cost
between 17% and 50% of the reported time depending on the cell and was unique to
this lane -- see the comment in `build`.  `sdtype`/`pdtype` therefore mean what
SPEC2 says they mean and never move the operand dtype: Q and K enter the QK gemm
as fp16 in every cell.

WHAT THE TWO DTYPE AXES ACTUALLY CONTROL -- read this before comparing numbers.

  Q and K enter `T.gemm` as fp16 in EVERY cell, accumulating into an fp32
  fragment.  That is not the axis under test: `sdtype` is defined as "the dtype
  the score tensor is kept in", and with an fp32 mma accumulator the score IS
  kept in fp32 unless something rounds it.  So `sdtype=fp16` is implemented as
  an explicit round of the scaled QK^T result through an fp16 fragment (FLASH)
  or through an fp16 global tensor (K3), and `sdtype=fp32` simply omits that
  round.  Nothing else moves.  `common2.score_bytes(d, score_dtype_size)`
  confirms the intent: the factor is priced in bytes of stored score.

  `pdtype` IS an operand dtype, because P is an operand of the PV matmul.
  pdtype=fp16 -> P and V are fp16 and PV runs on fp16 tensor cores.
  pdtype=fp32 -> P and V are fp32 and the PV matmul is an explicit fp32
                 multiply-add over shared memory, on the CUDA cores.  No
                 tensor core is involved.  That is the point of the arm.

  WHY NOT `T.gemm` FOR THE fp32 ARM -- this is a measured result, not taste.
  TileLang will happily accept fp32 operands, and on sm_89 it lowers them to
      tl::mma_sync<kTensorFloat32, kTensorFloat32, kFloat32, 16, 8, 8>
  i.e. TF32 tensor cores.  Read as "fp32 gets tensor cores for free" that
  would be wrong twice over.  TF32 keeps 10 operand mantissa bits, and
  TileLang converts by REINTERPRETING the fp32 bit pattern -- truncation, not
  round-to-nearest.  Truncation is biased, and P and V are both non-negative
  here, so every product is biased the same way and the bias does not cancel:

      probe, (128x64)@(64x128) of torch.rand operands, vs an fp64 reference
        TileLang T.gemm(fp32,fp32)         relative mean error  -6.508e-4
        emulated bit-truncated tf32        relative mean error  -6.503e-4
        emulated round-to-nearest tf32     relative mean error  +2.776e-6

  End to end that is a uniform ~-6.5e-4 relative bias on the output, i.e.
  ~3.3e-4 absolute on values near 0.5, against a gate budget of 1.5e-4.  The
  whole tensor fails.  Measured, with T.gemm wired into the fp32 arm:

      FLASH d=128   max_abs_err 4.120e-4, 100% of elements outside the gate
      FLASH d=256   max_abs_err 4.154e-4, 100% of elements outside the gate
      FLASH d=1024  max_abs_err 4.216e-4, 100% of elements outside the gate

  So the fp32-probability arm uses the fp32 FMA path.  It is what the label
  says it is, it passes, and it makes the arm measure the thing the arm exists
  to measure: what PV costs when it cannot have fp16 tensor cores.

sdtype=fp16 AT d=1024 CANNOT PASS THE GATE, AND NOT BECAUSE OF THIS CODE.

  The gate is 1e-4 + 1e-4*|ref|, i.e. ~1.5e-4 on outputs near 0.5.  Take the
  reference pipeline in exact fp32 and change exactly one thing -- round the
  scaled QK^T result to fp16 before the softmax sees it, which is what
  sdtype=fp16 is defined to mean -- and measure against the unrounded fp32
  answer.  No kernel involved (common2.sdpa_inputs, seed 0, dist=rand):

      d= 128  |s|max  4.31  fp16 ulp 0.0039  max_abs_err 4.63e-5  0 outside
      d= 256  |s|max  5.36  fp16 ulp 0.0039  max_abs_err 8.68e-5  0 outside
      d=1024  |s|max  9.30  fp16 ulp 0.0078  max_abs_err 1.66e-4  3.2e-8 outside

  d=1024 is over budget before any kernel exists.  The mechanism is a binade
  crossing, not a gradual drift: Q,K ~ U(0,1) put the mean score at d/4/sqrt(d)
  = sqrt(d)/4, so the peak score moves 4.3 -> 5.4 -> 9.3 and at d=1024 crosses
  8.  fp16 ulp doubles across that boundary while the softmax width (512
  terms) does not change, so the rounding noise that survives normalisation
  doubles too.  d=256 sits just under the cliff at 5.4 and passes with 42% of
  the budget to spare; d=1024 sits just over it.

  This lane measures 1.908e-4 (FLASH) and 1.872e-4 (K3) against that 1.656e-4
  floor -- about 15% above it, the excess being the fp16 Q/K operands, which
  every cell pays.  So the implementation is close to the best achievable and
  the cell is still unreachable.  It is reported as a failure, not tuned at.
  The one trick that would rescue it -- subtract the row max before rounding --
  is not available: it is part of the softmax, which by definition runs after
  the score is rounded, and in K3 the score has to survive a round trip through
  global memory in fp16 with no per-row state to carry.

WHY THE KERNEL SOURCE IS GENERATED AS TEXT.

  The PV matmul needs one fp32 accumulator fragment per output d-tile (8 of
  them at d=1024, where a single (Br, 1024) fragment cannot be the C operand of
  a gemm whose B operand is a (Bc, D_TILE) shared tile).  Selecting among them
  requires a PYTHON-level loop, and TileLang gives no way to write one:

    * a `for` statement inside the prim_func body is parsed by TVMScript into a
      TIR loop, so its induction variable is a tir.Var and cannot index a Python
      list of buffers;
    * factoring the chain into a plain helper function and calling it from the
      body does not work either -- it compiles, but the `T.gemm` calls made from
      the helper are silently DROPPED (measured: the accumulator comes back
      exactly zero, max_abs_err 35 against the reference on a probe matmul).
      TVMScript parses the body's AST; it does not follow calls.

  The incumbent d=1024 solution deals with this by hand-writing all eight
  accumulators.  That does not generalise to n_d in {1, 2, 8} x two algorithms,
  so here the prim_func is emitted as source text and parsed through
  `linecache` (which is what `inspect.getsource` reads).  The unrolling factor
  is then just a Python int.  Every schedule number in the emitted text comes
  from the tile table below -- nothing is searched.

TILES are fixed per (algo, d) and are IDENTICAL across the three dtype pairs,
so that a dtype effect can never be a tile change in disguise.  See `_TILES`.
"""
#   No `from __future__ import annotations`: TileLang resolves the T.Tensor
#   annotations against module globals, so stringised annotations break it.
import linecache
import math
import os
import time

import torch

import tilelang
import tilelang.language as T

import common2

_CACHE_ENABLED = os.environ.get("PHASE2_TL_CACHE", "0") == "1"
if not _CACHE_ENABLED:
    tilelang.disable_cache()

_LOG2E = 1.4426950408889634

_TORCH_DT = {"fp16": torch.float16, "fp32": torch.float32}
_TL_DT = {"fp16": "float16", "fp32": "float32"}


# ---------------------------------------------------------------- tile table ---
# (algo, d) -> schedule.  Constant across the three dtype pairs, by construction:
# nothing in the generated source reads a dtype to pick a tile.
#
# FLASH   Br x Bc score tile, D_TILE-wide slices of the head dim for both the QK
#         accumulation and the PV output accumulators.  n_d = d // D_TILE output
#         accumulators of (Br, D_TILE) fp32 live in registers.
#           d=1024: 8 x (64,128) fp32 = 64 K floats over 256 threads.  That is
#           the same register pressure as the incumbent 1.12x d=1024 kernel
#           (Br=64, Bc=128, D_TILE=128, 256 threads), which is why it is known
#           to fit.  Bc is 64 rather than 128 because the fp32-probability cell
#           needs an fp32 (Bc, D_TILE) V tile and an fp32 (Br, Bc+1) score
#           bridge, which is the widest of the three cells:
#             16384 Q + 16384 K + 16640 S + 32768 V + 512 scalars = 82688 B
#           against the 101376 B limit.  At Bc=128 the same cell wants 144 KB
#           and does not fit -- and the tile has to be identical in all three
#           cells or the dtype factor is confounded, so all three run Bc=64.
# K3      three independent gemm/reduction tiles, listed separately.
# K2      the SAME qk and pv tiles K3 carries at that head dim, so that a K2-vs-K3
#         comparison isolates exactly one thing -- whether S is written to global
#         memory -- and cannot be a tile change in disguise.  The fused
#         QK^T+softmax kernel's block owns a whole (BM, S) row block, held as
#         S//BN = 8 fp32 mma accumulators of (BM, BN); at BM=BN=64 over 128
#         threads that is 256 floats per thread, the same register pressure the
#         FLASH d=1024 kernel already carries (8 x (64,128) fp32 over 256
#         threads).  Shared memory is only the Qs/Ks pair, 2 x 8 KB x 2 stages.
_TILES = {
    ("FLASH", 128):  dict(Br=64, Bc=64, DT=128, threads=128),
    ("FLASH", 256):  dict(Br=64, Bc=64, DT=128, threads=128),
    ("FLASH", 1024): dict(Br=64, Bc=64, DT=128, threads=256),
    # qk: (BM,BN,BK,threads,stages)  sm: (rows,threads)  pv: (BM,BN,BK,threads,stages)
    ("K3", 128):  dict(qk=(64, 64, 64, 128, 2), sm=(8, 128), pv=(64, 64, 64, 128, 2)),
    ("K3", 256):  dict(qk=(64, 64, 64, 128, 2), sm=(8, 128), pv=(64, 64, 64, 128, 2)),
    ("K3", 1024): dict(qk=(64, 64, 64, 128, 2), sm=(8, 128), pv=(64, 64, 64, 128, 2)),
    # qk: (BM,BN,BK,threads,stages) -- the fused QK^T+softmax kernel
    ("K2", 128):  dict(qk=(64, 64, 64, 128, 2), pv=(64, 64, 64, 128, 2)),
    ("K2", 256):  dict(qk=(64, 64, 64, 128, 2), pv=(64, 64, 64, 128, 2)),
    ("K2", 1024): dict(qk=(64, 64, 64, 128, 2), pv=(64, 64, 64, 128, 2)),
}


def _compile(src: str, tag: str, out_idx):
    """Parse a generated prim_func and compile it.

    `inspect.getsource` -- which TVMScript calls on the decorated function --
    reads `linecache`, so registering the text there is enough to make an
    exec'd prim_func parseable.  The alternative (a hand-written body per
    unrolling factor) is what this exists to avoid.
    """
    fname = f"<tilelang-sdpa:{tag}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    g = {"T": T, "tilelang": tilelang}
    exec(compile(src, fname, "exec"), g)  # noqa: S102
    return tilelang.compile(g["main"], out_idx=out_idx, target="cuda"), src


# ---------------------------------------------------------------------- FLASH ---
def _src_flash(BH, S, D, Br, Bc, DT, threads, sdtype, pdtype):
    n_d = D // DT
    n_kv = S // Bc
    PD = _TL_DT[pdtype]
    scale = 1.0 / math.sqrt(D)
    L = []
    a = L.append
    a("@T.prim_func")
    # Q/K/V arrive in the fp32 the benchmark stores them in.  The fp16 operands
    # the QK gemm needs are produced by `T.copy` on the global->shared path --
    # the conversion rides the load and never becomes a host-side tensor.  See
    # `build()` for why materializing them instead was worth up to 50% of the
    # measured time.
    a(f"def main(Q: T.Tensor(({BH}, {S}, {D}), 'float32'),")
    a(f"         Kt: T.Tensor(({BH}, {S}, {D}), 'float32'),")
    a(f"         Vt: T.Tensor(({BH}, {S}, {D}), 'float32'),")
    a(f"         O: T.Tensor(({BH}, {S}, {D}), 'float32')):")
    a(f"    with T.Kernel({S // Br}, {BH}, threads={threads}) as (bx, by):")
    a(f"        Qs = T.alloc_shared(({Br}, {DT}), 'float16')")
    a(f"        Ks = T.alloc_shared(({Bc}, {DT}), 'float16')")
    if pdtype == "fp16":
        a(f"        Ss = T.alloc_shared(({Br}, {Bc}), '{PD}')")
    else:
        # +1 float of row padding.  Without it the fp32 FMA reads Ss[i, kk]
        # with a 64-float row stride, so bank = (64i + kk) % 32 = kk % 32 is
        # the SAME bank for every thread in the warp with 8 distinct addresses
        # -- an 8-way conflict on the hottest load in the kernel.  Stride 65
        # makes it (i + kk) % 32 and spreads across all 32 banks.
        a(f"        Ss = T.alloc_shared(({Br}, {Bc + 1}), 'float32')")
    a(f"        Vs = T.alloc_shared(({Bc}, {DT}), '{PD}')")
    a(f"        acc_s = T.alloc_fragment(({Br}, {Bc}), 'float32')")
    if pdtype == "fp16":
        # gemm-A operand fragment; the fp32 arm reads P straight out of shared
        a(f"        acc_p = T.alloc_fragment(({Br}, {Bc}), '{PD}')")
    if sdtype == "fp16":
        # the whole of the sdtype axis: a real round trip through fp16 storage
        a(f"        s16 = T.alloc_fragment(({Br}, {Bc}), 'float16')")
    for t in range(n_d):
        a(f"        acc_o{t} = T.alloc_fragment(({Br}, {DT}), 'float32')")
    for nm in ("m", "mp", "sc", "ss", "ls"):
        a(f"        {nm} = T.alloc_fragment(({Br},), 'float32')")
    if pdtype == "fp32":
        # Break the layout chain between the softmax state and the output
        # accumulator by routing the two row-broadcast scalars through shared
        # memory.
        #
        # `sc` and `ls` are (Br,) fragments whose row->thread mapping is fixed
        # by T.reduce_max over the QK mma C fragment; in that layout the four
        # n-dimension threads (tid & 3) all hold the same row, so `sc` is 4x
        # REPLICATED.  Written as `acc_o[i, j] *= sc[i]`, acc_o inherits the
        # replication: 256 floats per thread for an 8192-element tile over 128
        # threads, and every FMA and every epilogue store done four times over.
        # Measured cost of that shape: 239 ms at d=128 against 23 ms without.
        # The fp16 arm never sees it because its PV `T.gemm` anchors acc_o to
        # the mma C layout before the broadcast is inferred.
        #
        # Pinning acc_o with T.annotate_layout does not work -- it pushes the
        # conflict one step back and TileLang reports "Layout infer conflict
        # between mp and sc in T.Parallel loop", because `sc` is computed from
        # `mp` and so cannot move on its own.  A shared-memory hop has no
        # layout constraint at all, which is the point.
        a(f"        scs = T.alloc_shared(({Br},), 'float32')")
        a(f"        lss = T.alloc_shared(({Br},), 'float32')")
    for t in range(n_d):
        a(f"        T.fill(acc_o{t}, 0)")
    a("        T.fill(ls, 0)")
    a("        T.fill(m, -T.infinity('float32'))")
    a(f"        for kb in T.serial({n_kv}):")
    a("            T.clear(acc_s)")
    a(f"            for dq in T.serial({n_d}):")
    a(f"                T.copy(Q[by, bx * {Br}:(bx + 1) * {Br}, dq * {DT}:(dq + 1) * {DT}], Qs)")
    a(f"                T.copy(Kt[by, kb * {Bc}:(kb + 1) * {Bc}, dq * {DT}:(dq + 1) * {DT}], Ks)")
    a("                T.sync_threads()")
    a("                T.gemm(Qs, Ks, acc_s, transpose_B=True,")
    a("                       policy=T.GemmWarpPolicy.FullRow)")
    a(f"            for i, j in T.Parallel({Br}, {Bc}):")
    a(f"                acc_s[i, j] = acc_s[i, j] * T.float32({scale!r})")
    if sdtype == "fp16":
        a(f"            for i, j in T.Parallel({Br}, {Bc}):")
        a("                s16[i, j] = acc_s[i, j]")
        a(f"            for i, j in T.Parallel({Br}, {Bc}):")
        a("                acc_s[i, j] = s16[i, j]")
    a("            T.copy(m, mp)")
    a("            T.fill(m, -T.infinity('float32'))")
    a("            T.reduce_max(acc_s, m, dim=1, clear=False)")
    a(f"            for i in T.Parallel({Br}):")
    a("                m[i] = T.max(m[i], mp[i])")
    a(f"            for i in T.Parallel({Br}):")
    a(f"                sc[i] = T.exp2((mp[i] - m[i]) * T.float32({_LOG2E!r}))")
    a(f"            for i, j in T.Parallel({Br}, {Bc}):")
    a(f"                acc_s[i, j] = T.exp2((acc_s[i, j] - m[i]) * T.float32({_LOG2E!r}))")
    a("            T.reduce_sum(acc_s, ss, dim=1)")
    a(f"            for i in T.Parallel({Br}):")
    a("                ls[i] = ls[i] * sc[i] + ss[i]")
    # Layout bridge.  For pdtype=fp16 the reduce layout of acc_s is not the
    # gemm-A layout, so P goes fp32 fragment -> pdtype shared -> pdtype
    # fragment.  For pdtype=fp32 there is no gemm, so the shared copy IS the
    # operand and the second hop is dropped.
    if pdtype == "fp16":
        a("            T.copy(acc_s, Ss)")
        a("            T.sync_threads()")
        a("            T.copy(Ss, acc_p)")
    else:
        # explicit store: Ss is padded, so T.copy's whole-buffer shape match
        # does not apply
        a(f"            for i, j in T.Parallel({Br}, {Bc}):")
        a("                Ss[i, j] = acc_s[i, j]")
        a(f"            for i in T.Parallel({Br}):")
        a("                scs[i] = sc[i]")
        a("            T.sync_threads()")
    bcast = "sc[i]" if pdtype == "fp16" else "scs[i]"
    a(f"            for i, j in T.Parallel({Br}, {DT}):")
    for t in range(n_d):
        a(f"                acc_o{t}[i, j] = acc_o{t}[i, j] * {bcast}")
    for t in range(n_d):
        a(f"            T.copy(Vt[by, kb * {Bc}:(kb + 1) * {Bc}, "
          f"{t * DT}:{(t + 1) * DT}], Vs)")
        a("            T.sync_threads()")
        if pdtype == "fp16":
            a(f"            T.gemm(acc_p, Vs, acc_o{t}, policy=T.GemmWarpPolicy.FullRow)")
        else:
            # True fp32 FMA on the CUDA cores.  kk is the OUTER loop and the
            # T.Parallel body is a plain elementwise update: with kk inside the
            # T.Parallel, TileLang's layout inference read the nest as a
            # reduction and gave acc_o a 4x-REPLICATED fragment layout (256
            # floats per thread for a 8192-element tile over 128 threads), so
            # every FMA and every epilogue store was done four times.  Measured
            # cost of that shape: 239 ms at d=128.
            a(f"            for kk in T.serial({Bc}):")
            a(f"                for i, j in T.Parallel({Br}, {DT}):")
            a(f"                    acc_o{t}[i, j] = acc_o{t}[i, j] + Ss[i, kk] * Vs[kk, j]")
        a("            T.sync_threads()")
    if pdtype == "fp32":
        a(f"        for i in T.Parallel({Br}):")
        a("            lss[i] = ls[i]")
        a("        T.sync_threads()")
    norm = "ls[i]" if pdtype == "fp16" else "lss[i]"
    a(f"        for i, j in T.Parallel({Br}, {DT}):")
    for t in range(n_d):
        a(f"            acc_o{t}[i, j] = acc_o{t}[i, j] / {norm}")
    for t in range(n_d):
        a(f"        T.copy(acc_o{t}, O[by, bx * {Br}:(bx + 1) * {Br}, "
          f"{t * DT}:{(t + 1) * DT}])")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------- K3 ---
def _src_qk(BH, S, D, BM, BN, BK, threads, stages, sdtype):
    """S = Q@K^T * scale, written to global in sdtype.  This is the kernel whose
    output the arm exists to pay for."""
    SD = _TL_DT[sdtype]
    scale = 1.0 / math.sqrt(D)
    L = []
    a = L.append
    a("@T.prim_func")
    # fp32 in, fp16 shared tiles: the operand conversion rides the global->shared
    # copy, exactly as in FLASH.  Nothing is cast on the host.
    a(f"def main(Q: T.Tensor(({BH}, {S}, {D}), 'float32'),")
    a(f"         Kt: T.Tensor(({BH}, {S}, {D}), 'float32'),")
    a(f"         Sc: T.Tensor(({BH}, {S}, {S}), '{SD}')):")
    a(f"    with T.Kernel({S // BN}, {S // BM}, {BH}, threads={threads}) as (bx, by, bz):")
    a(f"        Qs = T.alloc_shared(({BM}, {BK}), 'float16')")
    a(f"        Ks = T.alloc_shared(({BN}, {BK}), 'float16')")
    a(f"        acc = T.alloc_fragment(({BM}, {BN}), 'float32')")
    a("        T.clear(acc)")
    a(f"        for ko in T.Pipelined({D // BK}, num_stages={stages}):")
    a(f"            T.copy(Q[bz, by * {BM}:(by + 1) * {BM}, ko * {BK}:(ko + 1) * {BK}], Qs)")
    a(f"            T.copy(Kt[bz, bx * {BN}:(bx + 1) * {BN}, ko * {BK}:(ko + 1) * {BK}], Ks)")
    a("            T.gemm(Qs, Ks, acc, transpose_B=True)")
    a(f"        for i, j in T.Parallel({BM}, {BN}):")
    a(f"            acc[i, j] = acc[i, j] * T.float32({scale!r})")
    # the store casts fp32 -> sdtype; that cast IS the sdtype factor
    a(f"        T.copy(acc, Sc[bz, by * {BM}:(by + 1) * {BM}, bx * {BN}:(bx + 1) * {BN}])")
    return "\n".join(L) + "\n"


def _src_softmax(BH, S, rows, threads, sdtype, pdtype):
    """P = softmax(S, dim=-1).  Reads sdtype, reduces in fp32, writes pdtype."""
    SD, PD = _TL_DT[sdtype], _TL_DT[pdtype]
    L = []
    a = L.append
    a("@T.prim_func")
    a(f"def main(Sc: T.Tensor(({BH}, {S}, {S}), '{SD}'),")
    a(f"         P: T.Tensor(({BH}, {S}, {S}), '{PD}')):")
    a(f"    with T.Kernel({S // rows}, {BH}, threads={threads}) as (bx, by):")
    a(f"        buf = T.alloc_fragment(({rows}, {S}), 'float32')")
    a(f"        mx = T.alloc_fragment(({rows},), 'float32')")
    a(f"        sm = T.alloc_fragment(({rows},), 'float32')")
    a(f"        T.copy(Sc[by, bx * {rows}:(bx + 1) * {rows}, 0:{S}], buf)")
    a("        T.reduce_max(buf, mx, dim=1, clear=True)")
    a(f"        for i, j in T.Parallel({rows}, {S}):")
    a("            buf[i, j] = T.exp(buf[i, j] - mx[i])")
    a("        T.reduce_sum(buf, sm, dim=1)")
    a(f"        for i, j in T.Parallel({rows}, {S}):")
    a("            buf[i, j] = buf[i, j] / sm[i]")
    a(f"        T.copy(buf, P[by, bx * {rows}:(bx + 1) * {rows}, 0:{S}])")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------- K2 ---
def _src_qk_softmax(BH, S, D, BM, BN, BK, threads, stages, sdtype, pdtype):
    """P = softmax(Q@K^T * scale, dim=-1), in ONE kernel.  S never reaches global.

    This is K3's first two kernels fused, and nothing else: the same qk tile, the
    same fp16 Q/K operands into `T.gemm` with an fp32 accumulator, the same fp32
    max/exp/sum, the same pdtype store.  The only difference from K3 is that the
    scaled score stays in the mma accumulator fragments instead of taking a round
    trip through global memory, which is exactly the quantity `K2 vs K3` exists
    to price (`common2.score_bytes`).

    The block owns a full (BM, S) row block, because softmax(dim=-1) needs the
    whole row and a block that owned only (BM, BN) of it would need cross-block
    communication -- which is what K3's separate softmax kernel is.  The row
    block is held as S//BN fp32 accumulators selected by a PYTHON loop, so the
    source is generated for the same reason FLASH's is (see the module
    docstring).

    `sdtype` keeps its meaning from the module docstring -- the dtype the score
    is KEPT in -- implemented as an explicit round through an fp16 fragment
    before the softmax sees it, byte for byte what FLASH does.  It changes no
    global traffic here (there is none for S, by construction), so K2's defining
    cells are the sdtype=fp32 ones; the fp16 cell is carried only so that K2 can
    be compared with FLASH at the same score precision.
    """
    PD = _TL_DT[pdtype]
    n_bn = S // BN
    scale = 1.0 / math.sqrt(D)
    L = []
    a = L.append
    a("@T.prim_func")
    a(f"def main(Q: T.Tensor(({BH}, {S}, {D}), 'float32'),")
    a(f"         Kt: T.Tensor(({BH}, {S}, {D}), 'float32'),")
    a(f"         P: T.Tensor(({BH}, {S}, {S}), '{PD}')):")
    a(f"    with T.Kernel({S // BM}, {BH}, threads={threads}) as (bx, by):")
    a(f"        Qs = T.alloc_shared(({BM}, {BK}), 'float16')")
    a(f"        Ks = T.alloc_shared(({BN}, {BK}), 'float16')")
    for t in range(n_bn):
        a(f"        acc{t} = T.alloc_fragment(({BM}, {BN}), 'float32')")
    if sdtype == "fp16":
        a(f"        s16 = T.alloc_fragment(({BM}, {BN}), 'float16')")
    for nm in ("m", "ss", "ls"):
        a(f"        {nm} = T.alloc_fragment(({BM},), 'float32')")
    for t in range(n_bn):
        a(f"        T.clear(acc{t})")
    # QK^T for the whole row block.  Q's (BM, BK) tile is re-read once per column
    # block; that is the same number of Q loads K3's qk kernel issues, since it
    # launches S//BN blocks per row block that each read the same Q tile.
    for t in range(n_bn):
        a(f"        for ko in T.Pipelined({D // BK}, num_stages={stages}):")
        a(f"            T.copy(Q[by, bx * {BM}:(bx + 1) * {BM}, ko * {BK}:(ko + 1) * {BK}], Qs)")
        a(f"            T.copy(Kt[by, {t * BN}:{(t + 1) * BN}, ko * {BK}:(ko + 1) * {BK}], Ks)")
        a(f"            T.gemm(Qs, Ks, acc{t}, transpose_B=True)")
    for t in range(n_bn):
        a(f"        for i, j in T.Parallel({BM}, {BN}):")
        a(f"            acc{t}[i, j] = acc{t}[i, j] * T.float32({scale!r})")
    if sdtype == "fp16":
        for t in range(n_bn):
            a(f"        for i, j in T.Parallel({BM}, {BN}):")
            a(f"            s16[i, j] = acc{t}[i, j]")
            a(f"        for i, j in T.Parallel({BM}, {BN}):")
            a(f"            acc{t}[i, j] = s16[i, j]")
    a("        T.fill(m, -T.infinity('float32'))")
    for t in range(n_bn):
        a(f"        T.reduce_max(acc{t}, m, dim=1, clear=False)")
    a("        T.fill(ls, 0)")
    for t in range(n_bn):
        a(f"        for i, j in T.Parallel({BM}, {BN}):")
        a(f"            acc{t}[i, j] = T.exp(acc{t}[i, j] - m[i])")
        a(f"        T.reduce_sum(acc{t}, ss, dim=1)")
        a(f"        for i in T.Parallel({BM}):")
        a("            ls[i] = ls[i] + ss[i]")
    for t in range(n_bn):
        a(f"        for i, j in T.Parallel({BM}, {BN}):")
        a(f"            acc{t}[i, j] = acc{t}[i, j] / ls[i]")
        # the store casts fp32 -> pdtype, the same cast K3's softmax kernel makes
        a(f"        T.copy(acc{t}, P[by, bx * {BM}:(bx + 1) * {BM}, "
          f"{t * BN}:{(t + 1) * BN}])")
    return "\n".join(L) + "\n"


def _src_pv(BH, S, D, BM, BN, BK, threads, stages, pdtype):
    """O = P@V.  Both operands are pdtype.

    fp16 -> `T.gemm`, fp16 tensor cores, fp32 accumulator.
    fp32 -> explicit fp32 FMA on the CUDA cores.  See the module docstring for
            why `T.gemm` is not used here: it would silently become TF32 with
            bit-truncated operands and bias the whole tensor out of the gate.
    """
    PD = _TL_DT[pdtype]
    L = []
    a = L.append
    a("@T.prim_func")
    a(f"def main(P: T.Tensor(({BH}, {S}, {S}), '{PD}'),")
    # P is produced by the previous kernel and is already pdtype; V comes
    # straight from the benchmark in fp32 and is converted on the way into
    # shared memory when pdtype is fp16.
    a(f"         Vt: T.Tensor(({BH}, {S}, {D}), 'float32'),")
    a(f"         O: T.Tensor(({BH}, {S}, {D}), 'float32')):")
    a(f"    with T.Kernel({D // BN}, {S // BM}, {BH}, threads={threads}) as (bx, by, bz):")
    # +1 float of row padding on the P tile for the fp32 arm: Ps[i, kk] at a
    # BK=64-float stride puts every thread of a warp on bank kk%32 at a
    # different address.  Same reason as the FLASH kernel.
    a(f"        Ps = T.alloc_shared(({BM}, {BK if pdtype == 'fp16' else BK + 1}), '{PD}')")
    a(f"        Vs = T.alloc_shared(({BK}, {BN}), '{PD}')")
    a(f"        acc = T.alloc_fragment(({BM}, {BN}), 'float32')")
    a("        T.clear(acc)")
    a(f"        for ko in T.Pipelined({S // BK}, num_stages={stages}):")
    if pdtype == "fp16":
        a(f"            T.copy(P[bz, by * {BM}:(by + 1) * {BM}, ko * {BK}:(ko + 1) * {BK}], Ps)")
    else:
        a(f"            for i, j in T.Parallel({BM}, {BK}):")
        a(f"                Ps[i, j] = P[bz, by * {BM} + i, ko * {BK} + j]")
    a(f"            T.copy(Vt[bz, ko * {BK}:(ko + 1) * {BK}, bx * {BN}:(bx + 1) * {BN}], Vs)")
    if pdtype == "fp16":
        a("            T.gemm(Ps, Vs, acc)")
    else:
        # kk outer -- see the FLASH kernel for why it must not be inside the
        # T.Parallel (replicated fragment layout)
        a(f"            for kk in T.serial({BK}):")
        a(f"                for i, j in T.Parallel({BM}, {BN}):")
        a("                    acc[i, j] = acc[i, j] + Ps[i, kk] * Vs[kk, j]")
    a(f"        T.copy(acc, O[bz, by * {BM}:(by + 1) * {BM}, bx * {BN}:(bx + 1) * {BN}])")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- build ---
def build(cfg) -> common2.Built2:
    algo = cfg.variant
    if algo not in ("K3", "K2", "FLASH"):
        raise KeyError(f"tilelang sdpa lane implements K3, K2 and FLASH, "
                       f"not {algo!r}")
    d = int(cfg.extra["d"])
    sdtype, pdtype = cfg.extra["sdtype"], cfg.extra["pdtype"]
    B, H, S = common2.S_B, common2.S_H, common2.S_S
    BH = B * H
    tile = _TILES[(algo, d)]
    p_torch = _TORCH_DT[pdtype]

    t0 = time.perf_counter()

    # `run` must be kernel launches and nothing else.  `.view` rather than
    # `.reshape`: q/k/v are contiguous (B,H,S,d) so this is a pure re-label of
    # the same storage, and `.view` RAISES if it ever stops being one, where
    # `.reshape` would silently start copying inside the timed region.  The fp16
    # operands the gemms want are produced by `T.copy` on the global->shared
    # path -- see `_src_flash`/`_src_qk`.
    #
    # This lane used to do `q.reshape(BH, S, d).half()` here, materializing fp16
    # copies of Q, K and V on every timed call.  Measured cost of that, run()
    # against the sum of the kernel times:
    #     FLASH d=1024 fp32/fp16   24.03 ms run, 16.27 ms kernels  (32% cast)
    #     K3    d=1024 fp32/fp16   22.47 ms run, 16.18 ms kernels  (28%)
    #     FLASH d=128  fp32/fp16    2.89 ms run,  1.46 ms kernels  (50%)
    # No other lane in the study does it, so every cross-DSL comparison built on
    # those numbers was wrong by up to 2x -- against this lane, but wrong.
    if algo == "FLASH":
        Br, Bc, DT, threads = tile["Br"], tile["Bc"], tile["DT"], tile["threads"]
        src = _src_flash(BH, S, d, Br, Bc, DT, threads, sdtype, pdtype)
        kf, _ = _compile(src, f"flash_d{d}_{sdtype}_{pdtype}", out_idx=[3])
        kernels = [("flash", kf)]
        sources = {"cuda_source": kf.get_kernel_source()}
        n_kernels = 1
        tile_str = (f"Br={Br} Bc={Bc} D_TILE={DT} threads={threads} "
                    f"n_d_acc={d // DT} n_kv={S // Bc}")

        def run(q, k, v):
            return kf(q.view(BH, S, d), k.view(BH, S, d),
                      v.view(BH, S, d)).view(B, H, S, d)

        def warm():
            qw = torch.zeros((BH, S, d), dtype=torch.float32, device="cuda")
            _ = kf(qw, qw, qw)
            torch.cuda.synchronize()
            del qw, _
    elif algo == "K2":
        qkT, pvT = tile["qk"], tile["pv"]
        s1 = _src_qk_softmax(BH, S, d, *qkT, sdtype, pdtype)
        s2 = _src_pv(BH, S, d, *pvT, pdtype)
        k1, _ = _compile(s1, f"qksm_d{d}_{sdtype}_{pdtype}", out_idx=[2])
        k2, _ = _compile(s2, f"pv_d{d}_{pdtype}", out_idx=[2])
        kernels = [("qk_softmax", k1), ("pv", k2)]
        sources = {"cuda_source": k1.get_kernel_source(),
                   "cuda_source_pv": k2.get_kernel_source()}
        n_kernels = 2
        tile_str = (f"qk+softmax {qkT[0]}x{qkT[1]}x{qkT[2]}/{qkT[3]}thr "
                    f"stages={qkT[4]} ({S // qkT[1]} score accumulators per "
                    f"block, full {S}-wide row); "
                    f"pv {pvT[0]}x{pvT[1]}x{pvT[2]}/{pvT[3]}thr stages={pvT[4]}")

        def run(q, k, v):
            p = k1(q.view(BH, S, d), k.view(BH, S, d))   # P, S never stored
            return k2(p, v.view(BH, S, d)).view(B, H, S, d)

        def warm():
            qw = torch.zeros((BH, S, d), dtype=torch.float32, device="cuda")
            pw = torch.zeros((BH, S, S), dtype=p_torch, device="cuda")
            _1 = k1(qw, qw)
            _2 = k2(pw, qw)
            torch.cuda.synchronize()
            del qw, pw, _1, _2
    else:
        qkT, smT, pvT = tile["qk"], tile["sm"], tile["pv"]
        s1 = _src_qk(BH, S, d, *qkT, sdtype)
        s2 = _src_softmax(BH, S, smT[0], smT[1], sdtype, pdtype)
        s3 = _src_pv(BH, S, d, *pvT, pdtype)
        k1, _ = _compile(s1, f"qk_d{d}_{sdtype}", out_idx=[2])
        k2, _ = _compile(s2, f"sm_d{d}_{sdtype}_{pdtype}", out_idx=[1])
        k3, _ = _compile(s3, f"pv_d{d}_{pdtype}", out_idx=[2])
        kernels = [("qk", k1), ("softmax", k2), ("pv", k3)]
        sources = {"cuda_source": k1.get_kernel_source(),
                   "cuda_source_softmax": k2.get_kernel_source(),
                   "cuda_source_pv": k3.get_kernel_source()}
        n_kernels = 3
        tile_str = (f"qk {qkT[0]}x{qkT[1]}x{qkT[2]}/{qkT[3]}thr "
                    f"stages={qkT[4]}; softmax {smT[0]} rows/{smT[1]}thr; "
                    f"pv {pvT[0]}x{pvT[1]}x{pvT[2]}/{pvT[3]}thr stages={pvT[4]}")

        def run(q, k, v):
            sc = k1(q.view(BH, S, d), k.view(BH, S, d))   # S materialized
            p = k2(sc)                                    # P materialized
            del sc
            return k3(p, v.view(BH, S, d)).view(B, H, S, d)

        def warm():
            qw = torch.zeros((BH, S, d), dtype=torch.float32, device="cuda")
            sw = torch.zeros((BH, S, S), dtype=_TORCH_DT[sdtype], device="cuda")
            pw = torch.zeros((BH, S, S), dtype=p_torch, device="cuda")
            _1 = k1(qw, qw)
            _2 = k2(sw)
            _3 = k3(pw, qw)
            torch.cuda.synchronize()
            del qw, sw, pw, _1, _2, _3

    # Force module load / first launch INSIDE the compile window, at the real
    # shape (the JIT is shape-specialised).  Warmed through the KERNELS, never
    # through `run`, so no host-side state is primed with a dummy tensor.
    warm()
    torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    if pdtype == "fp16":
        pv_note = ("PV: fp16 operands -> T.gemm on fp16 TENSOR CORES, fp32 "
                   "accumulator")
    else:
        pv_note = ("PV: fp32 operands -> explicit fp32 FMA over shared memory "
                   "on the CUDA CORES (no tensor core, no T.gemm), fp32 "
                   "accumulator. NOT tf32. T.gemm was rejected for this arm "
                   "because TileLang lowers fp32 operands to "
                   "tl::mma_sync<kTensorFloat32,kTensorFloat32,kFloat32,16,8,8> "
                   "by REINTERPRETING the fp32 bits, i.e. truncation rather "
                   "than round-to-nearest: measured -6.51e-4 relative bias "
                   "(bit-truncation emulation gives -6.50e-4, RNE +2.8e-6), "
                   "which put 100% of elements outside the 1e-4 gate "
                   "(max_abs_err 4.12e-4 / 4.15e-4 / 4.22e-4 at d=128/256/1024)")
    s_note = ("scores rounded to fp16 after the scale, before the softmax sees "
              "them" if sdtype == "fp16" else
              "scores kept in the fp32 mma accumulator, never rounded")

    artifacts = {
        "algo": algo,
        "score_dtype": sdtype,
        "prob_dtype": pdtype,
        "n_kernels": n_kernels,
        "tile": tile_str,
        "tilelang_version": tilelang.__version__,
        "tilelang_disk_cache": _CACHE_ENABLED,
        "backend_detail": (
            f"tilelang {tilelang.__version__} {algo} d={d} "
            f"sdtype={sdtype} pdtype={pdtype}; tile[{tile_str}]; "
            "Q/K/V enter the kernel as the fp32 the benchmark stores and are "
            "converted on the global->shared T.copy -- NO host-side cast in "
            "run(), which is kernel launches plus the output allocation only; "
            "QK^T: fp16 Q/K operands -> T.gemm fp32 accumulator in every cell "
            "(sdtype governs only the rounding of the RESULT); "
            f"{s_note}; softmax reduction in fp32 (exp2 with log2(e) folded in "
            "for FLASH, expf for K3/K2); "
            f"{pv_note}; output fp32; "
            "prim_func emitted as generated source (linecache) because "
            "TVMScript cannot Python-loop over accumulator buffers"),
    }
    artifacts.update(sources)

    notes = (f"tilelang {tilelang.__version__} sdpa {algo} d={d} "
             f"s={sdtype} p={pdtype}: {tile_str}")
    return common2.Built2(run=run, compile_s=compile_s, artifacts=artifacts,
                          notes=notes, n_kernels=n_kernels,
                          x_dtype=torch.float32)
