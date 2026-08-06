"""TileLang-only SDPA study. TWO axes, kept apart on purpose.

The study spec is explicit that these must not be merged:

  WITHIN-KERNEL ABSTRACTION AXIS -- S3-H / S3-M / S3-MP / S3-L.
      All four are the same single-kernel online-softmax algorithm with the same
      tile. Only how the kernel is written changes. This is the axis on which
      the phrase "abstraction costs X%" is meaningful.

  ALGORITHMIC DECOMPOSITION AXIS -- S1 / S2 / best-S3.
      Different numbers of kernels, different materialization, different V
      re-read counts, different occupancy. This is NOT an abstraction result and
      the report must not label it as one.

Comparing S1 against S3-H directly and calling the difference "abstraction" is
the specific error this file is arranged to prevent: those two differ in
materialization, register pressure, V re-reads and kernel count all at once.

--- the abstraction arms -------------------------------------------------------

S3-H   High. `T.Pipelined` over the KV loop, `T.gemm`, `T.reduce_max` /
       `T.reduce_sum`, and ONE output accumulator of (block_M, D_TILE). At
       d=1024 a (block_M, d) accumulator does not fit in registers, so the
       high-level formulation is forced into an OUTER loop over d-tiles -- and
       that loop redoes the entire QK^T for every d-tile. The n_d_tiles-fold
       repetition of QK work is not a strawman: it is what the natural
       high-level formulation degenerates into once d exceeds what one
       accumulator can hold. At d=128 there is one d-tile and the arm is a
       clean single pass, which is why the head-dim sweep is the whole point.

S3-M   Hybrid. Regular (un-pipelined) KV loop, one output accumulator PER
       d-tile so a single KV pass covers all of d, and an explicit fp32->shared
       ->fp16 layout bridge for the score tile. Still uses T.gemm and
       T.reduce_*. This is the shape of the shipped incumbent solution.

S3-MP  S3-M with `T.Pipelined` put back on the KV loop. Same manual accumulator
       structure, byte for byte -- the loop constructor is the ONLY token that
       differs from S3-M. Tests whether a generic software pipeline composes
       with a hand-managed V buffer that is deliberately re-read per d-tile.
       IT DOES NOT, ABOVE d=128; see "WHAT S3-MP MEASURES" below. That is the
       arm's result, not a defect in it.

S3-L   S3-M with the online-softmax reductions written as manual warp shuffles
       (`T.shfl_down`) instead of `T.reduce_max` / `T.reduce_sum`. Everything
       else identical. See "HOW S3-L REDUCES" below for why a manual reduction
       over a `T.gemm` accumulator has to go through shared memory first.

--- the decomposition arms -----------------------------------------------------

S1     Three kernels: QK^T -> softmax -> PV, S materialized in global memory.
S2     Two kernels: fused QK^T+softmax, then PV.

Both share `sdpa_tilelang`'s kernels so that S1 here and K3 in the cross-DSL
table are the same code, which is what lets the two studies be cross-read.

================================================================================
HOW S3-L REDUCES, AND WHY IT IS NOT JUST "T.reduce_* SPELLED OUT"
================================================================================

`acc_s` is the fp32 C fragment of a `T.gemm`. Its (row, col) -> (thread, lane,
register) mapping is chosen by TileLang's layout inference and is not visible in
the source, so a warp-shuffle reduction *cannot be written over it directly*: a
shuffle chain reduces across the 32 lanes of a warp, and nothing in the language
says which (i, j) those lanes hold.

The first version of this arm tried to dodge that by writing

    for i, jj in T.Parallel(bM, bN):
        if acc_s[i, jj] > m_cur[i]:
            m_cur[i] = acc_s[i, jj]

which is a data race -- every thread that owns any column of row i writes
`m_cur[i]`, and `m_cur` is itself a fragment, so each thread ends up holding the
max over only *its own* columns. The exponentials, the row sum and the running
normalizer are then each computed against a different per-thread shift, and the
result is not a softmax at all. TileLang's own checker says so at compile time
("Data race detected: `m_cur(i,)` is written by multiple threads in loop (jj,)")
and the gate said so at run time: max_abs_err 6.502e+00 against a ~1.5e-4 gate,
next to 4.27e-05 for S3-M, which is the same algorithm with library reductions.

So the manual arm stages the score tile into a *shared* fp32 buffer first, where
the (row, col) -> address mapping is defined by the program rather than by
layout inference, and reduces that:

    warp w owns rows [w*bM/nwarps, (w+1)*bM/nwarps)      (8 rows each at 64/8)
    lane l owns columns l, l+32, ...                      (2 columns each at 64)
    -> per-lane partial, then T.shfl_down 16/8/4/2/1 leaves the whole row in
       lane 0, then lane 0 writes the row's value to a (bM,) shared vector

Because a warp owns *whole rows*, the shuffle chain is the entire reduction and
no cross-warp tree is needed; the (bM,) shared vector exists to hand the result
back to the fragment world (`m_cur[i] = red[i]` under `T.Parallel`), which is
also the only layout-safe way to get a row-broadcast scalar into a fragment --
the same shared-memory hop `sdpa_tilelang`'s fp32 arm uses for `scs`/`lss`.

That is the honest cost of the manual form and it is what the arm measures: the
library reduction gets to reduce the fragment in place, the manual one pays a
round trip through shared memory, twice per KV block (once for the max, once for
the sum), because the fragment layout is not addressable from the source.

================================================================================
WHAT S3-MP MEASURES: A GENERIC PIPELINE DOES NOT COMPOSE WITH THIS STRUCTURE
================================================================================

S3-MP is S3-M with `T.serial(n_kv)` replaced by `T.Pipelined(n_kv,
num_stages=2)` and nothing else. It compiles and passes at d=128. At d=256 and
d=1024 it does not compile, and the reason is structural rather than incidental:

    InternalError: Pipeline planning error: Multiple writes to overlapping
    buffer regions detected. Stage 13 and stage 16 are both writing to buffer
    'V_s' with overlapping regions. This is not supported in pipeline planning.

The manual structure holds one output accumulator per d-tile and feeds them from
a SINGLE (block_N, D_TILE) V buffer that is re-loaded once per d-tile. With
n_d_tiles == 1 (d=128) that buffer is written once per KV iteration and the
planner multi-buffers it happily. With n_d_tiles > 1 it is written n_d_tiles
times per iteration, from n_d_tiles separate statements, and TileLang's pipeline
planner requires that each shared buffer be written by at most one statement in
the loop body.

The fix the planner wants is one V buffer per d-tile, which the pipeline then
double-buffers. That costs 2 * block_N * d * 2 bytes for V alone. Not an
estimate -- that exact variant was written out (same body, same reductions, one
V_s per d-tile) and compiled, and these are its `dyn_shared_memory_buf` values
at block_M=64, D_TILE=128, 256 threads, against sm_89's 101376 B per block:

    block_N   d      n_d_tiles     smem
        64     128       1        75776 B   fits  (== the shipped S3-MP: at
                                                   n_d_tiles=1 the two forms
                                                   are the same kernel)
        64     256       2       108544 B   over by  7168 B
        64    1024       8       305152 B   3.0x over
        32    1024       8       161792 B   1.6x over
        16    1024       8        90112 B   fits

The V term is invariant to how d is tiled -- block_N * d * 2 bytes of V are
touched per KV block however many d-tiles they are cut into -- so no choice of
D_TILE helps; only block_N moves it. Solving the budget at d=1024,

    4096*block_N (V, doubled) + 256*block_N (K_s) + 128*block_N (S_s)
        + 16384 (Q_s) + 2048 (reduce workspace)  <=  101376
    -> block_N <= 18.5, i.e. block_N = 16, which is what the table measures.

So the only tile at which a double-buffered S3-MP fits at d=1024 is a QUARTER of
the KV tile the other three arms run. Shrinking the shared tile to 16 to make
one arm compile would make the abstraction axis measure the tile instead, so it
is not done: the tile stays block_M=64, block_N=64, D_TILE=128, threads=256 for
all four arms, and S3-MP is reported as compiling only at d=128. Anything the
report says about S3-MP is therefore a statement about d=128 alone.

This is a result about the abstraction, not a gap in the study. The high-level
arm S3-H takes `T.Pipelined` at every head dim precisely BECAUSE it keeps one
accumulator and one V load per KV block; the hybrid arm buys a single-pass KV
loop with a hand-managed buffer and loses the generic pipeline for it. Both
halves of that trade are measured here.
"""
import math
import os
import time

import torch

import tilelang
import tilelang.language as T

import common2

if os.environ.get("PHASE2_TL_CACHE", "0") != "1":
    tilelang.disable_cache()

ABS_ARMS = {
    "S3-H":  "high: T.Pipelined KV loop, T.gemm, T.reduce_*, one output "
             "accumulator (outer d-tile loop re-does QK^T when d > D_TILE)",
    "S3-M":  "hybrid: regular KV loop, one accumulator per d-tile, explicit "
             "fp32->smem->fp16 layout bridge",
    "S3-MP": "hybrid + T.Pipelined over the same manual accumulator structure "
             "(compiles only where n_d_tiles == 1; see module docstring)",
    "S3-L":  "hybrid + manual T.shfl_down warp reductions for the online "
             "softmax, over a shared-memory staging of the score tile",
    "S1":    "three kernels: QK^T -> softmax -> PV (S materialized)",
    "S2":    "two kernels: fused QK^T+softmax, then PV",
}

_LOG2E = 1.4426950408889634

# Identity for the manual max reduction, in the per-thread register the shuffle
# chain runs on. A finite -FLT_MAX rather than -inf, matching the fused lane's
# F2/F3 arms: it never reaches the exponent (the running max `m_prv` is still
# seeded with -inf, and it is `m_prv` that feeds `exp2`), so all this constant
# has to do is lose every comparison.
_NEG_BIG = -3.402823466e+38


def _tile_for(d):
    """One tile shape per head dim, shared by EVERY arm at that head dim.

    Fixed here rather than per-arm so the abstraction axis cannot pick up a
    tile change: block_M=64, block_N=64, D_TILE=128, threads=256 at every head
    dim, so the tile is in fact constant across the whole study.

    block_N is 64, not the 128 an earlier version used, because the pipelined
    arms have to fit. Measured dynamic shared memory (`dyn_shared_memory_buf`
    read off the compiled TIR, not arithmetic over the alloc_shared calls --
    the multi-buffer `T.Pipelined` inserts is invisible to that arithmetic,
    which is how the block_N=128 tile came to be shipped as "about 98 KB, fits"):

                          block_N=128       block_N=64
        S3-M   any d        100352 B          59392 B
        S3-H   d=128        133120 B  FAIL    75776 B
        S3-H   d>=256       135168 B  FAIL    77824 B
        S3-MP  d=128        133120 B  FAIL    75776 B
        S3-L   any d        131328 B  FAIL    73984 B

    (both columns are the arms AS THEY STAND IN THIS FILE, i.e. the block_N=128
    column is what the shipped tile would cost the repaired S3-L, not what the
    broken one cost.) Against sm_89's 101376 B per block: at block_N=128 the
    un-pipelined library arm already sits at 99% of the budget (Q_s 16 K + K_s
    32 K + V_s 32 K + S_s 16 K + 2 K of reduce workspace), so neither
    `T.Pipelined`'s second buffer nor S3-L's fp32 staging tile can exist. Three
    of the four arms are over the limit there; the two pipelined ones are what
    the shipped file actually died on, at launch, with "Failed to set the
    allowed dynamic shared memory size to 133120". block_N=64 halves K_s, V_s
    and S_s and leaves ~25 KB of headroom, which is exactly one more V tile --
    what a 2-stage pipeline needs.

    The shrink is not free and is not hidden. Measured back to back in ONE
    process on the same kernel body (so only block_N moves), S3-M at d=128:

        block_N=128   smem 100352 B   2.4023 ms
        block_N= 64   smem  59392 B   2.6757 ms      +11.4%

    Half the KV tile is twice the KV iterations and half the reuse of Q_s. That
    11% is paid by all four arms equally, which is the point -- a tile that
    differed per arm would make the abstraction axis measure the tile.

    D_TILE stays 128 at every head dim, so n_d_tiles is 1 / 2 / 8 at d = 128 /
    256 / 1024 and the "one accumulator per d-tile" arms hold 1 / 2 / 8
    fragments of (64, 128) fp32. At d=1024 that is 256 floats per thread over
    256 threads -- the same register pressure the shipped incumbent d=1024
    kernel carries, which is why it is known to be reachable.
    """
    return dict(block_M=64, block_N=64, D_TILE=min(d, 128), threads=256)


def _smem_bytes(kernel):
    """The compiled kernel's real dynamic shared-memory request, in bytes.

    Read off the lowered TIR (`dyn_shared_memory_buf`) rather than recomputed
    from the tile, because the multi-buffering `T.Pipelined` adds is invisible
    to any hand arithmetic over the `alloc_shared` calls -- which is precisely
    how the block_N=128 tile came to be shipped as "about 98 KB, fits".
    """
    try:
        fn = next(iter(kernel.artifact.device_mod.functions.values()))
        return int(fn.attrs["dyn_shared_memory_buf"])
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- S3 arms ---
def _s3(batch, heads, seq, d, arm, pdtype):
    """One builder for all four S3 arms; the arm only switches the marked
    blocks, so nothing else can drift between them."""
    t = _tile_for(d)
    bM, bN, DT, th = t["block_M"], t["block_N"], t["D_TILE"], t["threads"]
    n_dt = d // DT
    n_kv = seq // bN
    nwarps = th // 32
    rows_per_warp = bM // nwarps          # 8 at 64 rows / 8 warps
    cols_per_lane = bN // 32              # 2 at block_N=64
    scale = (1.0 / math.sqrt(d)) * _LOG2E     # folded into exp2
    shape = [batch, heads, seq, d]
    pd = "float16" if pdtype == "fp16" else "float32"
    single_acc = arm == "S3-H"
    pipelined = arm in ("S3-H", "S3-MP")
    manual_red = arm == "S3-L"
    assert bM % nwarps == 0 and bN % 32 == 0, "manual warp reduction needs " \
        "block_M divisible by the warp count and block_N by 32"

    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        # Q/K/V arrive as fp32 and are converted on the global->shared path by
        # the `T.copy`s below, which write into fp16 shared tiles. They must NOT
        # be cast on the host: `S1`/`S2` delegate to the cross-DSL lane, which
        # converts on the load, so a host-side cast here would put a dtype
        # conversion on one side of the S1-vs-S3 comparison and not the other --
        # and that comparison is supposed to isolate the decomposition.
        def main(Q: T.Tensor(shape, "float32"),
                 K: T.Tensor(shape, "float32"),
                 V: T.Tensor(shape, "float32"),
                 O: T.Tensor(shape, "float32")):
            with T.Kernel(T.ceildiv(seq, bM), heads, batch, threads=th) as (bx, by, bz):
                Q_s = T.alloc_shared([bM, DT], "float16")
                K_s = T.alloc_shared([bN, DT], "float16")
                V_s = T.alloc_shared([bN, DT], "float16")
                S_s = T.alloc_shared([bM, bN], pd)
                acc_s = T.alloc_fragment([bM, bN], "float32")
                acc_c = T.alloc_fragment([bM, bN], pd)
                m_cur = T.alloc_fragment([bM], "float32")
                m_prv = T.alloc_fragment([bM], "float32")
                m_scl = T.alloc_fragment([bM], "float32")
                r_sum = T.alloc_fragment([bM], "float32")
                lsum = T.alloc_fragment([bM], "float32")

                # ---- S3-L only: the machinery a manual reduction needs -------
                # An fp32 staging copy of the score tile (the gemm C fragment's
                # layout is not addressable from the source, so the shuffle
                # chain cannot run on it), a (bM,) vector to carry the reduced
                # rows back into fragment-land, and one register of scratch.
                if manual_red:
                    S_f = T.alloc_shared([bM, bN], "float32")
                    red = T.alloc_shared([bM], "float32")
                    lred = T.alloc_local([1], "float32")
                    tid = T.get_thread_binding(0)
                    wid = tid >> 5
                    lane = tid & 31

                # ---- output accumulators -------------------------------------
                # S3-H holds exactly one and pays an outer d-tile loop for it.
                # The others hold one per d-tile so a single KV pass covers d --
                # exactly n_d_tiles of them, never more. An earlier version
                # allocated eight unconditionally and rescaled all eight every
                # KV block, so at d=128 the emitted kernel declared seven dead
                # `float a[32]` arrays (224 registers per thread) that only
                # ptxas's dead-code elimination stood between and a spill. The
                # count is now n_d_tiles = 1 / 2 / 8 at d = 128 / 256 / 1024,
                # checkable in `get_kernel_source()`.
                a0 = T.alloc_fragment([bM, DT], "float32")
                if not single_acc and n_dt > 1:
                    a1 = T.alloc_fragment([bM, DT], "float32")
                if not single_acc and n_dt > 2:
                    a2 = T.alloc_fragment([bM, DT], "float32")
                    a3 = T.alloc_fragment([bM, DT], "float32")
                    a4 = T.alloc_fragment([bM, DT], "float32")
                    a5 = T.alloc_fragment([bM, DT], "float32")
                    a6 = T.alloc_fragment([bM, DT], "float32")
                    a7 = T.alloc_fragment([bM, DT], "float32")

                # ---------------------------------------------------------------
                if single_acc:
                    # HIGH-LEVEL FORM. One accumulator, so d is walked on the
                    # OUTSIDE and every d-tile repeats the whole QK^T pass.
                    for dt in T.serial(n_dt):
                        T.clear(a0)
                        T.fill(lsum, 0)
                        T.fill(m_cur, -T.infinity("float32"))
                        for kb in T.Pipelined(n_kv, num_stages=2):
                            T.clear(acc_s)
                            for di in T.serial(n_dt):
                                T.copy(Q[bz, by, bx * bM, di * DT], Q_s)
                                T.copy(K[bz, by, kb * bN, di * DT], K_s)
                                T.sync_threads()
                                T.gemm(Q_s, K_s, acc_s, transpose_B=True,
                                       policy=T.GemmWarpPolicy.FullRow)
                            T.copy(m_cur, m_prv)
                            T.fill(m_cur, -T.infinity("float32"))
                            T.reduce_max(acc_s, m_cur, dim=1, clear=False)
                            for i in T.Parallel(bM):
                                m_cur[i] = T.max(m_cur[i], m_prv[i])
                                m_scl[i] = T.exp2(m_prv[i] * scale - m_cur[i] * scale)
                            for i, jj in T.Parallel(bM, bN):
                                acc_s[i, jj] = T.exp2(acc_s[i, jj] * scale
                                                      - m_cur[i] * scale)
                            T.reduce_sum(acc_s, r_sum, dim=1)
                            for i in T.Parallel(bM):
                                lsum[i] = lsum[i] * m_scl[i] + r_sum[i]
                            T.copy(acc_s, S_s)
                            T.sync_threads()
                            T.copy(S_s, acc_c)
                            for i, jj in T.Parallel(bM, DT):
                                a0[i, jj] *= m_scl[i]
                            T.copy(V[bz, by, kb * bN, dt * DT], V_s)
                            T.sync_threads()
                            T.gemm(acc_c, V_s, a0, policy=T.GemmWarpPolicy.FullRow)
                        for i, jj in T.Parallel(bM, DT):
                            O[bz, by, bx * bM + i, dt * DT + jj] = a0[i, jj] / lsum[i]
                else:
                    # HYBRID FORM. One KV pass; every d-tile accumulates into its
                    # own fragment, so QK^T is computed exactly once per KV block.
                    T.clear(a0)
                    if n_dt > 1:
                        T.clear(a1)
                    if n_dt > 2:
                        T.clear(a2); T.clear(a3); T.clear(a4)
                        T.clear(a5); T.clear(a6); T.clear(a7)
                    T.fill(lsum, 0)
                    T.fill(m_cur, -T.infinity("float32"))
                    # S3-MP differs from S3-M in exactly this constructor and
                    # nothing else. Above n_d_tiles == 1 the planner rejects the
                    # body; see the module docstring -- that rejection IS the
                    # arm's measurement.
                    for kb in (T.Pipelined(n_kv, num_stages=2) if pipelined
                               else T.serial(n_kv)):
                        T.clear(acc_s)
                        for di in T.serial(n_dt):
                            T.copy(Q[bz, by, bx * bM, di * DT], Q_s)
                            T.copy(K[bz, by, kb * bN, di * DT], K_s)
                            T.sync_threads()
                            T.gemm(Q_s, K_s, acc_s, transpose_B=True,
                                   policy=T.GemmWarpPolicy.FullRow)

                        # ---- online-softmax update: the ONLY thing S3-L changes
                        T.copy(m_cur, m_prv)
                        if manual_red:
                            # MANUAL ROW MAX. The score tile goes out to shared
                            # memory first because a shuffle chain cannot address
                            # a gemm C fragment (module docstring). One warp owns
                            # `rows_per_warp` whole rows, so the 5-step
                            # T.shfl_down chain IS the whole reduction and lane 0
                            # holds the row max at the end of it.
                            T.copy(acc_s, S_f)
                            T.sync_threads()
                            for r in T.serial(rows_per_warp):
                                lred[0] = T.float32(_NEG_BIG)
                                for c in T.serial(cols_per_lane):
                                    v = S_f[wid * rows_per_warp + r, c * 32 + lane]
                                    if v > lred[0]:
                                        lred[0] = v
                                lred[0] = T.max(lred[0], T.shfl_down(lred[0], 16))
                                lred[0] = T.max(lred[0], T.shfl_down(lred[0], 8))
                                lred[0] = T.max(lred[0], T.shfl_down(lred[0], 4))
                                lred[0] = T.max(lred[0], T.shfl_down(lred[0], 2))
                                lred[0] = T.max(lred[0], T.shfl_down(lred[0], 1))
                                if lane == 0:
                                    red[wid * rows_per_warp + r] = lred[0]
                            T.sync_threads()
                            # back into fragment-land; a shared hop is the only
                            # layout-safe way to broadcast a row scalar into a
                            # fragment whose layout inference already anchored it
                            for i in T.Parallel(bM):
                                m_cur[i] = red[i]
                            T.sync_threads()
                        else:
                            T.fill(m_cur, -T.infinity("float32"))
                            T.reduce_max(acc_s, m_cur, dim=1, clear=False)
                        for i in T.Parallel(bM):
                            m_cur[i] = T.max(m_cur[i], m_prv[i])
                            m_scl[i] = T.exp2(m_prv[i] * scale - m_cur[i] * scale)
                        for i, jj in T.Parallel(bM, bN):
                            acc_s[i, jj] = T.exp2(acc_s[i, jj] * scale
                                                  - m_cur[i] * scale)
                        if manual_red:
                            # MANUAL ROW SUM, same staging and the same warp
                            # mapping, with the max chain replaced by an add
                            # chain. Second round trip through shared memory per
                            # KV block; that is the arm's cost and it is real.
                            T.copy(acc_s, S_f)
                            T.sync_threads()
                            for r in T.serial(rows_per_warp):
                                lred[0] = T.float32(0)
                                for c in T.serial(cols_per_lane):
                                    lred[0] = lred[0] + S_f[wid * rows_per_warp + r,
                                                            c * 32 + lane]
                                lred[0] = lred[0] + T.shfl_down(lred[0], 16)
                                lred[0] = lred[0] + T.shfl_down(lred[0], 8)
                                lred[0] = lred[0] + T.shfl_down(lred[0], 4)
                                lred[0] = lred[0] + T.shfl_down(lred[0], 2)
                                lred[0] = lred[0] + T.shfl_down(lred[0], 1)
                                if lane == 0:
                                    red[wid * rows_per_warp + r] = lred[0]
                            T.sync_threads()
                            for i in T.Parallel(bM):
                                r_sum[i] = red[i]
                            T.sync_threads()
                        else:
                            T.reduce_sum(acc_s, r_sum, dim=1)
                        for i in T.Parallel(bM):
                            lsum[i] = lsum[i] * m_scl[i] + r_sum[i]

                        # explicit layout bridge: the fp32 score fragment cannot
                        # feed T.gemm as an A operand, so it goes out to shared
                        # memory and comes back in the probability dtype
                        T.copy(acc_s, S_s)
                        T.sync_threads()
                        T.copy(S_s, acc_c)

                        for i, jj in T.Parallel(bM, DT):
                            a0[i, jj] *= m_scl[i]
                            if n_dt > 1:
                                a1[i, jj] *= m_scl[i]
                            if n_dt > 2:
                                a2[i, jj] *= m_scl[i]; a3[i, jj] *= m_scl[i]
                                a4[i, jj] *= m_scl[i]; a5[i, jj] *= m_scl[i]
                                a6[i, jj] *= m_scl[i]; a7[i, jj] *= m_scl[i]

                        # V is re-read once per d-tile through a single buffer;
                        # that reuse is exactly what a generic pipeline fights
                        T.copy(V[bz, by, kb * bN, 0], V_s)
                        T.sync_threads()
                        T.gemm(acc_c, V_s, a0, policy=T.GemmWarpPolicy.FullRow)
                        if n_dt > 1:
                            T.copy(V[bz, by, kb * bN, 1 * DT], V_s)
                            T.sync_threads()
                            T.gemm(acc_c, V_s, a1, policy=T.GemmWarpPolicy.FullRow)
                        if n_dt > 2:
                            T.copy(V[bz, by, kb * bN, 2 * DT], V_s)
                            T.sync_threads()
                            T.gemm(acc_c, V_s, a2, policy=T.GemmWarpPolicy.FullRow)
                            T.copy(V[bz, by, kb * bN, 3 * DT], V_s)
                            T.sync_threads()
                            T.gemm(acc_c, V_s, a3, policy=T.GemmWarpPolicy.FullRow)
                            T.copy(V[bz, by, kb * bN, 4 * DT], V_s)
                            T.sync_threads()
                            T.gemm(acc_c, V_s, a4, policy=T.GemmWarpPolicy.FullRow)
                            T.copy(V[bz, by, kb * bN, 5 * DT], V_s)
                            T.sync_threads()
                            T.gemm(acc_c, V_s, a5, policy=T.GemmWarpPolicy.FullRow)
                            T.copy(V[bz, by, kb * bN, 6 * DT], V_s)
                            T.sync_threads()
                            T.gemm(acc_c, V_s, a6, policy=T.GemmWarpPolicy.FullRow)
                            T.copy(V[bz, by, kb * bN, 7 * DT], V_s)
                            T.sync_threads()
                            T.gemm(acc_c, V_s, a7, policy=T.GemmWarpPolicy.FullRow)

                    for i, jj in T.Parallel(bM, DT):
                        O[bz, by, bx * bM + i, jj] = a0[i, jj] / lsum[i]
                    if n_dt > 1:
                        for i, jj in T.Parallel(bM, DT):
                            O[bz, by, bx * bM + i, 1 * DT + jj] = a1[i, jj] / lsum[i]
                    if n_dt > 2:
                        for i, jj in T.Parallel(bM, DT):
                            O[bz, by, bx * bM + i, 2 * DT + jj] = a2[i, jj] / lsum[i]
                            O[bz, by, bx * bM + i, 3 * DT + jj] = a3[i, jj] / lsum[i]
                            O[bz, by, bx * bM + i, 4 * DT + jj] = a4[i, jj] / lsum[i]
                            O[bz, by, bx * bM + i, 5 * DT + jj] = a5[i, jj] / lsum[i]
                            O[bz, by, bx * bM + i, 6 * DT + jj] = a6[i, jj] / lsum[i]
                            O[bz, by, bx * bM + i, 7 * DT + jj] = a7[i, jj] / lsum[i]
        return main

    return _k(), t


def build(cfg) -> common2.Built2:
    arm = cfg.variant
    if arm not in ABS_ARMS:
        raise KeyError(f"unknown SDPA abstraction arm {arm!r}; "
                       f"known: {sorted(ABS_ARMS)}")
    d = cfg.extra["d"]
    sd, pd = cfg.extra["sdtype"], cfg.extra["pdtype"]
    B, H, S = common2.S_B, common2.S_H, common2.S_S

    t0 = time.perf_counter()
    smem = None
    if arm in ("S1", "S2"):
        # The decomposition arms delegate to the cross-DSL TileLang lane through
        # its ordinary build(cfg), so S1 here and K3 in the cross-DSL table are
        # literally the same object code. That is what lets a reader carry a
        # number from one table to the other; a re-implementation would only
        # look the same.
        from . import sdpa_tilelang as x
        sub = common2.make_sdpa_config(
            "tilelang", "K3" if arm == "S1" else "K2",
            extra={"d": d, "sdtype": sd, "pdtype": pd})
        built = x.build(sub)
        run, nk = built.run, built.n_kernels
        detail = built.artifacts.get("backend_detail", "")
        tile = built.artifacts.get("tile", "(see lane)")
    else:
        k3, tile = _s3(B, H, S, d, arm, pd)
        smem = _smem_bytes(k3)

        def run(q, k, v):
            return k3(q, k, v)
        nk = 1
        detail = ABS_ARMS[arm]

        # The dummy must carry the kernel's declared shape. A smaller one is
        # rejected by TileLang's packed-ABI check, so the warm-up would raise at
        # build time and the arm would never run at all.
        qw = torch.zeros((B, H, S, d), dtype=torch.float32, device="cuda")
        _ = k3(qw, qw, qw)
        torch.cuda.synchronize()
        del qw, _
        torch.cuda.empty_cache()
    compile_s = time.perf_counter() - t0

    tile_str = tile if isinstance(tile, str) else (
        f"block_M={tile['block_M']} block_N={tile['block_N']} "
        f"D_TILE={tile['D_TILE']} threads={tile['threads']} "
        f"n_d_tiles={d // tile['D_TILE']} n_kv={S // tile['block_N']}")

    artifacts = {"backend_detail": f"{detail}; tile={tile_str}; d={d} "
                                   f"score={sd} prob={pd}"
                                   + (f"; dyn_smem={smem} B of "
                                      f"{common2.SM89_MAX_SMEM_PER_BLOCK} B"
                                      if smem else ""),
                 "algo": arm, "score_dtype": sd, "prob_dtype": pd,
                 "tile": tile_str}
    if smem is not None:
        artifacts["shared_bytes"] = smem
    capture_dir = os.environ.get("TILELANG_ABSTRACTION_CAPTURE_DIR", "")
    if arm not in ("S1", "S2"):
        try:
            artifacts["cuda_source"] = k3.get_kernel_source()
        except Exception as e:  # noqa: BLE001 - retained as admission evidence
            artifacts["cuda_source_error"] = repr(e)
        if capture_dir:
            os.makedirs(capture_dir, exist_ok=True)
            base = os.path.join(capture_dir, cfg.key().replace("/", "_"))
            for extension, exporter in (("ptx", k3.export_ptx), ("sass", k3.export_sass)):
                path = base + "." + extension
                try:
                    exporter(path)
                    artifacts[extension + "_path"] = path
                except Exception as e:  # noqa: BLE001 - retained as admission evidence
                    artifacts[extension + "_error"] = repr(e)
            with open(base + ".cu", "w", encoding="utf-8") as handle:
                handle.write(artifacts.get("cuda_source", ""))
            artifacts["cuda_source_path"] = base + ".cu"

    return common2.Built2(
        run=run, compile_s=compile_s, n_kernels=nk,
        artifacts=artifacts,
        notes=f"tilelang sdpa abstraction arm {arm} at d={d} ({sd}/{pd})",
        x_dtype=torch.float32)
