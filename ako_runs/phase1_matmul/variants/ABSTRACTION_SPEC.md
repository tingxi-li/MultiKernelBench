# TileLang-only abstraction study — standard matmul

**Separate from the cross-DSL transfer study.** Mixing them confounds
abstraction level with algorithm and with hardware instruction path. This
study varies *only* the abstraction level at which the same algorithm on the
same hardware path is expressed.

Module: `variants/tilelang_abstraction.py`, registered as DSL `tilelang_abs`.
Same `build(cfg) -> common.Built` contract as `SPEC.md`; read that first.

## Held constant in every variant

- fp16 tensor-core operands, **fp32 output**
- `KC = 2048` in-block chunk flush
- `BM=128, BN=128, BK=32`
- 256 threads
- identical **pre-cast** fp16 inputs (`cfg.cast == "precast"`)
- plain 2-D grid, no swizzle, no cross-block split-K, no autotuning

## The variants

| variant | level | inner-K implementation |
|---|---|---|
| **TL-H1** | high | `T.Pipelined(..., num_stages=3)` + `T.copy` + `T.gemm` |
| **TL-H2** | high | same, `num_stages=1` |
| **TL-M1** | hybrid | regular K loop, explicit `T.copy`, explicit barriers, `T.gemm` |
| **TL-M2** | hybrid | explicit double-buffered shared storage and synchronization around `T.gemm` |
| **TL-S1** | SIMT control | same blocking and same fp32 chunking, but scalar/thread-level FMA |

`cfg.variant` is `H1|H2|M1|M2|S1`; `cfg.stages` and `cfg.arith` are pre-set from
`common.ABSTRACTION_SPECS` and must be honoured.

**Deliberately excluded:** `T.wgmma_gemm` and `T.tcgen05_gemm`. Those manual
asynchronous interfaces target Hopper and Blackwell; this host is Ada (sm_89).

**TL-S1 is a hardware control, not an abstraction measurement.** It removes
tensor cores, so `M1 − S1` measures the tensor-core contribution and must never
be reported as abstraction overhead.

## What each comparison isolates

| comparison | isolates |
|---|---|
| H1 vs H2 | the compiler's software pipeline |
| H2 vs M1 | high-level scheduling overhead, with pipelining disabled on both sides |
| H1 vs M2 | compiler-generated vs manually expressed pipeline |
| M1 vs S1 | the tensor-core contribution (hardware, not abstraction) |

### The H1-vs-M2 depth confound, and how it is removed

As specified, **H1 is 3-stage and M2 is 2-stage**, so the headline comparison
mixes "compiler-generated vs hand-written" with pipeline depth. Removing that
means running one arm at the other's depth — and only one of them can move:

- `num_stages` is a single integer to the compiler-managed pipeline, so **H1 at
  2 stages** is a config change. It runs as its own campaign
  (`jobs/abstraction_depth.json`, tag `abstraction_depth`) because `build()`
  refuses an off-spec depth unless `x_depth_control=1` is passed — accidental
  drift voids the study, so the opt-in is explicit rather than the check being
  permissive.
- **M2 at 3 stages does not exist as a configuration.** Its buffer parity is
  hand-unrolled at depth 2 (`_kernel_M2` asserts `KI % 2 == 0`); a third stage
  requires re-deriving the wait discipline. That asymmetry is not an obstacle to
  work around — it is evidence for conclusion 2 below, and is reported as such.

So the matched-depth comparison is **H1@stages=2 vs M2**, and the decision rule
is applied to that pair, not to the literal H1-vs-M2 pair.

## Decision rule (fixed in advance)

- **H1 ≈ M2 within 3% and similar SASS** → abstraction is mainly ergonomic
- **H1 > M2 by more than 10%** → compiler-managed pipeline is materially better
- **M2 > H1 by more than 10%** → the high-level abstraction leaves performance on the table
- **H1 ≫ H2** → the pipeline, not `T.gemm` alone, is the important TileLang feature

The rule has **a gap between 3% and 10%** and a measurement can land in it. If it
does, that is reported as "no verdict by the stated rule" plus the measured
direction — not resolved by widening a band after the fact. `abstraction_rule.py`
applies the thresholds mechanically for exactly this reason.

It also has a resolution floor: between-process spread at the frozen protocol is
~3.5%, so the 3% "ergonomic" band is *inside the noise*. A result in that band is
reported as consistent with equivalence, never as demonstrating it.

## Required evidence

Dump IR / PTX / SASS for all five variants (`inspect_code.py --dsl tilelang_abs
--variant H1` etc.). **Explicitly verify that H1 and M2 execute comparable MMA
counts and differ primarily in load scheduling.** If their HMMA counts differ
materially, they are not the same algorithm and the H1-vs-M2 comparison does not
mean what the decision rule assumes — say so rather than applying the rule.

## Two conclusions to report separately

1. **Abstraction efficiency** — performance of equivalent algorithms on
   equivalent instruction paths written at TL-H vs TL-M vs TL-L.
2. **Abstraction-enabled exploration** — whether the higher-level interface made
   it *cheaper to discover* split-K, fusion, layouts and pipeline configurations.
   This is a claim about search cost, not about kernel speed; support it with
   compile time per variant and with the number of edits each level required,
   not with runtime.
