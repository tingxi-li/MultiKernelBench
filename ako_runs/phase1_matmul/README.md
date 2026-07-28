# Phase 1 — the decisive standard-matmul experiment

Decomposes the published `0.59 → 0.79 → 1.11 → 4.13` cross-DSL GEMM spread
(`../SIX_OP_ANSWERS.md`) into its actual causes, by implementing **the same four
kernels in all four DSLs** and changing exactly one thing at a time.

The published spread blends at least four factors that were never separated:
fp16 vs tf32 vs fp32 arithmetic, accumulation strategy, software pipelining,
and the input distribution the tolerance gate is evaluated under. This directory
separates them.

## The four variants

| | arithmetic | K accumulation | pipeline | isolates |
|---|---|---|---|---|
| **A** | fp32, no tensor cores | one chain over all K | DSL-native | baseline |
| **B** | fp16 TC → fp32 acc | one chain over all K | off (1 stage) | fp16 tensor-core throughput *and* its error |
| **C** | fp16 TC → fp32 acc | `KC=2048` chunk flush | off (1 stage) | split-K |
| **D** | fp16 TC → fp32 acc | `KC=2048` chunk flush | 3 stages | software pipelining |

Read as differences:

- **B − A** → what fp16 tensor cores buy, and what they cost in accuracy
- **C − B** → is the chunk flush an accuracy enabler or a performance lever
- **D − C** → what the software pipeline buys
- **spread across DSLs at D** → residual compiler / code-generation advantage,
  with arithmetic, tiling and pipeline depth all held equal

## Matched geometry

`BM=128, BN=128, BK=32, threads=256`, plain 2-D grid with **no block swizzle**,
**no cross-block split-K**, **no autotuning**, operands **pre-cast to fp16
outside the timed region** unless the casting control says otherwise.

Three tile shapes are defined in `common.GEOMS`:

| name | BM | BN | BK | role |
|---|---|---|---|---|
| `primary` | 128 | 128 | 32 | the matched point; every default table uses it |
| `secondary` | 128 | 256 | 32 | second matched point, so no conclusion is hostage to one shape |
| `incumbent` | 128 | 256 | 64 | the shape the published tilelang winner used |

At `BN=128` each 128×128 output tile streams `(128+128)·8192·2 B = 4.2 MB` and
the grid is 512 blocks, so ~2.15 GB flows through L2 per launch; at `BN=256` it
is ~1.6 GB. The narrower tile is measurably more L2-bound, which compresses
differences between variants. That is a reason to report both, not a reason to
prefer either — and the two shapes do disagree about pipelining (see
`PHASE1_REPORT.md` §3).

**`incumbent` is not a matched point.** At `BK=64`, three pipeline stages need
`(128·64 + 64·256)·2·3 = 147456 B` of shared memory against sm_89's 101376 B
limit, so variant D cannot run there at `stages=3`. It is carried only for the
casting comparison, at `stages=2`.

## Measurement discipline

| requirement | how |
|---|---|
| absolute runtime primary | every table is ms; ratios are derived and labelled |
| serial on one GPU | `driver.py --gpu N`, one subprocess at a time |
| randomized variant order | `random.Random(--order-seed).shuffle` over (variant × rep) |
| identical saved inputs | `inputs/rand_seed0.pt`, `inputs/randn_seed0.pt` |
| ≥5 independent process runs | `--reps 5`; one variant per process, no JIT/cuBLAS/allocator leakage |
| median and CI | per-process median → median of medians, full range, t-based 95% CI |
| compile/search separated | `Built.compile_s`, reported in its own column |

Timing is cuda-event with the L2 thrashed before every trial, as in
`AKO4ALL/bench/kernelbench/bench.py` — but the warmup is **a fixed 2.0 seconds of
wall clock, not a fixed iteration count**. That is a deliberate departure from
the harness default and it matters: these cards idle at 210 MHz, soak thermally
under load, and never reach a true steady state (`stability.py` measures the
median *rising* monotonically with warmup depth, 0.886 → 1.023 ms from 50 to
1000 iterations). Because variant A runs ~4.5× longer than variant D, a fixed
iteration count would deliver ~4.5× more heat before the slow variant than
before the fast one and bias the comparison. Fixed-time warmup equalizes the
thermal state instead of the instruction count, and cuts between-process spread
from 27.6% (50 iters) to 3.5%.

## Why cross-run speedup is not used

`../standard_matrix_multiplication/tilelang/convergence.csv` logs
`runtime=1.07 ms` at `speedup=4.187` — implying a 4.48 ms reference — while the
triton cell's identity row implies 6.08 ms for the same reference. The
denominator moved 36% between runs. Every number here is therefore an absolute
runtime measured in the same campaign, and `torch.matmul` is re-measured as just
another variant (`--dsl torch`) rather than quoted from history.

Measured here under the frozen protocol: `torch.matmul` fp32 = **4.717 ms**
(29.1 TFLOP/s), and `torch.matmul` fp16 = **1.168 ms** — which *fails* the
harness gate with max error 1.821 and 72.8% of elements out of budget, roughly
8× worse than any fp16 variant in this study.

## Precision control

Two RMS-matched input distributions, both `RMS = 1/√3`:

| | mean | ‖C‖ typical | gate budget `1e-4 + 1e-4·|C|` |
|---|---|---|---|
| `torch.rand` (benchmark-faithful) | 0.5 | 2048 | **0.205** |
| `torch.randn × 1/√3` (zero-mean) | 0 | 24 | **0.0025** |

The same relative gate is **82× tighter** under `randn`. That ratio is a
property of the benchmark's input distribution, not of any kernel, and it is the
confound the fp16 path may be riding. `accuracy.py` measures every variant over
≥20 seeds under both, recording max/mean error, error quantiles, signed bias,
the fraction of elements violating the gate, and — against an fp64 ground truth —
how much of the "error" belongs to the fp32 oracle itself.

## Layout

```
common.py                   shapes, saved inputs, timing, gate, error stats, aggregation
variants/SPEC.md            the contract every DSL module implements
variants/ABSTRACTION_SPEC.md the TileLang-only study's contract (kept separate on purpose)
variants/*_gemm.py          one module per DSL: build(cfg) -> Built(run, compile_s, ...)
variants/tilelang_abstraction.py  the TL-H1/H2/M1/M2/S1 arms
runner.py                   time ONE variant in a fresh process -> JSON
driver.py                   randomized-order, multi-process campaign (+ busy-GPU preflight)
make_jobs.py                emits jobs/{matched,kc_sweep,casting,pipeline,native_tuned}.json
run_all.sh                  the frozen protocol, all six campaigns end to end
accuracy.py                 ≥20-seed × 2-distribution error campaign
stability.py                warmup-depth / thermal-soak characterization
inspect_code.py             PTX/SASS dump + HMMA/LDGSTS/LDSM/FFMA census, regs/spills/smem
profile_target.py           minimal ncu target (kernel only, no reference, no L2 thrash)
ncu_collect.py              tensor-pipe %, occupancy, stalls, DRAM/L2, achieved TC FLOP/s
analyze.py                  raw records -> the matched tables
sweep_tables.py             the sub-study tables
confirm_winners.py          re-measure native-tuning winners at the full 5-process protocol
abstraction_rule.py         applies the pre-registered 3%/10% decision rule mechanically
plots.py                    the figures
report_tables.py            precision + generated-code tables
build_report.py             injects every generated table into PHASE1_REPORT.md
```

## Running it

```bash
cd ako_runs/phase1_matmul

# everything, frozen protocol, serialized on one idle GPU (~2 h)
GPU=0 ./run_all.sh

# or one campaign at a time
python make_jobs.py
python driver.py --jobs jobs/matched.json --gpu 0 --reps 5 --tag matched \
                 --warmup-s 2.0 --trials 100
python analyze.py --tag matched --md results/matched/TABLES.md

# precision control (DSL-invariant: outputs are bit-identical across DSLs)
python accuracy.py --dsl tilelang --variant D --seeds 20

# generated code + hardware counters
python inspect_code.py --all
python ncu_collect.py --all --gpu 0

# native-tuning winners, re-measured at the reporting protocol
python confirm_winners.py --emit
python driver.py --jobs jobs/confirm.json --gpu 0 --reps 5 --tag confirm \
                 --warmup-s 2.0 --trials 100

# assemble
python plots.py && python build_report.py
```

`driver.py` refuses to start on a GPU that has another compute process on it.
That check exists because a concurrent job silently inflates every timing in the
run and the result looks perfectly well-formed afterwards — an earlier campaign
here was lost to exactly that and had to be discarded and re-run.
