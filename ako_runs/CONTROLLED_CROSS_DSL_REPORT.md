# Cross-DSL performance, optimization trajectories, transfer, and TileLang abstraction

> Corrections and current controlling interpretation: [`controlled_followup/REVIEW_ROUND2_RESPONSE_20260731.md`](controlled_followup/REVIEW_ROUND2_RESPONSE_20260731.md).

> Evidence snapshot: branch `cross-dsl-6op-ncu-redo`, commit `8854ae0`.
> This report audits the committed trajectories and controlled Phase-1/Phase-2
> artifacts; it does not claim an exhaustive search of any DSL.

## Executive summary

The repository does **not** establish a universal ordering such as
`CUDA >= Triton >= TileLang`, nor does it establish that any DSL has an
unconditionally higher performance ceiling. It establishes three narrower
results:

1. On memory-bound kernels, all four implementations normally converge near the
   same hardware traffic floor. Language choice changes the route and the cost of
   exploration more than the final runtime.
2. On tensor-core kernels, the original native searches diverged widely, but the
   later controlled experiments show that most of that divergence came from
   precision choices, missing optimization families, unequal search, and
   denominator choices. Once the same matmul recipe is implemented in all four
   DSLs, the four-way spread contracts from the published `7.0x` to `1.31x`.
3. TileLang's lower abstraction levels do not systematically improve generated
   kernel efficiency. High-level TileLang is equal or faster in the clean
   expression-level comparisons. Moving to a more explicit level helps when the
   *algorithmic structure* must change, as in large-head-dimension SDPA; that is
   different from low-level code intrinsically generating faster instructions.

The practical recommendation is therefore **high-level first, then descend only
after profiling identifies a structural or expressibility obstruction**. An
optimization idea normally transfers across DSLs; its exact schedule and the
engineering cost of realizing it often do not.

## 1. Scope and evidence hierarchy

This report combines two kinds of artifact that answer different questions.

### Native-search trajectories

The six-op `convergence.csv` files record what each DSL's own optimization run
happened to try and how much compile-plus-benchmark time it consumed. They answer:

> What did this particular search find, and how expensive were its trials?

They do **not** prove a language ceiling. Search breadth was unequal, some lanes
stopped after one custom kernel, several important optimization families were
never attempted, and current solution files have sometimes drifted away from the
solutions described by their convergence logs.

Start with:

- [`CONVERGENCE_PROTOCOL.md`](CONVERGENCE_PROTOCOL.md)
- [`SIX_OP_ANSWERS.md`](SIX_OP_ANSWERS.md), treated as the historical native-search interpretation
- each operation's `<dsl>/convergence.csv` and `ITERATIONS*.md`

### Controlled studies

Phase 1 and Phase 2 hold arithmetic, tile geometry, schedules, inputs, and
algorithms fixed, or vary one registered factor at a time. They answer:

> Does the apparent gap survive after the missing optimization is transferred?

These later studies should take precedence when they contradict the native-search
interpretation:

- [`phase1_matmul/PHASE1_REPORT.md`](phase1_matmul/PHASE1_REPORT.md)
- [`phase2_fused_sdpa/PHASE2_REPORT.built.md`](phase2_fused_sdpa/PHASE2_REPORT.built.md)
- implementation contracts in
  [`phase1_matmul/variants/SPEC.md`](phase1_matmul/variants/SPEC.md) and
  [`phase2_fused_sdpa/variants2/SPEC2.md`](phase2_fused_sdpa/variants2/SPEC2.md)

All measurements are on one RTX 6000 Ada (`sm_89`) and fixed problem shapes.
"Ceiling" below means the best demonstrated result in the tested space, never a
proof about all programs an expert might eventually write.

## 2. Does any DSL have a higher performance ceiling?

### Answer

**No universal ceiling ordering is supported.** In particular, the artifacts
directly refute the assumption that lower-level CUDA must be at least as fast as
Triton or TileLang.

The evidence splits by kernel regime:

| Regime | Controlled observation | Defensible conclusion |
|---|---|---|
| Memory-bound reduction/depthwise/normalization | Implementations converge near the same byte-traffic floor | No demonstrated DSL ceiling |
| Matched standard matmul | TileLang 1.048 ms, Triton 1.171, CUDA-PTX 1.253, CUDA-WMMA 1.376 | TileLang has a modest realization advantage for this fixed recipe, not a multi-x ceiling |
| Equal 19-point matmul search | TileLang 0.978 ms, Triton 1.078, CUDA-PTX 1.101, CUDA-WMMA 1.278 | Four-way demonstrated spread is 1.31x |
| Matched fused matmul | TileLang 1.598 ms, Triton 1.773, CUDA-PTX 1.897, CUDA-WMMA 2.031; PyTorch FP16 1.523 | Compiler-backed lanes realize this configuration better, but no custom lane has a lower median than the precision-matched vendor kernel |
| SDPA, `D=1024`, fastest passing custom cell | Triton K3 18.98 ms, CUDA-PTX K3 21.33, TileLang FLASH 25.57, CUDA-WMMA K3 30.45 | The winner changes with algorithm, dtype, and head dimension |

The standard-matmul control is decisive because all four DSLs implement the same
progression and produce bit-identical results:

```text
A: FP32 CUDA-core arithmetic
B: FP16 Tensor Cores, full-K accumulation
C: B + in-block KC=2048 accumulator flush
D: C + staged global-to-shared loads
```

Every DSL can use FP16 Tensor Cores. The mechanisms differ—`T.gemm`, `tl.dot`,
WMMA C++, or `mma.sync`/`ldmatrix`—but the hardware path is not TileLang-exclusive.
The published gap arose because the native Triton and CUDA searches had not all
reached that recipe.

The remaining 1.10--1.31x spread is real for this GPU, implementation set, and
search grid. It may reflect instruction scheduling and code generation, but it
does not justify "no matter how optimized, DSL X can never catch DSL Y."

There is also a useful representational caveat. If unrestricted CUDA is allowed
to embed or load the exact PTX/cubin emitted by another DSL, its theoretical
expressibility contains that generated program. That does not mean a human will
rediscover the same schedule within a realistic budget, and `ptxas` still owns
parts of register allocation and instruction scheduling. The experiments measure
practical finite-budget realization, not that theoretical inclusion relation.

Files to inspect:

- [`phase1_matmul/PHASE1_REPORT.md`](phase1_matmul/PHASE1_REPORT.md), Sections 3, 5, and 7
- [`phase1_matmul/variants/`](phase1_matmul/variants/) for the four matched implementations
- [`phase2_fused_sdpa/PHASE2_REPORT.built.md`](phase2_fused_sdpa/PHASE2_REPORT.built.md), Sections 2 and 4

## 3. How do optimization trajectories differ by DSL?

The main differences are in how candidate schedules are expressed and how costly
they are to test, not in exclusive access to FP16 Tensor Cores.

| DSL | Characteristic trajectory | Interface-specific capability or constraint | Observed result |
|---|---|---|---|
| Triton | Change constexpr tile/stage/warp parameters; use `tl.dot`; often attach an autotune grid | Fastest cold compilation in the controlled test; cache hints and compiler layouts are concise | Explores configurations cheaply, but the original matmul search missed FP16 + KC flush; a later fused artifact did find FP16 and became faster than the logged TileLang artifact |
| CUDA without inline PTX | Write explicit CUDA C++ and WMMA fragments; rebuild for each structural variant | WMMA hides accumulator element-to-coordinate mapping, restricting some register epilogues | High implementation and compile cost; original compute-heavy searches were shallow, but Phase 1 proves FP16 + KC flush is implementable and competitive |
| CUDA with inline PTX | Add direct `mma.sync`, `ldmatrix`, `cp.async`, and cache/store controls | Maximum local instruction control and known fragment mapping | Extra control did not create a universal win; async staging was neutral or harmful at the matched matmul tile, although other tiles can benefit |
| TileLang | Begin with `T.gemm`, `T.copy`, `T.Pipelined`, and `T.reduce_*`; optionally expose loops, buffers, async copies, or shuffles | Compiler owns fragment layout and can generate pipelines/reductions over it; some high-level pipeline compositions reject complex shared-buffer reuse | Easy parameter changes and safe layout generation; medium-level restructuring matters for large-D SDPA, while low-level reductions/pipelines are not faster by default |

### Inline example: standard matmul

The native trajectories looked DSL-specific:

- TileLang found FP16 `T.gemm`, then the KC flush, then a wider N tile by its
  fifth custom candidate (CSV row 6).
- Triton stayed in FP32/TF32 families and never tried FP16 + KC flush.
- CUDA-WMMA explored several TF32 fragment layouts but not the successful FP16 family.
- CUDA-PTX progressed from SGEMM to WMMA to `mma.sync`, then added chunking and
  buffering over ten custom variants.

Phase 1 then transplanted the same FP16/KC/pipeline recipe into every DSL. It
worked in all four. The earlier "TileLang-only" result was therefore mostly a
trajectory/discovery result, not a hard capability difference.

Files to inspect:

- [`standard_matrix_multiplication/tilelang/convergence.csv`](standard_matrix_multiplication/tilelang/convergence.csv)
- [`standard_matrix_multiplication/triton/convergence.csv`](standard_matrix_multiplication/triton/convergence.csv)
- [`standard_matrix_multiplication/cuda_noptx/convergence.csv`](standard_matrix_multiplication/cuda_noptx/convergence.csv)
- [`standard_matrix_multiplication/cuda_unlimited/convergence.csv`](standard_matrix_multiplication/cuda_unlimited/convergence.csv)

### Inline examples: unique mechanisms without unique ceilings

LayerNorm uses the same transferable two-pass/L2-residency idea in all four
lanes, but the realization differs. TileLang expresses a single cooperative
kernel with `T.sync_grid`; Triton adds a small third kernel so every apply block
does not repeat the final reduction; the CUDA lanes use explicit host-side row
loops. All reach essentially the same traffic roofline. `T.sync_grid` is therefore
a DSL-unique convenience in this experiment, not evidence of a unique final
performance level.

SDPA shows a different kind of uniqueness. Triton and TileLang can express and
retune fused FlashAttention-like kernels concisely; WMMA hides accumulator
coordinates; inline PTX exposes them but at much greater implementation cost.
Nevertheless, Phase 2 implements both the `FLASH` and three-kernel `K3`
algorithms across all four lanes, and the winning algorithm changes with head
dimension. The unique interfaces alter the search route, not access to a result
that no other lane can reproduce.

Files to inspect:

- [`layer_norm/tilelang/ITERATIONS.md`](layer_norm/tilelang/ITERATIONS.md)
- [`layer_norm/triton/ITERATIONS.md`](layer_norm/triton/ITERATIONS.md)
- [`scaled_dot_product_attention/triton/ITERATIONS_opus48.md`](scaled_dot_product_attention/triton/ITERATIONS_opus48.md)
- [`scaled_dot_product_attention/tilelang/ITERATIONS_opus48.md`](scaled_dot_product_attention/tilelang/ITERATIONS_opus48.md)

### Inline example: fused matmul artifact drift

The published 1.17 ms TileLang result belongs to the archived Opus artifact, but
the Phase-2 incumbent checker loads the mutable current `solution/` file. Those
files have different GEMM tiles, pipeline depths, weight layouts, and softmax
implementations. Triton's archived file was not a logged TF32 kernel: it was a
10-configuration kernel that loaded fp32 operands and cast tiles to fp16 inside
the kernel, with a fp32 GELU intermediate. The current file has 13 configurations,
precasts the activation, caches an fp16 weight, and writes an fp16 intermediate.

For example, archived TileLang uses `BM=128, BN=256, BK=32, stages=3` and reads
the native `(N,K)` weight through `transpose_B=True`; current TileLang uses
`BM=128, BN=128, BK=64, stages=2`, consumes a cached `(K,N)` weight, and replaces
the softmax implementation. This is a different program, not a fresh timing of
the logged winner.

Therefore, the current 1.202 ms Triton versus 1.669 ms TileLang result means
"the current files rank this way," not "the original 1.17 ms TileLang result was
disproved." A strict reproduction must benchmark both archived
`solution_opus48` files in the same randomized processes.

Files to inspect:

- [`phase2_fused_sdpa/fused_incumbent_check.py`](phase2_fused_sdpa/fused_incumbent_check.py)
- [`matmul_gelu_softmax/tilelang/solution/solution_opus48/matmul_gelu_softmax.py`](matmul_gelu_softmax/tilelang/solution/solution_opus48/matmul_gelu_softmax.py)
- [`matmul_gelu_softmax/tilelang/solution/matmul_gelu_softmax.py`](matmul_gelu_softmax/tilelang/solution/matmul_gelu_softmax.py)
- corresponding Triton files under [`matmul_gelu_softmax/triton/solution/`](matmul_gelu_softmax/triton/solution/)

## 4. Are optimization trajectories transferable?

### Answer

**Optimization principles transfer well; literal configurations do not.**

| Transfer layer | Examples | Transfer verdict |
|---|---|---|
| Algorithm/dataflow | FP16 operands with FP32 output, KC accumulator flush, fuse an epilogue, cache immutable weights, avoid materializing attention scores, L2-resident second pass | Usually transferable |
| Work mapping | Coalesced indexing, tile the inner dimension, choose one versus multiple kernels | Transferable after adapting to the DSL's programming model |
| Exact schedule | `BM/BN/BK`, warp count, pipeline depth, number of accumulators | Must be retuned |
| DSL primitive | `T.reduce_*`, Triton cache hints, WMMA fragments, inline `cp.async` | Requires translation and may not have a one-line equivalent |

Two controls show both sides.

**Successful transfer:** Phase 1 implements A/B/C/D in every DSL, with identical
numerics and dynamic Tensor Core work. The original four-way `7.0x` native-search
spread becomes `1.31x` after transfer and equal-budget tuning.

The source-level forms are different spellings of the same idea:

```python
# TileLang: compiler-managed load schedule around a tensor-core operation
for ko in T.Pipelined(KC // BK, num_stages=stages):
    T.copy(A[..., ko * BK], As)
    T.copy(B[ko * BK, ...], Bs)
    T.gemm(As, Bs, Cchunk)

# Triton: the outer loop explicitly resets the chunk accumulator
for _ in range(0, K // KC):
    chunk = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, KC // BK):
        chunk = tl.dot(a.to(tl.float16), b.to(tl.float16), chunk)
    acc += chunk

# CUDA-WMMA: the same accuracy repair is an explicit fragment drain
wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
flush_acc(acc_frag, fp32_register_accumulator);
```

See the exact implementations in
[`tilelang_gemm.py`](phase1_matmul/variants/tilelang_gemm.py),
[`triton_gemm.py`](phase1_matmul/variants/triton_gemm.py),
[`cuda_noptx_gemm.py`](phase1_matmul/variants/cuda_noptx_gemm.py), and
[`cuda_unlimited_gemm.py`](phase1_matmul/variants/cuda_unlimited_gemm.py).

This decomposition also resolves the earlier question about casting, K
splitting, and software pipelining:

| Isolated change | TileLang | Triton | CUDA-WMMA | CUDA-PTX | What it means |
|---|---:|---:|---:|---:|---|
| FP32 CUDA cores -> FP16 Tensor Cores | 6.20x | 4.52x | 4.64x | 4.88x | The dominant speed gain; portable to every DSL |
| Full-K accumulator -> `KC=2048` chunk flush | 0.95x | 0.72x | 0.71x | 0.93x | Slower, but repairs the FP16 full-K accuracy failure |
| Best staged load schedule vs one stage | 1.04x | 1.45x | 1.20x | 1.00x | Helpful in some realizations, not the universal main gain |

So the answer is **no** if "splitting K plus software pipelining is the main
source of TileLang's gain" is meant literally. The large gain is FP16
Tensor-Core arithmetic. Merely converting inputs is not the gain: moving
`.half()` into the timed region costs 0.207--0.221 ms. The repository's
"split-K" here is an **in-block periodic FP32 accumulator flush**, not
cross-block split-K parallelism; it primarily restores accuracy and actually
costs throughput in isolation.

**Non-transferable schedule:** at the matched matmul tile, the best gain over one
stage is 1.45x for Triton, 1.20x for CUDA-WMMA, 1.04x for TileLang, and 1.00x for
CUDA-PTX. Copying TileLang's three-stage setting verbatim can therefore regress
another DSL even though copying the *idea* "overlap loads and MMA" is sensible.

The same distinction appears in SDPA. `K3` and `FLASH` are implemented across all
four lanes, and all reproduce the head-dimension-dependent crossover. Their exact
best tiles and fastest algorithm differ.

## 5. Which DSL converges faster?

There are two different notions of convergence.

### Cost of one controlled trial

Cold compilation of the same fused-GEMM arm was:

| DSL | Cold build |
|---|---:|
| Triton | 1.19 s |
| TileLang | 6.01 s |
| CUDA-WMMA | 36.39 s |
| CUDA-PTX | 36.35 s |

This is the cleanest convergence-cost result: **Triton can test configurations
fastest, TileLang is next, and both CUDA lanes are much slower per source-changing
trial**. This measures one build, not the cost of discovering a candidate:
Triton's shipped 13-configuration autotune runs during warm-up and is not charged
to `compile_s`, and human/debug time is absent. CUDA and Triton have warm-cache
measurements; the shipped TileLang path did not expose a comparable warm state.

### Native trajectory to 95% of each cell's best custom result

The protocol's identity-inclusive metric can label a failed search "converged at
iteration 1" when no custom kernel catches the vendor baseline. To compare actual
search paths, the following is recomputed directly from `convergence.csv` after
excluding the identity. Each entry is:

```text
first custom variant within 5% of that cell's best logged custom result
/ total custom variants @ cumulative benchmark time at the threshold
```

| Operation | Triton | CUDA-WMMA | CUDA-PTX | TileLang |
|---|---:|---:|---:|---:|
| Layer norm | 2/2 @ 39.8 s | 2/5 @ 68.8 s | 1/2 @ 57.0 s | **1/3 @ 25.3 s** |
| Sum reduction | 1/2 @ 128.0 s | 1/3 @ 158.3 s | 1/3 @ 159.9 s | 1/4 @ 127.4 s |
| Standard matmul | 1/4 @ 20.4 s* | 3/7 @ 125.9 s* | 10/10 @ 406.9 s | **5/9 @ 48.6 s** |
| Fused matmul | 3/4 @ 47.3 s | 1/1 @ 47.7 s* | 1/1 @ 47.9 s* | **2/4 @ 22.6 s** |
| Depthwise convolution | 1/4 @ 26.9 s | 2/3 @ 103.2 s | 2/3 @ 107.7 s | **1/4 @ 24.1 s** |
| SDPA | 6/7 @ 686.0 s | 1/1 @ 212.0 s* | 1/1 @ 210.4 s* | **4/8 @ 355.5 s** |

`*` marks misleading early convergence: Triton and CUDA-WMMA standard matmul
missed the successful FP16/chunk family, while both CUDA fused and SDPA lanes
stopped after one custom implementation. The latter SDPA attempts only had to
beat a weak PyTorch fallback at `D=1024`. Fewer iterations can mean an efficient
successful search or premature convergence to a poor local family.

There is therefore no defensible global ranking by iteration count. Across the
six logged searches TileLang actually tried 32 custom variants, Triton 23, and
each CUDA lane 20. The robust advantage of Triton and TileLang is cheaper source
experimentation, not universally fewer trials.

Profiler-run counts cannot be ranked from the committed trajectories. Only five
CSV rows have a non-empty `ncu_key`—one TileLang LayerNorm, one CUDA-PTX sum, one
CUDA-WMMA matmul, and two Triton depthwise rows. The schema contains no `ncu_s`,
all `agent_s` fields are empty, and the prose mentions additional uncounted
profiles. Exact profile count and profiler time are unrecoverable.

Files to inspect:

- all six operation `convergence.csv` files
- [`CONVERGENCE_PROTOCOL.md`](CONVERGENCE_PROTOCOL.md), especially the declared ceiling metric
- [`tools/timed_bench.sh`](tools/timed_bench.sh), for the fields actually recorded
- [`phase2_fused_sdpa/PHASE2_REPORT.built.md`](phase2_fused_sdpa/PHASE2_REPORT.built.md), Section 1.3
- [`phase1_matmul/PHASE1_REPORT.md`](phase1_matmul/PHASE1_REPORT.md), cold-compile and equal-grid tables

## 6. Do TileLang's lower abstraction levels improve efficiency?

### Answer

**Not as a general rule. The clean comparisons run in the opposite direction.**

The tested levels need precise names. TL-H delegates load scheduling and
reductions to `T.Pipelined`, `T.copy`, `T.gemm`, and `T.reduce_*`. TL-M exposes
loops, buffers, barriers, or multiple accumulators while retaining compiler GEMM
and reduction primitives. The most explicit arms use `T.async_copy` or
`T.shfl_down`. There is no raw-`mma.sync` TileLang arm on Ada: the available
manual asynchronous GEMM interfaces target Hopper/Blackwell. The SIMT `S1` arm
removes Tensor Cores and is a hardware-path control, not an abstraction level.

| Experiment | Higher-level result | More explicit result | Interpretation |
|---|---:|---:|---|
| Matmul, matched two-stage pipeline | `T.Pipelined + T.gemm`: 0.995 ms | Manual double-buffer pipeline around `T.gemm`: 1.071 ms | Manual form is 7.7% slower; preregistered rule calls the magnitude inconclusive |
| Fused softmax kernel | `T.reduce_*`: 0.0881 ms | Best coalesced shuffle/cached-exp form: 0.0870 ms | Approximately equal; 1.2% is at timer resolution |
| SDPA reduction, same loop and tile | `T.reduce_*` | Manual `T.shfl_down` | Manual form is 3.3--13% slower because it stages the opaque GEMM fragment through shared memory |

The manual fused-softmax variants also expose an indexing decision that the
high-level `T.copy` path avoids. Using thread-contiguous rather than coalesced
indexing costs 2.66--2.78x. That is not an inherent penalty of low-level code,
but it is a mistake that the lower level makes expressible.

### Where descending a level does help

Large-D SDPA requires an algorithmic restructuring:

```text
S3-H: one output accumulator
      -> repeats QK^T for each D tile
      -> 90.34 ms at D=1024

S3-M: one accumulator per D tile in one KV pass
      -> 25.30 ms at D=1024
```

This supports **high-level first, then medium-level when the algorithm demands
it**. It does not support "lower level emits faster code for the same algorithm."
`S3-H` and `S3-M` perform different amounts of arithmetic. In the only SDPA pair
that holds the algorithm fixed (`S3-M` versus `S3-L`), the lower-level manual
reduction is slower.

Another limit appears in `S3-MP`: replacing the medium loop with
`T.Pipelined` works at `D=128` but does not compile at `D=256/1024` because the
pipeline planner cannot compose with repeated writes to the manually reused V
buffer within shared-memory capacity. A separate V buffer per D tile would need
108,544 bytes at `D=256` and 305,152 bytes at `D=1024`, above Ada's 101,376-byte
limit.

The evidence suggests the following workflow:

1. Express the baseline with `T.gemm`, `T.copy`, `T.reduce_*`, and parameterized
   `T.Pipelined`.
2. Profile work count, traffic, occupancy, and numerics.
3. Descend only to restructure accumulators/data reuse or to access an operation
   unavailable at the high level.
4. Re-run a matched high-versus-explicit control before attributing a gain to
   abstraction level.

This is an engineering recommendation, not a demonstrated convergence theorem.
The repository contains no randomized "start high" versus "start low" search,
no developer/debug-time measurement, and no profiler-run count by abstraction
level. What it demonstrates is that descending enabled the necessary SDPA
dataflow, while matched lower-level implementations did not run faster.

Files to inspect:

- [`phase1_matmul/variants/ABSTRACTION_SPEC.md`](phase1_matmul/variants/ABSTRACTION_SPEC.md)
- [`phase1_matmul/variants/tilelang_abstraction.py`](phase1_matmul/variants/tilelang_abstraction.py)
- [`phase2_fused_sdpa/variants2/fused_tilelang_abstraction.py`](phase2_fused_sdpa/variants2/fused_tilelang_abstraction.py)
- [`phase2_fused_sdpa/variants2/sdpa_tilelang_abstraction.py`](phase2_fused_sdpa/variants2/sdpa_tilelang_abstraction.py)

## 7. Do simpler kernels converge faster than complex kernels?

### Answer

**Usually in meaningful optimization steps, but not necessarily in recorded wall
time.**

- Sum reduction begins at the one-read bandwidth floor. Its identity baseline is
  already within 95% of every trajectory's final best.
- Depthwise convolution reaches its best family in two or three variants after
  moving to a coalesced direct kernel.
- Successful tensor-core searches require more decisions: dtype, accumulator
  precision, tensor-core instruction family, tile, pipeline, layout, fusion, and
  sometimes algorithm decomposition. Counting only custom candidates, TileLang
  matmul takes five variants, Triton fused matmul three, TileLang SDPA four, and
  Triton SDPA six to reach 95% of their respective best custom results.
  CUDA-PTX matmul takes ten.

The descriptive aggregate makes the pattern visible:

| Regime | Mean first-custom iteration to within 5% | Failed custom arms |
|---|---:|---:|
| Sum reduction | 1.0 | 0/12 |
| LayerNorm, hinted calibration | 1.5 | 0/12 |
| Depthwise convolution | 1.5 | 1/14 |
| Fused matmul | 1.75 | 0/10, but gate/artifact reliability is weak |
| SDPA | 3.0 | 4/17 |
| Standard matmul | 4.75 | 10/30 |

The raw seconds are not monotonic with complexity. Sum reduction spends roughly
62--97 seconds per benchmark even though its first row is already optimal, while
matmul trials can be shorter. SDPA time is dominated by repeatedly timing the
slow PyTorch reference. Complexity is therefore better compared by meaningful
variant count and decision axes than by `cum_compute_s` alone.

## 8. Can migrating DSL-1's strategy bring a slower DSL-2 to parity?

### Answer

**Often, yes; not automatically.**

The strongest example is standard matmul. Migrating TileLang's successful
precision/chunk/pipeline recipe into Triton and both CUDA lanes collapses the
headline spread to 1.31x. The same pattern appears in memory-bound kernels:
coalescing, inner-dimension tiling, and L2-resident second passes generally bring
previously lagging DSLs near the common traffic floor.

Parity can still fail for three reasons:

1. **The literal configuration is DSL-dependent.** Pipeline depth and tile shape
   must be retuned after migration.
2. **An interface may hide information.** WMMA does not expose accumulator
   coordinate mapping; TileLang's inferred GEMM fragment layout is likewise not
   directly reducible with an arbitrary manual warp shuffle.
3. **Code-generation residuals remain.** After normalized arithmetic and equal
   tuning, standard matmul still shows roughly 1.10x TileLang over Triton and a
   1.31x four-way spread.

A sound migration experiment should proceed as follows:

```text
Fix semantics and inputs
  -> normalize operand, accumulator, and output dtypes
  -> implement the same algorithm/work count
  -> match tile, threads, and memory layout
  -> verify dynamic MMA/work and bitwise or tolerance-equivalent output
  -> sweep each DSL's native schedule parameters under the same point budget
  -> remeasure winners in fresh processes
```

Without those controls, "strategy migration failed" can simply mean that the
foreign schedule was copied literally, or that one implementation paid casting,
caching, or materialization costs that another excluded.

## 9. Important limitations and reporting corrections

- Phase 1's equal-budget search intentionally excludes swizzles, persistent
  kernels, cross-block split-K, operand caching, and native autotuners. It is a
  controlled shared search space, not an exhaustive ceiling search.
- Phase 2 says the Phase-1 equal-budget result transfers to the fused GEMM because
  the two shapes have equal FLOP counts. Equal FLOPs do not preserve aspect ratio,
  tile count, cache behavior, or optimal scheduling. The fused *matched* campaign
  was run; an equal-budget fused search was not.
- The Phase-2 summary's fused "high versus low" comparison groups TileLang/Triton
  against hand-written CUDA. That confounds abstraction, compiler, and
  implementation. Within-TileLang paired arms are the sound basis for the
  abstraction conclusions here.
- The precision gates are highly operation-dependent. `torch.rand` makes the
  standard-matmul relative budget unusually generous; fused softmax outputs are
  so small that the absolute tolerance dominates; FP16 SDPA output can fail from
  representation rounding alone. Performance and correctness conclusions are
  benchmark-distribution-specific.
- Most controlled FP16 tables use pre-cast inputs. A real benchmark call starts
  from FP32; moving the cast into the timed region costs about 0.21 ms in Phase 1.
  It does not change which DSLs can use Tensor Cores, but it changes headline
  speedups.
- All results cover one Ada GPU and a small set of fixed shapes. Differences below
  roughly 4% are not resolved by the measured process-to-process variation.

## 10. Final conclusions

1. **No DSL is a universal performance upper bound.** More direct hardware access
   does not guarantee faster code.
2. **Most large native-search gaps are not proven ceilings.** The standard-matmul
   gap largely disappears after transferring precision and accumulation choices.
3. **Optimization ideas transfer more reliably than schedules.** Transfer dtype,
   dataflow, work mapping, and accuracy repairs; retune tiles and pipelines.
4. **Triton has the cheapest controlled new-configuration trial, followed by
   TileLang, then CUDA.** Native iteration counts mix search success with early
   stopping and cannot alone rank convergence.
5. **TileLang lower-level code is not intrinsically faster.** High-level forms are
   equal or faster for matched algorithms; medium-level control is valuable when
   it enables a better algorithmic structure.
6. **Simple roofline kernels need fewer meaningful decisions.** Complex tensor-core
   kernels require more search axes, but benchmark/reference cost can obscure that
   in wall-clock logs.
7. **Migration often closes the gap but does not guarantee parity.** Remaining
   differences come from schedule retuning, hidden layout information, compiler
   scheduling, and the tested search budget.

## 11. Artifact map

| Question | Primary files |
|---|---|
| Original six-op board and historical interpretation | [`SIX_OP_ANSWERS.md`](SIX_OP_ANSWERS.md), [`COMPUTE_FRONTIER_FINDINGS.md`](COMPUTE_FRONTIER_FINDINGS.md) |
| Measurement and stopping protocol | [`CONVERGENCE_PROTOCOL.md`](CONVERGENCE_PROTOCOL.md) |
| Per-DSL search paths | each operation's `<dsl>/convergence.csv` and `ITERATIONS*.md` |
| Controlled standard-matmul transfer | [`phase1_matmul/PHASE1_REPORT.md`](phase1_matmul/PHASE1_REPORT.md) |
| Matmul implementation contract | [`phase1_matmul/variants/SPEC.md`](phase1_matmul/variants/SPEC.md) |
| TileLang matmul abstraction levels | [`phase1_matmul/variants/ABSTRACTION_SPEC.md`](phase1_matmul/variants/ABSTRACTION_SPEC.md) |
| Controlled fused matmul and SDPA | [`phase2_fused_sdpa/PHASE2_REPORT.built.md`](phase2_fused_sdpa/PHASE2_REPORT.built.md) |
| Phase-2 factor definitions | [`phase2_fused_sdpa/variants2/SPEC2.md`](phase2_fused_sdpa/variants2/SPEC2.md) |
| Broader memory/index-kernel evidence | [`CROSS_DSL_FINDINGS.md`](CROSS_DSL_FINDINGS.md), with its later corrections/addenda |
