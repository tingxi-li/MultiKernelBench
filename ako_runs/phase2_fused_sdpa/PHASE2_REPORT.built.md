# Phase 2 — fused op and SDPA

Companion to `../phase1_matmul/PHASE1_REPORT.md`. Phase 1 asked whether the
published 7.0× spread across four DSLs on a plain GEMM was a code-generation
result; normalizing arithmetic and tile collapsed it to 1.31×. Phase 2 asks the
same question of two harder cells, where the published gains have more places to
come from: a fused `matmul + GELU + softmax`, and scaled dot-product attention.

Two questions are kept apart throughout, because merging them is the single
error that makes this kind of study meaningless:

* **Abstraction efficiency** — how fast is an *equivalent algorithm on an
  equivalent instruction path* when written at a high level versus a low one.
* **Abstraction-enabled exploration** — whether the higher-level interface made
  it cheaper to find the good schedule at all, and whether the lower-level one
  made it easier to ship a bad one.

Section 8 states them separately and does not mix them.

---

## 0. Controls, and what would falsify each claim

| claim | control that supports it | what would falsify it |
|---|---|---|
| the fused cell's published gain is mostly precision + weight caching, not codegen | torch fp16 with a cached weight, run on the same ladder, timed in the same processes | a DSL kernel beating precision-matched torch by a wide margin at the matched configuration |
| the ladder's features are cheap | `G → GB → GBG → GBGS` measured as four separate builds, not by subtraction | a large `GBGS − G` gap that survives the 5-process CI |
| the weight cache dominates | three weight levels, the third being a kernel that never materializes a transposed copy | `native` costing as much as `uncached` |
| this cell's gate is weak | four deliberately wrong tensors scored against the real gate | those tensors failing on most elements rather than on a fraction of a percent |
| the SDPA reference is not a naive fallback | every backend forced individually and timed | the default matching `math` and nothing else |
| the SDPA gate rejects fp16 output regardless of the kernel | the exact fp32 answer rounded to fp16 and back, scored against the real gate | that round trip passing, which would put the failure back on the kernel |
| the `d=1024` fp16-score cell is infeasible, not under-tuned | the whole attention in fp64 with only the scaled scores rounded to fp16 | that bound landing inside the gate, which would make the failures a tuning problem |
| softmax abstraction level is not what made the incumbent fast | the softmax kernel timed **alone**, and the manual arms built in both access patterns | a large gap between `F1` and a well-written manual arm |

Everything below is absolute runtime in milliseconds. Ratios appear only where a
ratio is the question, and always with the denominator named.

---

## 1. Measurement protocol

Imported from Phase 1 rather than re-derived — `common2.py` pulls `time_kernel`,
`gate_stats`, `median_ci`, `setup_cuda_env`, the gate constants and the
distributions directly out of `phase1_matmul/common.py`, so a Phase-2
millisecond and a Phase-1 millisecond are the same object.

* one variant per process; 5 independent processes per cell
* median of per-process medians, t-based 95% CI
* fixed **warm-up time** (2.0 s), not a fixed iteration count — the card ramps
  210 MHz → ~2.5 GHz and soaks, so a fixed count systematically favours whichever
  variant is already fast
* L2 flushed between trials
* randomized process order, fixed seed, one GPU, nothing else on that GPU
  (`driver2.py` aborts if the card is busy)
* compile time recorded per build, but **cold only for TileLang** — see §1.3,
  which is a correction to an earlier draft of this line
* correctness gate is the harness's own: `|ref−got| ≤ 1e-4 + 1e-4·|ref|`
  elementwise against an fp32 reference

Arm `G` of the fused ladder is the anchor to Phase 1: `2·1024·8192·8192` is
*exactly* `2·2048·8192·4096`, so the fused GEMM has the same arithmetic volume as
Phase 1's, at the same matched schedule.

Full arm-by-arm specification, including every deviation:
[`variants2/SPEC2.md`](variants2/SPEC2.md).

### 1.1 Two tables, and why both are needed

Every result below appears in one of two framings, and they answer different
questions:

* **Natively tuned** (§2.1, §4.2) — each lane's *shipped* artifact, whatever
  schedule its own search arrived at, re-measured here against both denominators.
  This is what the published numbers describe.
* **Matched configuration** (§2.3, §4.3) — every lane running the same geometry,
  the same schedule and the same arithmetic, so that what remains is attributable
  to the DSL rather than to the search.

A DSL can lead one table and trail the other, and in this study that happens.
Quoting either alone is how an abstraction-efficiency claim gets made from an
exploration result, or vice versa.

One honest caveat on the first framing: the shipped artifacts were **not**
produced under a search budget equalized by this study. They are whatever each
lane's original AKO run produced, and those runs differed in length and in how
many iterations were kept. So the natively-tuned table is a faithful record of
*what was shipped*, not a controlled experiment about what each DSL could reach
given equal effort. The matched table is the controlled one.

**The third framing — equal-budget native tuning — is inherited, not repeated.**
Phase 1 ran it: an identical 19-point grid per DSL, each lane's best point
promoted to its row, winners re-measured at the full protocol
(`PHASE1_REPORT.md` §6). It transfers to the fused GEMM by arithmetic identity —
arm `G` is `2·1024·8192·8192`, exactly Phase 1's `2·2048·8192·4096`, at the same
matched schedule (§1) — so Phase 1's equal-budget spread of **1.31×** is the
equal-budget answer for this study's GEMM as well, and it agrees with the
matched-configuration spread to two decimal places.

What that identity does *not* cover is every axis Phase 2 added, and none of
them received an equalized search here: the epilogue ladder, the weight-cache
factor, and all three SDPA factors (algorithm, dtype pair, head dim). For those,
only the two framings above are available. Where a lane's shipped artifact wins
on one of those axes, the honest reading is "this is what its own search
found", not "this is what its DSL affords" — §8.2 is written to that limit.

### 1.2 What may be compared with what

Several configurations are measured in more than one campaign. Those repeats are
the only direct evidence for how far a number travels between campaigns run at
different times on a card that soaks:

| configuration | fused_matched | fused_native | fused_epilogue | fused_cast | spread |
|---|---|---|---|---|---|
| cuda_unlimited/G/cached/smem | 1.8227 | -- | 1.7930 | -- | 1.66% |
| cuda_unlimited/GB/cached/smem | 1.8207 | -- | 1.8217 | -- | 0.06% |
| cuda_unlimited/GBG/cached/smem | 1.8452 | -- | 1.7879 | -- | 3.21% |
| cuda_unlimited/GBGS/cached/smem | 1.8975 | -- | 1.8227 | -- | 4.10% |
| torch:fp16/GBGS/cached/- | 1.5227 | -- | -- | 1.5227 | 0.00% |

5 anchor cells; worst drift **4.10%**, ABOVE Phase 1's measured 3.5% noise floor.

Consequence for reading this report: differences **within** one campaign are comparable at the CI shown; differences **between** campaigns carry an additional ~4% of uncertainty and any effect smaller than that is not resolvable across tables.

A second, independent estimate of the same quantity comes from the incumbent
checks (§2.1, §4.2), which were run once, then re-run later as five processes.
Every subject moved in the same direction between the two sittings, by **+6.5%
to +9.3%** — torch's own fp32 SDPA reference included, at +9.3%. Nothing about
the kernels changed; the card did. This is the reason no absolute millisecond in
this report is compared against one measured in a different sitting, and why
every ratio is formed from subjects timed in the same processes.

### 1.3 Compile time, and a correction

An earlier draft of the protocol list above claimed compile time was measured
"cold, with the DSL disk caches disabled". That was true of TileLang and of
nothing else, and reading the campaign's `compile_s` medians as a ranking
inverts the answer.

**Controlled** — arm `G`, 5 repetitions, a fresh empty cache directory per cold repetition (`compile_cold.py`):

| lane | cold build | warm build | cold ÷ warm |
|---|---|---|---|
| cuda_noptx | 36.39 s [36.00, 37.17] | 0.34 s [0.32, 0.34] | 108× |
| cuda_unlimited | 36.35 s [36.02, 36.91] | 0.35 s [0.34, 0.36] | 104× |
| tilelang | 6.01 s [5.12, 6.33] | n/a — cache disabled in the shipped build path; warm state not reachable | — |
| triton | 1.19 s [1.15, 1.25] | 0.63 s [0.61, 0.63] | 2× |

**Campaign census** — every build the campaigns performed, which is *not* a like-for-like ranking:

| lane | cache policy | builds | median | builds > 5s | of those, first-time source | repeat source |
|---|---|---|---|---|---|---|
| tilelang_abs | disabled | 185 | 7.97 s | 173 | 35 | 138 |
| tilelang | disabled | 155 | 7.29 s | 116 | 22 | 94 |
| triton | kept | 155 | 0.32 s | 0 | 0 | 0 |
| cuda_unlimited | kept | 175 | 0.20 s | 12 | 12 | 0 |
| cuda_noptx | kept | 135 | 0.20 s | 6 | 6 | 0 |
| torch | n/a | 150 | 0.10 s | 0 | 0 | 0 |

Cache policy: **tilelang** — `tilelang.disable_cache()` — every build is cold; **tilelang_abs** — `tilelang.disable_cache()` — every build is cold; **triton** — Triton's own cache, keyed by source hash; **cuda_noptx** — ninja, persistent `TORCH_EXTENSIONS_DIR`; **cuda_unlimited** — ninja, persistent `TORCH_EXTENSIONS_DIR`; **torch** — no compilation step.

Only TileLang calls `tilelang.disable_cache()`, so only TileLang recompiles on
every build. Triton keeps its own source-hash cache, and both CUDA lanes are
served by ninja out of the persistent `TORCH_EXTENSIONS_DIR` that Phase 1 pins.
The campaign census makes the mechanism visible directly, and the last two
columns are the whole argument. For the CUDA lanes, **every** build slower than
5 s is the first time that source was compiled (12 of 12 and 6 of 6) and **no
repeat build is ever slow** — their 0.20 s median is a cache hit, not a compile.
For TileLang the pattern is the opposite: 94 of its slow builds are *repeats* of
a source it had already compiled, which is precisely what a disabled cache looks
like. The two lanes are not being measured in the same state.

So the ~35× spread in the raw medians compares one cold lane against three warm
ones. Cold against cold, on the same arm, the ordering reverses: **TileLang
compiles ~6× faster than nvcc** (6.01 s vs 36.4 s), and Triton faster still
(1.19 s). Triton's cold cost had never been observed in the campaign at all —
0 of 155 builds exceeded 5 s — so it was measured here rather than left blank.

This is corroborated independently. Phase 1's equal-budget grid reports that
compiling its 19 configurations costs "the CUDA lanes around twelve minutes"
(`PHASE1_REPORT.md` §6) — ≈38 s per build, arrived at on a different op, in a
different campaign, and within 4% of the 36.4 s measured here.

Two limits on this. TileLang has no warm figure because its cache is disabled
inside the shipped build path, so reaching a warm state would mean editing the
artifact under measurement; it is reported as not-exercised rather than
estimated. And **search time is not measured anywhere in this study.** The
shipped Triton artifact carries a 13-config `@triton.autotune` decorator whose
first call sweeps all thirteen; that cost is paid inside the timed region's
warm-up, is invisible in `compile_s`, and no equivalent search cost was recorded
for any other lane. Compile time here is the cost of *one* build, not the cost of
arriving at the kernel — which is the quantity §8.2's exploration claim would
actually need, and does not have.

---

## 2. The fused op

### 2.1 What the published number is actually made of

The published cell reports, against a 6.88 ms fp32 reference: tilelang 1.17 ms
(5.29×), triton 2.91 ms (2.36×), cuda_unlimited 5.56 ms (1.24×), cuda_noptx
6.57 ms (1.05×).

Re-measured — every subject timed in the same process against both denominators,
the L2 flush as an explicit control, and the whole check repeated across five
independent processes (`fused_incumbent_check.py`, driven by `incumbent_reps.py`):

| what | ms (no L2 flush) | 95% CI | vs torch fp32 | vs torch fp16 | gate |
|---|---|---|---|---|---|
| torch_fp32 (flush_l2=True) | 6.885 | [6.349, 7.239] | | | |
| torch_fp32 (flush_l2=False) | 6.784 | [6.297, 7.035] | | | |
| torch_fp16 (flush_l2=True) | 1.698 | [1.564, 1.801] | | | |
| torch_fp16 (flush_l2=False) | 1.714 | [1.586, 1.827] | | | |
| tilelang | 1.669 | [1.576, 1.745] | 4.06x | 1.03x | pass |
| triton | 1.202 | [1.141, 1.230] | 5.64x | 1.43x | pass |
| cuda_noptx | 6.840 | [6.511, 6.968] | 0.99x | 0.25x | pass |
| cuda_unlimited | 4.816 | [4.586, 4.931] | 1.41x | 0.36x | **FAIL** (5.9e-04) |

median of per-process medians, t-based 95% CI, 5 independent processes. Intervals that overlap mean the ordering of those two rows is **not resolved** by this measurement.

Three things fall out.

**The fp32 denominator is doing most of the work.** Plain PyTorch, with nothing
but a `.half()` and a weight transposed once, lands at 3.96× of the published
5.29× — and its interval overlaps the shipped tilelang solution's, so which of
the two is faster is **not resolved** by five processes. That non-result is the
finding: the shipped kernel and a two-line PyTorch change are within noise of
each other, so no custom kernel is required to obtain most of the published gain.
An earlier draft of this line read "4.77× — beating the shipped tilelang
solution", from a single-process run; the ordering did not survive repetition,
which is why both incumbent checks now run five processes (`incumbent_reps.py`).

**The shipped ranking does not reproduce.** Published order is tilelang ≫ triton;
measured order is triton < tilelang. The shipped triton artifact is a 13-config
`@triton.autotune` fp16 kernel, newer than the last kept iteration in its own
`convergence.csv` (a tf32 kernel at 2.91 ms). The artifacts have drifted from
their logs, which is why everything here is re-measured rather than quoted.

**One shipped solution does not pass the gate.** `cuda_unlimited` returns a max
absolute error of ~6e-4 against a 1e-4 tolerance — about 6× over — while being
credited with 1.24×. (The exact figure moves between runs in the last digit,
because the kernel's tf32 accumulation order is not fixed; it is never close to
passing.) `cuda_noptx`'s solution is the identity — it calls the reference.

### 2.2 The gate for this cell is close to vacuous

Softmax over 8192 columns produces values around 1/8192 = 1.22e-4. The gate is
`1e-4 + 1e-4·|ref|`, and at that output scale the absolute term dominates
completely: the measured mean |ref| is 1.2207e-4 against a median tolerance of
1.0001e-4, so **the gate permits 82% per-element relative error**. Scoring
deliberately wrong tensors against the real gate:

Softmax output scale: mean 1.221e-04, max 7.426e-04; median gate tolerance 1.000e-04.

| substitute for the true answer | % of elements inside tolerance | gate |
|---|---|---|
| a constant tensor, every element = 1/8192 | 99.31% | fail |
| each row replaced by its own mean | 99.31% | fail |
| all zeros | 10.91% | fail |
| **the reference with its rows reversed** (every row's answer assigned to the wrong row) | 99.86% | fail |

Taking the correct answer and **reversing its rows** — so every row's
distribution is attached to the wrong row — leaves 99.86% of elements inside
tolerance. The gate rejects it only on a fraction of a percent of elements, and
only via the max-error criterion.

This is not a hypothetical concern about the benchmark. The triton agent
recorded it in its own convergence log: its winning iteration is captioned
*"tf32 tensor-core GEMM+gelu fused + row-softmax (tol forgiving via softmax)"*.
It is also visible in every table below, and the ladder isolates it exactly,
because the only difference between the failing and passing rows is the appended
softmax. Every lane, identical GEMM:

| arm | max abs err, all four DSL lanes | gate |
|---|---|---|
| `G` (GEMM only) | 5.14e-4 | **fail** |
| `GB` | 5.14e-4 | **fail** |
| `GBG` | 4.26e-4 | **fail** |
| `GBGS` (= `GBG` + softmax) | 1.53e-7 | pass |

A 3,400× drop in reported error from appending a normalization. The arithmetic
that produced the 5.14e-4 is unchanged and still present; softmax merely
compresses the output range until the absolute tolerance swallows it. Note also
that `torch:fp16` — the same op in the vendor library — fails those same three
arms *worse* than any DSL lane (1.17e-3 at `G`), which shows this is a property
of fp16 accumulation at K=8192 against an fp32 reference, not a defect in any
kernel written here. The benchmark's headline op is the one configuration in
the ladder where that error becomes invisible.

### 2.3 The ladder, at the matched configuration

Phase 1's primary geometry and variant-D schedule transplanted verbatim
(`BM=128 BN=128 BK=32`, 256 threads, fp16 tensor cores, `KC=2048`, 3-stage
pipeline). No autotuning anywhere.

Fused ladder, absolute ms (median of per-process medians), wcache=cached, cast=precast
`*` = fails the 1e-4 gate.

| lane | G (GEMM only) | GB (+ bias) | GBG (+ fused exact GELU) | GBGS (+ softmax (full op)) | GBGS-G |
|---|---|---|---|---|---|
| tilelang | 1.527* | 1.556* | 1.545* | 1.598 | +0.072 |
| triton | 1.672* | 1.680* | 1.730* | 1.773 | +0.100 |
| cuda_unlimited | 1.823* | 1.821* | 1.845* | 1.897 | +0.075 |
| cuda_noptx | 1.923* | 1.950* | 1.946* | 2.031 | +0.108 |
| torch:fp16 | 1.438* | 1.480* | 1.451* | 1.523 | +0.085 |
| torch:fp32 | 5.373 | 5.415 | 5.435 | 5.494 | +0.120 |

Fused ladder, absolute ms (median of per-process medians), wcache=uncached, cast=precast
`*` = fails the 1e-4 gate.

| lane | G (GEMM only) | GB (+ bias) | GBG (+ fused exact GELU) | GBGS (+ softmax (full op)) | GBGS-G |
|---|---|---|---|---|---|
| tilelang | 3.229* | 3.180* | 3.216* | 3.294 | +0.065 |
| triton | 3.255* | 3.400* | 3.448* | 3.486 | +0.231 |
| cuda_unlimited | 3.476* | 3.553* | 3.554* | 3.633 | +0.156 |
| cuda_noptx | 3.631* | 3.654* | 3.678* | 3.746 | +0.115 |
| torch:fp16 | 2.738* | 2.803* | 2.824* | 2.853 | +0.115 |
| torch:fp32 | 6.676 | 6.644 | 6.736 | 6.842 | +0.166 |

The ladder's features are nearly free. Adding a bias, a fused exact-erf GELU,
and an entire second kernel for the softmax costs 0.07–0.23 ms on a 1.5–3.7 ms
op — 4–7%. Fusion is not where the published gain lives.

And at the matched configuration, *no DSL beats precision-matched torch*:

Speedup at arm GBGS, wcache=cached: which denominator you divide by decides the answer

| lane | ms | vs torch fp32 | vs torch fp16 |
|---|---|---|---|
| tilelang | 1.598 | 3.44x | 0.95x |
| triton | 1.773 | 3.10x | 0.86x |
| cuda_unlimited | 1.897 | 2.90x | 0.80x |
| cuda_noptx | 2.031 | 2.70x | 0.75x |
| torch:fp16 | 1.523 | 3.61x | 1.00x |
| torch:fp32 | 5.494 | 1.00x | 0.28x |

Read the two ratio columns together. Divided by the fp32 reference every lane
looks like a 2.7–3.4× win. Divided by the same torch code with `.half()` in
front of it, every lane is a loss. The entire published effect for this cell,
at a matched schedule, is the arithmetic change plus the weight cache.

### 2.4 The weight factor

The study asks for a two-way cached/uncached factor. A third level is carried,
because without it the uncached arm is a straw man: it pays for a transpose that
a competent uncached kernel would never perform.

Weight factor at arm GBGS, absolute ms

| lane | cached | uncached | native | cache worth | native gap |
|---|---|---|---|---|---|
| tilelang | 1.598 | 3.294 | 2.327 | +1.696 | +0.728 |
| triton | 1.773 | 3.486 | 2.304 | +1.713 | +0.531 |
| cuda_unlimited | 1.897 | 3.633 | -- | +1.735 | -- |
| cuda_noptx | 2.031 | 3.746 | -- | +1.715 | -- |
| torch:fp16 | 1.523 | 2.853 | 1.908 | +1.330 | +0.386 |
| torch:fp32 | 5.494 | 6.842 | -- | +1.349 | -- |

`cache worth` = uncached - cached: what re-converting the weight every call costs.
`native gap`   = native  - cached: what a kernel that never materializes a transposed copy gives up. It is the honest floor for 'uncached'; the difference between the two columns is work the cache is credited with but that a competent uncached kernel simply does not do.

The cache is credited with ~1.70 ms when measured against the literal uncached
transpose, but a kernel that simply consumes `W` in its stored `(N,K)` layout
gives up only 0.53–0.73 ms. So roughly 60% of what the cache appears to be worth
is work that a competent uncached implementation does not do in the first place.

Decomposed further, on the torch lane where the pieces can be timed separately:

| piece | ms |
|---|---|
| `W.half().t().contiguous()` — the conversion, if repeated | 1.42 |
| GEMM with the pre-transposed `(K,N)` fp16 weight (NN) | 0.72 |
| GEMM with the native `(N,K)` fp16 weight (NT) | 1.21 |

The cache buys two distinct things that the published number reports as one: it
removes a 1.42 ms conversion, *and* it hands cuBLAS a layout that is 1.67×
faster. Only the first is amortization; the second is a real, permanent
property of the layout.

### 2.5 Epilogue asymmetry between the two CUDA lanes

A WMMA accumulator's `frag.x[e] → (row, col)` mapping is not part of the API
contract, so `cuda_noptx` cannot apply a column-indexed bias in registers and
must stage the tile through shared memory. `cuda_unlimited` wrote its own
`ldmatrix`/`mma.sync` pairing and therefore knows the mapping, so it can add the
bias directly to accumulator registers. Both are measured:

| arm | epilogue=smem (matched) | epilogue=regs (native) | regs advantage |
|---|---|---|---|
| G | 1.793 | 1.802 | -0.009 |
| GB | 1.822 | 1.779 | +0.043 |
| GBG | 1.788 | 1.797 | -0.009 |
| GBGS | 1.823 | 1.831 | -0.008 |

This is a genuine capability difference created by abstraction level, and it is
reported on the *decomposition* side of the ledger, not as "the compiler is
slower".

### 2.6 The activation cast

| lane | cast=precast | cast=in_region | activation cast cost |
|---|---|---|---|
| tilelang | 1.598 | 1.641 | +0.042 |
| triton | 1.773 | 1.814 | +0.041 |
| cuda_unlimited | 1.897 | 1.937 | +0.039 |
| cuda_noptx | 2.031 | 2.043 | +0.012 |

### 2.7 What the hardware counters say about the four GEMMs

Per-kernel Nsight Compute census, steady-state launches only. Durations are
counters-only and are **not** the study's runtimes (Phase 1 established that
ncu's durations sit at the cold-clock transient); the resource columns are what
this table is for.

| lane | epilogue | kernel | regs | occupancy | smem/block | DRAM | grid |
|---|---|---|---|---|---|---|---|
| tilelang | - | main_kernel | 238 | 16.6% | 49152 B | 0.59 GB | 512 |
| tilelang | - | main_kernel | 80 | 43.0% | 64 B | 0.03 GB | 1024 |
| triton | - | _fused_gemm_kernel | 255 | 16.7% | 65536 B | 0.59 GB | 512 |
| triton | - | _softmax_kernel | 54 | 60.7% | 32 B | 0.03 GB | 1024 |
| cuda_unlimited | - | mma_gemm | 168 | 16.7% | 67584 B | 0.59 GB | 512 |
| cuda_unlimited | - | softmax_kernel | 48 | 64.4% | 64 B | 0.03 GB | 1024 |
| cuda_noptx | - | fused_kernel | 167 | 16.7% | 67584 B | 0.59 GB | 512 |
| cuda_noptx | - | softmax_kernel | 48 | 64.4% | 64 B | 0.03 GB | 1024 |
| cuda_unlimited | smem | mma_gemm | 168 | 16.7% | 67584 B | 0.59 GB | 512 |
| cuda_unlimited | smem | softmax_kernel | 48 | 64.7% | 64 B | 0.03 GB | 1024 |
| cuda_unlimited | regs | mma_gemm | 168 | 16.7% | 67584 B | 0.59 GB | 512 |
| cuda_unlimited | regs | softmax_kernel | 48 | 64.5% | 64 B | 0.03 GB | 1024 |

One-shot host-side weight conversion is excluded (it launches once, not once per call); it is reported separately in the JSON as `setup_dram_GB`.

The four lanes move **identical DRAM traffic** (0.59 GB in the GEMM, 0.03 GB in
the softmax) and land at **identical occupancy** — every GEMM at 16.6–16.7%,
which on sm_89 is exactly one 256-thread block resident per SM. They arrive there
by different routes: tilelang and triton are register-limited (238 and 255
registers × 256 threads against a 65536-register file), while the two CUDA lanes
are limited twice over, by registers and by a 67584 B shared-memory footprint
that cannot fit two blocks in 101376 B. Same envelope, different binding
constraint.

So neither bandwidth nor occupancy explains the spread between them, and register
pressure runs *inversely* to performance: the two fastest lanes carry the most
registers (triton at the 255 ceiling, tilelang at 238) while the two slowest
carry the fewest (168 and 167). The ranking is set by instruction scheduling
inside an identical resource envelope, not by a resource limit — which is the
concrete mechanism behind §2.3's finding that the compiler-backed lanes lead the
hand-written ones. Spending registers aggressively is what the compilers did
right, and it is the opposite of the usual "reduce register pressure" heuristic.

Note also that `cuda_unlimited`'s two epilogues compile to the **same 168
registers**. The register-resident bias does not reduce register pressure; it
removes a shared-memory round trip. That is consistent with §2.5's small and
bias-conditional effect.

## 3. TileLang abstraction study — softmax reduction level

Everything held fixed except the reduction: the GEMM is the matched kernel with
a `T.Parallel` bias + erf-GELU epilogue, the softmax is one row per block at 256
threads, output fp32.

The whole-op number cannot answer this question. The softmax is roughly 5% of
the op, so a 3% difference between reduction styles is 0.002 ms — under the
noise floor. The softmax kernel is therefore timed **alone**, on the real
post-GELU scratch so the reduction sees the real value distribution.

The manual arms are built in **both** access patterns. `F2/F3/F4` index
`X[bx, tid*ept + k]` — each thread walks 32 consecutive floats, so a warp
touches 32 separate 128 B segments and the loads do not coalesce. That is the
indexing the shipped incumbent uses. `F2c/F3c/F4c` are identical in every other
respect and index `X[bx, k*th + tid]`.

| arm | reduction | access pattern | softmax kernel alone (ms) | whole op (ms) |
|---|---|---|---|---|
| F1 | T.reduce_max / T.reduce_sum | n/a (T.copy) | 0.0881 | 1.549 |
| F2 | manual smem tree | thread-contiguous | 0.2703 | 1.591 |
| F3 | manual warp shuffle, exp recomputed | thread-contiguous | 0.2714 | 1.604 |
| F4 | warp shuffle + cached exp (incumbent) | thread-contiguous | 0.2314 | 1.576 |
| F2c | manual smem tree, coalesced | coalesced | 0.0983 | 1.561 |
| F3c | manual warp shuffle, coalesced | coalesced | 0.0975 | 1.559 |
| F4c | warp shuffle + cached exp, coalesced | coalesced | 0.0870 | 1.563 |

**Abstraction efficiency: approximately zero.** `F1` — a two-line
`T.reduce_max`/`T.reduce_sum` — is 0.0881 ms against `F4c`'s 0.0870 ms, a
hand-written warp-shuffle reduction with cached exponentials: **1.2% apart**,
and the high-level form is the slower one by an amount comparable to the timer's
1.024 µs quantum. Dropping from a shared-memory tree to warp shuffles
(`F2c → F3c`) is worth 0.8%. Caching the exponentials (`F3c → F4c`) is worth
10.7%, and that is an algorithmic change — one fewer `exp` per element — not an
abstraction one.

**What the low-level form actually bought was the opportunity to get the access
pattern wrong**, and the shipped incumbent took it. Holding the reduction style
fixed and changing only the index expression:

| reduction style | non-coalesced | coalesced | penalty |
|---|---|---|---|
| smem tree | `F2` 0.2703 | `F2c` 0.0983 | 2.75× |
| warp shuffle | `F3` 0.2714 | `F3c` 0.0975 | 2.78× |
| warp shuffle + cached exp | `F4` 0.2314 | `F4c` 0.0870 | 2.66× |

The indexing choice is worth **2.7×**; every abstraction-level choice on the
same rows is worth under 11%, and the top-to-bottom choice is worth 1.2%. The
high-level form gets coalescing from `T.copy` and *cannot express the mistake* —
which is the one place in this study where the abstraction level demonstrably
changed the outcome, and it did so by removing a degree of freedom rather than
by generating better code.

So the conclusion the existing artifact suggested — "high-level versus
warp-level softmax contributes only ~1–3%, therefore softmax abstraction should
not be credited for the overall result" — is confirmed, and is if anything
understated: at the whole-op level the entire `F1 … F4c` span is 1.549–1.604 ms,
a 3.6% spread that is itself dominated by the indexing factor rather than the
abstraction one, while the weight cache (§2.4) moves the same op from 1.549 ms
to 3.197 ms. One factor is worth 3.6% at its widest and ~1% on the axis actually
under study; the other is worth 106%.

---

## 4. SDPA

### 4.1 What the reference actually runs

The recurring claim about this cell is that the reference falls back to the
naive `math` backend at `D=1024`, making the published speedups meaningless.
Measured by forcing each backend individually (`sdpa_reference_audit.py`):

| head dim | dtype | default (ms) | attributed to | flash | mem_efficient | math |
|---|---|---|---|---|---|---|
| 128 | float32 | 5.56 | mem_efficient | unavailable | 5.67 | 13.05 |
| 256 | float32 | 13.97 | mem_efficient | unavailable | 13.81 | 20.32 |
| 1024 | float32 | 56.90 | mem_efficient | unavailable | 55.55 | 67.20 |
| 128 | float16 | 1.39 | flash | 1.35 | 2.01 | 17.03 |
| 256 | float16 | 2.96 | flash | 3.10 | 5.60 | 25.00 |
| 1024 | float16 | 22.43 | mem_efficient | unavailable | 21.83 | 82.89 |

The claim is **false as stated**. The `math` backend is never selected at any
head dim or dtype. FlashAttention is genuinely unavailable at `D=1024` (its
head-dim cap is 256), but the mem-efficient backend is available and is what the
default path selects, at 55.55 ms against `math`'s 67.20 ms. The reference is a
real tiled kernel, not a materializing fallback.

What *is* true, and matters more, is that FlashAttention is unavailable at
**every** head dim in fp32 — it requires fp16/bf16 — so the published fp32
denominator is always the mem-efficient backend. The published comparison is
therefore an fp32 mem-efficient kernel against an fp16 custom kernel, and the
dtype, not the backend, is what the ratio is measuring.

### 4.2 The incumbent solutions, precision-matched

| what | ms | 95% CI | vs torch fp32 | vs torch fp16 | gate |
|---|---|---|---|---|---|
| torch_sdpa_fp32 | 61.580 | [61.421, 61.670] | | | _published denominator_ |
| torch_sdpa_fp16 | 23.159 | [23.076, 23.326] | | | _precision-matched denominator_ |
| tilelang | 24.398 | [24.352, 24.492] | 2.52x | 0.95x | pass |
| triton | 36.207 | [36.025, 36.297] | 1.70x | 0.64x | pass |
| cuda_noptx | 62.500 | [62.341, 62.680] | 0.99x | 0.37x | pass |
| cuda_unlimited | 55.283 | [55.192, 55.376] | 1.11x | 0.42x | pass |

median of per-process medians, t-based 95% CI, 5 independent processes. Intervals that overlap mean the ordering of those two rows is **not resolved** by this measurement.

The published tilelang gain for this cell is 2.78×; re-measured it is 2.52×.
The fp16 cast alone accounts for 2.66× — *more than the entire measured gain*.
`torch` handed operands that are **already fp16**, with no custom kernel at all,
runs at 23.16 ms and is faster than all four shipped solutions (24.40 / 36.21 /
55.28 / 62.50). Against that denominator every shipped solution is a loss:
0.95× / 0.64× / 0.42× / 0.37×. Unlike the fused cell's near-tie, these
orderings are resolved: no shipped solution's interval reaches torch fp16's.

Note carefully what that 23.16 ms is and is not. It is torch's fp16 attention
*kernel*, timed on operands converted beforehand — the right denominator for
"how good is torch's kernel", and the one used here because the shipped solutions
also cast inside their own `run()`. It is **not** what torch achieves on this
benchmark's actual fp32 inputs, which is measured separately in §4.3 and is
slower, because the conversion then has to be materialized. The two numbers
answer different questions and are never compared to each other in this report.

`cuda_noptx` shipped the identity — it calls `F.scaled_dot_product_attention` —
on a stated precision concern: *"FP16 internals: FP16 Q/K dot product errors
exceed atol=1e-4 at D=1024"*. That claim is **correct about the thing it names
and the wrong conclusion to draw from it.** §4.4 derives the floor independently:
with the scores rounded to fp16, `D=1024` bottoms out at 1.65e-4 against a
~1.5e-4 budget, so an all-fp16 path really is unreachable at that head dim — the
concern was right. But the operands and the scores are separable factors. Keeping
Q/K as fp16 tensor-core *operands* while accumulating and storing the scores in
fp32 passes the same gate at 4.6e-5, which is what the tilelang solution does and
what the `fp32/fp16` arms below do. The lane gave up 2.5× of available
performance by treating one precision decision as though it forced the other.

### 4.3 Cross-DSL: two algorithms × three head dims × two dtypes

Both algorithms are semantically identical in every lane. `K3` materializes the
`(B,H,S,S)` score tensor to global memory; `FLASH` never does. The tile is held
identical across the three dtype pairs at a given `(algo, d)`, so the dtype
factor cannot be confounded with a tile change.

**One convention had to be forced before this table meant anything.** All four
lanes take `Q,K,V` as fp32 kernel parameters and convert on the global→shared
path. The tilelang lane originally converted on the *host*, inside the timed
region, and no other lane did — worth up to 50% of the measured time at `d=128`.
That is not cheating (it inflated its own numbers) but it makes a cross-DSL table
meaningless, so it was fixed before any number below was taken. Every
`max_abs_err` is bit-identical across the fix, confirming only the placement of
the conversion changed. The fix is not a free win: reading fp32 operands doubles
the global bytes, and at `K3`/`d=1024` that costs slightly more than the removed
host cast saved. It is here for comparability, not for speed.

**The same rule is applied to the denominators, and it costs torch more.** Every
lane in this table receives the benchmark's fp32 `Q,K,V`. A custom kernel can
convert on the global→shared path, so the conversion rides a load it was going to
issue anyway. `F.scaled_dot_product_attention` cannot: its fp16 path requires
fp16 tensors, so the cast must be materialized to global memory first — three
tensors read at fp32 and rewritten at fp16 — and `TORCH_F16` below therefore pays
a real cost that the custom lanes avoid by construction.

That is not an artifact; it is one of the few concrete architectural advantages
of writing the kernel yourself that this study found, and it is reported as such.
But it means the fp16 denominator must be read as *"what torch achieves on fp32
inputs"*, not as *"how fast torch's fp16 kernel is"*. The latter is the 23.16 ms
figure in §4.2, and the honest comparison at `d=1024` is against all three:

| denominator | what it measures | gate |
|---|---|---|
| `TORCH_F32` | the published denominator | passes |
| `TORCH_F16` (below) | best torch path from fp32 inputs, cast included | **fails** (§4.4) |
| torch fp16, operands pre-cast (§4.2) | torch's fp16 kernel alone | **fails** (§4.4) |

#### What the head-dim sweep shows

This is the one place in the study where custom kernels beat precision-matched
torch — and the sweep says exactly when:

| head dim | is FlashAttention available to torch? | best gate-passing custom kernel vs `TORCH_F16` |
|---|---|---|
| 128 | **yes** | 1.12× |
| 256 | **yes** | 1.01× |
| 1024 | **no** (head-dim cap is 256) | **1.74×** |

Where a good vendor kernel exists, four DSLs and two hand-written CUDA lanes
converge on parity with it. Where one does not — above FlashAttention's head-dim
cap, leaving torch on the mem-efficient backend — the same code wins by 1.74×.
The advantage is not a property of the language; it is the size of the hole in
the vendor library. This is why the study runs the smaller head dims at all: at
`d=1024` alone the conclusion would have read as a DSL result.

**The algorithm ranking also inverts across the sweep, in every lane at once.**
At `d=128` the single fused kernel wins everywhere (`FLASH` 2.49–3.79 ms against
`K3` 3.94–6.19). At `d=1024` it loses in three lanes of four — `cuda_unlimited`
21.33 vs 29.93, `triton` 18.98 vs 49.92, `cuda_noptx` 30.45 vs 34.83 — and ties
in the fourth. Never materializing the `(B,H,S,S)` score tensor stops paying once
the fused accumulator's register pressure and the repeated V traffic cost more
than the write. That is an algorithmic result, reproduced independently in four
lanes, and it agrees with the TileLang-only decomposition axis in §5.2.

**One lane is not internally consistent across dtypes.** `triton` is the fastest
lane at `d=1024` with fp32 scores (18.98 ms) and by far the slowest with fp32
probabilities (101.01 ms `K3`, 65.40 ms `FLASH`; the next worst is 55.48). Its
`FLASH fp32/fp32` at `d=256` is 43.30 ms against 10.24–10.70 elsewhere. The cause
is §4.4's tf32 finding: `pdtype=fp32` forbids fp16 tensor cores, and rather than
accept tf32's one-sided truncation the lane falls back to IEEE fp32 on CUDA
cores. That is the correct choice for accuracy and an expensive one for speed,
and it is why the fp32/fp32 column should not be read as a codegen ranking.

**head_dim = 128**

| lane | K3 fp32/fp32 | K3 fp32/fp16 | K3 fp16/fp16 | FLASH fp32/fp32 | FLASH fp32/fp16 | FLASH fp16/fp16 |
|---|---|---|---|---|---|---|
| cuda_noptx | 9.35 | 6.19 | 5.51 | 5.81 | 3.79 | 3.79 |
| cuda_unlimited | 8.38 | 5.19 | 4.28 | 5.12 | 2.85 | 2.78 |
| tilelang | 9.53 | 5.51 | 4.95 | 5.27 | 3.00 | 3.03 |
| triton | 15.49 | 5.25 | 3.94 | 4.28 | 2.49 | 2.54 |

| denominator | ms | best gate-passing custom kernel vs it |
|---|---|---|
| torch fp32 (published denominator) | 6.39 | 2.57x |
| torch fp16 (precision-matched) | 2.78 | 1.12x |
| torch, math backend forced | 14.41 | 5.79x |

**head_dim = 256**

| lane | K3 fp32/fp32 | K3 fp32/fp16 | K3 fp16/fp16 | FLASH fp32/fp32 | FLASH fp32/fp16 | FLASH fp16/fp16 |
|---|---|---|---|---|---|---|
| cuda_noptx | 15.31 | 9.64 | 8.99 | 10.70 | 7.11 | 7.14 |
| cuda_unlimited | 12.89 | 7.63 | 6.73 | 10.24 | 5.79 | 5.73 |
| tilelang | 14.95 | 8.48 | 7.88 | 10.28 | 5.99 | 6.03 |
| triton | 28.09 | 6.59 | 5.97 | 43.30 | 7.39 | 7.40 |

| denominator | ms | best gate-passing custom kernel vs it |
|---|---|---|
| torch fp32 (published denominator) | 15.40 | 2.69x |
| torch fp16 (precision-matched) | 5.80 | 1.01x |
| torch, math backend forced | 22.52 | 3.93x |

**head_dim = 1024**

| lane | K3 fp32/fp32 | K3 fp32/fp16 | K3 fp16/fp16 | FLASH fp32/fp32 | FLASH fp32/fp16 | FLASH fp16/fp16 |
|---|---|---|---|---|---|---|
| cuda_noptx | 51.23 | 30.45 | 29.74* | 53.45 | 34.83 | 34.78* |
| cuda_unlimited | 39.73 | 21.33 | 20.46* | 55.48 | 29.93 | 29.94* |
| tilelang | 49.86 | 26.27 | 25.64* | 55.50 | 25.57 | 25.76* |
| triton | 101.01 | 18.98 | 18.08* | 65.40 | 49.92 | 50.26* |

| denominator | ms | best gate-passing custom kernel vs it |
|---|---|---|
| torch fp32 (published denominator) | 60.71 | 3.20x |
| torch fp16 (precision-matched) | 33.00 | 1.74x |
| torch, math backend forced | 71.97 | 3.79x |

`*` = fails the 1e-4 gate.

The `best gate-passing custom kernel` column uses the fastest cell in the table above at that head dim that actually passes the gate, so a number that only exists because it is wrong cannot become the speedup.

### 4.4 The score dtype, and one cell that no implementation can reach

The spec asks for score and probability dtypes to be fixed independently. Two of
the nine (algorithm × dtype) cells at `d=1024` are **not reachable by any
implementation**, and that is a property of the benchmark's inputs, not of the
kernels. All four lanes fail exactly the same two cells, independently.

The floor was derived directly rather than inferred from failures: computing the
entire attention in fp64 and rounding *only the scaled scores* to fp16 — the most
accurate implementation of "fp16 scores" that can exist — still leaves

| head dim | elements over the gate | best achievable max error |
|---|---|---|
| 128 | 0 / 6.7e7 | comfortably inside |
| 256 | 0 / 1.3e8 | comfortably inside |
| 1024 | 19 / 5.37e8 | **1.6527e-4** vs a ~1.5e-4 budget |

The cause is the input distribution. `torch.rand` gives `Q,K ~ U(0,1)`, so every
score is a sum of `d` non-negative products and the peak score grows like
`√d/4`. At `d=1024` that crosses 8, which is exactly where fp16's ulp doubles.
The gate is then tighter than the representation. Nothing a kernel author can do
changes this; the two cells are reported as failures with the reason attached,
not tuned at.

This is the same effect as §2.2 seen from the other side. There, softmax
compressed the output until the tolerance became meaningless; here, the input
distribution inflates an intermediate until the tolerance becomes unsatisfiable.
Both are properties of the harness's fixed `torch.rand` inputs, and both decide
correctness verdicts that read as though they were about the kernel.

**And the precision-matched denominator does not pass this gate at all.**
`TORCH_F16` — `F.scaled_dot_product_attention` on `.half()` operands, upcast to
fp32 on return — fails at every head dim, at a max error of 2.79e-4 with 19.1% of
elements outside tolerance. That is not an accuracy defect in torch's attention.
Take the *exact* fp32 answer, store it in fp16, and read it back, changing
nothing else:

| head dim | max abs err | elements outside the gate | gate |
|---|---|---|---|
| 128 | 2.4414e-4 | 19.1% | fail |
| 1024 | 2.4414e-4 | 19.1% | fail |

Identical failure fraction, and 2.4414e-4 is exactly half an fp16 ulp at 0.5.
The attention output is a softmax-weighted average of `V ~ U(0,1)`, so it sits at
`|ref| ≈ 0.5`, where the gate allows `1e-4 + 1e-4·0.5 = 1.5e-4` — less than fp16's
own rounding. **Storing the answer in fp16 is by itself a gate failure, before any
kernel runs.** The custom lanes pass only because they write fp32 output and keep
the scores in fp32.

So on this op "match the reference's precision" and "pass the harness's gate" are
mutually exclusive, and a speedup over `TORCH_F16` is a speedup over an
implementation the harness would reject. Both denominators are therefore carried
throughout §4.3 and neither is presented alone: fp32 is the only gate-passing
torch configuration, and fp16 is the only precision-matched one.

Read together with §2.2 this is one finding, not two. The same fixed absolute
tolerance of `1e-4` is applied to a fused-op output of magnitude 1.2e-4 and an
SDPA output of magnitude 0.5 — a factor of 4096 apart. At the small scale it
permits 82% per-element relative error; at the large scale it rejects the
correctly rounded answer. A fixed absolute tolerance cannot mean the same thing
for two ops whose outputs differ by three orders of magnitude, and in this
benchmark it does not.

A second precision finding, found independently by the tilelang and triton lanes:
on sm_89, fp32 operands entering a tensor-core op are **bit-truncated to tf32,
not rounded**. The measured one-sided relative error is −6.4e-4 against −2.9e-6
for IEEE fp32, and because the contraction is over non-negative terms it does not
cancel. A tf32 "fp32-scores" arm would therefore have been ~130× *less* accurate
than the fp16 arm it was supposed to bound. Both lanes use IEEE fp32 on CUDA
cores for the fp32-score arms instead, which is why those arms are slower than a
naive reading of "fp32 tensor cores" would predict.

### 4.5 Per-kernel counters

The spec asks for score-tensor DRAM traffic, register pressure and occupancy per
kernel — the only honest way to compare a three-kernel algorithm against a
one-kernel one, since a single fused number would hide exactly the thing being
measured.

**The materialization tax is a constant, and that is the whole story of the
algorithm inversion.** The score tensor `S` is `(B,H,S,S)` fp32 = 1.07 GB and the
probabilities `P` are 0.54 GB at fp16, *whatever the head dimension is*. `K3`
must write `S`, read it, write `P` and read it — 3.22 GB of traffic that does not
depend on `d` at all. `FLASH` moves none of it. Measured, averaged over the four
lanes:

| head dim | algo | kernels | total algorithm DRAM | of which score tensor | vs `FLASH` |
|---|---|---|---|---|---|
| 128 | `FLASH` | 1 | 1.05 GB | 0.00 GB (0%) | — |
| 128 | `K2` | 2 | 2.80 GB | 1.07 GB (38%) | 2.7× |
| 128 | `K3` | 3 | 4.17 GB | 3.22 GB (**77%**) | 4.0× |
| 256 | `FLASH` | 1 | 2.12 GB | 0.00 GB (0%) | — |
| 256 | `K2` | 2 | 4.35 GB | 1.07 GB (25%) | 2.1× |
| 256 | `K3` | 3 | 5.24 GB | 3.22 GB (61%) | 2.5× |
| 1024 | `FLASH` | 1 | 9.07 GB | 0.00 GB (0%) | — |
| 1024 | `K2` | 2 | 12.10 GB | 1.07 GB (9%) | 1.33× |
| 1024 | `K3` | 3 | 12.44 GB | 3.22 GB (**26%**) | 1.37× |

At `d=128` materializing the scores *is* the algorithm's memory cost — 77% of it —
and `K3` moves 4× what `FLASH` does. At `d=1024` the same 3.22 GB is 26%, because
the operand traffic has grown around it, and the gap narrows to 1.37×. The fused
kernel's advantage decays not because it got worse but because the penalty it
avoids stopped being the dominant term. `K2` sits exactly where its structure
predicts: it never creates `S`, so it pays 1.07 GB instead of 3.22 GB, one third
of `K3`'s tax — which is why `S2` beats `S1` at every head dim in §5.2.

**But DRAM is not what decides the ranking at `d=1024`, and the counters say so.**
There `K3` still moves 1.37× more traffic than `FLASH` and is nonetheless *faster*
in three lanes of four (§4.3). The reason is in the occupancy column: the fused
kernel carries one register budget for the whole algorithm — 238–255 registers at
15–16.5% occupancy — while the decomposition lets each kernel take the budget it
actually needs, and the softmax kernel, which is pure bandwidth, runs at **19–22
registers and 97–98% occupancy** in every lane. Fusion buys away a fixed 3.22 GB
and pays for it with a register budget sized by the worst phase. Which side wins
is a function of the head dimension, not of the language or the abstraction level
— all four lanes reproduce the same crossover.

| lane | algo | d | scores/probs | kernel | regs | occupancy | DRAM | share of algo DRAM |
|---|---|---|---|---|---|---|---|---|
| tilelang | K3 | 128 | fp32/fp16 | main_kernel | 90 | 24.6% | 1.57 GB | 38% |
| tilelang | K3 | 128 | fp32/fp16 | main_kernel | 56 | 73.3% | 1.56 GB | 37% |
| tilelang | K3 | 128 | fp32/fp16 | main_kernel | 254 | 16.4% | 1.05 GB | 25% |
| tilelang | FLASH | 128 | fp32/fp16 | main_kernel | 238 | 8.3% | 1.05 GB | 100% |
| triton | K3 | 128 | fp32/fp16 | _k3_qk_kernel | 248 | 16.4% | 1.56 GB | 37% |
| triton | K3 | 128 | fp32/fp16 | _k3_softmax_kernel | 19 | 98.1% | 1.56 GB | 38% |
| triton | K3 | 128 | fp32/fp16 | _k3_pv_kernel | 124 | 33.0% | 1.04 GB | 25% |
| triton | FLASH | 128 | fp32/fp16 | _flash_kernel | 128 | 16.6% | 1.05 GB | 100% |
| cuda_unlimited | K3 | 128 | fp32/fp16 | qk3_kernel | 144 | 16.3% | 1.55 GB | 37% |
| cuda_unlimited | K3 | 128 | fp32/fp16 | sm3_kernel | 22 | 97.0% | 1.57 GB | 38% |
| cuda_unlimited | K3 | 128 | fp32/fp16 | pv3_kernel | 146 | 16.5% | 1.05 GB | 25% |
| cuda_unlimited | FLASH | 128 | fp32/fp16 | flash_kernel | 254 | 16.3% | 1.05 GB | 100% |
| cuda_noptx | K3 | 128 | fp32/fp16 | k3_qk_kernel | 72 | 41.3% | 1.56 GB | 37% |
| cuda_noptx | K3 | 128 | fp32/fp16 | k3_softmax_kernel | 21 | 97.2% | 1.57 GB | 38% |
| cuda_noptx | K3 | 128 | fp32/fp16 | k3_pv_kernel | 80 | 40.8% | 1.05 GB | 25% |
| cuda_noptx | FLASH | 128 | fp32/fp16 | flash_kernel | 128 | 16.7% | 1.05 GB | 100% |
| tilelang | K2 | 128 | fp32/fp16 | main_kernel | 255 | 16.5% | 1.75 GB | 62% |
| tilelang | K2 | 128 | fp32/fp16 | main_kernel | 254 | 16.6% | 1.05 GB | 38% |
| tilelang | K3 | 256 | fp32/fp16 | main_kernel | 166 | 24.9% | 2.11 GB | 40% |
| tilelang | K3 | 256 | fp32/fp16 | main_kernel | 56 | 73.3% | 1.56 GB | 30% |
| tilelang | K3 | 256 | fp32/fp16 | main_kernel | 254 | 16.6% | 1.58 GB | 30% |
| tilelang | FLASH | 256 | fp32/fp16 | main_kernel | 255 | 8.4% | 2.12 GB | 100% |
| triton | K3 | 256 | fp32/fp16 | _k3_qk_kernel | 236 | 16.5% | 2.10 GB | 40% |
| triton | K3 | 256 | fp32/fp16 | _k3_softmax_kernel | 19 | 96.7% | 1.56 GB | 30% |
| triton | K3 | 256 | fp32/fp16 | _k3_pv_kernel | 124 | 32.9% | 1.57 GB | 30% |
| triton | FLASH | 256 | fp32/fp16 | _flash_kernel | 203 | 16.5% | 2.12 GB | 100% |
| cuda_unlimited | K3 | 256 | fp32/fp16 | qk3_kernel | 142 | 16.4% | 2.10 GB | 40% |
| cuda_unlimited | K3 | 256 | fp32/fp16 | sm3_kernel | 22 | 97.4% | 1.57 GB | 30% |
| cuda_unlimited | K3 | 256 | fp32/fp16 | pv3_kernel | 146 | 16.6% | 1.58 GB | 30% |
| cuda_unlimited | FLASH | 256 | fp32/fp16 | flash_kernel | 245 | 16.5% | 2.12 GB | 100% |
| cuda_noptx | K3 | 256 | fp32/fp16 | k3_qk_kernel | 72 | 39.8% | 2.10 GB | 40% |
| cuda_noptx | K3 | 256 | fp32/fp16 | k3_softmax_kernel | 21 | 97.3% | 1.57 GB | 30% |
| cuda_noptx | K3 | 256 | fp32/fp16 | k3_pv_kernel | 80 | 41.2% | 1.58 GB | 30% |
| cuda_noptx | FLASH | 256 | fp32/fp16 | flash_kernel | 180 | 16.7% | 2.12 GB | 100% |
| tilelang | K2 | 256 | fp32/fp16 | main_kernel | 255 | 17.8% | 2.77 GB | 64% |
| tilelang | K2 | 256 | fp32/fp16 | main_kernel | 254 | 16.4% | 1.58 GB | 36% |
| tilelang | K3 | 1024 | fp32/fp16 | main_kernel | 126 | 24.8% | 5.35 GB | 44% |
| tilelang | K3 | 1024 | fp32/fp16 | main_kernel | 56 | 73.3% | 1.56 GB | 13% |
| tilelang | K3 | 1024 | fp32/fp16 | main_kernel | 254 | 16.4% | 5.13 GB | 43% |
| tilelang | FLASH | 1024 | fp32/fp16 | main_kernel | 255 | 15.0% | 8.96 GB | 100% |
| triton | K3 | 1024 | fp32/fp16 | _k3_qk_kernel | 236 | 16.6% | 5.71 GB | 46% |
| triton | K3 | 1024 | fp32/fp16 | _k3_softmax_kernel | 19 | 98.5% | 1.67 GB | 13% |
| triton | K3 | 1024 | fp32/fp16 | _k3_pv_kernel | 124 | 33.2% | 5.12 GB | 41% |
| triton | FLASH | 1024 | fp32/fp16 | _flash_kernel | 255 | 15.8% | 9.18 GB | 100% |
| cuda_unlimited | K3 | 1024 | fp32/fp16 | qk3_kernel | 128 | 33.0% | 5.73 GB | 46% |
| cuda_unlimited | K3 | 1024 | fp32/fp16 | sm3_kernel | 22 | 97.1% | 1.68 GB | 13% |
| cuda_unlimited | K3 | 1024 | fp32/fp16 | pv3_kernel | 146 | 16.5% | 5.13 GB | 41% |
| cuda_unlimited | FLASH | 1024 | fp32/fp16 | flash_kernel | 128 | 29.9% | 9.56 GB | 100% |
| cuda_noptx | K3 | 1024 | fp32/fp16 | k3_qk_kernel | 72 | 41.3% | 5.81 GB | 46% |
| cuda_noptx | K3 | 1024 | fp32/fp16 | k3_softmax_kernel | 21 | 96.8% | 1.68 GB | 13% |
| cuda_noptx | K3 | 1024 | fp32/fp16 | k3_pv_kernel | 80 | 41.1% | 5.17 GB | 41% |
| cuda_noptx | FLASH | 1024 | fp32/fp16 | flash_kernel | 254 | 14.5% | 8.56 GB | 100% |
| tilelang | K2 | 1024 | fp32/fp16 | main_kernel | 255 | 15.8% | 6.97 GB | 58% |
| tilelang | K2 | 1024 | fp32/fp16 | main_kernel | 254 | 16.5% | 5.13 GB | 42% |

---

## 5. TileLang SDPA abstraction — two axes, kept apart

Comparing `S1` (three kernels) against `S3-H` (one fused kernel) and calling the
difference "abstraction" is the specific error this section is arranged to
prevent: those two differ in materialization, register pressure, V re-read count
and kernel count all at once.

All arms share one tile — `block_M=64, block_N=64, D_TILE=128, threads=256` — at
every head dim. It is *smaller* than the shipped incumbent's `block_N=128`, and
that shrink is a reported cost, not a free choice: at `block_N=128` the
un-pipelined library arm already occupies 100352 B of sm_89's 101376 B, so
neither the pipelined arms (133120 B) nor `S3-L`'s fp32 staging tile (131328 B)
could be built at all. Measured back to back on an identical body, the shrink
costs `S3-M` +11.4% at `d=128`. All four arms pay it equally, which is what the
axis requires.

**Within-kernel abstraction axis.** One algorithm, one tile (`block_M=64, block_N=64, D_TILE=128, threads=256`) at every head dim; only how the kernel is written changes.

| arm | what changes | d=128 | d=256 | d=1024 |
|---|---|---|---|---|
| S3-H | single accumulator, `T.Pipelined`, `T.reduce_*` | 3.01 | 8.66 | 90.34 |
| S3-M | one accumulator per d-tile, plain loop, `T.reduce_*` | 3.08 | 6.11 | 25.30 |
| S3-MP | `S3-M` + `T.Pipelined` on the same manual structure | 3.00 | **does not build** | **does not build** |
| S3-L | `S3-M` + manual `T.shfl_down` reductions | 3.47 | 6.38 | 26.12 |

Of those four, only **`S3-M` vs `S3-L`** isolates expression level alone — same loop structure, same accumulators, library reduction versus hand-written warp shuffles:

| head dim | `S3-M` (`T.reduce_*`) | `S3-L` (manual shuffles) | manual costs |
|---|---|---|---|
| 128 | 3.08 | 3.47 | +13.0% |
| 256 | 6.11 | 6.38 | +4.3% |
| 1024 | 25.30 | 26.12 | +3.3% |

**Algorithmic decomposition axis — this is NOT an abstraction result.** These differ in kernel count, materialization, V re-reads and occupancy all at once.

| arm | what changes | d=128 | d=256 | d=1024 |
|---|---|---|---|---|
| S1 | three kernels, `S` materialized to global | 5.23 | 8.43 | 26.05 |
| S2 | two kernels, `S` never leaves registers | 4.58 | 7.77 | 24.85 |
| S3-M | one accumulator per d-tile, plain loop, `T.reduce_*` | 3.08 | 6.11 | 25.30 |

| head dim | best `S3` | best decomposed | fused wins by |
|---|---|---|---|
| 128 | S3-MP (3.00) | S2 (4.58) | 1.53x |
| 256 | S3-M (6.11) | S2 (7.77) | 1.27x |
| 1024 | S3-M (25.30) | S2 (24.85) | **loses**, 1.02x slower |

### 5.1 Reading the within-kernel axis correctly

The four `S3` arms are *not* four points on one abstraction scale, and treating
them as one is the same category error the section header warns about, one level
down.

**`S3-M` vs `S3-L` is the only pair that isolates expression level.** Same loop
structure, same accumulators, same tile; a library reduction versus a
hand-written warp-shuffle chain. The manual form is **slower at every head dim**,
and by the most at the smallest one, where the reduction is the largest share of
the kernel. This is the third independent measurement in this study — after the
fused GEMM lanes (§2.3) and the fused softmax ladder (§3) — in which dropping to
the lower-level formulation of the *same* computation costs a few percent rather
than gaining any.

The reason is concrete and is a real property of the abstraction, not an
implementation slip. `acc_s` is a `T.gemm` C fragment, and its
`(row, col) → (thread, lane, register)` mapping is chosen by TileLang's layout
inference and is **not visible in the source**. A shuffle chain reduces across
the 32 lanes of a warp, and nothing in the language says which `(i, j)` those
lanes hold — so a warp reduction cannot be written over the fragment at all. The
manual arm must first stage the score tile into a *shared* fp32 buffer, where
addressing is defined by the program, and reduce that. It pays a shared-memory
round trip twice per KV block that the library reduction, which reduces the
fragment in place, does not. Going lower-level here does not buy access to the
hardware; it buys a detour around information the compiler has and the programmer
does not.

**`S3-H` vs `S3-M` is not an expression difference at all**, and the head-dim
sweep is what makes that visible. At `d=128` the two are equivalent. As `d`
grows they diverge sharply, and the cause is structural: `S3-H` keeps one
`(block_M, D_TILE)` accumulator, so once `d` exceeds what one accumulator holds
the high-level formulation is forced into an **outer loop over d-tiles that
recomputes the entire `QK^T` for every d-tile** — `n_d_tiles` = 1, 2 and 8 at the
three head dims. `S3-M`'s per-d-tile accumulators cover all of `d` in a single KV
pass. That is a different amount of arithmetic, not a different way of writing
the same arithmetic.

It is still a genuine finding about abstraction, but it belongs on the
*exploration* ledger, not the efficiency one: the natural high-level formulation
degenerates at large `d`, and nothing warns the author. It is not a strawman —
it is what the obvious code does — but it must not be quoted as "high-level
costs 3.7×".

**`S3-MP` answers its question by failing to build.** It is `S3-M` with
`T.serial` replaced by `T.Pipelined` and no other token changed. At `d=128` it
compiles and is indistinguishable from `S3-M` — the generic pipeline buys
nothing. Above `d=128` it does not compile: TileLang's planner requires each
shared buffer be written by at most one statement in a pipelined body, and the
manual structure re-loads one V buffer once per d-tile. The arrangement the
planner wants — one V buffer per d-tile — was built and compiled, and needs
108544 B at `d=256` (over budget) and 305152 B at `d=1024` (3.0× over); only
`block_N ≤ 16` would fit, a quarter of the other arms' KV tile. The prediction
that generic pipelining would conflict with hand-managed V reuse is confirmed in
a stronger form than stated: it does not regress, it becomes inexpressible. Every
`S3-MP` number here is a statement about `d=128` alone.

### 5.2 The decomposition axis, reported separately

Only after the above is the fused-versus-decomposed comparison meaningful, and it
inverts with head dimension. At small `d` the fused kernel wins clearly — it
makes one pass over K and V and never materializes `S`. At `d=1024` that
advantage is gone: the fused accumulator's register pressure and the `S3` arms'
repeated V traffic cost about what materializing `S` costs, and the decomposed
forms catch up. The prediction that the decomposed form "may still win at
`D=1024` because large regular GEMMs achieve better occupancy" is essentially
borne out, though as a convergence rather than a rout.

Note which decomposed form: `S2` (two kernels, `S` never written to global) is
the better of the two at every head dim, and `S2` versus `S1` is a clean
one-factor comparison because they share the *same* `PV` kernel object code —
the only difference is whether `S` is materialized. That is the cost of
materialization, measured directly.

---

## 6. Limitations

* `wcache=native` is not implemented in the two hand-written CUDA lanes. A
  `(N,K)` B operand needs a transposed WMMA fragment load or a different
  `ldmatrix` staging — a rewrite, not a factor level. Building it would have
  confounded the factor with a kernel change, so the level is carried only where
  it is a config change.
* `K2` is absent from the cross-DSL SDPA table by design; the two-kernel
  decomposition is on the TileLang algorithm axis as `S2`.
* The shipped artifacts have drifted from their convergence logs (§2.1), so
  published per-DSL numbers are re-measured rather than quoted. Where a published
  number is cited it is labelled as published.
* Every number here is one GPU (RTX 6000 Ada, sm_89), one problem shape per cell,
  and `torch.rand` inputs. Phase 1 showed the input distribution decides whether
  a reduced-precision path passes the gate at all; §2.2 shows it also decides
  whether the gate means anything.
* Six harness bugs were found during development. All six produce
  clean-looking wrong numbers rather than errors, which is why they are named
  here — each is worth checking for in any similar harness:
  1. A build-time warm-up that ran through `run()` and primed the weight cache
     with a `torch.zeros` dummy. Every timed call then multiplied by zeros: the
     op returned all-zeros and got *faster* (0.89 ms), which looks like a win.
  2. `is not` used to compare two `data_ptr()` values. CPython does not intern
     large ints, so the identity test was always true and the cached arm was
     silently the uncached one. This produced a *plausible* wrong answer —
     cached 2.40 vs uncached 2.52 ms — and was only caught because the
     conversion alone is known to cost 1.42 ms, which the gap could not contain.
  3. The aggregator did not key on `soft_only`, so the softmax-only cells
     (~0.09 ms) and the whole-op cells (~1.5 ms) were pooled into single
     "results" whose 95% CI spanned both populations.
  4. The Nsight Compute collector folded launches by kernel *name*. TileLang
     names every generated kernel `main_kernel`, so its GEMM and its softmax
     collided and only the last survived — the GEMM, 0.84 ms and 0.59 GB of
     DRAM, was simply absent from the census. K3's three SDPA kernels would have
     collapsed the same way, which is precisely the per-kernel breakdown the
     study exists to report. Launches are now folded on
     `(name, grid, block, registers)`.
  5. The worst-behaved of the five, because it degrades *gracefully*: one report
     table selected its cells with a hard-coded key tuple. When the aggregation
     key gained a field (bug 3's fix), that tuple stopped matching, every lookup
     returned `None`, and the table rendered a tidy **"(campaign not run)"** —
     with all 25 of its records sitting on disk. The build's own `--check` did
     not flag it either, because the output was well-formed. Cells are now
     selected by field rather than by tuple, and `--check` treats a bare
     parenthetical as a failure. A pipeline that says "not measured" when it
     means "not retrieved" is worse than one that crashes.
  6. `compile_s` was recorded uniformly across lanes that were **not in the same
     cache state** (§1.3). Only TileLang disables its cache; the others were
     served from warm caches, so the median compile times formed a 35× spread
     that ranked the lanes in the reverse of the true order. Nothing errored and
     no number was individually wrong — the defect was entirely in what the set
     of them was taken to mean. Cold and warm are now measured deliberately and
     separately, and the campaign census reports which state each build was in.
* Two further issues were found in measurement lanes rather than in the harness,
  both of which would have confounded a comparison the study exists to make:
  * The tilelang SDPA lane converted Q/K/V to fp16 on the host inside the timed
    region while the other three lanes converted on the global→shared path
    (§4.3). It inflated only its own numbers, so it was not cheating, but it
    would have made every cross-DSL SDPA comparison wrong by up to 2×.
  * The TileLang SDPA *abstraction* lane had the same defect while `S1`/`S2`
    delegate to the (already fixed) cross-DSL lane — so a host-side cast would
    have sat on the `S3` side of the `S1`-vs-`S3` comparison and not the other,
    which is precisely the confound §5 is arranged to prevent.
* **This tree is not under version control**, which means the pre-fix state of a
  lane cannot be recovered and a before/after claim about a lane's own numbers is
  not independently reproducible. Only the current state is. Where before/after
  figures appear in this report they are the change author's, corroborated by an
  independent re-measurement of the *after* column and by the invariant that
  should hold across the change (identical `max_abs_err`), not by re-running the
  old code.

---

## 7. Reproducing

```bash
cd ako_runs/phase2_fused_sdpa
python make_jobs2.py

# context: what the shipped artifacts and the references actually do.
# `incumbent_reps.py` runs both incumbent checks as 5 independent processes and
# folds them; running either script directly gives a single unreplicated number.
python incumbent_reps.py --gpu 0 --reps 5
python sdpa_reference_audit.py

# compile cost, cold and warm, with the caches under control (§1.3).
# Host-side work, so it can share a card that is not timing anything.
python compile_cold.py --gpu 1 --reps 5

# the campaigns (each aborts if the GPU is busy)
python driver2.py --op fused --jobs jobs/fused_matched.json     --gpu 0 --reps 5 --tag fused_matched
python driver2.py --op fused --jobs jobs/fused_native.json      --gpu 0 --reps 5 --tag fused_native
python driver2.py --op fused --jobs jobs/fused_epilogue.json    --gpu 0 --reps 5 --tag fused_epilogue
python driver2.py --op fused --jobs jobs/fused_cast.json        --gpu 0 --reps 5 --tag fused_cast
python driver2.py --op fused --jobs jobs/fused_abstraction.json --gpu 0 --reps 5 --tag fused_abstraction
python driver2.py --op sdpa  --jobs jobs/sdpa_cross.json        --gpu 0 --reps 5 --tag sdpa_cross
python driver2.py --op sdpa  --jobs jobs/sdpa_abstraction.json  --gpu 0 --reps 5 --tag sdpa_abstraction

python build_report2.py          # fills this document's placeholders
```

---

## 8. The two conclusions, stated separately

### 8.1 Abstraction efficiency

**The question.** How fast is an *equivalent algorithm on an equivalent
instruction path* when written at a high level versus a low one? Every
comparison in this subsection holds the algorithm, the tile, the arithmetic and
the inputs fixed and changes only how the code is expressed.

**The answer, in four independent measurements: approximately zero, and never in
favour of the lower level.**

| what was held fixed | high level | low level | low level costs |
|---|---|---|---|
| fused GEMM, matched tile/schedule/arithmetic (§2.3) | best compiler lane 1.598 ms | best hand-written lane 1.897 ms | **+19%** |
| fused softmax, same GEMM, same 256 threads (§3) | `T.reduce_*` 0.0881 ms | warp shuffles + cached `exp` 0.0870 ms | −1.2% |
| SDPA online softmax, same loop, same tile (§5.1) | `T.reduce_*` (`S3-M`) | `T.shfl_down` (`S3-L`) | **+3.3 to +13.0%** |

On the fused GEMM the two groups do not interleave at all: both compiler-backed
lanes are faster than both hand-written ones — tilelang 1.598 < triton 1.773 <
`cuda_unlimited` 1.897 < `cuda_noptx` 2.031 ms. Given the same geometry and
schedule the DSLs were given, writing raw `mma.sync`/`ldmatrix` PTX bought
nothing, and writing WMMA C++ cost 27% against the best compiler lane.
And the counters say why (§2.7): all four GEMMs move identical DRAM (0.59 GB) and
sit at identical occupancy (one 256-thread block per SM), so no resource was
unlocked by going lower. Register pressure runs *inversely* to speed — 238 and
255 in the two fastest lanes against 168 and 167 in the two slowest. What the
compilers did right was spend registers aggressively, which is the opposite of
the usual hand-optimization heuristic.

The SDPA case shows the mechanism most clearly, and it is a property of the
abstraction rather than an implementation slip. A warp-shuffle reduction
*cannot* be written over a `T.gemm` accumulator, because the fragment's
`(row, col) → (thread, lane, register)` mapping is chosen by layout inference and
is invisible in the source. The manual arm must first stage the score tile
through shared memory, where addressing is defined by the program, and pays that
round trip twice per KV block. Dropping a level did not buy closer access to the
hardware; it bought a detour around information the compiler had and the
programmer did not.

**Where abstraction level did change outcomes, it did so by removing a degree of
freedom rather than by generating better code.** The manual softmax forms expose
the index expression, and the shipped incumbent chose a non-coalescing one. That
single choice costs **2.66–2.78×** — more than every abstraction-level effect in
this study combined and multiplied. The high-level form gets coalescing from
`T.copy` and cannot express the mistake.

**One genuine capability difference, reported on the other ledger.** `cuda_noptx`
cannot apply a column-indexed bias in accumulator registers, because WMMA's
fragment→(row,col) mapping is not part of its API contract; it must stage the
tile through shared memory. That is abstraction constraining what is
*expressible*, and it is real — but it is worth ~1.5%, and only when a bias is
present (§2.5).

**So the defensible statement is not "high-level code is as fast as low-level
code."** It is: *at a matched configuration the abstraction level was worth about
1%, in the compilers' favour, while the errors the lower level made available
were worth 2.7×.* Scope: one GPU (RTX 6000 Ada, sm_89), two op classes, one shape
per cell. It says nothing about ceilings an expert could reach given unlimited
time — only about what these four interfaces produce from the same specification.

### 8.2 Abstraction-enabled exploration

**The question.** Did the higher-level interface make it *cheaper to discover*
split-K, fusion, layouts and pipeline configurations? This is about the search,
not the generated code, and it is where the two DSLs actually differ from the two
CUDA lanes.

**It did, and the evidence is mostly negative space — what the low-level lanes
never explored.**

* In tilelang and triton, changing `KC`, `stages`, the tile, or the arithmetic is
  a parameter edit. In the CUDA lanes the same change is a rewrite of the
  pipeline and the fragment staging. The `wcache=native` arm could not be built
  in either CUDA lane for exactly this reason (§6): consuming `W` in its stored
  `(N,K)` layout needs different `ldmatrix` staging, which is a new kernel, not a
  factor level.
* **The per-iteration cost of exploring differs by ~6×, and it is measured**
  (§1.3). A kernel the search has not seen before costs TileLang 6.0 s to
  compile and nvcc 36.4 s, cold against cold on the same arm; Triton 1.2 s.
  Over Phase 1's 19-point grid that is ~1.9 minutes against ~11.5 minutes
  *before any measurement is taken* — so at a fixed wall-clock budget the DSL
  lanes get roughly six times as many trials, and Triton thirty. This is the one
  component of the search cost this study actually quantified; the rest (§1.3)
  it did not.
* Both CUDA lanes shipped worse artifacts than both DSL lanes on both ops, and
  `cuda_noptx` shipped **the identity on both** — it calls the reference. A lane
  that spends its budget on staging code has less left for the search.
* `S3-MP` (§5.1) is the sharpest instance: adding `T.Pipelined` to a
  hand-managed multi-accumulator structure is a one-token edit that *cannot
  compile* above `d=128`. The high-level construct and the manual structure
  compose only where the manual structure happens to look high-level.

**But the more useful finding is that the exploration advantage was mostly spent
on things that are not compiler properties, and sometimes spent badly.**

What the search on the fused op actually found, ranked by what it was worth:

| lever | worth | is it an abstraction property? |
|---|---|---|
| fp32 → fp16 arithmetic | ~3.5× | no — a dtype decision |
| host-side weight caching | 1.549 → 3.197 ms | no — a Python-level cache |
| coalescing the softmax index | 2.7× *(if you get it wrong)* | only in that the DSL prevents the error |
| the entire fusion ladder (bias, GELU, softmax) | +0.07 to +0.23 ms | no — 4–7% of the op |
| softmax reduction abstraction level | ~1% | yes — and it is the smallest term |

And three cases where cheap iteration produced a confidently wrong result:

* The incumbent tilelang softmax settled on a non-coalescing index costing 2.7×.
  Cheap iteration explores more configurations; it does not evaluate them better.
* `cuda_noptx` abandoned fp16 for SDPA on a precision claim that was **correct
  about what it named and the wrong conclusion to draw** — fp16 *scores* really
  are unreachable at `d=1024` (§4.4), but fp16 *operands* with fp32 scores pass
  comfortably. It gave up ~2.5× by treating one precision decision as forcing the
  other, and shipped the identity.
* One shipped solution (`cuda_unlimited`, fused) does not pass the gate at all
  while being credited with 1.24×, and the shipped triton artifact is newer than
  its own convergence log. Search records and search products had drifted apart.

**The one place custom kernels genuinely won on the merits was SDPA at
`d=1024`** — and the head-dim sweep says precisely why. Against
precision-matched torch:

| head dim | FlashAttention available to torch? | best gate-passing custom kernel |
|---|---|---|
| 128 | yes | 1.12× |
| 256 | yes | 1.01× |
| 1024 | **no** — head-dim cap is 256 | **1.74×** |

Where a good vendor kernel exists, every lane converges on parity with it. Where
one does not, the same code wins by 1.74×. On the fused op, where cuBLAS is
near-optimal at every arm, no DSL beat precision-matched torch anywhere
(0.75–0.95×). **The advantage tracked the size of the hole in the vendor library,
not the abstraction level of the language used to fill it** — and had this study
run only the `d=1024` cell the spec asked to be wary of, that would have read as
a DSL result.

**The combined statement.** The published gains for these two cells decompose
into an arithmetic change, a host-side cache, a denominator choice, and — in one
of two ops — genuine headroom left by a missing vendor kernel. The abstraction
level under study is the smallest term in that decomposition on the fused op and
is not the operative term on SDPA either. Where abstraction demonstrably mattered
was in what it made *impossible to express wrongly*, and in what the low-level
lanes consequently never got around to trying.

---

## Appendix A — every fused cell

| lane | arm | wcache | epi | cast | timed | ms | ci95 | n | gate | maxerr |
|---|---|---|---|---|---|---|---|---|---|---|
| cuda_noptx | G | cached | - | precast | full | 1.9231 | 1.8417-1.9721 | 5/5 | FAIL | 5.14e-04 |
| cuda_noptx | G | uncached | - | precast | full | 3.6306 | 3.5650-3.6762 | 5/5 | FAIL | 5.14e-04 |
| cuda_noptx | GB | cached | - | precast | full | 1.9497 | 1.8383-2.0169 | 5/5 | FAIL | 5.14e-04 |
| cuda_noptx | GB | uncached | - | precast | full | 3.6541 | 3.4919-3.7480 | 5/5 | FAIL | 5.14e-04 |
| cuda_noptx | GBG | cached | - | precast | full | 1.9456 | 1.8589-2.0217 | 5/5 | FAIL | 4.26e-04 |
| cuda_noptx | GBG | uncached | - | precast | full | 3.6782 | 3.5514-3.7372 | 5/5 | FAIL | 4.26e-04 |
| cuda_noptx | GBGS | cached | - | in_region | full | 2.0429 | 2.0113-2.0845 | 5/5 | pass | 1.53e-07 |
| cuda_noptx | GBGS | cached | - | precast | full | 2.0311 | 1.9506-2.0850 | 5/5 | pass | 1.53e-07 |
| cuda_noptx | GBGS | uncached | - | precast | full | 3.7458 | 3.6197-3.8367 | 5/5 | pass | 1.53e-07 |
| cuda_unlimited | G | cached | - | precast | full | 1.8227 | 1.7388-1.8683 | 5/5 | FAIL | 5.14e-04 |
| cuda_unlimited | G | cached | regs | precast | full | 1.8017 | 1.7299-1.8256 | 5/5 | FAIL | 5.14e-04 |
| cuda_unlimited | G | cached | smem | precast | full | 1.7930 | 1.7441-1.8225 | 5/5 | FAIL | 5.14e-04 |
| cuda_unlimited | G | uncached | - | precast | full | 3.4765 | 3.3731-3.5580 | 5/5 | FAIL | 5.14e-04 |
| cuda_unlimited | GB | cached | - | precast | full | 1.8207 | 1.7521-1.8446 | 5/5 | FAIL | 5.14e-04 |
| cuda_unlimited | GB | cached | regs | precast | full | 1.7787 | 1.7238-1.8352 | 5/5 | FAIL | 5.14e-04 |
| cuda_unlimited | GB | cached | smem | precast | full | 1.8217 | 1.7681-1.8516 | 5/5 | FAIL | 5.14e-04 |
| cuda_unlimited | GB | uncached | - | precast | full | 3.5528 | 3.4200-3.6247 | 5/5 | FAIL | 5.14e-04 |
| cuda_unlimited | GBG | cached | - | precast | full | 1.8452 | 1.7224-1.8784 | 5/5 | FAIL | 4.26e-04 |
| cuda_unlimited | GBG | cached | regs | precast | full | 1.7971 | 1.7339-1.8270 | 5/5 | FAIL | 4.26e-04 |
| cuda_unlimited | GBG | cached | smem | precast | full | 1.7879 | 1.7365-1.8279 | 5/5 | FAIL | 4.26e-04 |
| cuda_unlimited | GBG | uncached | - | precast | full | 3.5538 | 3.4071-3.6591 | 5/5 | FAIL | 4.26e-04 |
| cuda_unlimited | GBGS | cached | - | in_region | full | 1.9369 | 1.9140-1.9593 | 5/5 | pass | 1.53e-07 |
| cuda_unlimited | GBGS | cached | - | precast | full | 1.8975 | 1.7675-1.9351 | 5/5 | pass | 1.53e-07 |
| cuda_unlimited | GBGS | cached | regs | precast | full | 1.8308 | 1.7779-1.8974 | 5/5 | pass | 1.53e-07 |
| cuda_unlimited | GBGS | cached | smem | precast | full | 1.8227 | 1.7941-1.8753 | 5/5 | pass | 1.53e-07 |
| cuda_unlimited | GBGS | uncached | - | precast | full | 3.6326 | 3.4794-3.7122 | 5/5 | pass | 1.53e-07 |
| tilelang | G | cached | - | precast | full | 1.5268 | 1.4602-1.5740 | 5/5 | FAIL | 5.14e-04 |
| tilelang | G | native | - | precast | full | 2.2774 | 2.1904-2.3117 | 5/5 | FAIL | 5.14e-04 |
| tilelang | G | uncached | - | precast | full | 3.2287 | 3.1818-3.2628 | 5/5 | FAIL | 5.14e-04 |
| tilelang | GB | cached | - | precast | full | 1.5565 | 1.5207-1.5800 | 5/5 | FAIL | 5.14e-04 |
| tilelang | GB | native | - | precast | full | 2.2564 | 2.0928-2.3626 | 5/5 | FAIL | 5.14e-04 |
| tilelang | GB | uncached | - | precast | full | 3.1795 | 3.1089-3.2757 | 5/5 | FAIL | 5.14e-04 |
| tilelang | GBG | cached | - | precast | full | 1.5452 | 1.5199-1.5703 | 5/5 | FAIL | 4.26e-04 |
| tilelang | GBG | native | - | precast | full | 2.2692 | 2.1977-2.3000 | 5/5 | FAIL | 4.26e-04 |
| tilelang | GBG | uncached | - | precast | full | 3.2159 | 3.1974-3.2548 | 5/5 | FAIL | 4.26e-04 |
| tilelang | GBGS | cached | - | in_region | full | 1.6410 | 1.6046-1.6679 | 5/5 | pass | 1.53e-07 |
| tilelang | GBGS | cached | - | precast | full | 1.5985 | 1.5471-1.6197 | 5/5 | pass | 1.53e-07 |
| tilelang | GBGS | native | - | precast | full | 2.3265 | 2.3204-2.3455 | 5/5 | pass | 1.53e-07 |
| tilelang | GBGS | uncached | - | precast | full | 3.2940 | 3.2538-3.3274 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F1 | cached | - | precast | full | 1.5493 | 1.5354-1.5673 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F1 | cached | - | precast | softmax | 0.0881 | 0.0881-0.0881 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F1 | uncached | - | precast | full | 3.1969 | 3.1823-3.2163 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F2 | cached | - | precast | full | 1.5913 | 1.5846-1.6040 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F2 | cached | - | precast | softmax | 0.2703 | 0.2700-0.2714 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F2 | uncached | - | precast | full | 3.3239 | 3.3228-3.3256 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F2c | cached | - | precast | full | 1.5606 | 1.5527-1.5894 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F2c | cached | - | precast | softmax | 0.0983 | 0.0983-0.0983 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F2c | uncached | - | precast | full | 3.2061 | 3.1956-3.2226 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F3 | cached | - | precast | full | 1.6036 | 1.5886-1.6102 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F3 | cached | - | precast | softmax | 0.2714 | 0.2708-0.2721 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F3 | uncached | - | precast | full | 3.3198 | 3.3142-3.3245 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F3c | cached | - | precast | full | 1.5585 | 1.5504-1.5865 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F3c | cached | - | precast | softmax | 0.0975 | 0.0971-0.0981 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F3c | uncached | - | precast | full | 3.2000 | 3.1765-3.2222 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F4 | cached | - | precast | full | 1.5764 | 1.5713-1.5808 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F4 | cached | - | precast | softmax | 0.2314 | 0.2314-0.2314 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F4 | uncached | - | precast | full | 3.2773 | 3.2695-3.2816 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F4c | cached | - | precast | full | 1.5631 | 1.5349-1.5713 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F4c | cached | - | precast | softmax | 0.0870 | 0.0870-0.0870 | 5/5 | pass | 1.53e-07 |
| tilelang_abs | F4c | uncached | - | precast | full | 3.1974 | 3.1918-3.2097 | 5/5 | pass | 1.53e-07 |
| torch:fp16 | G | cached | - | precast | full | 1.4377 | 1.3962-1.4583 | 5/5 | FAIL | 1.17e-03 |
| torch:fp16 | G | native | - | precast | full | 1.8258 | 1.7574-1.8655 | 5/5 | FAIL | 1.17e-03 |
| torch:fp16 | G | uncached | - | precast | full | 2.7382 | 2.6781-2.8003 | 5/5 | FAIL | 1.17e-03 |
| torch:fp16 | GB | cached | - | precast | full | 1.4797 | 1.4400-1.5083 | 5/5 | FAIL | 1.52e-03 |
| torch:fp16 | GB | native | - | precast | full | 1.8412 | 1.7330-1.8899 | 5/5 | FAIL | 1.52e-03 |
| torch:fp16 | GB | uncached | - | precast | full | 2.8027 | 2.7056-2.8618 | 5/5 | FAIL | 1.52e-03 |
| torch:fp16 | GBG | cached | - | precast | full | 1.4510 | 1.3608-1.4978 | 5/5 | FAIL | 1.93e-03 |
| torch:fp16 | GBG | native | - | precast | full | 1.8852 | 1.8417-1.9244 | 5/5 | FAIL | 1.93e-03 |
| torch:fp16 | GBG | uncached | - | precast | full | 2.8242 | 2.7843-2.8514 | 5/5 | FAIL | 1.93e-03 |
| torch:fp16 | GBGS | cached | - | precast | full | 1.5227 | 1.5135-1.5311 | 10/10 | pass | 8.29e-07 |
| torch:fp16 | GBGS | native | - | precast | full | 1.9082 | 1.8722-1.9397 | 5/5 | pass | 8.29e-07 |
| torch:fp16 | GBGS | uncached | - | precast | full | 2.8529 | 2.8051-2.8818 | 5/5 | pass | 8.29e-07 |
| torch:fp32 | G | cached | - | precast | full | 5.3734 | 5.3146-5.3942 | 5/5 | pass | 7.51e-06 |
| torch:fp32 | G | uncached | - | precast | full | 6.6765 | 6.6184-6.7323 | 5/5 | pass | 7.51e-06 |
| torch:fp32 | GB | cached | - | precast | full | 5.4149 | 5.2414-5.5241 | 5/5 | pass | 7.51e-06 |
| torch:fp32 | GB | uncached | - | precast | full | 6.6442 | 6.5635-6.7743 | 5/5 | pass | 7.51e-06 |
| torch:fp32 | GBG | cached | - | precast | full | 5.4349 | 5.3702-5.5321 | 5/5 | pass | 8.11e-06 |
| torch:fp32 | GBG | uncached | - | precast | full | 6.7359 | 6.6065-6.8023 | 5/5 | pass | 8.11e-06 |
| torch:fp32 | GBGS | cached | - | precast | full | 5.4938 | 5.3210-5.6075 | 5/5 | pass | 3.41e-09 |
| torch:fp32 | GBGS | uncached | - | precast | full | 6.8424 | 6.7485-6.9264 | 5/5 | pass | 3.41e-09 |
| triton | G | cached | - | precast | full | 1.6722 | 1.6187-1.7163 | 5/5 | FAIL | 5.14e-04 |
| triton | G | native | - | precast | full | 2.2067 | 2.1412-2.2911 | 5/5 | FAIL | 5.14e-04 |
| triton | G | uncached | - | precast | full | 3.2548 | 3.1990-3.4333 | 5/5 | FAIL | 5.14e-04 |
| triton | GB | cached | - | precast | full | 1.6804 | 1.6327-1.7086 | 5/5 | FAIL | 5.14e-04 |
| triton | GB | native | - | precast | full | 2.1745 | 2.0457-2.3332 | 5/5 | FAIL | 5.14e-04 |
| triton | GB | uncached | - | precast | full | 3.3997 | 3.3353-3.4299 | 5/5 | FAIL | 5.14e-04 |
| triton | GBG | cached | - | precast | full | 1.7295 | 1.6855-1.7834 | 5/5 | FAIL | 4.26e-04 |
| triton | GBG | native | - | precast | full | 2.2415 | 2.1088-2.3618 | 5/5 | FAIL | 4.26e-04 |
| triton | GBG | uncached | - | precast | full | 3.4483 | 3.3460-3.5345 | 5/5 | FAIL | 4.26e-04 |
| triton | GBGS | cached | - | in_region | full | 1.8135 | 1.7756-1.8530 | 5/5 | pass | 1.53e-07 |
| triton | GBGS | cached | - | precast | full | 1.7725 | 1.7443-1.7799 | 5/5 | pass | 1.53e-07 |
| triton | GBGS | native | - | precast | full | 2.3040 | 2.0929-2.4084 | 5/5 | pass | 1.53e-07 |
| triton | GBGS | uncached | - | precast | full | 3.4857 | 3.4106-3.5559 | 5/5 | pass | 1.53e-07 |

---

## Appendix B — every SDPA cell

Both campaigns, all head dims and dtype pairs, including the cells that
fail the gate and the `S3-MP` cells that do not build (§5.1).

| lane | arm | d | scores/probs | wcache | epi | cast | timed | ms | ci95 | n | gate | maxerr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuda_noptx | FLASH | 128 | fp16/fp16 | cached | - | precast | full | 3.7893 | 3.7514-3.8203 | 5/5 | pass | 6.10e-05 |
| cuda_noptx | FLASH | 128 | fp32/fp16 | cached | - | precast | full | 3.7862 | 3.7543-3.8661 | 5/5 | pass | 4.24e-05 |
| cuda_noptx | FLASH | 128 | fp32/fp32 | cached | - | precast | full | 5.8117 | 5.7182-5.8411 | 5/5 | pass | 5.57e-06 |
| cuda_noptx | FLASH | 256 | fp16/fp16 | cached | - | precast | full | 7.1434 | 7.1073-7.1914 | 5/5 | pass | 9.32e-05 |
| cuda_noptx | FLASH | 256 | fp32/fp16 | cached | - | precast | full | 7.1122 | 7.0470-7.1544 | 5/5 | pass | 4.20e-05 |
| cuda_noptx | FLASH | 256 | fp32/fp32 | cached | - | precast | full | 10.6988 | 10.5761-10.7376 | 5/5 | pass | 5.33e-06 |
| cuda_noptx | FLASH | 1024 | fp16/fp16 | cached | - | precast | full | 34.7848 | 34.7234-34.8846 | 5/5 | FAIL | 1.93e-04 |
| cuda_noptx | FLASH | 1024 | fp32/fp16 | cached | - | precast | full | 34.8344 | 33.2721-35.5327 | 5/5 | pass | 4.29e-05 |
| cuda_noptx | FLASH | 1024 | fp32/fp32 | cached | - | precast | full | 53.4513 | 53.1975-53.8680 | 5/5 | pass | 5.66e-06 |
| cuda_noptx | K3 | 128 | fp16/fp16 | cached | - | precast | full | 5.5112 | 5.4213-5.6054 | 5/5 | pass | 6.27e-05 |
| cuda_noptx | K3 | 128 | fp32/fp16 | cached | - | precast | full | 6.1942 | 6.1459-6.2285 | 5/5 | pass | 4.43e-05 |
| cuda_noptx | K3 | 128 | fp32/fp32 | cached | - | precast | full | 9.3486 | 9.2422-9.4259 | 5/5 | pass | 5.63e-06 |
| cuda_noptx | K3 | 256 | fp16/fp16 | cached | - | precast | full | 8.9917 | 8.8616-9.2726 | 5/5 | pass | 1.01e-04 |
| cuda_noptx | K3 | 256 | fp32/fp16 | cached | - | precast | full | 9.6374 | 8.9391-9.9442 | 5/5 | pass | 4.98e-05 |
| cuda_noptx | K3 | 256 | fp32/fp32 | cached | - | precast | full | 15.3124 | 15.2456-15.5041 | 5/5 | pass | 5.28e-06 |
| cuda_noptx | K3 | 1024 | fp16/fp16 | cached | - | precast | full | 29.7416 | 29.4419-30.1051 | 5/5 | FAIL | 1.87e-04 |
| cuda_noptx | K3 | 1024 | fp32/fp16 | cached | - | precast | full | 30.4461 | 29.9993-30.6762 | 5/5 | pass | 4.58e-05 |
| cuda_noptx | K3 | 1024 | fp32/fp32 | cached | - | precast | full | 51.2343 | 50.7582-51.6068 | 5/5 | pass | 5.66e-06 |
| cuda_unlimited | FLASH | 128 | fp16/fp16 | cached | - | precast | full | 2.7807 | 2.7627-2.8158 | 5/5 | pass | 5.87e-05 |
| cuda_unlimited | FLASH | 128 | fp32/fp16 | cached | - | precast | full | 2.8539 | 2.8234-2.8686 | 5/5 | pass | 4.48e-05 |
| cuda_unlimited | FLASH | 128 | fp32/fp32 | cached | - | precast | full | 5.1226 | 5.0347-5.1500 | 5/5 | pass | 5.72e-06 |
| cuda_unlimited | FLASH | 256 | fp16/fp16 | cached | - | precast | full | 5.7283 | 5.5828-5.8064 | 5/5 | pass | 9.05e-05 |
| cuda_unlimited | FLASH | 256 | fp32/fp16 | cached | - | precast | full | 5.7856 | 5.7554-5.7996 | 5/5 | pass | 4.44e-05 |
| cuda_unlimited | FLASH | 256 | fp32/fp32 | cached | - | precast | full | 10.2431 | 10.1272-10.3610 | 5/5 | pass | 5.36e-06 |
| cuda_unlimited | FLASH | 1024 | fp16/fp16 | cached | - | precast | full | 29.9361 | 29.6232-30.4235 | 5/5 | FAIL | 1.91e-04 |
| cuda_unlimited | FLASH | 1024 | fp32/fp16 | cached | - | precast | full | 29.9325 | 29.8174-30.1779 | 5/5 | pass | 4.56e-05 |
| cuda_unlimited | FLASH | 1024 | fp32/fp32 | cached | - | precast | full | 55.4808 | 54.6780-55.9941 | 5/5 | pass | 5.96e-06 |
| cuda_unlimited | K3 | 128 | fp16/fp16 | cached | - | precast | full | 4.2849 | 4.2696-4.2931 | 5/5 | pass | 6.27e-05 |
| cuda_unlimited | K3 | 128 | fp32/fp16 | cached | - | precast | full | 5.1948 | 5.1867-5.2102 | 5/5 | pass | 4.43e-05 |
| cuda_unlimited | K3 | 128 | fp32/fp32 | cached | - | precast | full | 8.3804 | 8.3200-8.4037 | 5/5 | pass | 5.63e-06 |
| cuda_unlimited | K3 | 256 | fp16/fp16 | cached | - | precast | full | 6.7251 | 6.6702-6.7829 | 5/5 | pass | 1.01e-04 |
| cuda_unlimited | K3 | 256 | fp32/fp16 | cached | - | precast | full | 7.6334 | 7.5779-7.7372 | 5/5 | pass | 4.98e-05 |
| cuda_unlimited | K3 | 256 | fp32/fp32 | cached | - | precast | full | 12.8927 | 12.8162-13.0404 | 5/5 | pass | 5.30e-06 |
| cuda_unlimited | K3 | 1024 | fp16/fp16 | cached | - | precast | full | 20.4559 | 20.1016-20.7861 | 5/5 | FAIL | 1.87e-04 |
| cuda_unlimited | K3 | 1024 | fp32/fp16 | cached | - | precast | full | 21.3345 | 21.1506-21.5917 | 5/5 | pass | 4.58e-05 |
| cuda_unlimited | K3 | 1024 | fp32/fp32 | cached | - | precast | full | 39.7322 | 39.3789-40.0884 | 5/5 | pass | 5.60e-06 |
| tilelang | FLASH | 128 | fp16/fp16 | cached | - | precast | full | 3.0254 | 2.9900-3.0516 | 5/5 | pass | 5.87e-05 |
| tilelang | FLASH | 128 | fp32/fp16 | cached | - | precast | full | 2.9952 | 2.9548-3.0418 | 5/5 | pass | 4.48e-05 |
| tilelang | FLASH | 128 | fp32/fp32 | cached | - | precast | full | 5.2685 | 5.2448-5.3064 | 5/5 | pass | 5.69e-06 |
| tilelang | FLASH | 256 | fp16/fp16 | cached | - | precast | full | 6.0257 | 5.9337-6.0688 | 5/5 | pass | 9.05e-05 |
| tilelang | FLASH | 256 | fp32/fp16 | cached | - | precast | full | 5.9863 | 5.9439-6.0430 | 5/5 | pass | 4.45e-05 |
| tilelang | FLASH | 256 | fp32/fp32 | cached | - | precast | full | 10.2815 | 10.1820-10.3650 | 5/5 | pass | 5.36e-06 |
| tilelang | FLASH | 1024 | fp16/fp16 | cached | - | precast | full | 25.7551 | 25.4170-25.9431 | 5/5 | FAIL | 1.91e-04 |
| tilelang | FLASH | 1024 | fp32/fp16 | cached | - | precast | full | 25.5734 | 25.4167-25.7202 | 5/5 | pass | 4.56e-05 |
| tilelang | FLASH | 1024 | fp32/fp32 | cached | - | precast | full | 55.5039 | 55.3131-55.6021 | 5/5 | pass | 5.96e-06 |
| tilelang | K3 | 128 | fp16/fp16 | cached | - | precast | full | 4.9516 | 4.8958-5.0042 | 5/5 | pass | 6.27e-05 |
| tilelang | K3 | 128 | fp32/fp16 | cached | - | precast | full | 5.5096 | 5.3915-5.6028 | 5/5 | pass | 4.43e-05 |
| tilelang | K3 | 128 | fp32/fp32 | cached | - | precast | full | 9.5273 | 9.5015-9.5566 | 5/5 | pass | 5.66e-06 |
| tilelang | K3 | 256 | fp16/fp16 | cached | - | precast | full | 7.8751 | 7.8490-7.9266 | 5/5 | pass | 1.01e-04 |
| tilelang | K3 | 256 | fp32/fp16 | cached | - | precast | full | 8.4767 | 8.3521-8.6318 | 5/5 | pass | 4.98e-05 |
| tilelang | K3 | 256 | fp32/fp32 | cached | - | precast | full | 14.9530 | 13.9280-15.2848 | 5/5 | pass | 5.28e-06 |
| tilelang | K3 | 1024 | fp16/fp16 | cached | - | precast | full | 25.6384 | 25.3073-26.0051 | 5/5 | FAIL | 1.87e-04 |
| tilelang | K3 | 1024 | fp32/fp16 | cached | - | precast | full | 26.2656 | 26.0956-26.7129 | 5/5 | pass | 4.58e-05 |
| tilelang | K3 | 1024 | fp32/fp32 | cached | - | precast | full | 49.8642 | 49.7243-50.1079 | 5/5 | pass | 5.66e-06 |
| tilelang_abs | S1 | 128 | fp32/fp16 | cached | - | precast | full | 5.2285 | 5.1865-5.2888 | 5/5 | pass | 4.43e-05 |
| tilelang_abs | S1 | 256 | fp32/fp16 | cached | - | precast | full | 8.4296 | 8.2332-8.5094 | 5/5 | pass | 4.98e-05 |
| tilelang_abs | S1 | 1024 | fp32/fp16 | cached | - | precast | full | 26.0470 | 25.7242-26.3890 | 5/5 | pass | 4.58e-05 |
| tilelang_abs | S2 | 128 | fp32/fp16 | cached | - | precast | full | 4.5850 | 4.5170-4.6064 | 5/5 | pass | 4.43e-05 |
| tilelang_abs | S2 | 256 | fp32/fp16 | cached | - | precast | full | 7.7747 | 7.7202-7.8814 | 5/5 | pass | 4.98e-05 |
| tilelang_abs | S2 | 1024 | fp32/fp16 | cached | - | precast | full | 24.8458 | 24.6917-24.9631 | 5/5 | pass | 4.58e-05 |
| tilelang_abs | S3-H | 128 | fp32/fp16 | cached | - | precast | full | 3.0070 | 2.9755-3.0616 | 5/5 | pass | 4.49e-05 |
| tilelang_abs | S3-H | 256 | fp32/fp16 | cached | - | precast | full | 8.6554 | 8.5223-8.7844 | 5/5 | pass | 4.44e-05 |
| tilelang_abs | S3-H | 1024 | fp32/fp16 | cached | - | precast | full | 90.3414 | 89.7440-91.0549 | 5/5 | pass | 4.56e-05 |
| tilelang_abs | S3-L | 128 | fp32/fp16 | cached | - | precast | full | 3.4749 | 3.4226-3.5332 | 5/5 | pass | 4.49e-05 |
| tilelang_abs | S3-L | 256 | fp32/fp16 | cached | - | precast | full | 6.3754 | 6.3367-6.5014 | 5/5 | pass | 4.44e-05 |
| tilelang_abs | S3-L | 1024 | fp32/fp16 | cached | - | precast | full | 26.1242 | 25.9589-26.3136 | 5/5 | pass | 4.56e-05 |
| tilelang_abs | S3-M | 128 | fp32/fp16 | cached | - | precast | full | 3.0752 | 3.0337-3.0968 | 5/5 | pass | 4.49e-05 |
| tilelang_abs | S3-M | 256 | fp32/fp16 | cached | - | precast | full | 6.1097 | 5.9959-6.1793 | 5/5 | pass | 4.44e-05 |
| tilelang_abs | S3-M | 1024 | fp32/fp16 | cached | - | precast | full | 25.3000 | 25.1537-25.4776 | 5/5 | pass | 4.56e-05 |
| tilelang_abs | S3-MP | 128 | fp32/fp16 | cached | - | precast | full | 3.0049 | 2.9794-3.0174 | 5/5 | pass | 4.49e-05 |
| tilelang_abs | S3-MP | 256 | fp32/fp16 | cached | - | precast | full | FAILED | | 0/5 | | InternalError: Pipeline planning error: Multiple writes to o |
| tilelang_abs | S3-MP | 1024 | fp32/fp16 | cached | - | precast | full | FAILED | | 0/5 | | InternalError: Pipeline planning error: Multiple writes to o |
| torch:fp16 | TORCH_F16 | 128 | fp32/fp32 | cached | - | precast | full | 2.7832 | 2.7581-2.8385 | 5/5 | FAIL | 2.79e-04 |
| torch:fp16 | TORCH_F16 | 256 | fp32/fp32 | cached | - | precast | full | 5.7979 | 5.7391-5.8536 | 5/5 | FAIL | 2.80e-04 |
| torch:fp16 | TORCH_F16 | 1024 | fp32/fp32 | cached | - | precast | full | 32.9953 | 32.8664-33.1847 | 5/5 | FAIL | 2.78e-04 |
| torch:fp16 | TORCH_F32 | 128 | fp32/fp32 | cached | - | precast | full | 6.3908 | 6.3271-6.4565 | 5/5 | pass | 1.43e-06 |
| torch:fp16 | TORCH_F32 | 256 | fp32/fp32 | cached | - | precast | full | 15.4046 | 15.2902-15.4908 | 5/5 | pass | 1.37e-06 |
| torch:fp16 | TORCH_F32 | 1024 | fp32/fp32 | cached | - | precast | full | 60.7114 | 60.3231-61.2399 | 5/5 | pass | 1.43e-06 |
| torch:fp16 | TORCH_MATH | 128 | fp32/fp32 | cached | - | precast | full | 14.4148 | 14.2741-14.5551 | 5/5 | pass | 4.77e-07 |
| torch:fp16 | TORCH_MATH | 256 | fp32/fp32 | cached | - | precast | full | 22.5172 | 22.4116-22.6089 | 5/5 | pass | 0.00e+00 |
| torch:fp16 | TORCH_MATH | 1024 | fp32/fp32 | cached | - | precast | full | 71.9713 | 71.5233-73.0534 | 5/5 | pass | 5.36e-07 |
| triton | FLASH | 128 | fp16/fp16 | cached | - | precast | full | 2.5405 | 2.5402-2.5413 | 5/5 | pass | 5.92e-05 |
| triton | FLASH | 128 | fp32/fp16 | cached | - | precast | full | 2.4914 | 2.4424-2.5418 | 5/5 | pass | 4.36e-05 |
| triton | FLASH | 128 | fp32/fp32 | cached | - | precast | full | 4.2849 | 4.2151-4.3363 | 5/5 | pass | 5.69e-06 |
| triton | FLASH | 256 | fp16/fp16 | cached | - | precast | full | 7.4030 | 7.2887-7.4520 | 5/5 | pass | 8.89e-05 |
| triton | FLASH | 256 | fp32/fp16 | cached | - | precast | full | 7.3948 | 7.3085-7.4834 | 5/5 | pass | 4.53e-05 |
| triton | FLASH | 256 | fp32/fp32 | cached | - | precast | full | 43.2963 | 42.2189-43.9351 | 5/5 | pass | 5.19e-06 |
| triton | FLASH | 1024 | fp16/fp16 | cached | - | precast | full | 50.2630 | 49.5922-50.5573 | 5/5 | FAIL | 1.90e-04 |
| triton | FLASH | 1024 | fp32/fp16 | cached | - | precast | full | 49.9159 | 49.6664-50.1061 | 5/5 | pass | 4.58e-05 |
| triton | FLASH | 1024 | fp32/fp32 | cached | - | precast | full | 65.3972 | 65.1326-65.6758 | 5/5 | pass | 5.96e-06 |
| triton | K3 | 128 | fp16/fp16 | cached | - | precast | full | 3.9438 | 3.9433-3.9446 | 5/5 | pass | 6.27e-05 |
| triton | K3 | 128 | fp32/fp16 | cached | - | precast | full | 5.2511 | 5.2495-5.2520 | 5/5 | pass | 4.43e-05 |
| triton | K3 | 128 | fp32/fp32 | cached | - | precast | full | 15.4896 | 14.9313-15.7499 | 5/5 | pass | 5.63e-06 |
| triton | K3 | 256 | fp16/fp16 | cached | - | precast | full | 5.9684 | 5.9336-6.0446 | 5/5 | pass | 1.01e-04 |
| triton | K3 | 256 | fp32/fp16 | cached | - | precast | full | 6.5915 | 6.5868-6.5978 | 5/5 | pass | 4.98e-05 |
| triton | K3 | 256 | fp32/fp32 | cached | - | precast | full | 28.0950 | 27.8578-28.2606 | 5/5 | pass | 5.28e-06 |
| triton | K3 | 1024 | fp16/fp16 | cached | - | precast | full | 18.0823 | 17.9580-18.2058 | 5/5 | FAIL | 1.87e-04 |
| triton | K3 | 1024 | fp32/fp16 | cached | - | precast | full | 18.9783 | 18.7892-19.0493 | 5/5 | pass | 4.58e-05 |
| triton | K3 | 1024 | fp32/fp32 | cached | - | precast | full | 101.0068 | 100.2996-102.5288 | 5/5 | pass | 5.66e-06 |
