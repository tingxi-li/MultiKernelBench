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

<!--TABLE_ANCHOR-->

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

<!--TABLE_COMPILE-->

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

<!--TABLE_INCUMBENT_FUSED-->

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

<!--TABLE_GATE-->

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

<!--TABLE_LADDER-->

The ladder's features are nearly free. Adding a bias, a fused exact-erf GELU,
and an entire second kernel for the softmax costs 0.07–0.23 ms on a 1.5–3.7 ms
op — 4–7%. Fusion is not where the published gain lives.

And at the matched configuration, *no DSL beats precision-matched torch*:

<!--TABLE_SPEEDUP-->

Read the two ratio columns together. Divided by the fp32 reference every lane
looks like a 2.7–3.4× win. Divided by the same torch code with `.half()` in
front of it, every lane is a loss. The entire published effect for this cell,
at a matched schedule, is the arithmetic change plus the weight cache.

### 2.4 The weight factor

The study asks for a two-way cached/uncached factor. A third level is carried,
because without it the uncached arm is a straw man: it pays for a transpose that
a competent uncached kernel would never perform.

<!--TABLE_WCACHE-->

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

<!--TABLE_EPILOGUE-->

This is a genuine capability difference created by abstraction level, and it is
reported on the *decomposition* side of the ledger, not as "the compiler is
slower".

### 2.6 The activation cast

<!--TABLE_CAST-->

### 2.7 What the hardware counters say about the four GEMMs

Per-kernel Nsight Compute census, steady-state launches only. Durations are
counters-only and are **not** the study's runtimes (Phase 1 established that
ncu's durations sit at the cold-clock transient); the resource columns are what
this table is for.

<!--TABLE_NCU_FUSED-->

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

<!--TABLE_FABS-->

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

<!--TABLE_SDPA_AUDIT-->

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

<!--TABLE_SDPA_INCUMBENT-->

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

<!--TABLE_SDPA_CROSS-->

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

<!--TABLE_NCU_SDPA-->

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

<!--TABLE_SDPA_ABS-->

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

<!--TABLE_RAW-->

---

## Appendix B — every SDPA cell

Both campaigns, all head dims and dtype pairs, including the cells that
fail the gate and the `S3-MP` cells that do not build (§5.1).

<!--TABLE_RAW_SDPA-->
