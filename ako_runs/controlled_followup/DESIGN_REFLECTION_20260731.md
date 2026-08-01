# Design reflection — the controlled cross-DSL study

**Date:** 2026-07-31 · **Branch:** `cross-dsl-6op-ncu-redo` · **Occasion:** completion of
`fused_epilogue_crossed_v1` (result tag `crossed_v1r1`)

This document answers three questions:

1. Are the experiments rigorous and under reasonable control?
2. Could the design better isolate the factor we actually want to study?
3. Are the resulting research questions worth answering, and what is the plan?

Every number below was recomputed from the sealed artifacts. Claims are marked
**[m]** where I recomputed them for this document, **[a]** where they are asserted
in repo prose that I did not re-derive.

---

## 0. The short version

The **correctness machinery is the best thing in this program** and is close to
publishable on its own. The **provenance discipline is now genuinely strong** —
`crossed_v1r1` is the first campaign whose launch commit was pushed and verified on
GitHub *before* the first GPU process. The **statistical instrument is competent but
was estimating the wrong quantity**; the fix was free and has been applied (§1.3) —
it leaves every verdict intact while turning the campaign's central null from
"undetectable within ±16 %" into "bounded at −4 %/+5 %".

The serious problem is elsewhere. **The design does not isolate the factor it names.**
The only epilogue strategy expressible in all four lanes runs a *byte-identical shared
CUDA kernel* for the entire epilogue, so the published "lane effects" measure GEMM
codegen with a common additive term — not what the four languages can express about
fusion. Separately, **the feasibility table — the campaign's most interesting result —
partly encodes which lane's author checked a CUDA return code**, not which lane can
express a strategy.

Both are cheap to fix. Neither is fixed by more replicates, and neither is visible in
the summary JSON.

---

## 1. Q1 — Is this rigorous and under reasonable control?

| Axis | Grade | Justification |
|---|---|---|
| Correctness gating | **A−** | Candidate-independent anchors, 4 adversarial distributions × 64 held-out seeds × 2 gate views, fp64 semantic reference, fixed-zero non-finite/negative thresholds, fail-closed. It caught 4 garbage-output cells. Docked for §1.5. |
| Provenance | **A−** | 1,428/1,428 record hashes verify **[m]**; source frozen and remote-verified 0.089 s before the first GPU process. Docked because the results are still untracked in git. |
| Preregistration | **B−** | Hash-bound `campaign.json`, no-fallback declared in advance, `g01` a genuine pre-declared control. But self-attested against the author's own remote, with no third-party timestamp. |
| Statistical inference | **B−** | Exact order-statistic intervals, paired randomized blocks, admirable refusal to over-read. But the estimator targets a transient (§1.3), and the unpaired ratio was published beside the paired one and is the one that got quoted (§1.4). |
| Measurement control | **C** | Fresh process per row, one physical GPU, joint order randomization, equal-*time* warmup, L2 flush, matched cast placement — all correct instincts, undone by no clock lock, no thermal gate, and a statistic computed over a non-stationary window. |
| Factor isolation | **D+** | §1.1 and §1.2. The treatment is not separated from the harness. |
| Reporting currency | **C+** | Scope lines and `claim_limit`s are exemplary. But at the time of writing, three controlling documents still deny that this campaign ran. |

**The single worst weakness is factor isolation**, and it is also the cheapest to fix.

### 1.1 The only four-lane-common strategy shares one epilogue kernel [m]

`global_intermediate` is the sole strategy legal in all four lanes, so it carries every
lane contrast the campaign reports. Its definition is *"a lane-native GEMM writes fp32
global scratch, then **one common CUDA kernel** performs bias, exact-erf GELU, and
row-softmax"* (`fused_epilogue_crossed_v1/README.md:9`).

That "common kernel" is literally common. Over all `global_intermediate` confirmation
records, `build_metadata.artifacts.postprocess.cuda_source_sha256` takes exactly **one**
value across all four lanes:

```
cuda_noptx      129e4a188a6bf0ecc51cbe4d6af7b8249f2f74057187d3a62c35255e12e31a1e
cuda_unlimited  129e4a188a6bf0ecc51cbe4d6af7b8249f2f74057187d3a62c35255e12e31a1e
tilelang        129e4a188a6bf0ecc51cbe4d6af7b8249f2f74057187d3a62c35255e12e31a1e
triton          129e4a188a6bf0ecc51cbe4d6af7b8249f2f74057187d3a62c35255e12e31a1e
```

So the headline numbers — TileLang leading Triton / CUDA-unlimited / CUDA-noptx by
1.08× / 1.16× / 1.23× at `g01` — decompose as *(lane-native GEMM) + (identical CUDA
epilogue)*. They are a **GEMM codegen contrast with a common additive term**. They are
not evidence about how well each language expresses a fused epilogue, which is the
question the campaign is named after.

This is construct invalidity, not noise. No amount of replication touches it.

### 1.2 The feasibility taxonomy partly measures error-handling hygiene [m]

The 40 non-passing measured outcomes look like a rich strategy × lane interaction. They
reduce to **two arithmetic facts**.

**Cause A — the fp32 epilogue tile exceeds sm_89's 99 KiB dynamic shared memory.**
`FSMEM_TILE_BYTES = BM*(BN+4)*4`. The grids where this exceeds 101,376 B are exactly
`{g05…g12}`, and that set matches the failures exactly:

| grid | BM×BN | `BM*(BN+4)*4` | over 99 KiB |
|---|---|---|---|
| g00–g04 | 128×128 | 67,584 | — |
| g05–g08 | 128×256 | 133,120 | **yes** |
| g09–g12 | 256×128 | 135,168 | **yes** |
| g13–g15 | 64×128 | 33,792 | — |
| g16–g18 | 128×64 | 34,816 | — |

- `smem_staged` × `tilelang` BUILD_FAILED: `g05…g12` ✓
- `smem_staged` × `cuda_noptx` BUILD_FAILED: `g05…g12` ✓
- `cuda_unlimited` LAUNCH_FAILED + GATE_FAILED: `g05…g12` ✓ (**both** strategies)

**Cause B — Triton's stages=4 operand pipeline.** `4*(BM*BK + BK*BN)*2` = 98,304 B for
`g07, g11, g15, g18` and 65,536 B for `g02`. Triton fails on the first four and passes
`g02` — i.e. four of the five stages=4 points, distinguished by operand footprint, with
error `Required: 106496, Hardware limit: 101376`.

**Why the same physical cause produces three different outcome labels.** `cuda_noptx`
routes through `cuda_fused_common.py:155-159`, which wraps the attribute call in a check:

```cpp
cudaError_t e = cudaFuncSetAttribute(..., FSMEM_BYTES);
TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute(", FSMEM_BYTES, ") failed: ", ...);
```

`cuda_unlimited` does not (`fused_cuda_unlimited.py:117-119`):

```cpp
cudaFuncSetAttribute(mma_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, FSMEM_BYTES);
attr_set = true;                      // return value discarded
...
mma_gemm<<<grid, block, FSMEM_BYTES, stream>>>(...);   // launches anyway
```

The lane that checks reports **BUILD_FAILED**; the lane that does not reports
**LAUNCH_FAILED** or **GATE_FAILED** with garbage output. Same wall, three labels.

**Worse: 8 of those cells are a pure artifact.** `FSMEM_BYTES` is
`max(FSMEM_TILE_BYTES, SMEM_BYTES)` and `EPILOGUE_DEVICE` is always compiled in
(`fused_cuda_unlimited.py:195`: *"gelu_exact is used by both epilogues, so
EPILOGUE_DEVICE is always in"*). So **`register_fused` × `cuda_unlimited` requests the
full fp32 tile of dynamic shared memory it never uses**, and dies on grids its own
strategy never touches. The recorded `shared_bytes` field reports only `smem` for the
`regs` arm (`:228`), which is why these failures look inexplicable in the summary — the
metadata under-reports what was actually requested.

**Consequence:** the `feasibility_strategy_x_lane_interactions` table — the campaign's
most novel-looking result — currently mixes language capability with (i) whether the
author checked a return code and (ii) a macro leaking into a strategy that does not use
it. It must be re-run before it is quoted.

### 1.3 The timing statistic estimates a transient, not a steady state [m]

Process medians are bimodal: **46 of 60** cell × distribution groups have a largest
adjacent gap >5 % of the cell median, **18 of 60** exceed 10 %. By lane the median gap is
TileLang 11.1 %, Triton 7.9 %, CUDA-unlimited 6.5 %, CUDA-noptx 5.2 %.

The worst offender is `global_intermediate.tilelang.g01` — *the baseline of every lane
contrast* — whose 15 process medians split into disjoint ~1.28 ms and ~1.58 ms clusters.

It is **not** a compiler effect: `cuda_source.sha256`, grid, block, and `shared_bytes`
are byte-identical across all 15 processes. It is **not** DVFS or thermal: over the 30
processes of that cell, the warmup phase immediately preceding the timed window runs at

| | ms/iter during warmup | median of timed window |
|---|---|---|
| "fast" processes (n=13) | 1.5566 | 1.2749 |
| "slow" processes (n=17) | 1.5523 | 1.5841 |

— **0.28 % apart in warmup, 24.3 % apart in the timed window.** A clock or temperature
cause would show in both.

The mechanism is visible in the trial trajectories. Within the 100 timed trials the two
modes move in **opposite directions**:

- "fast" processes: first decile 1.1193 ms → last decile 1.4463 ms (**+29.2 %**)
- "slow" processes: first decile 1.6595 ms → last decile 1.5550 ms (**−6.3 %**)

Both converge toward ~1.45–1.55 ms. The "fast mode" is not a faster steady state — it is
a **transient**, and the median over a window containing it is not estimating steady-state
latency at all. Whether a process reads fast depends on how much of its window was still
decaying.

**The fix is free and the data is already on disk.** Recomputing the same paired
per-block estimator on the settled tail (trials 60–99):

| contrast at `g01` | full window (0–99) | settled tail (60–99) | Δ point | CI shrink |
|---|---|---|---|---|
| triton / tilelang | 1.0878 [1.0757, 1.3102] | 1.0919 [1.0720, 1.1383] | +0.38 % | **3.5×** |
| cuda_unlimited / tilelang | 1.1708 [1.1525, 1.4249] | 1.1777 [1.1415, 1.2217] | +0.59 % | **3.4×** |
| cuda_noptx / tilelang | 1.2464 [1.2267, 1.4484] | 1.2599 [1.2414, 1.2689] | +1.08 % | **8.1×** |

Point estimates move ≤1.3 %; intervals shrink 3.4–8.1×. **Zero GPU-hours.** This is a
precision problem, not a bias problem — which is exactly why re-plumbing the harness is
the wrong first move and re-analysis is the right one.

**This has now been run** over all ten contrasts and all thirty stability ratios —
`reanalysis_tail_v1.py` → `results/crossed_v1r1/reanalysis_tail_v1.json`
(sha256 `e7296ce10585169e7dcf1f0d66c6bc59bf079d367d30491c46f42a8b3d03d1e5`). It is a
diagnostic overlay: the published full-window summary remains controlling, and no gate,
threshold, selection, or cell definition changed. Results:

- **Every verdict is stable.** All three lane effects still exclude 1.0; no strategy
  effect does. The direction of the campaign's conclusions survives.
- **Median precision gain 3.2×** (range 0.7–8.1×), max point-estimate shift **1.08 %**.
  One contrast (`cuda_unlimited` × `global_intermediate`) gets *wider* at 0.7×, which is
  the honest outcome where there was no drift to remove and the tail simply has fewer
  trials.
- **The strategy null becomes informative.** `tilelang` × `smem_staged` goes from
  [0.9819, 1.1517] — unable to exclude a 15 % slowdown — to **[0.9960, 1.0491]**;
  `tilelang` × `global_intermediate` from [0.8387, 1.0452] — unable to exclude a 16 %
  speedup — to **[0.9638, 1.0485]**. The null is now bounded at roughly −4 % to +5 %
  rather than ±16 %. **That is the difference between an uninformative null and a
  publishable one, and it cost nothing.**
- **Within-window drift** exceeds 5 % between first and last decile in **12 of 60** cell ×
  distribution groups, worst +12 %, and it is *not* TileLang-specific (worst offenders are
  CUDA-noptx and CUDA-unlimited). The non-stationarity is a property of the harness, not
  of one lane.

### 1.4 The largest reported "distribution effect" is a sampling artifact [m]

`distribution_stability` reports `per_cell_signed_over_positive = 0.8319` for
`global_intermediate.tilelang.g01` — a 17 % apparent input-distribution effect, and the
campaign's most eye-catching stability number. It is an artifact of §1.3:

| statistic | value |
|---|---|
| fast-mode draws, positive arm | 5 / 15 |
| fast-mode draws, signed arm | 8 / 15 |
| published unpaired median-of-medians ratio | **0.8319** |
| within fast mode | **0.9868** |
| within slow mode | **1.0029** |
| paired per-block median (already in the same JSON object) | **0.9873** [0.7987, 1.2250] |

The mode fraction crossed 50 % between two independent draws of 15 processes, moving an
*unpaired* median across the mode gap. There is no distribution effect here. **The
correct number was sitting in the same file.** Any document quoting 0.8319 needs an
erratum before it propagates.

The overlay from §1.3 settles this across the whole campaign. Using the paired estimator
on the settled tail, the **largest** deviation from unity among all 30 cells falls from
**16.8 % to 2.4 %**, and the number of cells whose interval excludes 1.0 falls from 3 to
**1**. *This campaign provides no evidence that the input distribution changes fused-op
timing.* Whatever support exists for the distribution-sensitivity thesis lives in the
Phase-1 tensor-core artifacts, and the two must not be conflated (§3, RQ(e)).

### 1.5 Every gate decision is 3.7 % from flipping [m]

All 150 GATE_PASSED cells bind on a single metric, `conformance_mixed/row_sum_error_max`,
taking only **three distinct values** (0.96337 ×105, 0.95952 ×30, 0.95128 ×15) at
95.1–96.3 % of threshold. The threshold in
`robust_gate/calibration/gate_spec_fused_v2.json` is `5e-07`, rounded up from
`observed_anchor_max × safety_factor = 3.630482550143199e-07 × 1.25 = 4.538e-07`.

**At the unrounded calibrated value every one of the 150 cells fails and the campaign
produces no timing data at all.** The spec is upstream and predates the campaign, so this
is not tuning — but the entire feasibility split rests on a rounding decision in a file
this campaign does not own, and the binding metric is not one anyone would call
"correctness". This should be reported to whoever owns the gate spec.

### 1.6 Verdict on Q1

Rigorous in the dimensions the program has been consciously working on — correctness,
provenance, preregistration discipline, refusal to over-read. Not yet under control in
two dimensions it has not been looking at: the *estimator* (§1.3–1.4) and the *treatment*
(§1.1–1.2). The reporting-currency failure the program has flagged for itself twice is
live again right now.

---

## 2. Q2 — Could the design better isolate the factor?

### 2.1 What is the treatment, actually?

Three distinct estimands are running under the single label "lane":

- **E1 — expressibility.** *Can this strategy be written at all in this language?*
  Identified by the feasibility arm, and this is the campaign's real contribution — but
  currently contaminated by §1.2 and by the 38 declared-unsupported cells.
- **E2 — attainable performance holding algorithm and effort fixed.** *Gestured at, not
  identified*, because of §1.1: the only common strategy shares its epilogue kernel.
- **E3 — attainable performance under free search.** Not this campaign's design at all
  (that is Phase 1/2), and the one most other papers already measure.

### 2.2 What is confounded with "lane"

| Confounder | Status | Controllable? |
|---|---|---|
| **Shared epilogue kernel** in the only common strategy | §1.1 | **Yes — decompose the strategy (§2.4).** |
| **Error-checking discipline** per lane builder | §1.2 | **Yes — one header, already written.** |
| Tensor-core mechanism: `wmma` (noptx) vs inline `mma.sync` (unlimited) vs `tl.dot` → `mma.sync` (triton) vs `T.gemm` (tilelang) | Structural | Measurable, not removable — Triton and CUDA-unlimited emit the *same* instruction, so a Triton–unlimited contrast is nearly a pure compiler contrast; CUDA-noptx is a different API. |
| Backend compiler (nvcc / LLVM / TVM) | Structural | Not removable; name it. |
| Author and per-lane tuning effort | Uncontrolled | Randomizable in principle (`reciprocal_v2` is the instrument). |
| Strategy factor is *read* by only some lane builders | Partial | Auditable — assert every builder consumes `cfg.extra["epilogue"]`. |

### 2.3 The 38 UNSUPPORTED cells are declarations, not measurements

Two grounds are declared in `core.py`:

- **`register_fused` × `cuda_noptx`:** *"nvcuda::wmma fragment element layout is opaque;
  no safe register epilogue exists without inline PTX."*
- **`smem_staged` × `triton`:** *"the pinned Triton API has no explicit user-managed
  shared-memory accumulator allocation."*

The Triton ground is solid — Triton genuinely does not expose a user-managed shared
accumulator, and the alternative *is* the `global_intermediate` strategy.

The CUDA-noptx ground is **weaker than it reads and needs a falsification test.** GELU is
elementwise and needs no fragment index mapping at all; bias can ride a second
same-type accumulator fragment without knowing the layout. Furthermore
`fused_cuda_noptx.py` never reads `cfg.extra["epilogue"]` — the declaration is doing
double duty as "not implemented". And `nvcuda::wmma` is the *deprecated* path, so the
claim as written reduces to "this deprecated API has no portable element mapping", not
"CUDA cannot express this". A reviewer with CUTLASS experience raises both objections
immediately.

**This test costs ~3 author-hours and <0.2 GPU-h, and it decides whether the reachability
result has content.** Better to find it ourselves than in review.

### 2.4 The highest-leverage design change

**Add a fourth strategy: `register_fused + common_postprocess`** — apply bias and GELU in
registers, then run the *same* `postprocess.py` kernel with `HAS_BIAS`/`HAS_GELU`
compiled out.

Today `ratio_to_register = 1.0267` for TileLang is one number covering two simultaneous
changes: *where bias/GELU is applied* **and** *whose softmax runs*. The fourth strategy
holds the softmax implementation fixed and varies only epilogue placement — which is the
contrast the campaign was designed to make. Cost: one `#define` and one builder branch,
**1–2 GPU-h**.

This ranks above re-plumbing the timing harness, for four reasons:
(i) the settled-tail re-analysis already buys 3.4–8.1× precision for free (§1.3);
(ii) the nuisance is a precision problem, not a bias problem — point estimates move ≤1.3 %
against effects of 8–23 %;
(iii) fresh-process isolation is a *working* control and this repo's own bug ledger
contains six harness bugs of exactly the class it prevents;
(iv) a perfect instrument measuring the wrong estimand is still measuring the wrong
estimand.

### 2.5 Ranked design changes

| # | Change | Isolates | Cost |
|---|---|---|---|
| 1 | Fourth strategy `register_fused + common_postprocess` (§2.4) | epilogue placement from softmax implementation | 1–2 GPU-h |
| 2 | Settled-tail / stationarity-gated statistic, as **re-analysis** — **already done**, §1.3 | 3.2× median precision, retroactively | **0 GPU-h** |
| 3 | Fix `cuda_unlimited` error checking; make `FSMEM_BYTES` epilogue-conditional; re-run the audit | capability from error-handling hygiene | ~8 GPU-h |
| 4 | Falsify both UNSUPPORTED grounds (§2.3) | declaration from measurement | <1 GPU-h |
| 5 | Sham duplicate-label negative control + resolution-floor publication gate | an empirical floor below which no effect may be reported | ~5 % of campaign GPU time |
| 6 | Second architecture, feasibility matrix only | hardware-bound from language-bound | 1 GPU-day + hardware |

### 2.6 Explicitly rejected

- **"Expand the grid set so a non-empty all-factor common-feasible region exists."**
  Arithmetically impossible: `common_feasible_definition` requires all four lanes, and
  two strata have one lane unsupported at *every* grid. `[]` was forced before any GPU
  ran. The only real options are to relax the definition to *all supported* lanes, or to
  add a strategy every lane can express (§2.4 does this).
- **Same-process round-robin timing.** Unvalidated fix for an unidentified mechanism, and
  it surrenders the isolation control that catches JIT/allocator/cache leakage. Defer
  behind a 20-minute diagnostic (device pointers + `ncu` L2 hit rate on a subsample).

---

## 3. Q3 — Do these questions matter?

External work was checked directly; the four papers below were fetched and verified to
say what is attributed to them.

| Prior work | What it owns |
|---|---|
| Sarkar, *The Correctness Illusion in LLM-Generated GPU Kernels*, arXiv:2606.20128 (2026-06-18) | Loose tolerance / fixed-shape validation certifying buggy kernels; 9/9 seeded bugs, 15/15 controls, extended to 26 ops × 5 architectures. |
| Redko et al., *Prior Knowledge or Search?*, arXiv:2605.19782 (2026-05-19) | *"CUDA improves monotonically under iterative feedback, while TVM IR actively degrades"* — cross-language LLM search divergence. |
| Hari, Balaji, Damani, Huang, Kozyrakis, arXiv:2603.29010 (2026-03-30) | μCUTLASS + Speed-of-Light guidance; 0.40× → 1.27× → 1.56×. **Explicitly does *not* hold optimization strategy fixed across abstraction levels.** |
| Yadav, Zhao, Kumar, *Evaluating CUDA Tile on Hopper and Blackwell*, arXiv:2604.23466 (2026-04-25) | Same CuTile attention kernel: 2.5× FlashAttention-2 on B200 but **53 %** of it on RTX PRO 6000 — abstraction rankings invert across GPU generations. |

### Triage

| RQ | Verdict |
|---|---|
| **(a) A reachability / expressibility frontier — which optimizations are expressible at all per DSL** | **Survives. The only one that clearly does, and it is the paper.** No tile-DSL work reports an *expressibility denominator*; everyone reports best-effort speedups over whatever was expressible. The intellectual ancestor is Pennycook's performance-portability metric with its Φ=0-if-unsupported rule — cite it or look naive. **Contingent on §2.3 and §1.2:** as it stands the frontier rests on two declarations and an unchecked return code. |
| **(b) Does DSL choice matter once strategy is held fixed?** | **Survives only as the null, and only after the free re-analysis.** Killed as a *lane* claim by §1.1 (shared epilogue) and by single-GPU scope against arXiv:2604.23466, which shows orderings inverting across architectures. What does survive, and is genuinely interesting, is the **dissociation**: once a strategy is expressible in two lanes, lane moves ≤1.23×, strategy ≈1.00×, interaction null — *while the feasibility interaction is maximal*. Note the irony worth stating in the paper: fixing the instrument is what buys the negative result. |
| **(c) Does an optimization recipe transfer across DSLs?** | **Survives conceptually; least occupied after (a).** Prior work measures portability of an *implementation across hardware*; nobody measures portability of an *idea across languages*. The deliverable is a taxonomy — idea-portable / schedule-portable / lane-locked — that outlives this hardware. Zero data exists, and the instrument (`reciprocal_v2`) has a launch-blocking defect (§4). |
| **(d) Do LLM search trajectories converge across DSLs?** | **Kill as standalone.** arXiv:2605.19782 has the cross-language result already, and novelty decays monthly. Only surviving angle: *trajectories conditioned on the reachability frontier* — do agents burn budget attempting strategies their lane cannot express? That is (a)+(d) and nobody has it. |
| **(e) Do correctness gates change which wins are real?** | **The general claim is taken** (arXiv:2606.20128, with 5-architecture validation). The sharper mechanism is still ours and is a better paper: *the input distribution, not the tolerance, does the work — `torch.rand`'s all-positive budget carries an fp16 win that is legal under every stated rule*. **But this campaign does not support it** — per §1.4 its distribution effect is ≈0.99, i.e. nothing. That evidence lives in the Phase-1 tensor-core artifacts. Given this repo's errata history, conflating the two is exactly what a hostile reviewer will hunt for. Requires a public-corpus replication (KernelBench / kernelbot data; `rand` vs `randn`; fp64 reference; report what fraction of published wins invert) or it should be dropped. |

### What generalizes

The transferable contribution is **not** "TileLang is fast on Ada." It is:

1. **Expressibility before speed** — a comparison of programming systems must report the
   denominator of what each system could express, not only the numerator of what it ran.
   This generalizes to any DSL/compiler comparison on any hardware.
2. **The shrinking-effect-size ladder.** For matmul, the four-way spread contracts from a
   published **7.0×** to **1.31×** once arithmetic and tile are normalized and search
   budget is equalized (`PHASE1_REPORT.md:748,768`), with three independent routes to the
   same number. The fused op reproduces the pattern at a similar magnitude — 1.2505× at
   frontier-closure v3, ~1.23× at `g01` here — though these are *different operations*, so
   the ladder is a consistent pattern rather than a single monotone series, and the `g01`
   rung must be re-derived after §1.1 before it is quoted. **This is a methodology result
   about the field's headline numbers, and it is the most transferable thing this program
   has produced.**
3. **The gate/distribution calibration appendix**, positioned as concurrent with
   arXiv:2606.20128 and arXiv:2604.22032, not competing with them.

**Recommended shape: one paper, not five.**

### What does not generalize

Every absolute latency, the specific 19-point grid, the sm_89 99 KiB wall, and the lane
ordering. arXiv:2604.23466 is direct evidence that the last of these inverts across
hardware. Without a second architecture, RQ(a) is a single-machine anecdote and RQ(b) is
not defensible.

---

## 4. Planned-but-not-done — status and verdicts

Three blockers recorded in `RUN_20260731.md` are **stale as of now**: the sandbox *can*
reach the driver, all four frozen RTX 6000 Ada UUIDs are visible and idle, and the
crossed campaign did launch. Genuinely absent: `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

### Tier 0 — free, do today

| # | Item | Cost |
|---|---|---|
| 0.1 | ~~Settled-tail re-analysis of the 900 existing raw records~~ — **DONE 2026-07-31.** `reanalysis_tail_v1.py` → `results/crossed_v1r1/reanalysis_tail_v1.json`. All verdicts stable; median 3.2× precision gain; the strategy null is now bounded at −4 %/+5 %; the distribution effect collapses to ≤2.4 % (§1.3–1.4). | 0 GPU-h |
| 0.2 | **Commit the 1,618 result files** and push. Every hash chain currently proves *result → frozen source*; nothing proves *result → external record*. | minutes |
| 0.3 | **Correct the stale ledger**: `RUN_20260731.md:38`, `controlled_followup/README.md:14`. Do **not** edit `campaign.json` (hash-bound to 1,428 receipts) or the sealed evidence index — add an adjacent errata note. | minutes |
| 0.4 | **Erratum for `0.8319`** → paired 0.9873 [0.7987, 1.2250] (§1.4), and republish the feasibility taxonomy with the corrected two-cause split (§1.2). | 0 |
| 0.5 | **Report the gate-saturation finding** (§1.5) to the owner of `gate_spec_fused_v2.json`. | 0 |

### Tier 1 — decides whether there is a paper

| # | Item | Cost |
|---|---|---|
| 1.1 | **Falsify both UNSUPPORTED grounds** (§2.3). Highest priority: it decides whether RQ(a) has content or is a Python dict literal. | <1 GPU-h + 6–10 author-h |
| 1.2 | **Fix `cuda_unlimited` error checking, make `FSMEM_BYTES` epilogue-conditional, re-run the audit** (§1.2). Recovers 8 `register_fused` cells lost to a macro they never use. | ~8 GPU-h |
| 1.3 | **Add the fourth strategy** (§2.4). The actual answer to Q2. | 1–2 GPU-h |
| 1.4 | **Adopt `legacy_cuda_harness_fix/checked_cuda_launch.h` at real call sites** + a lint asserting new CUDA wrappers include it. It currently has zero call sites — an unadopted safety header reads as "fixed" while the defect stays live, and §1.2 shows the just-completed campaign ran on the defective wrapper. | ~1 h |
| 1.5 | **Sham duplicate-label negative control** + resolution-floor publication gate, as standing policy. | ~5 % of GPU time |

### Tier 2 — external validity, currently planned nowhere

| # | Item | Cost |
|---|---|---|
| 2.1 | **Second architecture for the feasibility matrix only.** The largest external-validity hole in the program — it has no directory, no manifest, no campaign anywhere in this subtree. | 1 GPU-day + hardware |
| 2.2 | **Public-corpus gate replication** for RQ(e), or drop RQ(e). | days, low GPU |

### Tier 3 — the three stalled campaigns

**`reciprocal_v2` — REDESIGN, then RUN. Highest scientific value of the three.**
It is the only design that makes home-field bias falsifiable. Three things must change
first:
- **The retune grid does not identify origin.** The 24 retune plans collapse to **2**
  distinct candidate lists: `cuda_noptx`, `cuda_unlimited` and `tilelang` destinations
  share one hash, only `triton` differs. For 2 origins × 3 destinations per translator the
  plan is byte-identical except the `recipe_origin` string. The retuned arm shrinks the
  interaction toward zero *by construction* — and a null interaction is exactly what the
  prereg reads as "a real compiler effect". Either make the grid a neighborhood of the
  *origin* recipe, or restrict the interaction estimand to the `literal` arm.
- **The execution runner does not exist** and is not listed among the fail-closed
  preconditions. Write it or disclose it.
- **The translator factor needs an honest decision.** If a genuinely independent second
  translator cannot be sourced, drop the factor to one level, re-freeze at 24 cells, and
  forfeit the translator-skill bound in writing. Do not fabricate two translators from one
  agent — `production_v1/README.md:20` already forbids this.
- No budget field exists anywhere in `reciprocal_v2/`. Record one.

**`effort_frontier_v1` — REDESIGN, then PILOT.** Best-specified estimand in the program
and the controller layer is genuinely finished, but the executor registry blocks all five
lanes, not just cuBLASLt, so a 3-lane descope forfeits *"does the industrial low-level path
pay?"* — the part that is interesting outside this repo — while saving little engineering.
Build two executors, run 2 lanes × 2 replicates × 2 checkpoints as a pilot to prove the
effort clock and failure ordering, then decide the full campaign. Pre-launch: add a
per-trajectory token **cap** (usage is already instrumented; the cap is missing); record
lane×GPU imbalance as a named limitation rather than "fixing" it (the manifest is already
maximally balanced and the proposed rotation breaks `launch.py:56-58`); and relax the
immutable-model-revision gate to "alias + response-header model string + resolution
timestamp, with model drift named as a threat to validity", or it is a permanent veto.

**`convergence_v2` — KILL as designed.** Widest gap between apparent and actual readiness
in the subtree: 21 passing tests and six lock files, but no executor, no trajectory
driver, no worktree provisioner, and nothing that writes `results/outcomes.jsonl` — none
of which appears in the declared non-negotiable blockers. The primary outcome is
**unpreregistered**: `--tau-s` is a required CLI float with no value anywhere in `locks/`,
`manifests/`, or `gates/`. A post-hoc τ is precisely the researcher degree of freedom this
program exists to eliminate. Novelty is decaying against arXiv:2605.19782. **Kill the
128-trajectory prompt extension outright** (40 % of budget for the least interesting
framing effect). If revived: preregister the event definition and τ first; run the
sum/SDPA gate calibrations as standalone deliverables (they have value independently);
smoke-test one executor on 4 trajectories; then re-scope as *"trajectories conditioned on
the reachability frontier"*, which is the only surviving angle.

**`provenance_supplement_v1` — effectively done.** `supplement_lock.json` records
`parent_missing_from_primary_count = 23`, `locked_union_count = 49`; archives exist. Only
residual action is the push in Tier 0.2. Note that the "complete" evidence bundle omits
`runner2.py` (the actual timing loop) and `common2.py` (the fp32 reference oracle) —
i.e. it omits the measurement apparatus. The supplement closes this; land it.

---

## 5. One-line summary

The correctness gate deserves publication on its own merits; the reachability frontier is
the paper, and it currently rests on two declarations and an unchecked CUDA return code
that a day's work could resolve; the performance numbers measure GEMM codegen through a
shared epilogue kernel and should not be quoted as a DSL result; the timing instrument
needs a free statistic change rather than a re-plumbing; and every remaining blocker in
this subtree is exactly one of — a missing executor, a missing second party, or a
`git push`.
