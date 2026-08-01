# Adversarial review of `CONTROLLED_CROSS_DSL_REPORT.md`

> Corrections and current controlling interpretation: [`controlled_followup/REVIEW_ROUND2_RESPONSE_20260731.md`](controlled_followup/REVIEW_ROUND2_RESPONSE_20260731.md).

> Target: `ako_runs/CONTROLLED_CROSS_DSL_REPORT.md` at commit `8854ae0` (branch
> `cross-dsl-6op-ncu-redo`), together with its evidence base
> (`phase1_matmul/PHASE1_REPORT.md`, `phase2_fused_sdpa/PHASE2_REPORT.built.md`,
> `SIX_OP_ANSWERS.md`, `CONVERGENCE_PROTOCOL.md`, the 24 `<op>/<dsl>/convergence.csv`
> cells, and the raw result JSONs).
>
> Method: five independent adversarial reviews (research merit, statistical rigor,
> confound/control audit, artifact-vs-claim number verification, external validity),
> followed by nine independent skeptic passes that attempted to *refute* the
> strongest criticisms against the artifacts on disk. Every number quoted below was
> re-derived from the committed CSVs/JSONs, not taken from the report's prose.
> Criticisms are labeled **[CONFIRMED]** / **[PARTIALLY CONFIRMED]** when they
> survived that refutation pass.

---

## Verdict in brief

The controlled core (Phase 1, most of Phase 2) is a genuinely strong experiment —
bit-identical cross-DSL outputs, thermal-soak protocol, 5-process medians with CIs,
winner re-confirmation, and every load-bearing number reproduces exactly from the raw
artifacts. The report's *negative/existence* claims (no universal DSL ceiling; the
published 7× GEMM spread was mostly discovery artifact) are well supported **as
scoped**. Two things keep it short of a rigorous research contribution today:
(1) the trajectory/"convergence" half rests on **one stochastic LLM-agent run per
(op, DSL) cell**, with the searcher never disclosed, per-op prompt hints uncontrolled,
and lanes run on different GPUs in different clock states — so Sections 3/5/7 describe
particular agent runs, not DSL properties; and (2) the headline quantitative claim
(7.0×→1.31×) is fully executed for exactly one operation and is gate-legal only under
`torch.rand`'s all-positive input distribution. The report also undersells its actual
novelty: the transplant-and-retune audit methodology and the two-sided tolerance-gate
pathology are the publishable contributions; several of the seven headline conclusions
are careful confirmations of known results.

---

# Part A — Q1: Research insights, data sufficiency, intellectual merit

## A.1 Scorecard for the seven final conclusions (report §10)

| # | Conclusion | Verdict | Why |
|---|---|---|---|
| 1 | No DSL is a universal performance upper bound | **Well supported** (as an existence/counterexample result, scoped to sm_89 + these implementations) | Bit-identical-output control makes "compiler lanes beat hand lanes at matched config" credible. But two of the four-way orderings quoted rest on overlapping CIs (see B.2-5), and the memory-bound row of the §2 table is native-search + ncu evidence, not a controlled study. |
| 2 | Most large native gaps are not proven ceilings | **Well supported qualitatively; quantitatively n=1 op** | The full transplant + equal-budget protocol exists only for standard matmul (1.31×). The fused op *inherits* equal-budget via a FLOP-identity argument the report itself declares invalid in §9; the SDPA "matched" spread at d=1024 is 1.60× (18.98/21.33/25.57/30.45) with the winner flipping across head dims — the collapse magnitude is op-dependent and the report never says so. **[CONFIRMED]** |
| 3 | Ideas transfer, schedules don't | **Well supported — but confirms Halide/TVM/Ansor-era folklore** | The A/B/C/D decomposition is clean. As a *contribution* it needs positioning against the schedule-transfer literature; currently no related-work framing exists anywhere. |
| 4 | Triton cheapest trial, then TileLang, then CUDA | **Well supported on the narrow compile-cost claim** | `compile_cold.json` backs 1.19/6.01/36.4/36.4 s at n=5 each. The trajectory half is correctly self-limited by the report — but see the searcher confound (B.2-1): the native iteration/seconds numbers are single agent runs on different cards. |
| 5 | TileLang lower level not intrinsically faster | **Plausible but underpowered** | The preregistered rule returned *no verdict* on the key H1-vs-M2 pair (+7.7% sits in the rule's own 3–10% gap); the softmax pair (1.2%) is at timer resolution; the only clean signal (S3-M vs S3-L, +3.3–13%) is mechanistically TileLang-specific — the manual arm is forced through smem because `T.gemm`'s fragment layout is opaque. That supports "TileLang's lower levels are hobbled by its own opaque layouts", not the general claim. No manual-async arm exists on Ada (interfaces target Hopper), so the conclusion is untested exactly where readers will apply it. |
| 6 | Simple kernels need fewer meaningful decisions | **Not supported as stated — near-tautological** | A descriptive mean over four single stochastic LLM trajectories per op, mixing a *hinted* calibration op (layer_norm) and 1/1 cells where CUDA lanes stopped after one variant against a weak fallback. "Meaningful decisions" has no outcome-independent definition. Weakest of the seven; dilutes the rest. |
| 7 | Migration often closes the gap, not automatically | **Supported as hedged, but "often" rests on ~1.5 fully controlled op families** | Matmul fully; fused matched-only; SDPA transplanted but tile-unmatched. |

## A.2 Where the intellectual merit actually lives (currently undersold)

The report leads with its least novel material. The defensible, publishable
contributions are:

1. **The transplant-and-retune audit methodology.** The sharpest claim this artifact
   set supports: *apparent cross-DSL performance gaps produced by LLM-driven kernel
   search are mostly discovery artifacts, and a fixed protocol (normalize arithmetic →
   transplant recipe → verify bit-identical/dynamic-work equivalence → retune
   schedules under equal budget → re-measure winners in fresh processes) collapses
   them* (7.0×→1.31× on matmul, by three independent routes). This is a
   methodology-plus-cautionary-tale paper for the KernelBench/LLM-kernel-generation
   era, and no published DSL comparison we know of has the bit-identical-output
   control. Ironically, the strongest available evidence for this thesis — the repo's
   own two-searcher replication (`OPUS48_VS_SONNET46_REPORT.md`), where identical
   (op, DSL) cells swing 1.7–2.1× when only the model changes — is never cited by the
   report.
2. **The two-sided tolerance-gate pathology.** One fixed `1e-4 + 1e-4·|ref|` gate
   simultaneously (a) permits 82% per-element relative error at softmax output scale —
   a row-*reversed* answer passes 99.86% of elements — and (b) rejects the correctly
   rounded fp16 storage of the *exact* answer at SDPA scale (19.1% of elements), with
   two (algo, dtype) cells proven unreachable by any implementation via an fp64 bound.
   Plus: fp16 GEMM is gate-legal only under `torch.rand`'s all-positive inputs
   (0/20 passes under RMS-matched `randn`, 78.3% elements out). This is an exportable
   benchmark-design result about KernelBench-style gates, quantified in both failure
   directions.
3. **The vendor-library-hole quantification.** Custom kernels beat precision-matched
   torch by 1.12×/1.01×/**1.74×** at d=128/256/1024 — the advantage tracks
   FlashAttention's head-dim cap, not the DSL — with the FLASH/K3 crossover reproduced
   in all four lanes. Converts "LLM kernels beat PyTorch" folklore into a falsifiable
   statement about where headroom lives.

Supporting novelty: the split-K finding (in-block flush is an *accuracy* lever that
costs throughput; bias exactly linear in KC), and the warm/cold compile-cache
inversion (six named harness bugs that produce clean-looking wrong numbers).

## A.3 What is confirmation of known results

Conclusions 1/3/4/6/7 largely re-confirm on n=1 GPU: the Triton paper's
cuBLAS/hand-CUDA parity claim; the autotuning literature's "algorithms transfer,
schedules retune" (Halide/TVM/Ansor); JIT-vs-nvcc compile cost; memory-bound kernels
tying at roofline. That is fine — replications with better controls have value — but
the report contains **no related-work positioning at all**, so a reader cannot tell
which findings are claimed as new. A program committee would reject the current
framing for exactly this reason.

## A.4 Is the data sufficient for the conclusions?

- **Sufficient** for the narrow negative claims as literally scoped in the report:
  no universal ceiling ordering *on this GPU, these implementations, this input
  distribution*; the published 7× matmul spread was not a codegen gap.
- **Not sufficient** for: any convergence-rate ranking (n=1 trajectory/cell,
  hinted/hinted-against prompts, mixed GPUs — B.2-1/2/3); the 1.31× as a *general*
  post-transfer spread (n=1 op — A.1-2); the ~1.10× "TileLang compiler intrinsically
  better" residual as a DSL property (single toolchain snapshot tilelang 0.1.11 /
  triton 3.6.0 / nvcc 13.1; home-recipe anchoring uncontrolled — B.2-7); conclusion 5
  beyond sm_89; conclusion 6 in any form.
- **Missing entirely**: idiomatic-ceiling anchors (no CUTLASS/cuBLASLt, no
  `@triton.autotune` arm, no TileLang-autotuner arm). Best sustained rate is ~131
  TFLOP/s (209 in counter runs) — an interior point of the tensor-core roofline, so
  the 1.31× spread is a spread among interior points, and the search-space
  suppression is asymmetric (autotuner exclusion suppresses Triton's core idiom;
  swizzle/cross-block-split-K/persistent exclusion suppresses the CUDA lanes').

## A.5 What a skeptical reviewer demands before this is publishable

1. Searcher replication (k≥3 trajectories/cell, ≥2 models) or explicit downgrade of
   all trajectory claims to case-study status.
2. A second wide-gap op through the *full* transplant + equal-budget protocol.
3. A distribution-robust gate rerun (signed inputs + absolute-error floor) — does
   1.31× survive when fp16's legality loophole closes?
4. Idiomatic-ceiling reference lanes to anchor "ceiling" against the machine.
5. One second architecture (Hopper — which also unlocks the missing manual-async
   TileLang arm) for the two decisive tables.
6. Related-work section; lead with the methodology + gate findings as the claimed
   contributions.

---

# Part B — Q2: Are the experiments controlled? Variables, constants, isolation

## B.1 Control matrix by evidence tier

Factors: **C** = held constant, **T** = treatment (deliberately varied), **X** =
varied incidentally (confound), **U** = unknown/unrecorded.

| Factor | Native 6-op search | P1 matched A/B/C/D | P1 equal 19-pt grid | P2 fused matched | P2 SDPA cross-DSL |
|---|---|---|---|---|---|
| DSL / compiler stack | T | T | T | T | T |
| Algorithm & work count | **X** (each lane free) | C (bit-identical, ncu-verified) | C | C (ladder G→GBGS) | T (K3/FLASH registered) |
| Operand/acc/output dtype | **X** | C | C | C | T (registered factor) |
| Tile geometry / stages / threads | **X** | C (imposed; stages=3 known lane-biased, later corrected) | T (shared grid; triton built 15/19) | C (verbatim P1-D transplant; **no** per-lane depth sweep on fused shape) | **X** (each lane picks own tile per (algo,d) — SPEC2 concession) |
| Input shapes & distribution | C (`torch.rand`) | C (+ `randn` control measured) | C | C | C (d ∈ {128,256,1024} swept) |
| Correctness gate | C op-wise, **X** across ops (meaning varies 4096× with output scale) | C | C | C (near-vacuous at softmax scale — measured) | C (unsatisfiable in 2 cells — measured) |
| Search budget & stop rule | **X** (uniform on paper; adjudicated by agent belief; iteration caps differ per op) | n/a | C (points; **X** in time — 20 s vs ~12 min compile) | n/a (inherited, invalidly) | n/a |
| Searcher (LLM model, seed) | **U/X** (n=1 Opus 4.8 run/cell; layer_norm unattributed; Sonnet campaign exists, diverges, unused) | n/a | n/a | n/a | n/a |
| Prompt/hints per cell | **X** (anti-hint on matmul, answer on layer_norm, pro-hint on SDPA) | n/a | n/a | n/a | n/a |
| GPU unit & clock state | **X/U** (lanes on different cards of a 4-GPU host; no `gpu` column; identity baselines differ 27–42%) | C (one pinned idle GPU, preflight-guarded) | C | C | C |
| Warmup/thermal state | **X** (single runs @200 iters, mid-transient) | C (2.0 s fixed-time soak, derived from data) | C | C | C |
| Cache state (compile) | **X** (mixed warm/cold) | C (measured separately, cold & warm) | C | C (corrected in §1.3) | C |
| Measurement replication | **X** (n=1 per row) | C (5 proc, randomized, CIs) | C (2-proc rank → 5-proc confirm) | C | C |
| Artifact identity (pinning) | **X** (solutions drifted from logs — caught) | C within campaign; **U** over time (no source hashing, tree not in VCS) | same | same | same |

Reading: **Phase 1 matched is near-exemplary; the native tier has confounds in almost
every row; Phase 2 is matched for the fused ladder but only algorithm/dtype-matched
for SDPA.** The report's evidence-hierarchy structure (controlled overrides native) is
the right response — the failures below are where native-tier material still leaks
into DSL-level claims, or where controlled-tier headlines outrun their own rules.

## B.2 Confounds and defects, ranked (with verification status)

1. **The searcher is the dominant uncontrolled variable, and it is undisclosed.**
   **[CONFIRMED core]** Every native trajectory is one stochastic LLM-agent run per
   cell (Opus 4.8 for the five compute ops; layer_norm's searcher is unrecorded
   anywhere). The report never says the searches were LLM-driven, never names a
   model, never states n=1, and never cites `OPUS48_VS_SONNET46_REPORT.md`, whose
   second complete campaign swings identical cells by 1.7–2.1×
   (matmul_gelu/triton 2.36→5.07; sdpa/cuda_noptx 1.03→1.75; sdpa/tilelang 2.57→3.41)
   — larger than most cross-DSL deltas in §5/§7. This cuts both ways: it invalidates
   §3/§5/§7 as DSL properties *and* is unused ammunition for conclusion 2 (native
   gaps don't replicate across searchers ⇒ discovery artifacts). Note the Sonnet
   campaign logged no compute clock, so it cannot be pooled into §5 — but it can and
   should be cited as replication evidence for instability.
2. **"All measurements are on one RTX 6000 Ada" is false for the native tier.**
   **[PARTIALLY CONFIRMED — direction right, mechanism refined]** The tilelang lane's
   *identity* (vendor) baselines are 27–42% faster than the other three lanes on every
   compute-bound op (cuBLAS matmul 4.46 vs 6.08/6.09/6.09 ms; cuDNN depthwise 3.39 vs
   5.84; F.sdpa 57.7 vs 80.6; fused 5.98 vs 6.88) while memory-bound identities match
   (those were serialized on GPU3 per `tools/HINTS_CONVERGENCE.md`). The repo's own
   `COMPUTE_FRONTIER_FINDINGS.md` C4 documents GPU-0/1/2 slow-clock state. So §5's
   cross-lane `cum_compute_s` and any cross-lane reading of native `runtime_ms` mix
   DSL signal with card/clock state; `convergence.csv` has no `gpu` column, so it
   cannot be corrected post-hoc. The report's generic "denominator choices" concession
   understates this.
3. **Per-cell prompt hints injected different, sometimes wrong priors — never listed
   as a factor.** **[CONFIRMED]** All standard_matmul cells were told *"~1× is the
   physical ceiling. Confirm the floor within the cap and STOP"* (iteration cap 2) —
   falsified by the eventual 4.13×; SDPA cells were told *"flash-style fusion is a
   real win"* (cap 6); layer_norm cells were given the answer (committed floors
   1.95–2.16×, and the tilelang log stops citing the ~2.10× calibration ceiling). §5
   bolds the hinted layer_norm cell as TileLang's fastest convergence with no hint
   annotation (the sole "hinted" label appears once, in §7). Search-breadth
   differences attributed to DSL trajectory are partly attributable to what each
   cell's prompt asserted.
4. **The §5 convergence recomputation applies a 5% threshold to single-run
   measurements with ~10% demonstrated noise.** **[CONFIRMED]** Each convergence row
   is one bench at `--num-warmup 200` — the depth Phase 1's own stability study
   measured at 10.2% between-process spread (mid-transient) for a ~1 ms kernel. The
   headline TileLang matmul cell ("5/9 @ 48.6 s" vs "4/9 @ 39.3 s") is decided by a
   **0.10%** margin (row-5 speedup 3.9735 vs threshold 3.9776). The table is exactly
   recomputable (verified, all 24 cells) but its cell boundaries are far below
   measurement resolution.
5. **Strict four-way orderings asserted where adjacent CIs overlap, against the
   reports' own reading rules.** **[PARTIALLY CONFIRMED — "preregistered" was an
   overstatement; the CI-overlap convention is stated, not preregistered]**
   P1 confirmed grid: triton [1.041, 1.102] vs cuda_unlimited [1.078, 1.116];
   P2 fused matched: triton [1.7443, 1.7799] vs cuda_unlimited [1.7675, 1.9351] —
   both quoted as strict chains in §2 ("tilelang < triton < CUDA-PTX < CUDA-WMMA"),
   while Phase 2's own captions say overlapping intervals mean the ordering is
   unresolved. The spread *endpoints* (1.31×, and tilelang-vs-triton) are solid; the
   middle of both chains is not.
6. **The fp16 legality of the headline result is distribution-scoped, and the
   headline doesn't carry the scope.** **[PARTIALLY CONFIRMED — §9 concedes it
   generically; the 78% figure never reaches the summary]** Every gate-passing fp16
   variant behind 7.0×→1.31× passes only under `torch.rand`; under RMS-matched
   `randn`, 0/20 passes at every KC (78.26–78.29% elements out; fp32 fails too but at
   0.004% — a 20,000× gap). No distribution-robust gate was ever run (declared out of
   scope). Under a corrected gate the *recipes themselves* could change (bf16, deeper
   split-K), not just the numbers — the 1.31× could dissolve rather than shift.
7. **Home-recipe anchoring of the ~1.10× residual.** The matched recipe and the
   19-point grid axes (BM/BN/BK/stages/KC) are the family TileLang's own search
   discovered. Every other lane realizing the discoverer's recipe slightly worse is
   the expected outcome of transplant *direction* alone; no reverse transplant (e.g.
   Triton's autotuned fp16 family, CUDA-PTX's mma.sync progression into other lanes)
   was run. "Is TileLang's compiler intrinsically better? Yes, but by ~1.10×" outruns
   this control. (The residual is real and replicated *within* the tested family —
   the skeptic pass on the noise-floor version of this attack found the CIs separate
   cleanly — but "intrinsically" is unearned.)
8. **The §2 summary table mixes control tiers under one "Controlled observation"
   header.** **[PARTIALLY CONFIRMED — report discloses the SDPA tile freedom in §4,
   not beside the table]** The SDPA row is algorithm/dtype-matched but *not*
   tile-matched (SPEC2: each lane compiles its own per-(algo,d) tile); the
   memory-bound row is native-search + ncu evidence with no matched-implementation
   tier at all (and layer_norm's four-lane convergence is partly by construction —
   lanes were pointed at committed floors). Both sit next to genuinely
   geometry-matched GEMM rows.
9. **The native stop rule is adjudicated by the agent's own beliefs.** Stop branch 2
   ("ncu confirms the binding roofline") let the Opus triton matmul lane stop on a
   belief ("tensor cores blocked by the fp32 gate") that the tilelang lane falsified
   in the same campaign. "Converged in fewer trials" is unidentifiable against "gave
   up earlier" under a belief-conditioned stop rule; any future convergence
   comparison needs an exogenous budget.
10. **No artifact pinning; Phase 2 tree not under version control.** No source hash
    in any result row; the incumbent checker loads mutable `solution/` files; the
    drift class already falsified one published ranking (tilelang ≫ triton on the
    fused op). The one decisive reproduction the report itself prescribes —
    re-benchmarking both archived `solution_opus48` fused files in the same
    randomized processes — was never run, and it is one loader argument away from the
    existing harness.
11. **Noise-floor bookkeeping.** The "~4%" floor is a single-configuration point
    estimate (tilelang/D, n=5, 3.5%); observed per-cell spreads reach 5.7% (tilelang/B)
    and 6.3% (cuda_unlimited/D); cross-campaign anchor drift is 4.10% (above the
    floor). An asymmetric-standards charge (7.7% called "inconclusive" while 10.2% is
    called "real") was **partially refuted**: the 10% materiality band was
    preregistered only for the within-TileLang abstraction question, and the decisive
    cross-DSL gaps have non-overlapping same-campaign CIs — but no band was ever
    stated for cross-DSL claims, and the report should state one framework and grade
    everything under it. Also: point estimates are medians-of-medians while CIs are
    t-intervals on the *mean* of medians (tilelang cold compile: 6.01 quoted, CI
    centered on 5.73).
12. **Smaller disclosure gaps.** GPU clocks were free-running everywhere (locking
    needs root, denied — `tools/refresh_all_floors.py`); never mentioned in the
    report. "Equal 19-point search" — triton built 15/19 points; one CUDA winner
    rank-flipped at confirmation (both facts are in Phase 1 but not beside the §2
    row). The decisive "bit-identical outputs" control has **no committed script** —
    it is corroborated indirectly (error stats identical to 17 significant digits
    across lanes, `matched/summary.json`), which is very strong, but an audit report
    should distinguish "verified by committed code" from "corroborated".

## B.3 What is genuinely well controlled (credit where due)

- **Equivalence controls most published comparisons lack**: bit-identical outputs at
  every A/B/C/D variant across all four DSLs; dynamic ncu work verification (exactly
  2³⁶ FFMAs on A; 33,554,432 m16n8k16 MMAs on D); adversarial one-hot permutation
  tests; SASS-verified tensor-core-free A.
- **Nuisance handling derived from data, not assumed**: the 50→2000-iter warmup sweep
  exposing the thermal transient (27.6%→2.4% spread) and motivating fixed-*time*
  soak; L2 thrash per trial; one variant per process; randomized (job × rep) order
  with fixed seed; idle-GPU preflight aborts; anchor cells across campaigns (1.2%
  worst drift in P1, 4.10% in P2, escalated into a reading rule).
- **Winner's curse explicitly handled**: 2-process grid search re-confirmed at 5
  processes; the +2.6–5.7% one-sided selection bias quantified; the cuda_noptx rank
  flip reported rather than hidden.
- **A built-in positive control**: the in-region `.half()` cost lands at +0.207 to
  +0.221 ms in all five lanes *including torch* — a ±7% invariance check on the
  timing rig.
- **Self-falsification discipline**: Phase 2 opens with a claim-by-claim
  falsification table; six harness bugs that produced clean-looking wrong numbers are
  named with mechanisms; the single-run incumbent check that flipped a sign on
  re-measurement was upgraded to 5 processes with the old artifacts retained.
- **Every checked number reproduces.** Independent recomputation verified: all 24
  cells of the §5 convergence table; custom-variant totals (32/23/20/20); the
  five-row ncu_key census; the matched/equal-grid/fused/SDPA tables from raw JSONs
  (including gate-fail exclusions); all §4 isolated-change multipliers; the cold
  compile medians; the artifact-drift diffs; the §6 abstraction and smem numbers.
  Zero numerical discrepancies were found. This is rare and worth stating.

## B.4 How to isolate the variables — the rigorous design

**Principle: for each research question, exactly one factor is the treatment; the
searcher, the prompt, the hardware state, and the artifact identity must be either
frozen or randomized-and-replicated — never sampled once.**

1. **Convergence/trajectory questions (RQ: which DSL converges faster).** Treat the
   searcher as a random factor: DSL(4) × op(3: one memory-bound, matmul, SDPA) ×
   model(≥2: Opus 4.8, Sonnet 4.6) × seed(k≥3 trajectories/cell). One frozen,
   hint-free prompt template (no tier labels, no ceiling assertions, no ports). One
   exogenous stop rule: fixed budget in **compute_s** (the correct equalizer — compile
   cost is a DSL property per the protocol's own split-clock principle; trial-equal
   budgets subsidize slow compilers, token budgets measure the LLM). All benches
   serialized on one pinned GPU, or a `gpu` + clock-state column per row with
   within-card comparisons only. Populate `agent_s` and token spend as covariates
   (the column exists and is empty in every committed CSV). Report per-cell median
   and range of time-to-95% so the DSL effect is estimable against agent variance.
2. **Ceiling questions (RQ: does any DSL have a higher ceiling).** Add
   idiomatic-ceiling reference lanes: cuBLASLt/CUTLASS fp16-in/fp32-acc; a real
   `@triton.autotune` arm with sweep cost charged to search time; a
   TileLang-autotuner arm; one unrestricted best-effort variant per DSL (swizzle,
   persistent, cross-block split-K permitted). Report matched / shared-grid /
   idiomatic tables side by side, plus achieved fraction of the tensor-core roofline.
   Add the reverse-transplant control (each lane's native-best family implemented in
   every other lane at one shape) to de-anchor the ~1.10× residual.
3. **Correctness as a controlled factor, not an inherited loophole.** Rerun matched
   A/B/C/D + equal-budget tuning under a distribution-robust gate: RMS-matched signed
   `randn` (keep `rand` as legacy arm), gate = `max(atol_floor, rtol·|ref|)` with
   `atol_floor` preregistered per op from the measured fp32-oracle-vs-fp64 error
   (accuracy.py already computes it). Let every DSL re-search its recipe under the
   new gate.
4. **Statistical framework, stated once.** Preregister: direction is real iff
   same-campaign CIs separate; magnitude is material iff >10%; cross-campaign
   comparisons carry +4% drift; n≈10 processes wherever 3–5% calls are needed (power
   the design to the claims). Match estimator to interval (median + bootstrap CI, or
   mean + t-CI). Re-grade the 7.7% abstraction penalty and the 10.2% cross-DSL
   residual under the same bands, whatever they turn out to be.
5. **Provenance and pinning.** SHA256 of the exact benchmarked source in every
   result row (`timed_bench.sh`, runners); phase trees under git; model/prompt-hash/
   GPU/campaign-id columns in `convergence.csv`; retroactively annotate what is
   recoverable and flag layer_norm as unattributable.
6. **Scope-completing replications** (cheapest first): tile-matched SDPA point
   (fix block_M/N/D_TILE/threads across lanes, as Phase 1 did for GEMM); per-lane
   stages sweep on the fused shape (Phase 1 proved the imposed depth is lane-biased);
   the archived-vs-current fused re-benchmark; 3-shape matmul grid (aspect ratio,
   short-K); one Hopper part for the two decisive tables (also unlocks the missing
   manual-async TileLang arm); one small-L2 GPU for the layer_norm 2-pass lever
   (working set > L2 kills the mechanism behind the memory-bound tie).

---

# Consolidated pros

1. Bit-identical cross-DSL outputs + dynamic work verification — the control that
   makes "pure codegen comparison" true rather than asserted.
2. Measurement protocol near state of the art for a single host: thermal-soak
   fixed-time warmup derived from a stability study, 5-process medians, CIs, L2
   thrashing, randomized order, idle-GPU aborts, anchor-cell drift monitoring.
3. Winner's-curse handling with quantified selection bias and an honestly reported
   rank flip.
4. Evidence-hierarchy discipline: the report demotes its own repo's earlier headline
   boards, corrects the published fused ranking after catching artifact drift, and
   re-measures instead of quoting.
5. Preregistered decision rules honored against the authors' interest (+7.7% → "no
   verdict").
6. The tolerance-gate pathology, quantified in both failure directions — an
   exportable benchmark-design contribution.
7. The vendor-library-hole result (1.12×/1.01×/1.74× tracking FlashAttention's
   head-dim cap) with a four-lane-reproduced algorithm crossover.
8. Six harness bugs named with mechanisms; falsification criteria stated up front.
9. Perfect artifact-number fidelity: every checked number in the report reproduces
   exactly from committed raw data.
10. Engineering cost partially quantified (device-LOC 44/52/355/412; cold compile
    1.19/6.01/36.4 s; 19-point grid ≈20 s vs ≈12 min) — with the warm-cache confound
    caught before it inverted a conclusion.

# Consolidated cons (ranked)

| # | Con | Severity | Status |
|---|---|---|---|
| 1 | Searcher (LLM, n=1/cell, model undisclosed; diverging second campaign unused; layer_norm searcher unattributable) confounds §3/§5/§7 and conclusions 4/6/7 | High | CONFIRMED (core) |
| 2 | "One GPU" scope false for native tier: lanes on different cards/clock states; 27–42% identity-baseline offsets; no gpu column | Medium-high | PARTIALLY CONFIRMED |
| 3 | Per-op prompt hints (anti-hint on matmul, answer on layer_norm, pro-hint on SDPA) never registered as a factor; hinted cell bolded in §5 | Medium | CONFIRMED |
| 4 | §5 convergence table: 5% threshold on single-run data with ~10% noise; headline cell decided by 0.10% | Medium | CONFIRMED |
| 5 | 7.0×→1.31× collapse fully executed for one op; fused equal-budget "inherited" by an argument the report itself rejects; SDPA spread is 1.60× with winner flips — op-dependence never stated | Medium | CONFIRMED |
| 6 | Strict orderings quoted across overlapping CIs in two headline tables, against the reports' own reading conventions | Medium | PARTIALLY CONFIRMED |
| 7 | fp16 legality (hence the headline result) is torch.rand-specific; 78% randn failure never reaches the summary; no corrected gate run | Medium | PARTIALLY CONFIRMED (conceded generically in §9) |
| 8 | Home-recipe anchoring: ~1.10× "intrinsically better" measured only on TileLang's own discovered family; no reverse transplant | Medium | Not independently verified (design-level) |
| 9 | No idiomatic-ceiling lanes; 1.31× is a spread among interior points; exclusion harm is asymmetric across DSLs | Medium | Factual basis verified |
| 10 | §2 table mixes control tiers under one header (SDPA not tile-matched; memory-bound row is native evidence) | Low-medium | PARTIALLY CONFIRMED (disclosed elsewhere) |
| 11 | Belief-conditioned stop rule makes "converged" vs "gave up" unidentifiable | Low-medium | Verified from logs |
| 12 | No artifact hashing; Phase 2 tree outside VCS; prescribed archived-vs-current fused re-benchmark never run | Low-medium | Verified |
| 13 | Noise floor is a single-config point estimate; estimator/CI mismatch; free-running clocks undisclosed; "equal 19-point" caveats not beside the row; bit-identical check not committed as code | Low | Verified |
| 14 | Conclusion 6 near-tautological; conclusion 5 generalized past its TileLang-specific mechanism and sm_89 | Low-medium | Analysis-level |
| 15 | No related-work positioning; contributions buried (methodology + gate pathology led by folklore confirmations) | Medium (for publication) | Judgment |

# TODO list (deduplicated, prioritized)

**P0 — fixes that change believability at near-zero experimental cost**
1. Disclose the searcher in §1: LLM agent, model versions, n=1-per-cell design;
   relabel §3/§5/§7 as agent-run properties; import the Opus-vs-Sonnet board
   divergence as replication evidence for conclusion 2. (Text-only fix; data already
   in repo.)
2. Re-flag the two CI-overlapping orderings (P1 confirmed grid, P2 fused matched) as
   partially unresolved — or add processes (n≈10–15) to the four middle cells until
   the intervals separate.
3. State the op-dependence of the collapse: report per-op post-transfer spreads
   (matmul 1.31×, fused — not run, SDPA 1.60× tile-unmatched) instead of one number.
4. Add the lane-environment audit for the native tier (identity-baseline table per
   op per lane; scope "one GPU" to the controlled studies), and annotate the §5
   table: single-run cells, ~10% noise, 0.10%-margin cell flagged, layer_norm hinted,
   per-lane GPU/clock caveat.
5. Pin artifacts: SHA256 of benchmarked source in every result row; put
   `phase2_fused_sdpa/` (and `phase1_matmul/`) under git; run the archived
   `solution_opus48` vs current fused re-benchmark the report itself prescribes.

**P1 — the experiments that convert audit into contribution**
6. Full transplant + equal-budget protocol on a second wide-gap op (fused
   matmul_gelu_softmax 19-point grid across all four lanes).
7. Distribution-robust gate rerun of matched A/B/C/D + tuning (signed randn +
   preregistered absolute floor; DSLs allowed to re-search recipes).
8. Idiomatic-ceiling lanes (cuBLASLt/CUTLASS, @triton.autotune with charged search
   cost, TileLang autotuner, one unrestricted arm per DSL) + reverse-transplant
   control for the 1.10× residual.
9. Searcher-replication factorial for convergence claims: k≥3 seeds × ≥2 models ×
   hint-free frozen prompts × exogenous compute_s budget × one pinned GPU;
   populate agent_s/tokens.
10. Tile-matched SDPA cross-DSL point + per-lane stages sweep on the fused shape.
11. One consistent, preregistered statistical framework (direction/magnitude bands,
    powered n, estimator-CI match), applied to all magnitude claims.

**P2 — scope and packaging**
12. Second architecture (Hopper) for matched A/B/C/D + SDPA sweep (adds the missing
    manual-async arm); one small-L2 GPU for the memory-bound mechanism; small
    preregistered shape grid for the tensor-core ops.
13. Commit the cross-DSL bitwise-equality check as a script + output artifact.
14. Restructure the write-up: lead with the transplant-audit methodology and the
    gate pathology as claimed contributions; add related work (Triton, Halide/TVM/
    Ansor, KernelBench/LLM-kernel-gen, FlashAttention, TileLang); demote conclusion 6
    or redefine "meaningful decisions" outcome-independently; scope conclusion 5 to
    sm_89 + tilelang 0.1.11 explicitly.
15. Quantify the effort axis the conclusions invoke: tokens, failed builds, debug
    iterations, cost-to-first-passing-kernel per cell — instrumentation, not new
    experiments.

---

## Bottom line for the two questions asked

**Q1 (insight/merit):** The data supports the report's narrow negative claims and a
strong methodology thesis it never claims explicitly. As it stands, this is an
excellent internal audit and *not yet* a research contribution: the headline
quantitative result is n=1-op and distribution-scoped, the convergence story is
n=1-trajectory, and the novelty is unpositioned. The path to intellectual merit is
real and comparatively short (P0 items are mostly text; P1 items 6–8 are the decisive
experiments): the claim "cross-DSL gaps produced by LLM-driven kernel search are
mostly discovery artifacts; a transplant-and-retune protocol with bit-identical-output
controls collapses them, and benchmark tolerance gates decide outcomes at both ends of
the output scale" is publishable at an MLSys/ML-for-systems venue with those additions.

**Q2 (control):** Phase 1 matched is near-exemplary; the equal-budget grid and the
fused ladder are well controlled with disclosed deviations; the SDPA cross-DSL table
is algorithm/dtype-matched only; the native six-op tier is uncontrolled in searcher,
prompt, hardware assignment, and replication — which the report structurally
acknowledges by the evidence hierarchy but then leaks by presenting trajectory
statistics, bolded convergence winners, and a "one GPU" scope statement it does not
hold. The isolation recipe is in B.4; the two structural fixes that matter most are
**treat the searcher as a random factor with replication** and **anchor "ceiling"
against idiomatic/vendor reference lanes**.
