# Round-2 review — the post-review experiment suite

> Targets: `fused_closure_v2`, `fused_reachability_v2`, `fused_frontier_closure_v3`,
> `archived_current_fused_v1`, `robust_gate/audits/*`, `provenance/evidence_v2`,
> `REVIEW_RESPONSE_20260730.md`, `ERRATA_20260730.md` (commits `898d965`, `d34ef71`).
> Method: six adversarial auditors re-derived every headline number from raw records,
> recomputed hash chains, diffed the producing code, and checked registered rules
> against what was executed. 59 factual checks; every author-stated number reproduced
> exactly. The three questions asked of this review are answered in §1–§3.

---

## 0. Verdict in brief

Measurement-level rigor is now the best in the program: in-campaign contract-matched
torch controls, randomized interleaved blocks, paired-ratio sign tests with exact
intervals and Holm correction, same-session internal reproduction of the old result
(the v3 session re-measures the old recipes at 1.4446×, statistically reproducing the
historical 1.4482× before showing the new 1.2505×), frozen gates never touched, and
scope language that mostly refuses its own temptations. The response document even
corrects this review series twice, both times validly (9-point strict intersection;
`epilogue=smem` was never expression-matched), and its rejection of my "one torch
lane" fix for a two-control design was the right call — both historical torch arms
failed the frozen gate 256/256 and would have poisoned the comparison.

The remaining problems are inferential, not arithmetical: a robustness dichotomy that
its own data cannot support (1/256 vs 0/256 on *disjoint* seeds, Fisher p = 0.5); a
frontier whose celebrated kernels failed same-day fresh-seed stress; an
engineering-attention asymmetry baked into the 1.2505× spread; a legality verdict
that is binary where the data spans 1.19× to 2,283× over threshold; and — ironically
for an append-only regime — controlling documents that are *stale against their own
repo*: the response and errata say the matmul-v4 instrument audit "contributes no
completed result," while a completed 66,852-record audit with the program's single
most consequential finding (every Phase-1 A/B/C/D kernel fails the frozen v4 gate,
including fp32 A) sits finished on disk, its raw records committed one minute after
those documents.

---

## 1. Q1 — Are the experiments rigorous and under reasonable control?

**At the measurement level: yes, with verified substance.**

- **Closure v2** answers the round-1 demand and exceeds it: 3 torch arms + 6 custom
  arms interleaved at randomized positions in 15 serialized blocks on one pinned GPU
  (zero process overlaps, UUID-bound); `torch_contract_fp32` implements the fused-v2
  contract exactly and passed 512/512 gate records; the two historical
  half-arithmetic torch arms failed 256/256 per gate and were structurally excluded
  *before* the run. The ratios 0.86385 [0.8584, 0.8883] and 0.86317 [0.8527, 0.8704]
  are CI-resolved below 1 (Holm p 0.0074/0.0020) and correctly scoped as
  contemporaneous-session, not a vendor-library claim.
- **Frontier closure v3** contains its own positive control: the old four recipes
  re-measured in the same 120/120 session give 1.4446 [1.4413, 1.4545] —
  reproducing the historical 1.4482× — before the new-recipe spread 1.2505
  [1.2436, 1.2565] is computed. That is the correct same-session way to quantify the
  reachability term, and "part, not all" is faithful: removing the smem-allocation
  barrier bought the fixed CUDA recipes 13.8%/21.1%, and the compiler recipes still
  win all six fixed contrasts (0.80–0.91, all intervals below 1).
- **The v4 instrument audit** is a real sensitivity instrument now: 7,844
  designated-wrong rows all rejected (42,240 threshold checks recomputed, zero
  mismatches), including near-threshold probes at 1.25× and 2×; the 640 "exact
  exclusions" are registered no-op transforms on bitwise-zero references whose
  passing is the *correct* outcome. The historical-kernel contact ran under the
  byte-frozen spec: **A/B/C/D all fail** — with an inverted and mechanistically
  authenticated signature: all three *signed* distributions pass; the failures are on
  `legacy_u01` (the torch.rand analog itself) and `opposing_means`, led by
  `abs_signed_bias` (B up to 9,322× over; C/D 2,283×; fp32 A fails q32 at
  1.19–3.63×), with B/C bias ratio 4.084 ≈ K/KC and C/D bitwise-identical — exactly
  the accumulation-drift-∝-E[a·b]·K mechanism Phase 1 derived. Note what this
  means: the drift the gate catches is worst on the *benchmark's own all-positive
  distribution*; the old gate's absolute-tolerance budget was hiding it there.
- **Provenance** is materially improved: analyzers and preregistrations are now
  hash-bound in prelaunch receipts, GPU clocks/temperature/power/nvcc are captured
  per record, three of four campaigns commit raw results as plain files, and the
  sealed bundle (SHA verified, 471 entries) plus 23/23 document-quoted hashes all
  reproduce.

**Where control still falls short (ranked):**

1. **The robustness contrast is statistically unsupported as framed.** All four old
   winners failing 1/256 fresh gain-16 seeds vindicates round-1's thin-margin flag —
   but three of the four fail on the *same seed* (197; Triton on 186), so the event
   is ~2 hard inputs on ~3 distinct streams, not four independent fragilities. The
   streamed epilogue's 3,072/3,072 collapses to 256 effective samples (per-seed
   row-sums bit-identical across 6 grids × 2 gates — disclosed), its worst headroom
   (0.697%) is *thinner* than the headroom that just broke (3.7–4.9%), it has more
   near-threshold seeds (9 vs 5–7), and it was never run on the two killer inputs.
   1/256 vs 0/256 on disjoint seeds is Fisher p = 0.5. "Old fragile, new robust" is
   not established; "both sit on a razor's edge of an extremely tight row-sum
   threshold" is what the data shows.
2. **Frontier/robustness tension.** The 0.86× ratios and half of the 1.2505× spread
   are earned by kernels the same day's stress fails. Timing eligibility was
   deliberately kept at the original frozen 4×64 split (defensible, disclosed), but
   the fragility caveat lives in adjacent prose, not inline with the headline tables.
3. **Engineering-attention asymmetry.** Only the CUDA lanes received the
   oracle-informed post-review redesign; the compiler lanes kept blind-grid-frozen
   artifacts. Direction is conservative (extra effort went to the losers, who still
   lost), but 1.2505× is a snapshot of asymmetric effort, not an equilibrium recipe
   frontier — and no document names this asymmetry explicitly.
4. **Legality is binary where it should be a margin.** "Not v4-legal" lumps fp32 A at
   1.19× over threshold with C/D at 2,283× over. No search has ever run with the
   gate *in the loop*, so "the gate outlaws the fp16-accumulation class" vs "these
   kernels need one more flush level" is undecided. Also scoped honestly but worth
   repeating: kernel contact covered the Triton lane only, extending to other lanes
   via the Phase-1 bit-identity premise.
5. **Document currency defect.** The response/errata (committed 21:07Z) deny a
   completed result that finished at 20:55Z and was committed at 21:08Z; the run
   status quoted upstream ("r1 live, r2/r3 queued") is likewise stale — r0–r3 all
   completed, 0 failures each. Append-only discipline has produced controlling
   narratives that lag their own evidence; the program's biggest finding currently
   exists only in an uncommitted summary plus committed raw records.
6. **Residual provenance gaps.** Preregistration ordering is self-attested (receipts
   + mtimes; everything entered git post-hoc in one commit; no external timestamp);
   `fused_reachability_v2/results/` is still gitignored (raw tree exists in git only
   inside its evidence tarball); the three historical reports contain **no pointer to
   the errata** that corrects them; the old report's "archived Triton = logged TF32
   kernel" mischaracterization survives (the archived artifact is a 10-config
   fp16-cast kernel — the −45% drift mechanism is operand/intermediate precision and
   weight caching, not tf32→fp16 MMA); the unchecked `cudaFuncSetAttribute` bug in
   the old fused harness is unfixed (the new candidate code does check it).
7. **The monoculture stands.** Every timed number in the program remains torch.rand,
   seed 0, one shape, one GPU model, one implementer. Correctness was proven
   distribution-sensitive; timing distribution-sensitivity has never been measured
   once.

**Answer:** rigorous and well-controlled at the measurement layer — as good as this
kind of single-host study gets — with the remaining risk concentrated in *inference
design* (asymmetric effort, disjoint-seed comparisons, post-hoc binary legality,
self-attested preregistration) and in *document currency*, not in the numbers.

---

## 2. Q2 — The sharper isolating designs

The current campaigns measure frozen artifacts well; the open research questions are
about *processes* (search, transfer, engineering effort). Factor-isolation per RQ:

**RQ1 — Does any DSL have a higher finite-budget realization frontier?**
As posed over frozen recipes it is not well-posed; budget was never denominated in a
common currency (19 shared grid points for compiler lanes; grid + one oracle-informed
redesign for CUDA; an unoptimized author implementation for torch). *Isolating
design:* make the frontier a property of the (DSL, searcher, budget) triple — run the
search itself as the experiment, with the frozen gate in the loop (fail-closed per
candidate), one pinned GPU, identical scaffold modulo DSL, budget = completed-compute
seconds, ≥5 independent search seeds per DSL. Observable: the *distribution* of
best-legal-confirmed latency per search; decide by order-statistic CIs.

**RQ2 — Decompose apparent gaps into discovery / codegen / reachability / legality.**
One mechanism (the smem barrier) is now quantified same-session; the rest are not.
*Isolating designs:* (a) **Reachability as a crossed factor, not a lane property**:
epilogue strategy {register-fused, smem-staged, streamed-two-kernel} × lane(4) ×
grid(19) in one session — every lane implements every strategy (the compiler lanes
can trivially express the two-kernel form). Feasibility becomes a cell outcome;
realization = the gap restricted to commonly-feasible cells. Decisive observable:
whether compiler lanes *also* lose cells under forced smem staging (Triton's g15/g18
OutOfResources failures hint yes — which would prove the barrier is
strategy-intrinsic, not language-intrinsic). (b) **Legality in the loop, margins not
binaries**: re-run the equal-budget search with the v4/fused-v2 gate as a
per-candidate constraint and report max-over-threshold ratios, yielding a per-DSL
accuracy–latency Pareto front. Decisive observable: if the legal fronts coincide,
the historical speed gaps were legality artifacts. (c) **Codegen share** via
expression-matched transplant of each discovered winner into every other lane
(the fused reciprocal that does not yet exist).

**RQ3 — Home-field/recipe-origin bias.** The reciprocal scaffold is the
best-designed campaign in the repo but cannot observe the most policy-relevant
reversal: its origin set is {tilelang, triton} — no CUDA-native origin. *Fix:* add a
third origin card (CUTLASS-default tiling, or the streamed-epilogue recipe itself),
making origin a 3-level factor; double-source destination implementations to bound
translator skill. Decisive observable: sign reversal of the destination effect across
origins (interaction) = anchoring; a destination effect invariant to origin and
retuning = a real finite-budget compiler effect.

**RQ4 — Which DSL converges faster under an LLM searcher?** Still zero data. *Fix
before launch:* swap one same-family alias (gpt-5.6-sol/terra) for a cross-family
model — the repo's own Opus-vs-Sonnet observation (1.7–2.1× per-cell swings) *is* the
power analysis, and a within-family factor is known-weaker than the effect it must
detect; add per-trajectory GPU pinning (absent from the manifests); keep
hints-as-treatment (the best idea in the prereg). Observable: survival curves of
time-to-first-gate-legal candidate within x% of a frozen reference (log-rank), plus
token-normalized curves; 8–10 replicates per cell if trajectories are cheap.

**RQ5 — Do lower abstraction levels pay?** Currently answerable only as "this
author's hand-CUDA, including one targeted redesign, loses by 9–20%, and inline PTX
recovers about half the deficit." *Isolating design:* lanes = {torch contract,
cuBLASLt/CUTLASS epilogue-fusion (the actual industrial low-level path, absent since
P1-8), Triton, TileLang, hand-CUDA}, each with logged effort (human hours + agent
tokens) at 3 checkpoint budgets; report latency-vs-cumulative-effort frontier
*curves*, not endpoints. That is the only form in which the question has an answer.

Cross-cutting fixes: same-seed stress when comparing old vs new candidates (run the
streamed epilogue on seeds 186/197); margin-ratio reporting everywhere a gate issues
verdicts; one timing campaign under a signed input distribution to test whether
*performance* rankings are distribution-stable; commit preregistrations before
execution (a push or third-party timestamp costs nothing).

---

## 3. Q3 — Community value and generalizable intellectual merit

Ranked by transferable value per unit of remaining work:

1. **Benchmark-gate design methodology — highest, and already substantially earned.**
   The arc is complete and self-contained: a vacuous absolute-tolerance gate (passes
   row-reversed answers) → anchor-calibrated, distribution-aware gates with frozen
   thresholds, disjoint seed namespaces, negative controls, and preregistered
   acceptance → the demonstration that the *entire celebrated frontier is
   gate-illegal under the fair instrument, failing worst on the benchmark's own
   input distribution*, and that "passed the robust gate" was split-specific (1/256
   fresh-seed failures at 4% headroom). This is precisely the failure mode of the
   KernelBench/LLM-kernel-generation genre, and none of it depends on any blocked
   campaign. Carrier: an MLSys/benchmark-track methods paper + a reusable
   gate-calibration library; the instrument audit (7,844 rejections, exact-zero
   exclusion controls, real-kernel contact with mechanism-authenticated failures) is
   the empirical core.
2. **LLM-guided-search evaluation methodology — high, but contingent.** The
   n=1-trajectory fallacy, the measured 1.7–2.1× searcher sensitivity, exogenous
   compute-clock budgets, hints-as-treatment, and gate-in-loop search are a needed
   corrective to how LLM kernel-search papers currently report "DSL X converges
   faster." Worth a methodology paper *with* the executed 120+48-trajectory dataset;
   a workshop note without it; near-zero if run with same-family models.
3. **Compiler-vs-hand economics — medium, currently repo-local.** "Two compiler
   artifacts beat a contract-matched torch by 1.16×; beat redesigned hand-CUDA by
   9–20% on one op/shape/GPU" is a specimen, not economics. It becomes publishable
   with the RQ5 effort-frontier design, a vendor-expert arm, and a second
   architecture.
4. **Scheduling-language design guidance — small but real.** The one genuine nugget:
   compiler lanes' register-resident epilogues occupy tile regions that naive
   hand-staging cannot reach, and the escape is an *algorithm* change (streaming),
   not a tuning change — a concrete lesson about epilogue expressiveness worth a
   section in paper (1), plus the inverted drift signature (signed inputs pass,
   all-positive inputs fail) as a benchmark-design vignette.
5. **Repo-local (valuable hygiene, not science):** the archived-vs-current drift
   result (+9%/−45%, opposite directions — cross-artifact history is uninterpretable
   as DSL effects), torch-anchor contract matching, and the specific constants
   (0.86×, 1.2505×, 1.31×, 1.448×), all now correctly scoped by the authors as
   artifact-set properties.

---

## 4. Priority actions

1. **Fix document currency** (hours): update response/errata to the completed
   matmul-v4 audit — the program's most consequential finding is currently denied by
   its own controlling documents; commit `results/summary.json`; refresh the run
   status.
2. **Close the robustness comparison properly** (one GPU-hour): run the streamed
   epilogue on seeds 186/197 and a same-seed 512-seed stress over all frontier
   cells; report margin ratios; put an inline fragility flag beside the 0.86× and
   1.2505× tables.
3. **Make the errata discoverable**: one pointer line in each historical report
   (this does not violate append-only in any meaningful sense); fix the "logged
   TF32" mischaracterization while there.
4. **Symmetry pass** (cheap): let the compiler lanes attempt the streamed two-kernel
   form, converting reachability into a crossed factor (§2 RQ2a).
5. **Reciprocal**: add a CUDA-native origin card, then unblock — it remains the
   single highest-value pending experiment (de-anchors the grid *and* runs the first
   gate-in-loop legality re-search via the KC ladder).
6. **Convergence**: cross-family model + GPU pinning before spending 120
   trajectories on a known-weak factor.
7. Housekeeping: commit `fused_reachability_v2/results/` as plain files; fix the
   unchecked `cudaFuncSetAttribute` in the old fused harness; take a post-campaign
   provenance snapshot with the new clock/nvcc capture.
