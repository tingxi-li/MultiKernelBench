# From Expressibility to Trustworthy Performance

## Current evidence and experimental lessons from a controlled cross-DSL GPU study

**Date:** 2026-08-04

**Status:** Interim, non-frozen technical synthesis. This is not the
policy-gated paper.

**Controlling fused result:** `crossed_v2r3`

**Repository state reviewed:** commit
`a204bf13486376db9a0d787c3e2cafa6d983a029` on
`cross-dsl-6op-ncu-redo`.

## Executive summary

The most useful outcome of this program is not an Ada-specific ranking of GPU
languages. It is an experimental framework for separating five questions that
benchmark reports often collapse into one:

1. Can a programming system express the optimization strategy?
2. If it can, which launch configurations are feasible on the named hardware?
3. If a configuration launches, does the complete operation satisfy a
   candidate-independent correctness contract?
4. If it is correct, what factor is actually changed by the performance
   comparison?
5. Is the measured effect larger than the demonstrated resolution of the
   timing instrument?

The framework is supported by a complete crossed campaign, not only by design
prose. The controlling Ada audit retained all 304 requested cells: 249
`GATE_PASSED`, 36 `BUILD_FAILED`, and 19 `UNSUPPORTED`, with no launch or gate
failures. All 249 eligible cells were screened in exactly two fresh processes
(498 records). A mechanical rule selected 44 cells, which were confirmed in
1,380 fresh-process records, including 60 duplicate-label sham records. The
complete 2,638-entry evidence archive verifies independently, and a separate
analysis rederivation was byte-identical to the sealed final summary.

The community-level conclusions are methodological:

- Report the **expressibility denominator** before reporting speed.
- Preserve the causal stage of failure; source-level unsupported, resource
  failure, launch failure, and incorrect output are different observations.
- Change one mechanism at a time; replication cannot repair a confounded
  treatment.
- Separate search from inference, inspect timing trajectories, and freeze a
  late-window estimand prospectively.
- Measure the benchmark's empirical resolution with a true null and require a
  claimed effect's whole interval to clear it.
- Treat gate calibration, analyzer code, failed runs, source identity, and
  hardware identity as parts of the experimental instrument.

These principles generalize to compiler, DSL, autotuning, and systems
benchmarking. The numerical Ada frontier does not. A non-sm_89 feasibility
matrix is still absent, so cross-architecture reachability and the paper remain
blocked by policy.

## 1. Current state of evidence

| Evidence thread | Completed evidence | Current interpretation |
|---|---|---|
| Historical arithmetic/tile normalization diagnostic | Under its historical `torch.rand` gate, the matched matmul study implements the same A/B/C/D recipe in four lanes and contracts a published spread of about 7.0× to 1.31× through three related Phase-1 analyses. | The later complete matmul-v4 instrument audit found every A/B/C/D artifact failed all 512 `legacy_u01` and all 512 `opposing_means` inputs. The 1.31× frontier is therefore not v4-legal; it illustrates a normalization method and the need to put the current gate inside the search, not a controlling performance result. |
| Crossed fused reachability | `4 strategies × 4 lanes × 19 grids = 304` retained audit outcomes. Three strategies share 15 four-lane-reachable grids; `smem_staged` has no four-lane-common grid. | Reachability is a matrix and an intersection, not a scalar success rate. Counts are bound to the Ada hardware, toolchain, source, and grid. |
| Correctness-first performance | 249 cells each pass four cases, 64 held-out seeds, and two gate views before timing: 127,488 bound gate rows. | Timing eligibility means conformance to a frozen finite gate, not universal correctness. Failed builds remain visible in the feasibility denominator. |
| Confirmed Ada timing | 498 screen records; mechanical `N=44` selection; 1,380 confirmation records; trials 60–99 control. | Only cell-specific, factor-isolated positive-distribution contrasts whose entire interval clears the sham floor are reportable. There is no generalized lane ranking. |
| Instrument resolution | Two labels bind one byte-identical implementation in 60 records. The resulting floor is `0.014957935591828302` in absolute log-ratio. | Statistical exclusion of one is insufficient when an effect is no larger than demonstrated label/process noise. |
| Auditability | Three failed frozen revisions have incident receipts; the controlling archive has 2,638 entries and verifies independently. | A frozen analysis failure is a failed experiment requiring a new lock, not permission to repair a result tree post hoc. |

The historical matched-matmul evidence is in
[`PHASE1_REPORT.md`](../phase1_matmul/PHASE1_REPORT.md); its current correctness
qualification is in
[`REVIEW_ROUND2_RESPONSE_20260731.md`](REVIEW_ROUND2_RESPONSE_20260731.md).
The controlling fused evidence and exact claim limits are in the
[`crossed_v2r3` result memo](fused_epilogue_crossed_v2/RESULT_CROSSED_V2R3_20260802.md)
and
[`final_summary.json`](fused_epilogue_crossed_v2/results/crossed_v2r3/final_summary.json).

## 2. Intellectual merit and generalizable insights

### 2.1 Expressibility before speed

Performance-only comparisons condition on implementations that were successfully
written, built, and launched. That conditioning hides the denominator and creates
survivorship bias: a system can appear fast because difficult strategies never
enter its timing set.

The crossed design instead reports a finite **reachability frontier**. Source
expressibility is resolved first; every requested strategy/lane/grid cell is then
retained through build, launch, correctness, and timing eligibility. The observed
Ada result is informative because it preserves the stages:

- 19 cells are source-unsupported by the measured Triton explicit shared-memory
  public-API limitation;
- 36 cells are build/setup failures at particular configurations;
- 249 cells complete the operation and pass the gate.

The four `register_common_postprocess.triton.{g07,g11,g15,g18}` requests make
the distinction concrete: each needs 106,496 B of shared memory where the device
permits 101,376 B. They are source-expressible strategy requests that hit a
grid-specific resource ceiling, so they remain `BUILD_FAILED`; they are not
relabeled unsupported.

**General lesson:** publish both the capability denominator and the performance
numerator. Use a staged failure taxonomy and report the intersection of feasible
design spaces across systems.

**Boundary:** the taxonomy generalizes. The 249/36/19 census and 15-grid common
frontier are measurements of these sources and tools on RTX 6000 Ada, not
language theorems.

### 2.2 Normalize mechanism and search effort inside the correctness contract

The matched matmul program supplies both a useful decomposition method and a
cautionary negative result. Under its historical `torch.rand` gate, holding
arithmetic, tile, and recipe fixed reduced a published approximately 7.0×
four-way spread to 1.31× through three related Phase-1 analyses.

That phase also suggested that search accessibility can differ substantially,
but its compile-cost observations came from one cold compile per lane and an
extrapolation to the grid. They are a hypothesis for a future effort study, not
a replicated or controlling effort frontier.

The later matmul-v4 audit changes the interpretation. It completed all 66,852
expected instrument records and found that every Phase-1 A/B/C/D artifact failed
all 512 `legacy_u01` and all 512 `opposing_means` inputs, while passing the four
signed cases. The historical 1.31× frontier is therefore not v4-legal. It cannot
support a current realization-gap claim without a new gate-in-loop search.
Agreement among implementations, even on bit-identical evaluated outputs, did
not establish correctness because every implementation could share the same
unexercised failure.

**General lesson:** decompose a headline speedup into arithmetic, algorithm,
schedule, code generation, search budget, and expression cost, but run the
current frozen correctness contract inside search and selection. “Best found,”
“same mechanism realized,” and “correct under the current gate” are different
estimands.

**Boundary:** the 7.0× to 1.31× contraction is a historical diagnostic under
its original gate. It is evidence for the decomposition methodology and for the
danger of stale gates, not controlling timing evidence or a current lane
comparison.

### 2.3 Failure handling is part of the measurement instrument

The predecessor showed that identical hardware resource limits can acquire
different scientific labels depending on wrapper hygiene. A checked CUDA
attribute call stopped as `BUILD_FAILED`; an unchecked call proceeded to a bad
launch or garbage output. Eight CUDA-unlimited register cells were also lost
because a shared-memory request from another epilogue path leaked into a strategy
that did not use it.

The corrective campaign standardizes checked launches, makes resource requests
strategy-conditional, scans generated wrappers for the checked-launch header,
and retains the eight recovered `register_fused.cuda_unlimited.g05` through
`g12` cells.

**General lesson:** outcome labels are produced by code. Before interpreting a
feasibility matrix, audit whether every implementation checks the same build,
attribute, launch, and synchronization boundaries. A taxonomy is only as valid
as the instrumentation that emits it.

### 2.4 Factor isolation is more important than more replication

The predecessor's common four-lane strategy changed two mechanisms together:
epilogue placement and softmax implementation. The corrective fourth strategy,
`register_common_postprocess`, applies bias and exact GELU in lane-native
accumulator code but binds the same checked common CUDA softmax source body as
`global_intermediate`, with bias and GELU disabled by compile-time switches.
Their within-lane, within-grid contrast therefore moves bias/GELU between the
producing kernel and postprocess while holding the row-softmax source body and
launch geometry fixed.

The result is scientifically more useful than a universal winner. In Triton,
register-common/global is 1.064220 at `g01` with interval
`[1.017226, 1.096966]`, but 0.978482 at `g05` with interval
`[0.970593, 0.982281]`. The direction reverses across launch geometries, and both
intervals clear the empirical resolution floor.

**General lesson:** identify a mechanism with a contrast that changes only that
mechanism. If direction changes across configurations, report the interaction;
do not average it into “register is faster” or “language X is faster.”

### 2.5 A benchmark needs an empirical publication-resolution floor

A conventional confidence interval answers a sampling question under a model;
it does not show that the complete measurement pipeline can distinguish the
effect from zero. The corrective campaign therefore runs `sham_a` and `sham_b`
as two labels for one byte-identical implementation:

| Distribution | Sham median ratio | Exact median interval |
|---|---:|---:|
| Positive | 1.001340 | [0.988614, 1.012375] |
| Withheld-signed | 1.002677 | [0.994867, 1.015070] |

The largest absolute log endpoint across both intervals defines a floor of
`0.014957935591828302`. A contrast is reportable only if its entire interval
lies beyond that floor. Only 18 of 28 preregistered positive-distribution
contrasts meet this rule: two placement contrasts and 16 cell-specific lane
contrasts.

**General lesson:** include a true null that traverses the same build, launch,
process, ordering, and analysis path as real treatments. Make its observed
resolution a publication rule, not a decorative diagnostic.

**Boundary:** a duplicate-label sham measures label, process, ordering, and
harness variation. It cannot detect a systematic bias shared by both labels.

### 2.6 Correctness thresholds require calibration, stress testing, and margin disclosure

The robust gate separates semantic quality from contract conformance. Thresholds
are calibrated from candidate-independent named anchors, never candidate
outputs; calibration and validation seeds are domain-separated; missing,
duplicate, malformed, NaN, or Inf records fail closed. The gate reports cases
separately rather than treating tensor elements or gate views as independent
replicates.

The same-seed stress experiment illustrates why this matters. Ten locally
source-frozen candidates were evaluated on 512 shared inputs under two gate
views, producing
10,240 complete records but an effective sample size of 512. Every candidate
failed at least one shared seed. All six paired old/new CUDA comparisons had zero
discordances, so changing the candidate generation did not remove the observed
failure on the shared inputs. Its protocol was not pushed or externally
timestamped before launch, so it is a paired diagnostic rather than
preregistered confirmatory evidence.

The post-campaign
[`FUSED_V2_GATE_SATURATION_20260731.md`](robust_gate/FUSED_V2_GATE_SATURATION_20260731.md)
adds an equally important negative lesson. All 150 crossed-v1r1 passes were only
3.7–4.9% below the frozen row-sum threshold
and would fail at the unrounded calibrated value. The threshold predated the
campaign and was not changed, so this is sensitivity disclosure, not evidence
of tuning. The gate-saturation report was delivered in the owner's repository
and assigned to the owner in
[GitHub issue #2](https://github.com/tingxi-li/MultiKernelBench/issues/2);
explicit acknowledgment is still pending.

**General lesson:** correctness is not a boolean oracle handed down from outside
the study. Publish how thresholds were calibrated, the effective independent
sample size, which metric binds, and the margin to the boundary. Never adapt a
frozen threshold using candidate or performance results.

### 2.7 Failed analyses are evidence, not trash

Three frozen revisions failed at different stages:

- `crossed_v2` stopped before retaining a cell outcome because a diagnostic
  divided by exact-zero frozen thresholds;
- `crossed_v2r1` retained all 304 audit outcomes but its analyzer wrongly treated
  measured support as success at every grid;
- `crossed_v2r2` completed the audit and 498-record screen but its analyzer lacked
  an imported sham constant and stopped before confirmation selection.

Each failure has a content-addressed incident receipt, an exact retained census,
and a successor tag. The r1 closure contains 561 files and 869,639,781 bytes; the
r2 closure contains 1,064 files and 805,139,083 bytes. Neither selection nor
timing was promoted through an overlay. The controlling campaign was rerun under
a new source bundle and lock.

**General lesson:** freeze the analyzer and orchestration with the experimental
source. When a frozen defect is discovered, preserve the closure, state the
scientific consequence, repair under a new lock, and rerun every downstream
phase that could have been influenced.

## 3. Why the evidence is well controlled

| Threat to validity | Control used | Why it is needed |
|---|---|---|
| Survivorship bias | Complete strategy × lane × grid audit before timing | Failed and unsupported cells remain in the denominator instead of disappearing from the comparison. |
| “Not implemented” presented as “not expressible” | Two 19-grid source-retaining support probes with common-failure requirements | A source-level limitation must be measured and reproducible, not only asserted in a table. |
| Resource limits presented as language limits | Separate `UNSUPPORTED`, `BUILD_FAILED`, `LAUNCH_FAILED`, and `GATE_FAILED` states | The stage of failure determines the defensible causal interpretation. |
| Confounded performance treatment | Fourth strategy holds the row-softmax source body and launch geometry fixed while moving bias/GELU placement | More repetitions reduce noise but cannot identify which of two changed mechanisms caused an effect. |
| Candidate-adaptive correctness | Candidate-independent anchors and locked validation seeds | Prevents grading candidates against thresholds derived from their own errors. |
| Input leakage and pseudoreplication | Domain-separated seed namespaces; shared-seed pairing; effective `n` is seeds, not tensors, gates, or candidates | Prevents accidental overlap and overstated precision. |
| Winner's curse and manual substitution | Two-process screen, mechanical top-two-plus-`g01` selection, fresh confirmation | Search measurements choose candidates; independent processes estimate effects. |
| Temporal drift | Frozen randomized block order, paired within-block ratios, fresh processes | Reduces confounding between candidate/distribution and campaign time while isolating process state. |
| Non-stationary timed-window risk | Predecessor trajectory audit; trials 60–99 frozen prospectively as primary | Nominal warmup does not prove that early trials are settled; the late window is a preregistered response to that risk, not proof of stationarity. |
| Effects below benchmark resolution | Byte-identical duplicate-label sham and whole-interval publication floor | A statistically non-null effect is not reported if the instrument demonstrates noise of the same magnitude. |
| Post-hoc source or analysis changes | Prelaunch commit pushed and verified upstream; source/lock hashes in every phase | Binds the measurements to the code and analysis plan that existed before execution. |
| Selective repair of failed runs | Incident closures and new locked successor tags | Makes deviations auditable and removes the freedom to promote favorable partial results. |

## 4. Worked experimental examples

### Example A — denominator-complete reachability

**Setup.** The frozen operation is GEMM+bias+exact-erf-GELU+row-softmax at
`M=1024`, `N=8192`, `K=8192`, using fp16 operands and fp32 accumulation. The
factors are four strategies, four lanes, and 19 launch grids.

**Design.** The two questionable expressibility claims are resolved before the
campaign with source-retaining 19-grid probes. The full Cartesian product is
then audited. Each cell receives exactly one terminal stage label, and only a
complete two-kernel, gate-passing operation can enter timing.

**Conduct.** CUDA-no-PTX register placement passed all 19 probes. Triton explicit
user-managed shared memory produced the same captured public-API limitation on
all 19 probes. The subsequent
[`audit_summary.json`](fused_epilogue_crossed_v2/results/crossed_v2r3/audit_summary.json)
retained 304/304 outcomes and 127,488 gate rows for its 249 passing cells.

**Reasoning.** The full product prevents a fast survivor from standing in for a
language. The probe requirement distinguishes API expressibility from missing
engineering. The terminal-stage taxonomy distinguishes source limitations from
resource ceilings and incorrect output.

**Observed result.** Three strategies have a 15-grid intersection across all
four lanes. `smem_staged` has no four-lane-common grid. This is the measured Ada
reachability frontier.

**Community use.** Apply the same pattern to compiler features, vectorization,
sparsity, quantization, or distributed schedules: enumerate the requested
design space, retain every terminal outcome, and report both per-system support
and the shared intersection.

### Example B — isolating placement from postprocess implementation

**Setup.** `global_intermediate` writes fp32 scratch and lets a common CUDA body
perform bias, GELU, and softmax. `register_common_postprocess` performs bias and
GELU on lane-native accumulator values, then binds the same softmax source body
with the bias/GELU switches disabled.

**Design.** Compare the two strategies inside the same lane and grid. The common
postprocess path holds softmax implementation fixed, leaving bias/GELU placement
as the intended treatment.

**Conduct.** All gate-legal cells were screened twice. Confirmation selection was
mechanical. Placement effects were computed as paired block ratios on the
settled-tail process medians, and the whole exact interval had to clear the sham
floor. The sealed estimates are in
[`final_summary.json`](fused_epilogue_crossed_v2/results/crossed_v2r3/final_summary.json).

**Reasoning.** The predecessor changed placement and softmax ownership together.
No number of additional repetitions could separate those mechanisms. Adding one
targeted strategy changes the estimand; that is higher leverage than simply
making the old estimate more precise.

**Observed result.** Only two placement contrasts are reportable above the floor,
and their directions differ between `g01` and `g05`.

**Community use.** When a “system” treatment bundles compiler, library, algorithm,
and schedule changes, add a bridging arm that holds all but one component fixed.
Report interactions when the direction depends on configuration.

### Example C — candidate-independent correctness and shared-seed stress

**Setup.** Every launchable fused cell must pass these four cases:
`legacy_u01_gain1`, `signed_normal_gain1`, `rademacher_gain4`, and
`signed_normal_gain16`. Each case uses 64 locked validation seeds under both
`semantic_mixed` and `conformance_mixed` gates. The separate matmul-v4 recovery
provides a powered calibration example.

**Design.** The semantic view evaluates the original input contract through an
fp64 reference. The conformance view evaluates the implementation's declared
fp16/fp32 boundaries. Thresholds come from candidate-independent named anchors.
Fixed structural thresholds for non-finite and negative-probability counts are
zero. Any missing or malformed evidence fails closed. After matmul v3 failed
2/64 `opposing_means` validation seeds, v4 used a fresh namespace with 640
calibration and 512 validation seeds for each of six cases; validation stayed
quarantined until calibration completed and the gate specification was frozen
and hashed.

**Conduct.** The fused campaign consumed 512 gate rows per passing cell. Matmul
v4 completed 23,040 calibration records and passed all 9,216 locked validation
records under its frozen rule. In the
[`same-seed stress study`](robust_gate/audits/fused_same_seed_stress_v2/results/summary.json),
ten candidates shared the same 512 seeds under two views: 10,240 records, but
effective `n=512`. Candidate comparisons were paired by seed.
The stress protocol was locally source-frozen but was not pushed or externally
timestamped before launch, so it is diagnostic rather than preregistered
confirmatory evidence.

**Reasoning.** Multiple adversarial distributions exercise sign, cancellation,
and scale. Separate semantic and conformance views distinguish disagreement with
the mathematical operation from disagreement with a declared precision
contract. Shared seeds remove sample-composition noise from candidate
comparisons. Counting seeds—not tensor elements, candidates, or duplicated gate
views—avoids pseudoreplication. Matmul v4's calibration size was fixed because
`24 × 0.99^640 = 0.0386132 < 0.05` for its 24 case/metric families per gate;
zero failures among 512 validation seeds gives the preregistered
Bonferroni-adjusted per-case upper bound `0.009307`. The full rationale is in
[`V4_PREREGISTRATION.md`](robust_gate/V4_PREREGISTRATION.md).

**Observed result.** Every stress candidate fails at least one shared seed; the
six paired old/new CUDA comparisons have zero discordances. The separate
saturation analysis shows why margin disclosure is still necessary even for a
frozen, candidate-independent gate.

**Community use.** Calibrate without candidate outputs, lock a disjoint holdout,
pair candidates on shared inputs, declare the independent sampling unit, and
publish failure margins rather than only pass/fail totals.

### Example D — search, settled-tail confirmation, and sham resolution

**Setup.** Screen each of 249 eligible cells in two fresh processes on physical
GPU 0. Select the two fastest legal cells plus legal `g01` within each
strategy/lane, deduplicated. Confirm 44 selected cells under positive and
withheld-signed distributions in 15 randomized blocks. Each process performs a
two-second warmup, flushes L2, and retains 100 trials.

**Design.** Trials 60–99 are the preregistered primary window; the full window and
first/last-decile drift are diagnostic. Both distributions are randomized inside
the same blocks. Two sham labels for one implementation traverse the identical
pipeline.

**Conduct.** Screening produced 498/498 records. Confirmation produced
`44 × 2 × 15 = 1,320` cell records plus
`2 labels × 2 distributions × 15 = 60` sham records, totaling 1,380/1,380.
Inference uses 15 process-level paired ratios and exact order-statistic intervals,
not 1,500 within-process trials as independent samples. Record counts and
contrasts are sealed in the
[`final summary`](fused_epilogue_crossed_v2/results/crossed_v2r3/final_summary.json).

**Reasoning.** Separate screening and confirmation reduces direct reuse of
optimization noise. Forced `g01` retains a common preregistered control even when
it is not top two. Randomized order limits fixed-order confounding, and
within-block pairing controls block-level shifts; neither guarantees that all
slow drift is removed. Fresh processes reduce JIT, allocator, and cache-state
leakage. The settled window responds to
predecessor trajectories showing that completed warmup did not remove timed-window
transients. The sham prevents publication below observed instrument resolution.

**Observed result.** The sham intervals center near one and yield the 0.014958
log-ratio floor. Eighteen cell-specific positive-distribution contrasts meet the
preregistered reporting rule; the others are not reported as effects.

**Community use.** Treat microbenchmarking as an experiment with selection,
temporal order, sampling units, timing trajectories, and a detection floor—not as a
single call returning a stable scalar.

### Example E — frozen incidents and corrective reruns

**Setup.** Source, campaign definition, analyzer, dependency closure, hardware
identity, and launch rules are hash-bound before GPU execution and verified on
the configured upstream.

**Design.** A frozen defect cannot be repaired inside retained results. The
incident policy requires an exact closure receipt, a non-controlling label, a
new source lock, and fresh execution of every downstream phase.

**Conduct.** The three failures were stopped at their actual boundaries: before
cell retention, after the complete audit, and after the complete screen. Their
receipts record exceptions, file counts, byte counts, hashes, outcome censuses,
and scientific effects. No r1 timing existed; no r2 confirmation existed; no
post-hoc selection was reused.

**Reasoning.** Analyzer fixes can change eligibility, selection, or reported
effects. Reusing partial output after seeing it creates a researcher degree of
freedom even if the underlying measurements are byte-identical. A new locked
run makes the correction observable and falsifiable.

**Observed result.** `crossed_v2r3` completed every phase, and its independently
verified final summary matches a separate rederivation byte for byte. The
[`crossed_v2r3` result memo](fused_epilogue_crossed_v2/RESULT_CROSSED_V2R3_20260802.md)
binds the incident chain and controlling evidence.

**Community use.** Preregister the recovery rule, not only the success path.
Preserved failures make debugging history part of scientific provenance.

## 5. A reusable protocol for controlled systems comparisons

### Before execution

1. State the estimand: expressibility, fixed-mechanism realization, or free-search
   attainment.
2. Freeze the complete requested denominator, including configurations expected
   to fail.
3. Define terminal stages and standardize error checks across implementations.
4. Calibrate correctness from candidate-independent anchors and freeze disjoint,
   domain-separated validation inputs.
5. Hash the source, analyzer, dependency closure, campaign, hardware contract,
   and expected record counts; push and verify them before execution.
6. Preregister selection, timing window, block order, sampling unit, interval,
   sham, and reporting floor.

### During execution

1. Retain every requested cell and its terminal outcome.
2. Require correctness before timing without removing failures from the
   feasibility denominator.
3. Use fresh processes and frozen randomized block order when process state and
   slow drift are plausible.
4. Separate search/screen measurements from confirmation measurements.
5. Emit exact expected-count, hardware-identity, and content-hash receipts for
   every phase.
6. Stop on an integrity failure; do not improvise replacements.

### During analysis and reporting

1. Re-derive eligibility and selection from retained evidence.
2. Inspect trial trajectories and distinguish the preregistered primary window
   from drift diagnostics; do not infer stationarity from warmup alone.
3. Use the true independent unit—often process or seed—not tensor elements or
   repeated views.
4. Pair observations inside randomized blocks when comparing treatments.
5. Require the complete uncertainty interval to clear an empirical sham floor.
6. Report the reachability matrix, common frontier, failure stages, gate margins,
   effect intervals, and excluded claims.
7. Preserve failed frozen runs under content-addressed incident receipts and use
   new locks for corrections.

## 6. Limits and work still required

The report deliberately does not claim more than the evidence supports:

- All controlling crossed-v2 performance evidence covers one fused shape on
  NVIDIA RTX 6000 Ada, compute capability 8.9, driver `610.43.02`. The Phase-1
  matmul timing discussed above covers a different shape and remains a
  historical legacy-gate diagnostic.
- The 19-grid frontier is hardware-, toolchain-, source-, and budget-bound; it is
  not a theoretical ceiling of any language.
- Timing inference covers 44 mechanically selected cells, not an exhaustive
  performance ranking of all 249 feasible cells.
- Reportable timing effects use the positive `rand`, seed-0 distribution. The
  withheld-signed measurements are diagnostics, so no cross-distribution timing
  generalization is made.
- The 28 preregistered intervals are interpreted per contrast; this report makes
  no familywise-error claim across them.
- Clock and temperature telemetry were recorded at preflight, but application
  clocks were not locked and no thermal admission band was enforced. Residual
  environmental variation remains possible.
- Passing 512 frozen gate rows demonstrates conformance to that gate, not
  correctness for all possible inputs.
- The sham bounds observable null variation in this harness but not shared
  systematic bias.
- No absolute latency, lane ordering, or placement ordering is generalized.
- The former `0.8319` distribution claim is excluded. RQ(e) remains dropped
  unless a new preregistered public-corpus study is funded and executed.
- Convergence, reciprocal-transfer, and effort-frontier programs contribute no
  controlling result here.
- As of the 2026-08-04 inspection, the host exposes four RTX 6000 Ada devices and
  no non-sm_89 GPU. The second architecture therefore remains blocked.
- The later-work policy authorizes only a hardware-bound 304-cell feasibility
  audit on that second architecture. It forbids second-architecture timing and
  delays the one paper until the matrix is sealed and independently verified.

The governing scope is recorded in
[`LATER_WORK_POLICY_20260731.md`](LATER_WORK_POLICY_20260731.md).

## 7. Evidence anchors

| Artifact | Binding |
|---|---|
| Crossed-v2 source bundle | `abe6873e89686ca94a2dc096b633feff615c7dbfd8fb4e91546850ed26cae20b` |
| Crossed-v2 launch lock | `206b75387acf4d60b688a8e19b3aff094a8ef69d5a0a1b72269d8b24f48be1b6` |
| [`audit_summary.json`](fused_epilogue_crossed_v2/results/crossed_v2r3/audit_summary.json) | `096a9833d47f858752fd4a31bbee3db7df525d7d36e91593bd6a011b05ff86aa` |
| [`confirmation_selection.json`](fused_epilogue_crossed_v2/results/crossed_v2r3/confirmation_selection.json) | `a008ba2a5e6a81432eee7e94e7ebb34eeb90e3b88031f901cf10129a4be432ca` |
| [`final_summary.json`](fused_epilogue_crossed_v2/results/crossed_v2r3/final_summary.json) | `b38e01849024c285e14ea1eba7477c85443c68ca54b08a885c32f2413fa5ca03` |
| Complete evidence archive | `4e6dda75b0516fd1cac928ec1a30ec84e03ec0774be10c178ef3b4d9e650bded` |
| Complete evidence index | `3340a86f4f64404a96461fa5bb5262ff3e1cebacd645d4bd63aaa1bedda212c5` |
| [`matmul-v4 instrument summary`](robust_gate/audits/matmul_v4_instrument_v1/results/summary.json) | `125e66bbd76012e49695b1226ff39943d52bb61da9a44842b549a53b1f678ebd` |
| [`matmul-v4 margin report`](robust_gate/audits/matmul_v4_instrument_v1/results/margin_report_v2.json) | `884a7cd513ae50f6ec58859da9185737fcc8fbb54e900503e4522c324d33e3af` |
| [`crossed_v2` incident](fused_epilogue_crossed_v2/INCIDENT_CROSSED_V2_20260801.json) | receipt `4a385b3b45544069f13a54f78df8fc12b6f74919b76c87e83bd1d91463247eb7` |
| [`crossed_v2r1` incident](fused_epilogue_crossed_v2/INCIDENT_CROSSED_V2R1_20260801.json) | closure `08a201af5390e47318270386a75c2018492fd561e4271d4dcb76415f72549868` |
| [`crossed_v2r2` incident](fused_epilogue_crossed_v2/INCIDENT_CROSSED_V2R2_20260801.json) | closure `0b2e9538fde871955c15fb3022c46743118b50fea11ff7617a7e5ab1fd573e2e` |
| [`same-seed stress summary`](robust_gate/audits/fused_same_seed_stress_v2/results/summary.json) | `d786cf0e1a42ba1680c132f7e859ca5152b34a691b4712161218670405806758` |
| [`crossed-v1 gate-saturation report`](robust_gate/FUSED_V2_GATE_SATURATION_20260731.md) | `60911716ce6ae2fdd3b754ee9218bd51e4a9001760bd1612224f4392d2fdd9a6` |
| [`crossed-v1 settled-tail overlay`](fused_epilogue_crossed_v1/results/crossed_v1r1/reanalysis_tail_v1.json) | `e7296ce10585169e7dcf1f0d66c6bc59bf079d367d30491c46f42a8b3d03d1e5` |

## 8. Conclusion

The central contribution is an experimental contract for trustworthy systems
comparison. It makes the denominator visible, separates capability from
resource and correctness outcomes, isolates mechanisms before measuring them,
uses candidate-independent gates before timing, treats timing trajectories and
resolution as empirical questions, and makes every correction auditable.

The Ada campaign demonstrates that the contract is executable at nontrivial
scale. It also demonstrates why restraint is part of rigor: placement and lane
directions can reverse across grids, 10 of 28 preregistered contrasts do not
clear the instrument floor, crossed-v1r1 margins were only 3.7–4.9% below the
frozen row-sum threshold, and a complete single-architecture study is still
insufficient for cross-architecture claims.

For the community, the durable result is therefore not “which DSL won.” It is a
reusable way to ask what was expressible, what was actually held fixed, what was
correct, what the instrument could resolve, and what evidence would make the
answer independently auditable.
