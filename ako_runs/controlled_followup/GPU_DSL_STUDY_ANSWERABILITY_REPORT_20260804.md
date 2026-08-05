# What Can a GPU DSL Study Actually Answer?

## Answerable questions, controlled experiments, and current claim boundaries

**Date:** 2026-08-04

**Status:** Interim, non-frozen technical report. This is not the policy-gated
paper.

**Controlling fused result:** `crossed_v2r3`

**Repository state reviewed:** commit
`4e8c49a7696a933da50f0e1e06d8e31f48643bcb` on
`cross-dsl-6op-ncu-redo`.

## Executive answer

Yes, a GPU DSL study can answer important questions, but the scientifically
useful questions are narrower than “which DSL is best?” The present evidence can
answer, on the named Ada hardware and frozen software stack:

1. whether a requested optimization strategy is source-expressible through a
   pinned public implementation path;
2. which requested launch configurations build, launch, and pass a frozen
   correctness contract;
3. where a feasibility failure occurs and whether it is an API limitation, a
   hardware-resource ceiling, a launch failure, or incorrect output;
4. whether one isolated implementation mechanism changes performance at a
   specific lane and grid;
5. whether a cell-specific lane contrast is larger than the demonstrated
   resolution of the measurement pipeline; and
6. whether historical artifacts still satisfy a newer, candidate-independent
   correctness contract.

Those answers have community value because they replace a single winner table
with a reproducible account of capability, feasibility, correctness, and
realization. The controlling campaign audited all 304 requested fused cells,
retaining 249 `GATE_PASSED`, 36 `BUILD_FAILED`, and 19 `UNSUPPORTED` outcomes.
It then produced 498 screen records and 1,380 confirmation records, including 60
duplicate-label sham records. Its 2,638-entry evidence archive verifies
independently.

The same study cannot currently answer which DSL is generally fastest, whether
the Ada reachability frontier persists on another architecture, whether recipes
transfer across DSLs, how engineering effort changes the frontier, or whether
the timing results generalize across input distributions. Those are missing
experiments, not inconvenient qualifications.

## 1. The answerable unit of a GPU DSL claim

A defensible GPU DSL result is indexed by more than a language name. Its minimum
answerable unit is:

```text
(operation, strategy, lane, grid, hardware, toolchain,
 correctness contract, input distribution, estimator)
```

The study can generalize over a component only if that component was varied or
otherwise justified. Holding hardware fixed cannot establish architecture
portability. Timing one input distribution cannot establish distribution
stability. Comparing two implementations that change both placement and
postprocessing cannot isolate either mechanism.

This gives a useful evidence ladder:

```text
requested cell
  -> source-expressible?
  -> builds and launches on the named hardware?
  -> passes the frozen correctness contract?
  -> is selected without manual substitution?
  -> is confirmed in fresh processes?
  -> does its whole interval clear the sham resolution floor?
```

Skipping a rung changes the question. A source limitation is not a slow kernel;
a resource-bound build is not a language limitation; a gate-passing cell is not
universally correct; and a confidence interval that excludes one is not
necessarily resolvable by the full benchmark pipeline.

## 2. Questions this study can answer now

| Research question | Current evidence-backed answer | Permissible scope |
|---|---|---|
| Can the previously disputed lane/strategy paths be expressed? | The source-retaining probes establish the pinned CUDA-no-PTX register path across all 19 grids and a common Triton explicit-smem public-API limitation across all 19 grids. | These implementations, APIs, versions, and strategies; not a theorem about CUDA, Triton, or TileLang. |
| What is the finite Ada reachability frontier? | The full audit retains 304/304 outcomes. `register_fused`, `global_intermediate`, and `register_common_postprocess` share 15 four-lane-reachable grids. `smem_staged` has no four-lane-common grid. | RTX 6000 Ada, the frozen 19-grid budget, source bundle, and toolchains. |
| Why are requested cells unavailable? | There are 19 source-level `UNSUPPORTED` cells and 36 grid-specific `BUILD_FAILED` cells. Four register-common Triton requests need 106,496 B of shared memory when 101,376 B is available. | The terminal stage and recorded cause are answerable; intrinsic language ceilings are not. |
| Do launchable cells meet the named operation contract? | All 249 timed-eligible cells pass four cases, 64 held-out seeds, and two gate views, yielding 127,488 bound gate rows. | Conformance to this finite frozen gate, not correctness for every input. |
| Does bias/GELU placement affect time when row softmax is held fixed? | Yes, locally. In Triton, register-common/global is 1.064220 at `g01` and 0.978482 at `g05`; both exact intervals clear the sham floor and their directions reverse. | Two cell-specific positive-distribution placement contrasts, not a universal placement ranking. |
| Are there resolvable lane-realization differences? | Sixteen preregistered cell-specific lane contrasts meet the reporting rule. | Named strategy/grid cells on positive `rand`, seed 0; no generalized lane ordering. |
| What resolution can the timing pipeline demonstrate? | Duplicate labels for one byte-identical implementation yield an absolute log-ratio floor of `0.014957935591828302`. | Label/process/order/harness variation in this campaign; not shared systematic bias. |
| Are the historical Phase-1 A/B/C/D matmul artifacts legal under matmul-v4? | No. The 66,852-record audit shows every route fails all 512 `legacy_u01` and all 512 `opposing_means` inputs while passing the four signed cases. | These frozen artifacts under matmul-v4; not all fp16-accumulation implementations. |
| Can the evidence and analysis be independently checked? | Yes. The archive verifies all 2,638 entries, and a separate rederivation of the final summary is byte-identical. | Reproducibility of the retained evidence and analyzer, not external validity. |

The exact fused results and boundaries are recorded in the
[`crossed_v2r3` result memo](fused_epilogue_crossed_v2/RESULT_CROSSED_V2R3_20260802.md).
The broader current-state synthesis is
[`CURRENT_STATE_REPORT_20260804.md`](CURRENT_STATE_REPORT_20260804.md).

## 3. Intellectual merit and generalizable knowledge

### 3.1 Expressibility is a denominator, not a footnote

Performance tables usually condition on code that was successfully written,
built, and launched. That hides strategies a system could not express or could
not realize within the hardware budget. The surviving implementations then look
like the whole design space.

The generalizable contribution is to publish both the requested denominator and
the timed numerator. This converts “unsupported” from an informal excuse into a
measured outcome and exposes the intersection on which fair performance
comparisons are possible.

### 3.2 Reachability is set-valued

A DSL does not have one scalar capability score. Its observable capability is a
set of reachable `(strategy, grid, hardware, toolchain)` cells. Cross-DSL
comparison depends on the intersection of those sets. A strategy may be
source-expressible but resource-infeasible at a particular grid, so support and
success must remain separate axes.

This set-valued view generalizes to sparsity systems, quantization DSLs,
distributed runtimes, vectorizers, and compiler feature studies. It prevents a
best surviving point from impersonating a whole programming system.

### 3.3 Mechanism interactions are a result

The placement contrast changes direction between Triton `g01` and `g05`.
Therefore the answer is not “register placement wins” or “global placement
wins.” The answer is that placement interacts with launch geometry within this
implementation family.

This is useful community knowledge: when effects reverse across legal
configurations, averaging them into a global ranking discards the mechanism the
experiment revealed.

### 3.4 Correctness belongs inside optimization

The historical matmul study showed a 7.0× to 1.31× shrinking spread under its
legacy gate. Matmul-v4 later demonstrated that all four A/B/C/D artifacts share
failures on two registered cases. The historical number remains a diagnostic of
normalization methodology, but it is not a current correctness-qualified
frontier.

The durable lesson is stronger than the old timing number: arithmetic, schedule,
and search normalization must occur inside the current correctness contract.
Bit-identical agreement among implementations does not establish correctness if
they share the same unexercised error.

### 3.5 Measurement resolution should be empirical

A statistical interval describes sampling uncertainty under an analysis model.
It does not establish how much variation the complete build/process/order/timing
pipeline creates under a true null. Duplicate labels for one byte-identical
implementation provide that null.

Requiring a real contrast's whole interval to clear the sham floor turns
resolution into a preregistered publication rule. The idea generalizes to any
systems benchmark where process state, clocks, compilation, ordering, or harness
logic can create small apparent effects.

### 3.6 Failures of the experiment are part of the evidence

The three frozen predecessor attempts failed before cell retention, after the
complete audit, and after the complete screen. Preserving each failure under an
incident receipt makes analyzer and orchestration defects visible. Repairing
under a new lock prevents a favorable partial result from being silently
promoted.

This provides a general recovery rule for empirical systems work: freeze the
analysis with the experiment, preserve the failed closure, identify its
scientific consequence, and rerun every downstream phase that the defect could
have influenced.

## 4. Why these answers are experimentally controlled

| Threat | Control | Reasoning |
|---|---|---|
| Survivorship bias | Audit the complete `4 × 4 × 19` product before timing. | Failed and unsupported requests remain in the denominator. |
| Unsupported-by-declaration | Run source-retaining 19-grid support probes. | Repeated captured evidence distinguishes a pinned API limitation from omitted engineering. |
| Resource failure mislabeled as language failure | Preserve separate terminal states. | Failure stage determines the defensible causal interpretation. |
| Confounded treatment | Add register-common as a bridge that holds row-softmax source and launch geometry fixed. | Replication cannot identify which mechanism mattered when two mechanisms change together. |
| Candidate-adaptive correctness | Calibrate from named, candidate-independent anchors and lock domain-separated validation seeds. | Candidate errors cannot set their own acceptance thresholds. |
| Search reuse and manual choice | Two-process screen, mechanical top-two-plus-`g01` selection, fresh confirmation. | Search chooses candidates; separate processes estimate the selected contrasts. |
| Pseudoreplication | Use process-level paired ratios and seed-level pairing. | Trials, tensor elements, candidates, and duplicated gate views are not independent experimental units. |
| Timing transients and order | Freeze randomized blocks and trials 60–99 prospectively; retain full-window and drift diagnostics. | Warmup alone does not prove that early timed trials are settled. |
| Effects below instrument resolution | Run a byte-identical duplicate-label sham. | A small non-null interval is not published when the pipeline exhibits null variation of the same size. |
| Post-hoc analysis repair | Bind source, analyzer, lock, hardware identity, counts, and incident policy before execution. | Analysis changes can alter eligibility, selection, and reported effects. |

These controls reduce specific threats; they do not remove all uncertainty.
Application clocks were not locked, no thermal admission band was enforced, the
28 intervals are per-contrast rather than familywise intervals, and the
reportable timing effects use one positive `rand`, seed-0 input.

## 5. Worked experimental examples

### Example A — Can a DSL express the requested strategy?

**Setup.** Freeze one fused operation,
GEMM+bias+exact-erf-GELU+row-softmax at `M=1024`, `N=8192`, `K=8192`, with four
strategies, four lanes, and 19 launch grids.

**Design.** Resolve the two questionable source-level claims with 19-grid
source-retaining probes, then audit the entire 304-cell Cartesian product. Give
every cell exactly one terminal outcome.

**Conduct.** CUDA-no-PTX register placement passes all 19 probes. Triton explicit
user-managed shared memory produces the same captured public-API limitation on
all 19 probes. The full
[`audit summary`](fused_epilogue_crossed_v2/results/crossed_v2r3/audit_summary.json)
retains 304 outcomes and 127,488 gate rows for 249 passing cells.

**Reasoning.** The complete product prevents fast survivors from standing in for
a DSL. Separate probes distinguish source expressibility from engineering
omission, while terminal states distinguish an API boundary from a resource
ceiling or incorrect result.

**Answer.** Three strategies share 15 reachable grids across all four lanes;
`smem_staged` has no four-lane-common grid.

**Boundary.** This is a finite Ada/toolchain/source frontier, not a theoretical
language ceiling.

### Example B — Does epilogue placement affect performance?

**Setup.** Compare `global_intermediate`, which applies bias/GELU in the common
postprocess, with `register_common_postprocess`, which moves bias/GELU into
lane-native accumulator code. Both bind the same checked row-softmax source body
and launch geometry.

**Design.** Form within-lane, within-grid paired contrasts so placement is the
intended changed factor. Admit only gate-passing cells, select confirmation cells
mechanically, and require the whole interval to clear the sham floor.

**Conduct.** Every gate-legal cell is screened in two fresh processes.
Confirmation uses 15 randomized blocks and process-level settled-tail medians.
The controlling estimates are in the
[`final summary`](fused_epilogue_crossed_v2/results/crossed_v2r3/final_summary.json).

**Reasoning.** The predecessor changed placement and softmax ownership together.
More repetitions would only estimate that confounded bundle more precisely. The
bridge strategy changes the estimand by holding row softmax fixed.

**Answer.** In Triton, register-common/global is 1.064220 at `g01`, interval
`[1.017226, 1.096966]`, and 0.978482 at `g05`, interval
`[0.970593, 0.982281]`. Placement interacts with grid.

**Boundary.** Only these two positive-distribution placement contrasts meet the
reporting rule; no universal placement order follows.

### Example C — Do historical optimized artifacts remain correct?

**Setup.** Take the four frozen Phase-1 A/B/C/D matmul routes and evaluate them
with matmul-v4's six cases, candidate-independent named anchors, and semantic and
conformance views.

**Design.** After v3 failed 2/64 `opposing_means` validation seeds, create a new
namespace with 640 calibration and 512 validation seeds per case. Quarantine
validation until calibration is complete and the gate specification is frozen.
The calibration size satisfies `24 × 0.99^640 = 0.0386132 < 0.05` for the 24
case/metric families per gate; zero failures among 512 validation seeds gives
the preregistered Bonferroni-adjusted per-case upper bound `0.009307`.

**Conduct.** The instrument audit completes all 66,852 expected unique records
with no missing, duplicate, unexpected, or binding failures. The
[`matmul-v4 summary`](robust_gate/audits/matmul_v4_instrument_v1/results/summary.json)
records every A/B/C/D route failing all 512 `legacy_u01` and all 512
`opposing_means` inputs while passing all four signed cases.

**Reasoning.** The new namespace prevents reuse of failed holdout information.
Candidate-independent anchors prevent self-grading. Case-level reporting avoids
pooling heterogeneous inputs into an artificially reassuring success rate.

**Answer.** The historical 1.31× frontier is not matmul-v4-legal and cannot be a
current performance conclusion without a new gate-in-loop search.

**Boundary.** The result rejects these artifacts under this gate; it does not
reject every fp16-accumulation design.

### Example D — Which timing contrasts are actually resolvable?

**Setup.** Screen 249 gate-legal cells in two fresh processes each. Select the
top two legal cells plus legal `g01` per strategy/lane, deduplicate to 44 cells,
and confirm both positive and withheld-signed inputs in 15 randomized blocks.
Each process records 100 trials after a two-second warmup and L2 flush.

**Design.** Separate screen from confirmation, freeze trials 60–99 as the primary
window, pair comparisons within blocks, and send two labels for one byte-identical
implementation through the same pipeline.

**Conduct.** Screening completes 498/498 records. Confirmation completes
`44 × 2 × 15 = 1,320` cell records plus 60 sham records, for 1,380/1,380.
Inference uses 15 process-level paired ratios, not 1,500 trials as independent
samples.

**Reasoning.** Separate confirmation limits direct reuse of search noise.
Mechanical selection removes manual substitutions. Randomization limits
fixed-order confounding; pairing controls block-level shifts. The late window is
a prospective response to predecessor timing trajectories, not proof of
stationarity. The sham asks whether the entire pipeline can resolve the claimed
effect.

**Answer.** The sham yields a 0.014958 absolute log-ratio floor. Eighteen of 28
preregistered positive-distribution contrasts meet the reporting rule: two
placement and 16 lane contrasts.

**Boundary.** These are per-contrast intervals on one positive timing input.
Withheld-signed results are diagnostic, clocks were not locked, and the sham
cannot expose bias shared by both labels.

### Example E — Can a frozen experiment recover from analyzer defects?

**Setup.** Bind campaign source, analyzer, dependencies, hardware identity,
expected counts, and recovery rules before GPU execution.

**Design.** If frozen code fails, preserve its exact closure, mark it
non-controlling, repair under a new source bundle and lock, and rerun every
downstream phase the defect could influence.

**Conduct.** `crossed_v2` stops before cell retention; `crossed_v2r1` retains the
complete audit but rejects valid resource failures; `crossed_v2r2` completes the
audit and screen but fails before confirmation selection. No partial selection
or timing is promoted by overlay. `crossed_v2r3` runs every phase fresh.

**Reasoning.** An analyzer defect can change the feasibility census, selection,
or final claims. Repairing a viewed result tree introduces a researcher degree
of freedom even if some measurements remain byte-identical.

**Answer.** The successor completes, its archive verifies independently, and its
rederived final summary is byte-identical. The
[`result memo`](fused_epilogue_crossed_v2/RESULT_CROSSED_V2R3_20260802.md)
binds the incident chain.

**Boundary.** This establishes auditability of the retained experiment, not
cross-hardware reproducibility.

## 6. Questions the present study cannot answer

| Question | Why it is unanswered | Experiment required |
|---|---|---|
| Is the reachability frontier architecture-general? | Only sm_89 Ada evidence exists. | Run the policy-authorized support probes and 304-cell feasibility audit on a frozen non-sm_89 successor. |
| Which DSL is generally fastest? | One fused shape, one architecture, selected cells, and cell-specific interactions cannot identify a universal order. | A preregistered multi-operation, multi-shape, multi-architecture study with explicit estimands and common reachable subsets. |
| Is any observed unsupported state a theoretical language ceiling? | The probes cover pinned public implementation paths and versions. | Broader API/version coverage or a formal language-capability argument. |
| How much do author effort and search cost move the frontier? | Compile observations are descriptive and the effort-frontier program has no controlling result. | Randomized or balanced effort budgets with replicated search and a declared effort estimand. |
| Do optimization recipes transfer across DSLs? | Reciprocal transfer has no controlling execution result. | Balanced donor/destination translations, frozen semantic equivalence, equal retuning budgets, and fresh confirmation. |
| Do search trajectories converge across DSLs? | The convergence program is retired unlaunched. | A newly justified design conditioned on the reachability frontier; the retired protocol contributes no evidence. |
| Do timing conclusions generalize across input distributions? | Reportable effects use positive `rand`, seed 0; withheld-signed timing is diagnostic. | The separately funded public-corpus paired `rand`/`randn` study required by policy. |
| Are gate-passing implementations universally correct? | The gate covers finite cases and seeds. | Formal verification or a broader independently powered validation domain. |

The governing
[`later-work policy`](LATER_WORK_POLICY_20260731.md) authorizes only
second-architecture feasibility, not timing. It also keeps RQ(e), convergence,
reciprocal-transfer, and effort-frontier claims out of the current paper scope.

## 7. A reusable protocol for future GPU DSL studies

1. Define the requested denominator before implementation begins.
2. Separate source expressibility, build/setup, launch, correctness, and timing
   into terminal stages.
3. Use source-retaining probes for disputed capability claims.
4. Calibrate correctness from candidate-independent anchors, freeze disjoint
   validation inputs, and fail closed on malformed evidence.
5. Add bridge treatments until each performance contrast changes one intended
   mechanism.
6. Audit the full requested product before selecting timing candidates.
7. Separate search from confirmation and derive selection mechanically.
8. Identify the real independent unit, usually a process or seed rather than a
   trial or tensor element.
9. Inspect timing trajectories, preregister the primary window, randomize order,
   and pair comparisons within blocks.
10. Run a true-null sham through the complete pipeline and make its empirical
    resolution a reporting rule.
11. Bind source, analyzer, hardware, expected counts, and evidence hashes before
    execution.
12. Preserve failed frozen runs and repair only under new locks.

## 8. Current state and next legitimate claim

The Ada campaign is complete and controlling as `crossed_v2r3`. The gate
saturation diagnostic has been delivered in the owner's repository through
[GitHub issue #2](https://github.com/tingxi-li/MultiKernelBench/issues/2), which
is open and assigned; explicit acknowledgment remains pending.

The current host exposes four RTX 6000 Ada devices and no non-sm_89 GPU. The next
legitimate expansion is therefore blocked: a hardware-bound second-architecture
304-cell feasibility matrix. Policy forbids performance timing on that second
architecture and delays the one paper until the matrix is sealed and verified.

## 9. Evidence anchors

| Artifact | SHA-256 or binding |
|---|---|
| Crossed-v2 source bundle | `abe6873e89686ca94a2dc096b633feff615c7dbfd8fb4e91546850ed26cae20b` |
| Crossed-v2 launch lock | `206b75387acf4d60b688a8e19b3aff094a8ef69d5a0a1b72269d8b24f48be1b6` |
| [`audit_summary.json`](fused_epilogue_crossed_v2/results/crossed_v2r3/audit_summary.json) | `096a9833d47f858752fd4a31bbee3db7df525d7d36e91593bd6a011b05ff86aa` |
| [`confirmation_selection.json`](fused_epilogue_crossed_v2/results/crossed_v2r3/confirmation_selection.json) | `a008ba2a5e6a81432eee7e94e7ebb34eeb90e3b88031f901cf10129a4be432ca` |
| [`final_summary.json`](fused_epilogue_crossed_v2/results/crossed_v2r3/final_summary.json) | `b38e01849024c285e14ea1eba7477c85443c68ca54b08a885c32f2413fa5ca03` |
| Complete evidence archive | `4e6dda75b0516fd1cac928ec1a30ec84e03ec0774be10c178ef3b4d9e650bded` |
| Complete evidence index | `3340a86f4f64404a96461fa5bb5262ff3e1cebacd645d4bd63aaa1bedda212c5` |
| [`matmul-v4 summary`](robust_gate/audits/matmul_v4_instrument_v1/results/summary.json) | `125e66bbd76012e49695b1226ff39943d52bb61da9a44842b549a53b1f678ebd` |
| [`gate-saturation report`](robust_gate/FUSED_V2_GATE_SATURATION_20260731.md) | `60911716ce6ae2fdd3b754ee9218bd51e4a9001760bd1612224f4392d2fdd9a6` |

## Conclusion

A GPU DSL study can answer more than a speed contest and less than a language
theorem. Its strongest answers concern the boundaries between expressibility,
hardware feasibility, correctness, and factor-isolated realization. Those
answers become useful to the community when the denominator is complete, the
failure stages are preserved, the treatment changes one mechanism, and the
measurement floor is demonstrated rather than assumed.

The current evidence supports that form of answer on Ada. It does not support a
universal winner. The distinction is not rhetorical restraint; it is the main
scientific result.
