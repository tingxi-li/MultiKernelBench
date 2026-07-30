# Response to the controlled follow-up review

> Date: 2026-07-30
>
> Review: [`REVIEW_20260730.md`](REVIEW_20260730.md)
>
> Completed-run ledger: [`RUN_20260730.md`](RUN_20260730.md)
>
> Historical-report overlay: [`ERRATA_20260730.md`](ERRATA_20260730.md)

This is an append-only response. It does not alter the completed campaign's
launch receipts, result files, frozen gates, analysis sources, or provenance
snapshots. The review's rederivation of the run-ledger claims is accepted.
Several review-level derived quantities and interpretations need correction or
narrower wording; the remaining issues are recorded below as prospective work
rather than backfilled into the run.

## Corrections to the review

### The strict all-four feasible intersection has nine points

The review's 11-point “common-feasible subset” is the intersection of the two
CUDA lanes, not all four lanes. Triton also fails `g15` and `g18`. The strict
all-four intersection is:

```text
g00, g01, g02, g03, g04, g13, g14, g16, g17
```

The two-process screen minima on those nine points are unchanged:

| DSL | Point | Screen median (ms) |
|---|---:|---:|
| TileLang | `g03` | 1.473024 |
| Triton | `g00` | 1.413376 |
| CUDA without inline PTX | `g04` | 1.829376 |
| CUDA with inline PTX | `g02` | 1.768448 |

The resulting exploratory screen spread is `1.2943307x`. It has two processes
per cell and is not evidentially equivalent to the five-process confirmed
frontier spread (`1.4481937x`). The fixed, five-process `g01` comparison has a
`1.2733267x` point spread.

### The epilogue was not expression-matched across all four lanes

The manifest declares `epilogue=smem`, but that field does not impose the same
implementation on every builder:

- TileLang applies bias and exact GELU in its accumulator fragment, then stores.
- Triton applies bias and exact GELU in its accumulator, then stores.
- CUDA without inline PTX stages the full fragment tile through shared memory.
- CUDA with inline PTX selects a shared-memory or register epilogue, but the
  current common macro unconditionally reserves a full `BM*(BN+4)*4` fp32 tile;
  the current register path therefore still requests that allocation.

Consequently, the 133,120--135,168-byte failures show reachability of these
specific CUDA recipes and allocation macros on this GPU. They do not establish
language-intrinsic CUDA-versus-compiler reachability. The `1.4481937x` result is
an equal-point-budget **implementation/recipe frontier** over a shared nominal
grid, with both realization and recipe reachability in the estimand. It is not a
same-epilogue code-generation contrast.

### The historical torch fp16 anchor is contract-unmatched

The historical `torch:fp16` fused implementation casts the activation inside
the timed `run`, casts the bias to fp16, executes GELU and softmax in the half
path, then converts the result to fp32. The follow-up custom configurations use
a precast fp16 activation, while the frozen fused-v2 contract requires fp32
bias, probability, and output semantics.

The historical 1.522688 ms torch value is therefore a useful cross-campaign
diagnostic, but it is neither timed-region-matched nor arithmetic-contract-
matched to the new custom frontier. One exact rerun of that historical torch
implementation cannot certify “vendor beaten.” A closure experiment needs at
least two contemporaneous torch controls:

1. the historical fp16 arithmetic with activation precast outside the timer;
2. a true mixed-contract path with fp16 operands, fp32 accumulation/output, and
   the fp32 bias/GELU/softmax contract used by fused-v2.

Both must run with the custom cells in the same randomized session.

### Robust-gate terminology needs tightening

Matmul-v4 did not have campaign-scale, full-shape negative controls or contact
with Phase-1 DSL kernels. It did have CPU pipeline/unit negatives (including a
zeros candidate and NaN/shape metric preflight), and its `native_fp32` and
`native_mixed` holdouts execute real CUDA torch matmuls on disjoint seeds.
Those GPU holdouts are positive controls: their zero-failure result measures
positive coverage and false-rejection risk. The missing campaign-scale
quantity is rejection of wrong candidates, often described as false-acceptance
or false-positive behavior. Calling v4 itself a “false-positive-rate self-test”
is therefore inaccurate.

Fused `semantic_mixed` is explicitly an fp16-tier test. The adapter enforces
GBGS, `arith=fp16`, `cast=precast`, and a half activation. It measures closeness
to fp64 semantics under an fp16 contract; it is not evidence that the kernels
survive the q32 tier.

The thin row-sum margin is real but not unique to the TileLang winner. The raw
`1.25 * anchor_max` cutoff is `4.538103187679e-7`, while the preregistered
next-1-2-5 threshold is `5e-7`. Of 28,672 validation records, 292 exceed the raw
cutoff and none exceeds the registered threshold. Maxima for the four frontier
cells are:

| Cell | Maximum row-sum error | Headroom to `5e-7` |
|---|---:|---:|
| TileLang `g08` | `4.797578e-7` | 4.05% |
| Triton `g05` | `4.756419e-7` | 4.87% |
| CUDA-no-PTX `g04` | `4.816827e-7` | 3.66% |
| CUDA-unlimited `g04` | `4.816827e-7` | 3.66% |

The review's direction-loaded-retry warning remains appropriate prospectively,
but the retained evidence supports the sparse-v3 diagnosis. The fresh v4
calibration maximum (`0.000621776`) exceeds both v3 failing holdouts
(`0.000549492`, `0.000528148`); v4's validation maximum is `0.000562199`, or
56.2% of the frozen `0.001` threshold. Forty-two of 640 v4 opposing-means paired
maxima exceed `0.0004`; under the empirical draw model, a size-32 calibration
misses all of them with probability about 10.8%. This is supporting evidence,
not permission for another validation-driven retry.

### Reciprocal transfer currently covers standard matmul only

The reciprocal scaffold's workload is Phase-1 standard matmul. If completed, it
can test home-recipe anchoring and robust-gate legality for the historical
`1.31x` matmul frontier. It cannot adjudicate recipe dependence of the new fused
`1.4481937x` frontier. That requires a separate fused reciprocal design with
fused recipe cards, epilogue contracts, and feasibility estimands.

## Disposition ledger

| Review issue | Disposition | Append-only response / required action |
|---|---|---|
| Historical reports omit the LLM searcher | **Upheld** | The errata discloses one stochastic agent run per cell, Opus 4.8 for five compute ops, and unknown LayerNorm attribution. The Sonnet campaign is sensitivity evidence, not a pooled convergence replicate. |
| Historical strict chains cross overlapping CIs | **Upheld** | Treat both chains as point-estimate orders. The endpoint spreads remain reportable; their overlapping middle comparisons are unresolved. |
| No explicit per-op spread/scope table | **Upheld** | The errata supplies matmul, fused, and tile-unmatched SDPA rows, with gate and feasibility scope. |
| Native GPU/lane/clock scope is misleading | **Upheld; irreparable retrospectively** | Native rows lack a GPU field and cannot be normalized post hoc. Scope “one GPU” to controlled studies and treat native runtime/compute comparisons as confounded agent-run observations. |
| `1.448x` conflates realization and reachability | **Upheld and narrowed** | Use the corrected nine-point strict intersection and label it screen-only. Call the headline an implementation/recipe frontier, not intrinsic DSL reachability. |
| Equal shared-memory epilogue | **Corrected** | The manifest label is nominal; TileLang/Triton accumulator epilogues and CUDA staging/allocation are not expression-matched. |
| Historical vendor anchor likely beaten | **Hypothesis retained; proposed one-lane closure rejected** | The old torch lane is also contract-unmatched. Run the two torch controls above in a contemporaneous randomized session. |
| Fused timing uses only `rand`, seed 0 | **Upheld** | Carry this scope beside every performance number. Robust signed inputs were correctness-only, not timed performance inputs. |
| Matmul-v4 lacks negative controls | **Partly upheld; terminology corrected** | CPU negatives exist, and GPU positive controls are real CUDA matmuls. Full-shape wrong-candidate sensitivity and Phase-1 kernel contact remain missing. |
| Thin fused row-sum margin | **Upheld as a cross-lane issue** | Report 292/28,672 above the raw cutoff and the four maxima; do not single out TileLang. |
| v3-to-v4 retry is direction-loaded | **Upheld prospectively** | Preserve the retry history and the supporting sparse-sample diagnostics. No further validation-driven recalibration is acceptable. |
| Fused result provenance is bifurcated | **Upheld; post-hoc preservation added** | `provenance/evidence_v1/` preserves 251 essential ignored files in a deterministic bundle. It explicitly does not prove launch-time ordering or external timestamping. |
| Confirmation analyzer omitted from launch source binding | **Upheld** | Do not alter the old receipt. Future receipts must bind the analyzer and a machine-readable inference policy before launch. |
| Reciprocal campaign de-anchors both matmul and fused | **Corrected** | Current reciprocal manifests cover standard matmul only. |
| Convergence campaign is ready after three blockers | **Rejected** | The checked-in summary and ledger disagree; the reconciled blocker set below remains open. |
| `n=5` cannot resolve contested orders | **Upheld** | If ordering matters, preregister more processes and match estimator to interval. Do not infer an order from the current point estimates. |

## Post-hoc evidence preservation

The completed fused result tree remains ignored and the original provenance
snapshots exclude `results/` and `raw/`. The new
[`provenance/evidence_v1/`](provenance/evidence_v1/) layer selects all 152 screen
records, all 80 confirmation records, their launch/status/selection/summary
controls, and the three aggregate robust record streams (608, 3,520, and 28,672
rows). It excludes caches, binaries, scratch files, and redundant robust
per-record copies.

The deterministic archive and index are post-campaign preservation artifacts.
They verify current bytes and make those bytes portable; they do not prove that
the source tree was committed before execution, bind the missing analyzer at
launch, or provide a third-party timestamp. Those limitations must remain in
any publication narrative.

## Reconciled convergence blockers

`convergence/manifests/summary.json` records four blockers, while the run ledger
abbreviates that list and adds sampling-seed support. Launch readiness must use
the union, not either prose list:

1. resolve both model aliases to immutable provider revisions;
2. freeze and bind prompt templates plus system/tool contracts;
3. freeze a hidden robust gate before any trajectory can observe it;
4. provide a runner that records provider token usage and the controller's
   completed-compute clock;
5. verify provider sampling-seed support, or preregister independent stochastic
   requests without claiming deterministic seed control.

Per-trajectory GPU UUID pinning, serialization/randomization, and clock-state
telemetry are also required design amendments. The current two aliases belong
to one model family; retaining them scopes the estimand to within-family
searcher sensitivity. They cannot retroactively identify the historical
Opus-versus-Sonnet effect.

## Priority after documentation and preservation

1. Fix future source/evidence capture and bind analysis policy before another
   performance launch.
2. Run the two torch controls with frozen custom cells in one randomized
   campaign; optionally confirm the strict-intersection cells in that session.
3. Test frozen matmul-v4 against preregistered full-shape wrong candidates and
   Phase-1 A/B/C/D kernels.
4. Complete the archived-versus-current fused rebenchmark.
5. Unblock and execute standard-matmul reciprocal transfer; design a separate
   fused reciprocal campaign if the `1.448x` recipe dependence is in scope.
6. Amend the convergence design, then launch it only after one canonical
   validator reports every blocker closed.

## Post-review results (append-only, 2026-07-30)

The experiments requested after the review are now complete for fused closure,
CUDA reachability, fresh fused row-sum stress, and archived-versus-current fused
artifacts.  They do not rewrite the historical reports or mutate any frozen
threshold, validation split, selection, or launch receipt.  The separate
matmul-v4 instrumentation/negative-control audit remains active and contributes
no completed result to this section.

### Receipt-backed evidence ledger

| Evidence | Bound result and SHA-256 | Census |
|---|---|---:|
| Fused closure v2 | [`source_receipt.json`](fused_closure_v2/source_receipt.json), `d38ba70aa667ee873e3c4c75708881ddf87ab3f48beb6ddc10cfcf94126156c7`; [`gate_summary.json`](fused_closure_v2/results/gate_v1/gate_summary.json), `0d7955543c262a5dff7fcfd8a3466678f8c55b13b5290c48dd40c90328a068e4`; [`analysis_summary.json`](fused_closure_v2/results/performance_v1/analysis_summary.json), `072ce7ea841dad51fe63c50d33c7f02f1bc2cf8c12cc97f3d1d75d387d9bbcec` | gate `4,608/4,608`; performance `135/135`, zero failures |
| Streamed-epilogue reachability v2 | [`launch_lock.json`](fused_reachability_v2/launch_lock.json), `7bd24892e29e8d8af4b48a34b397434ed80bd59e2349ff714fb4babe0afcc240`, binding source bundle `ac077b9de182ea9596e97dafa6a6e2ee88771f3d282dec9ed7d19bde90078d2a`; [`screen_summary.json`](fused_reachability_v2/results/screen_analysis_v1/screen_summary.json), `e0013d3bbe1169d2a55d4154a1b1ef6cc48bc08563d5bb1694579e15d3daa753`; [`confirmation_summary.json`](fused_reachability_v2/results/confirmation_analysis_v1/confirmation_summary.json), `5b691fbacc772b3d27af743791b6aa3dd0edd8e7d63d125c56399f86b22e4c7f` | screen `32/32` records in `16/16` cells; robust `3,072/3,072`; confirmation `90/90` |
| Prior-winner row-sum stress | [`summary.json`](robust_gate/audits/fused_row_sum_stress_v1/results/summary.json), `df51077cf16f5a67abc6ab821bda99ecd87ee93b77a8e8cd3246cdcf38859891`; [`completion_receipt.json`](robust_gate/audits/fused_row_sum_stress_v1/receipts/completion_receipt.json), `fae663bca0ee2802cd6afcb24cb200a909da59099b5ce48896d54db53b39e39e` | `2,048/2,048` candidate-gate records, four candidates, two gates, 256 fresh seeds |
| Streamed-epilogue row-sum stress | [`summary.json`](robust_gate/audits/fused_reachability_row_sum_stress_v1/results/summary.json), `326cc75f7901ad602f239f75740ecd004e9749feee98fdeddfb0233698a0427a`; [`completion_receipt.json`](robust_gate/audits/fused_reachability_row_sum_stress_v1/receipts/completion_receipt.json), `9c48a157635fdb179b0021d38ce7bd642059fb54c6ed8f60d61ea6bb96766b62` | `3,072/3,072` candidate-gate records, six candidates, two gates, 256 fresh seeds |
| Archived/current fused v1 | [`summary.json`](archived_current_fused_v1/analysis/main_v1/summary.json), `d212b4c1be6045eb9d2c53d12777a26f9d182173a4c1f8bb0b14ecd68e968777`; [`completion_receipt.json`](archived_current_fused_v1/results/main_v1/completion_receipt.json), `5900d33335361a5672fcc54a3a5f5f53a4ac59937c6b10eac008b8b56f103f90` | `60/60`, zero failures |
| Original fused post-hoc preservation | [`evidence_index.json`](provenance/evidence_v1/evidence_index.json), `70e939efdad1b41aafc99f94588ee5795311482c7b455fb50261cf5a174aef96`; [`evidence_bundle.tar.gz`](provenance/evidence_v1/evidence_bundle.tar.gz), `6305573cbb13e47902d0b0cd58d629feea812678ca6e4b1e13e4884f96432ea6` | 251 files: 152 screen, 80 confirmation, and aggregate robust streams of 608, 3,520, and 28,672 rows |

The reachability evidence bundle contains 1,712 entries: its
[`complete_v1.index.json`](fused_reachability_v2/evidence/complete_v1.index.json)
has SHA-256
`d228ba509e02f2ed7e9a75346e2e960e3ec9fa6f8cfea943193da6a91b7f06cb`,
and its archive has SHA-256
`efcd54b68885b2bdd2c60202d68b858fbe3f40a8288d5815ef5fa7eeb6536572`.
The streamed row-sum audit's 25-entry index and archive have SHA-256 values
`00f985e92bbfdc5697d88b5e5334093d1e5c0124a82715c070c077bc6cb87cbe`
and `8acc2e8927f95761781688647cddcb5ef3e6dba9df2e9a8d650aced96aa41028`,
respectively.

### Same-contract Torch closure is positive but narrow

The closure gate covered nine candidates under both mixed gates, with 256
records per candidate-gate group.  The same-contract
`torch_contract_fp32`, TileLang, Triton, and strict-common CUDA controls passed
their frozen groups.  Both historical half-arithmetic Torch controls failed
`256/256` records under each gate, so they remain diagnostic and are excluded
from same-contract performance claims.

All nine candidates then completed 15 randomized performance blocks.  For the
two preregistered same-contract custom/Torch comparisons, the paired median
ratios were:

| Comparison (`custom / torch_contract_fp32`) | Median ratio | Exact interval | Holm-adjusted p | Equivalent `torch / custom` |
|---|---:|---:|---:|---:|
| TileLang `g08` | `0.863852956` | `[0.858407072, 0.888349475]` | `0.00738525` | `1.157604x` |
| Triton `g05` | `0.863172215` | `[0.852713156, 0.870408856]` | `0.00195313` | `1.158517x` |

Thus these two frozen custom artifacts were faster than this contemporaneous,
same-arithmetic-contract Torch control in this session.  This is not a general
vendor-library or latent-DSL-optimum claim.  The strict-common four-recipe
within-block spread was `1.280590725x` (reported compactly as `1.280591x`), with
exact interval `[1.279077909, 1.283655568]`.  Its estimand is the four frozen
recipes, not an intrinsic language effect.

### Fresh row-sum stress narrows the old robustness claim

The four prior winners passed the original frozen `4 x 64` validation split,
but each failed `1/256` fresh `signed_normal_gain16` seeds under the unchanged
registered `5e-7` row-sum threshold.  Because `semantic_mixed` and
`conformance_mixed` evaluate the same candidate output and share this row-sum
decision, each unique failing candidate-seed outcome is counted in both gate
groups; those paired gate outcomes are identical and non-independent.  The
observed maxima were `5.832095236e-7` (TileLang `g08`), `5.435876638e-7`
(Triton `g05`), and `5.856042182e-7` for both CUDA `g04` artifacts.

This does not retroactively change the frozen split or threshold.  It does mean
that the earlier phrase “passed the robust gate” must be scoped literally to
the original `4 x 64` split; it is not evidence of zero failure on fresh
gain-16 inputs.

The new shared streamed epilogue passed its disjoint 256-seed stress, but the
maximum row-sum error was `4.965130613e-7`, only `0.697%` below the registered
threshold.  Nine of the 256 unique seeds exceeded the lower descriptive raw
safety cutoff `4.538103188e-7`.  The row-sum outcomes were identical across all
six selected grids and both gates on every seed, so those 12 streams are
non-independent and must not be pooled as 12 independent zero-failure samples.
The threshold stayed fixed, and this correctness-only audit did not feed back
into performance selection.

### The streamed design removes the observed CUDA allocation barrier

The reachability screen made every formerly excluded `g05`--`g12` recipe
launch-reachable and legacy-correct in both CUDA lanes: all 16 cells and all 32
process records were eligible.  The top three cells per lane then passed the
complete frozen validation split, `1,536/1,536` records per lane and
`3,072/3,072` in aggregate.  This demonstrates that a two-kernel recipe which
writes fp32 GEMM accumulators to a global intermediate and applies the shared
fp32 bias/exact-GELU/softmax epilogue in a second kernel avoids the prior full-
tile shared-memory allocation failure.  It does not show a language-intrinsic
ceiling, nor is it an expression-matched one-kernel comparison with the
TileLang and Triton artifacts.

The 15-process confirmation results were:

| Lane | Grid | Median ms | Exact `[x4,x12]` interval ms |
|---|---:|---:|---:|
| CUDA-no-PTX | `g05` | `1.473071992` | `[1.426559985, 1.501183987]` |
| CUDA-no-PTX | `g09` | `1.481695950` | `[1.443840027, 1.527807951]` |
| CUDA-no-PTX | `g08` | `1.650688052` | `[1.630720019, 1.662464023]` |
| CUDA-unlimited | `g06` | `1.519616008` | `[1.500159979, 1.547263980]` |
| CUDA-unlimited | `g07` | `1.411072016` | `[1.388415992, 1.440256000]` |
| CUDA-unlimited | `g05` | `1.500159979` | `[1.462272048, 1.515519977]` |

CUDA-no-PTX's point winner is `g05`, but the frozen lane-minimum bootstrap
status is **UNRESOLVED** (`g05` winner probability `0.74221`, `g09` `0.25779`).
CUDA-unlimited's `g07` result is **RESOLVED** (winner probability `0.99974`).
The reported v1/v2 lane-minimum ratios, `1.267946x` for CUDA-no-PTX and
`1.279390x` for CUDA-unlimited, are descriptive only because they cross
campaigns and physical cards.  No cross-GPU ratio between the two v2 lanes is
inferred.

### Archived/current artifacts drifted in opposite directions

In 15 randomized paired blocks, TileLang's `current / archived` median ratio
was `1.090397588` with exact interval `[1.079490056, 1.097570182]`, so the
archived TileLang file was faster.  Triton's ratio was `0.553445467` with exact
interval `[0.543730207, 0.578872048]`, so the current Triton file was faster.
Both Holm-adjusted sign-test p-values were `0.0001220703`.  This opposite drift
is artifact-level evidence: each pair contains different programs, and the run
used a bound single-input correctness diagnostic rather than the full fused-v2
gate.  It therefore reinforces the version/provenance warning but cannot be
interpreted as an intrinsic TileLang-versus-Triton change.

Finally, the 251-file original fused archive remains post-hoc preservation.
Its verified hashes make the retained bytes portable; they do not repair the
old analyzer binding, establish preregistration, or supply an external
timestamp.

### Same-GPU closure quantifies the reachability term

The final mechanism closure placed the old CUDA common recipes, the new
streamed CUDA recipes, the two compiler-frontier recipes, and
`torch_contract_fp32` in one 15-block randomized session on physical GPU 3.
All `120/120` measurements completed without failure.  Imported timing
eligibility is limited to the original frozen four-case, 64-seed, two-gate
validation split; it is not a fresh-stress claim.

The paired old-to-new CUDA ratios were:

| Comparison (`streamed / old`) | Median ratio | Exact interval | Holm-adjusted p |
|---|---:|---:|---:|
| no-PTX `g05 / g04` | `0.862385` | `[0.861279, 0.867097]` | `0.000183105` |
| no-PTX `g09 / g04` | `0.862385` | `[0.856836, 0.867150]` | `0.000183105` |
| unlimited `g07 / g02` | `0.789461` | `[0.784999, 0.796866]` | `0.000183105` |

Thus removing the full-tile shared-memory epilogue allocation improved these
fixed CUDA recipes by about 13.8% (no-PTX) and 21.1% (unlimited) in the same
session.  It did not erase the compiler-recipe advantage.  TileLang `g08` and
Triton `g05` each had about 20% lower latency than streamed no-PTX and about 9%
lower latency than streamed unlimited; all six fixed compiler/CUDA contrasts had exact ratio
intervals below one and within-family Holm-adjusted `p=0.000366211`.

The contract-Torch relation is recipe-specific: streamed no-PTX `g05` and
`g09` were slower than Torch (ratios `1.093574` and `1.097494`), while streamed
unlimited `g07` was faster (`0.967406`, exact interval
`[0.959459, 0.980932]`, Holm-adjusted `p=0.000976562`).  The fixed four-recipe
frontier spread for TileLang `g08`, Triton `g05`, streamed no-PTX `g05`, and
streamed unlimited `g07` was `1.250500x`, exact interval
`[1.243647, 1.256544]`.  This is the same-GPU replacement for attempts to read
the old `1.4481937x` as pure realization: recipe reachability contributed
materially, while a smaller fixed-recipe realization gap remains.  It is not
an additive decomposition or a universal DSL-optimum claim, and the no-PTX
`g05`/`g09` winner remains unresolved.

The bound
[`analysis_summary.json`](fused_frontier_closure_v3/results/performance_v1/analysis_summary.json)
has SHA-256
`a16601328aa45b7c573c111c3505f6c1d708bf9d070f5d52364996a895a49404`.
The 319-entry deterministic evidence index and archive have SHA-256 values
`5e12dc8676d4694a05e6970bbd3b54ab9cace3b09976e3e558563ed1b5cf0771`
and `8b0a05a24013e4944be27accdbcbb7b9fe24bd027bade08fddfbe5c63dc44760`,
respectively.
