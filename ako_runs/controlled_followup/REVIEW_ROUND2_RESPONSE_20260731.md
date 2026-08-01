# Response to the round-2 controlled follow-up review

> Date: 2026-07-31
>
> Review: [`REVIEW_ROUND2_20260730.md`](REVIEW_ROUND2_20260730.md)
>
> Current run ledger: [`RUN_20260731.md`](RUN_20260731.md)
>
> Historical overlay: [`ERRATA_20260730.md`](ERRATA_20260730.md)

This document controls interpretations made after the round-2 review. It does
not refit a correctness threshold, alter a frozen result, or turn an unlaunched
design into evidence. The review's 59 reproduced factual checks are accepted.

## Resolution of the review findings

| Review finding | Disposition | Current interpretation/action |
|---|---|---|
| Matmul-v4 documents were stale | **Corrected** | The completed `66,852/66,852` audit is now in the controlling ledger; old “active/no result” wording is removed or explicitly superseded. |
| `1.6033x` was presented as the closest failure | **Corrected** | It was a group maximum. The per-record minimum failing utilization is `1.2133795796216873x`; the maximum is `9322.150545631303x`. |
| Old `1/256` versus streamed `0/256` used disjoint seeds | **Resolved empirically** | The corrected 512-seed paired campaign completed `10,240/10,240`. Every streamed CUDA cell fails the same old seed 197 as both old CUDA winners; all six old/new contrasts have zero discordances and Holm-adjusted `p=1.0`. No robustness improvement is supported. |
| Fragility caveats were separated from timing tables | **Corrected** | Robustness warnings now appear inline beside the `0.86x` and `1.250500x` results. |
| CUDA received asymmetric engineering attention | **Upheld** | `1.250500x` is an asymmetric-engineering artifact snapshot, not an equilibrium recipe frontier. |
| Historical reports did not expose their corrections | **Corrected with receipt** | Each report has one current-corrections pointer; before/after hashes are recorded. |
| Archived Triton was mislabeled as TF32 | **Corrected** | The archived artifact is a 10-configuration fp16-cast kernel; the current artifact also changes operand conversion, weight caching, intermediate precision, and its 13-configuration grid. |
| Old CUDA harness did not fail closed on launch setup | **Corrected prospectively** | Receipt-bound historical wrappers remain unchanged. A reusable checked-launch overlay and injected-failure tests govern descendants. |
| Timing remains one shape/model and `rand`, seed 0 | **Open limitation** | No cross-shape, cross-architecture, or timing-distribution claim is added by this response. |

## Matmul-v4 completion and continuous margins

The authoritative binary result remains
[`summary.json`](robust_gate/audits/matmul_v4_instrument_v1/results/summary.json),
SHA-256 `125e66bbd76012e49695b1226ff39943d52bb61da9a44842b549a53b1f678ebd`.
It contains all `66,852` expected unique records with zero missing, duplicate,
unexpected, or binding failures:

- all four frozen-threshold replication blocks pass (`36,864/36,864`);
- all registered synthetic decisions are correct (`8,484/8,484`), including
  fail-closed structural controls;
- every Phase-1 A/B/C/D route fails all 512 `legacy_u01` and all 512
  `opposing_means` inputs, while passing all four signed cases.

The historical `1.31x` A/B/C/D artifact frontier therefore is not v4-legal.
That statement is about these artifacts, not about every fp16-accumulation
implementation or a future gate-in-loop search.

The corrected diagnostic is
[`margin_report_v2.json`](robust_gate/audits/matmul_v4_instrument_v1/results/margin_report_v2.json),
SHA-256 `884a7cd513ae50f6ec58859da9185737fcc8fbb54e900503e4522c324d33e3af`.
It validates all required metric values and recorded decisions before reporting
one maximum frozen-threshold utilization per real-candidate record. Of `21,504`
records, `7,168` fail and `14,336` pass.

| Extreme | Candidate/case/gate/seed | Limiting metric | Value / threshold | Utilization |
|---|---|---|---:|---:|
| Closest failure | A / `opposing_means` / `semantic_q32` / 465 | `max_abs_err` | `0.012133795796216873 / 0.01` | `1.2133795796216873x` |
| Worst failure | B / `legacy_u01` / `conformance_mixed` / 251 | `abs_signed_bias` | `0.18644301091262605 / 0.00002` | `9322.150545631303x` |

Version 2 also records each group's failure counts, metric-specific failure
counts, linearly interpolated utilization quantiles, seed-indexed per-record
maxima, and nearest/worst failing records. Missing, nonnumeric, negative, or
nonfinite required metrics and gate-decision mismatches abort generation. The
diagnostic does not authorize threshold mutation.

## Headline timing results with inline qualifications

| Timing result | Estimate | Qualification that travels with the number |
|---|---:|---|
| TileLang `g08 / torch_contract_fp32` | `0.863852956` | Same-contract/same-session, but TileLang `g08` failed `1/256` later fresh stress seeds; eligibility is the original `4 x 64 x 2` split. |
| Triton `g05 / torch_contract_fp32` | `0.863172215` | Same-contract/same-session, but Triton `g05` failed `1/256` later fresh stress seeds; eligibility is the original split. |
| Four-recipe frontier spread | `1.250500x` | TileLang/Triton were grid-frozen while CUDA received an oracle-informed streamed redesign; compiler artifacts failed separate fresh stress and old/new stress seeds were disjoint. |

The first two rows remain valid timing measurements of frozen artifacts. They
do not establish a vendor-library ordering or fresh-stress-qualified frontier.
The third row validly shows that removing the observed CUDA allocation barrier
reduced the same-session spread from the internally reproduced old frontier,
but it does not isolate a language effect under equal engineering effort.

The two original old-winner and streamed row-sum campaigns remain useful
thin-margin diagnostics, but their disjoint seeds could not support an
old-versus-new dichotomy. The corrected same-seed v2 campaign now closes that
comparison. It completed all `10,240/10,240` records over 512 shared inputs,
ten candidates, and two non-independent gate views, with no collection or
binding failures. TileLang `g08` fails seed 197; Triton `g05` fails seeds 186,
260, and 392; both old CUDA winners and all six streamed CUDA cells fail seed
197. The eight CUDA candidates have identical per-seed metric vectors under
both gates. Accordingly, each of the six preregistered old/new CUDA contrasts
has `both_fail=1`, `both_pass=511`, zero discordances, and Holm-adjusted
`p=1.0`. The streamed redesign established better reachability and timing, not
better fresh-stress robustness.

The controlling same-seed artifact is
[`results/summary.json`](robust_gate/audits/fused_same_seed_stress_v2/results/summary.json),
SHA-256 `d786cf0e1a42ba1680c132f7e859ca5152b34a691b4712161218670405806758`;
the 70-entry evidence archive has SHA-256
`9f4b8be60744da896fc14bcdf03fb4cc49780656c85b4e7b41c7a3ea742a450c`.
The protocol was locally source-frozen before execution, but it was not pushed
or externally timestamped before launch; that provenance limitation travels
with the result.

## Provenance and implementation corrections

The report/review, Phase-1 report, and both Phase-2 report forms now contain one
pointer to this response. The top-level report additionally corrects the
archived Triton description. The exact pre-edit and post-edit SHA-256 values are
in
[`historical_document_corrections_20260731.json`](provenance/historical_document_corrections_20260731.json).
This makes the correction discoverable while retaining content-addressable
identities for the historical bytes.

Receipt-bound historical CUDA source remains unchanged. Future descendants use
[`checked_cuda_launch.h`](legacy_cuda_harness_fix/checked_cuda_launch.h) to
check both `cudaFuncSetAttribute` and the immediately preceding kernel launch.
Its CPU test compiles against injected CUDA/Torch shims and verifies that
attribute and launch failures raise rather than continue.

## Implemented round-2 experiment programs

> **Superseded in part, 2026-07-31 (later the same day).** The crossed epilogue row
> below was accurate when written and is no longer. That campaign was pushed,
> launched, and completed as result tag `crossed_v1r1`; its first launch (`crossed_v1`)
> sealed zero outcomes and was replaced by a reporting-only recovery. See
> [`RUN_20260731.md`](RUN_20260731.md) for the completed-evidence entry and
> [`DESIGN_REFLECTION_20260731.md`](DESIGN_REFLECTION_20260731.md) for its
> interpretation and limits. The other three rows still stand.

The four inferential follow-ups are now implemented as fail-closed programs,
but none is represented as an empirical result:

| Program | Implemented design | Current non-result state |
|---|---|---|
| ~~Crossed epilogue v1~~ **(now complete — see the note above)** | `3` strategies x `4` lanes x `19` grids = `228` cells; `190` checked-supported and `38` explicit `UNSUPPORTED`; four-GPU audit sharding, GPU-0 timing, full fused-v2 gate, randomized confirmation, common-feasible and difference-in-differences analysis | ~~Source bundle `f8d7745416c7254d10cfdd144d80d91409f745d521d0debb5f5f8cc87c9de0b6` is locally committed as `4ddfc88d5712959e12161da75ce991f8c9a20248`; launch is forbidden until that commit is present on the configured upstream.~~ The commit was pushed and verified on the upstream at `2026-07-31T16:20:20.276522Z`, 0.089 s before the first GPU process. Controlling artifact: `fused_epilogue_crossed_v1/results/crossed_v1r1/final_summary.json` (`98552e48…`). |
| Reciprocal transfer v2 | Three origins, four destinations, literal/retuned modes, two isolated translators; `48` audit and `48` primary cells; 19-attempt retuning; strict audit-screen-terminal-primary lifecycle; common KC ladder and origin-by-destination interaction analysis | The append-only production supplement has `2` isolation requests, `48` translation requests, and a `24`-cell KC plan. Its 77-entry `prereg_v2` bundle is `8bfa1d6d06b986836defeddac276e91bb75701a902412f40af6de765e5e868df`. Real isolated translations, v4 KC summaries, and pushed provenance remain absent. |
| Convergence v2 | `192` core plus `128` prompt-extension trajectories, eight replicates per cell balanced two per GPU, GPT-5.6-sol versus Claude Opus 4.8, hidden terminal gates, censoring, Kaplan-Meier/log-rank/RMST/Holm analysis; registered total `56.5333` completed-evaluation GPU-hours | All four Ada identities are receipt-bound. The current 40-file `prereg_v2` bundle is `399d9928d0df31dfce5cc52443283d5339a8263b480f5d6f848b7413aebf6b85`. Immutable provider revisions, completed sum/SDPA gates, provider credentials, reference latencies, and remote preregistration remain unresolved. |
| Effort frontier v1 | Four programmable lanes x five independent contexts, checkpoints at `0.5/2/8` active hours, a non-search torch control, hidden holdout, exact lane inference, and 15-block GPU-0 confirmation under positive and withheld signed inputs | The 21-entry preregistration bundle is `50d1642c1a7c22f38ba95b08d9b77fc7793e022d99cade3f052030beab267ad6`. The four-lane executor registry is correctly classified as an experimental treatment artifact; model resolution, credentials, and pushed provenance are also absent. |

The crossed launch guard was exercised and stopped before GPU work because the
local commit had not been pushed. *(It was subsequently pushed and the campaign
ran; the guard behaved exactly as designed, first refusing and then admitting the
launch on the same criterion.)* The other validators likewise name their real
missing treatment or external dependencies. No placeholder source, provider
event, KC pass, launch authorization, or performance record was created.

The reachability-v2 `results/` ignore rule has also been removed (runtime
`active.lock` files remain ignored), exposing 1,682 plain result/receipt files
to version control in addition to the preserved evidence archive.
