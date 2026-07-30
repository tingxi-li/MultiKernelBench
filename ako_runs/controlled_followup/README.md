# Controlled cross-DSL follow-up

This subtree is the prospective follow-up to `CONTROLLED_CROSS_DSL_REPORT.md`
and its cross-read.  It does not overwrite or reinterpret the historical
results.  New campaigns must write to this subtree, carry a campaign manifest,
and bind every result to the exact source used to produce it.

Execution status and evidence hashes are recorded in
[`RUN_20260730.md`](RUN_20260730.md). The fused equal-grid campaign and matmul
v4 correctness-gate experiment are complete. Reciprocal recipe transfer and
replicated convergence remain explicitly launch-blocked and have no claimed
performance results.

The launch order is dependency driven:

1. `provenance/`: freeze sources, environment, GPU identities, and protocol.
2. `robust_gate/`: calibrate conformance and semantic-quality gates without
   looking at candidate outputs, then validate them on locked seeds.
3. `fused_grid/`: run the full `GBGS` native/matched/equal-grid protocol.
4. reciprocal recipe transfers and replicated search trajectories only after
   the correctness contract is frozen.

The estimand is a **finite-budget realization frontier** under the recorded
hardware, toolchains, input contract, and search budget.  No result in this
subtree is evidence of a theoretical language ceiling.

## Result states

Every campaign directory uses these states:

- `planned`: manifest generated, not run;
- `screening`: correctness-first low-repetition search in progress;
- `confirming`: candidates frozen and independently remeasured;
- `complete`: all preregistered cells and analyses present;
- `censored`: a declared resource/safety cap was reached;
- `invalid`: a preregistered integrity or hardware rule failed.

Build failures and correctness failures are retained as outcomes.  Missing or
failed cells are never silently replaced with private configurations.
