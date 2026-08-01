# Controlled cross-DSL follow-up

This subtree is the follow-up to `CONTROLLED_CROSS_DSL_REPORT.md` and its
cross-reads. It does not overwrite frozen results or gates. Historical reports
carry a current-corrections pointer, with their pre/post identities preserved
in `provenance/historical_document_corrections_20260731.json`. New campaigns
must write to this subtree, carry a campaign manifest, and bind every result to
the exact source used to produce it.

Current interpretation and execution status are recorded in
[`REVIEW_ROUND2_RESPONSE_20260731.md`](REVIEW_ROUND2_RESPONSE_20260731.md) and
[`RUN_20260731.md`](RUN_20260731.md). The fused campaigns, the 66,852-record
matmul-v4 instrument audit, the corrected same-seed robustness campaign, and the
228-cell crossed epilogue campaign (result tag `crossed_v1r1`) are complete.
Its corrective 304-cell successor has completed both measured support probes:
285 cells are supported and 19 Triton explicit-smem cells are unsupported.
Reciprocal transfer now has a one-honest-
translator 24-cell corrective design and fail-closed runner admission; the
effort frontier has a non-controlling cuBLASLt/Triton pilot protocol; and
`convergence_v2` is retired unlaunched. None contributes a result.

Design critique and the prioritized plan for what remains — including
`RUN`/`REDESIGN`/`KILL` verdicts on those three programs — are in
[`DESIGN_REFLECTION_20260731.md`](DESIGN_REFLECTION_20260731.md). Read it before
quoting any lane-level performance number from the crossed campaign: the only
strategy expressible in all four lanes shares one CUDA epilogue kernel across
them, so those contrasts do not isolate the language factor.
Second-architecture, paper-scope, and default RQ(e) retirement rules are in
[`LATER_WORK_POLICY_20260731.md`](LATER_WORK_POLICY_20260731.md).

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
