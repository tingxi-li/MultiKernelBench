# TileLang abstraction v2 (A1/A2)

Status: CPU-only design protocol. It is not frozen and cannot launch GPU or model
work under the current policy.

`protocol.py` implements three fail-closed controls:

- A1 accepts a `TL-H`/`TL-M` pair into the runtime estimand only when operator,
  family, shape, gate/input contracts, algorithm, dtype, tile, pipeline depth,
  threads, instruction family, logical and dynamic work all match. Both arms
  must be `GATE_PASSED` and bind distinct implementations plus source,
  gate-receipt, IR/PTX/SASS, work and resource receipts. A mismatch is retained
  as `capability_only`; a missing binding is invalid. The design census is ready
  for a future multi-family analysis only with at least three families, three
  shapes per family, two implementers, both implementation orders, and a
  randomization-receipt binding for every pair.
- A1 classifies the confidence interval for `log(T_low/T_high)` against the
  hardware sham floor `delta_hw` and a preregistered equivalence bound
  `epsilon >= delta_hw`. Direction and equivalence are reported as orthogonal
  decisions because both can be true for an interval between the two bounds.
- A2 requires exactly `TL-H-only` and `TL-M-only` arms with identical candidate,
  wall-clock, hardware, task, feedback, searcher, prompt, tool, isolation and
  randomization bindings. Budgets must be positive. Its content-addressed,
  gate-legal 5% terminal reference stays hidden until search completion and
  uses a dataset distinct from tuning.

This version checks required SHA-256 bindings but does not pretend that a hash
string proves an artifact exists: `claim_scope` remains
`local_only_unverified`. A frozen successor must re-read and validate source,
gate, compiler-artifact, dynamic-work, and resource receipts before promoting
the claim.

The module deliberately contains no runner. Even with all future locks present,
`authorize_launch` refuses and requires a separately frozen successor campaign.

Run the check with:

```bash
python -m unittest ako_runs.controlled_followup.tilelang_abstraction_v2.test_protocol
```
