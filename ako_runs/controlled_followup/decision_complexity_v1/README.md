# Decision-complexity v1

CPU-only, design-only protocol for experiments C1/C2 in
`GPU_DSL_CLAIM_EXPERIMENT_DESIGN_20260805_ZH.md`.

It derives seven rows per `(task, searcher, replicate)`: one preregistered
observational-complexity trajectory, three randomized open-axis arms, two
byte-contract-matched label shams, and one valid-hint sensitivity control.
Complete-block order and GPU slots are mechanically randomized and balanced.
The exact manifest projection is validated byte-for-byte; axis domains,
dependency graphs and one shared gate-legal target are bound into each search
space. Tuning and terminal datasets must be distinct.

Analysis accepts only a complete, content-addressed ordered attempt ledger for
every trajectory. It derives the first terminal-legal candidate within 5%,
charges failed attempts, and rejects early censoring. C1, C2 and controls are
reported separately. Because no statistical successor lock or material
execution exists, every analysis is explicitly `design_only_noncontrolling`.
The outcome payload digest is a deterministic integrity check, not an external
terminal-evaluator receipt; a frozen successor must bind and verify that
independent receipt before any result can control a claim.

The pure survival-analysis primitives are reused from the preserved
`convergence_v2` code; this does not revive that retired campaign. This new
protocol always refuses launch. A future launch requires a new authorization,
material task/searcher bindings, power-derived replicates, GPU identities,
remote preregistration, and a frozen successor lock.

CPU check:

```bash
python -m unittest ako_runs.controlled_followup.decision_complexity_v1.test_protocol
```
