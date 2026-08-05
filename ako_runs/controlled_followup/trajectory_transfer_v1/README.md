# Trajectory-transfer v1: CPU-only T0/T1 protocol

Status: **design-only, deferred, and not launchable**.

This directory implements the smallest useful part of experiments T0/T1 without
modifying the sealed `reciprocal_v2` campaign. It validates a future donor-prefix
registry and deterministically derives:

1. the donor-order census
   `sum_o(L_o + 1) × 4 destinations × 2 modes × R translators`; and
2. a separate order-control census for trajectories with at least three steps and
   a dependency-valid order different from donor order.

The four destinations are `tilelang`, `triton`, `cuda_noptx`, and
`cuda_unlimited`. Both `literal` and `retuned` modes are retained, and every origin
therefore includes its DSL self-transfer as a pipeline positive control.

## Donor-prefix registry contract

The input is JSON with this shape (names are illustrative, not donor data):

```json
{
  "schema_version": 1,
  "record_type": "trajectory_transfer_donor_prefix_registry",
  "campaign_id": "a_frozen_successor_id",
  "operator": "operator_id",
  "policy_status": "deferred",
  "destinations": ["tilelang", "triton", "cuda_noptx", "cuda_unlimited"],
  "transfer_modes": ["literal", "retuned"],
  "translators": ["translator_a"],
  "order_seed": "freeze_this_seed",
  "gate_lock_sha256": "...",
  "input_contract_sha256": "...",
  "retuned_attempts_per_cell": 19,
  "retune_plan_sha256_by_origin": {"origin_id": "..."},
  "retune_attempt_plans_by_origin": {
    "origin_id": [{"attempt_index": 1, "config_sha256": "..."}]
  },
  "trajectories": [{
    "origin": "origin_id",
    "origin_dsl": "triton",
    "prefixes": [{
      "prefix_index": 0,
      "prefix_id": "origin_p0",
      "introduced_mechanisms": [],
      "depends_on_steps": [],
      "terminal_status": "GATE_PASSED",
      "source_sha256": "...",
      "gate_receipt_sha256": "...",
      "mechanism_audit_sha256": "...",
      "step_artifact_sha256": null
    }]
  }]
}
```

Prefix zero is the semantic baseline. Every donor prefix must be `GATE_PASSED`;
every later prefix must introduce exactly one named mechanism, and dependencies
may name only unique earlier step indexes. This rejects both an invalid donor and
the confounded “precision + tile + pipeline” step rather than trying to repair
either statistically.

Target outcomes retain `TRANSLATION_FAILED`, `AUDIT_FAILED`, `UNSUPPORTED`,
`BUILD_FAILED`, `LAUNCH_FAILED`, `GATE_FAILED`, or `GATE_PASSED`.
`UNSUPPORTED` requires a support-probe receipt. Only `GATE_PASSED` is timing
eligible. A literal self-transfer is a positive control only when its target
source is byte-identical to the donor source.
Literal cells receive one attempt. Retuned cells receive the frozen
origin-bound plan and `retuned_attempts_per_cell`; build failures consume that
budget and retuned results remain descriptive. Every plan must contain exactly
the frozen number of uniquely hashed, contiguous attempts and reproduce its
plan hash.

Order controls apply content-addressed, composable step artifacts from the
baseline and require a newly bound constructed source. A prefix hash is never
treated as if it were an independently composable step.

Derive a manifest on stdout:

```bash
python ako_runs/controlled_followup/trajectory_transfer_v1/protocol.py donor_prefixes.json
```

Run the CPU-only check:

```bash
python -m unittest ako_runs.controlled_followup.trajectory_transfer_v1.test_protocol
```

`--launch` always exits with status 2. Current policy explicitly defers transfer,
and this version validates hash bindings without re-reading the named material
artifacts. No build/call/gate/timing ABI is frozen here. A future launch requires
an artifact re-hash/receipt validator and a new frozen successor, not an in-place
relaxation.
