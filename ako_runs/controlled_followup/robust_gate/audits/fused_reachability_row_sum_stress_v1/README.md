# Fused reachability row-sum stress v1

This append-only audit applies the unchanged fused-v2 mixed gates to all six
`fused_reachability_v2` screen-selected candidates on 256 fresh
`signed_normal_gain16` inputs.  Its namespace is disjoint from both the
registered 64-seed validation split and `fused_row_sum_stress_v1`.

The audit is correctness-only.  Candidate wall time is retained only as an
operational diagnostic; no performance sampling or inference is authorized.
The frozen screen selection cannot be changed using these results.  Registered
gate failures and the lower, unrounded `observed_anchor_max * safety_factor`
row-sum exceedances are reported separately for every candidate and gate.  The
latter are descriptive and never alter the registered `5e-7` threshold.

Protocol:

```bash
python -m ako_runs.controlled_followup.robust_gate.audits.fused_reachability_row_sum_stress_v1.freeze --freeze
python -m ako_runs.controlled_followup.robust_gate.audits.fused_reachability_row_sum_stress_v1.freeze --prepare-launch
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 python -m ako_runs.controlled_followup.robust_gate.audits.fused_reachability_row_sum_stress_v1.runner
python -m ako_runs.controlled_followup.robust_gate.audits.fused_reachability_row_sum_stress_v1.analyze
python -m ako_runs.controlled_followup.robust_gate.audits.fused_reachability_row_sum_stress_v1.capture_evidence build --name complete_v1
python -m ako_runs.controlled_followup.robust_gate.audits.fused_reachability_row_sum_stress_v1.capture_evidence verify --index ako_runs/controlled_followup/robust_gate/audits/fused_reachability_row_sum_stress_v1/evidence/complete_v1.index.json
```

The GPU command must run only after physical GPU 2 is idle.  The runner checks
the frozen UUID/model/compute capability and refuses a busy device.
