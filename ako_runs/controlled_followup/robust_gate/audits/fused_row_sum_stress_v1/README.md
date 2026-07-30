# Fused-v2 row-sum stress audit

This append-only campaign isolates the fused-v2 gate's thin row-normalization
margin without reopening its threshold. It binds the original fused-v2 gate and
the already content-addressed fused-grid adapter.

The deterministic boundary arm verifies the actual decision surface at N=8192:
4.5e-7 passes the unrounded safety diagnostic, 4.7e-7 occupies the intentional
1-2-5 rounding gap and still passes the registered 5e-7 threshold, and 5.1e-7
is rejected solely by `row_sum_error_max`.

The GPU arm runs the four previously selected DSL winners on 256 fresh
`signed_normal_gain16` seeds under both mixed gates. Registered gate failures
and exceedances of the descriptive unrounded `anchor_max * 1.25` value are
reported separately. Neither outcome authorizes a threshold change.

Run tests, freeze sources, and create the launch receipt before opening raw
results:

```bash
python -m unittest discover -s ako_runs/controlled_followup/robust_gate/audits/fused_row_sum_stress_v1/tests -t . -v
python -m ako_runs.controlled_followup.robust_gate.audits.fused_row_sum_stress_v1.freeze --freeze
python -m ako_runs.controlled_followup.robust_gate.audits.fused_row_sum_stress_v1.freeze --prepare-launch
```

The receipt contains the exact CPU boundary and GPU-1 winner commands. Raw rows
are fsynced into retained `.partial` files and never overwritten. After both
workloads complete, run the package's `analyze` module.
