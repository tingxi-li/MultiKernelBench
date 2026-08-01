# Fused-v2 gate saturation report

> Date: 2026-07-31
>
> Action: notify the owner of `calibration/gate_spec_fused_v2.json`. This is a
> diagnostic report; it does not authorize or make a threshold change.

All 150 `GATE_PASSED` cells in crossed epilogue result tag `crossed_v1r1`
bind on `row_sum_error_max` under the mixed gate views. The frozen threshold is
`5e-07`. Only three maximum-over-threshold ratios occur:

| Ratio | Observed maximum | Cells | Headroom to `5e-07` |
|---:|---:|---:|---:|
| `0.9633653546003984` | `4.8168267730019920e-07` | 105 | 3.6634645399601595% |
| `0.9595156558184215` | `4.7975782790921075e-07` | 30 | 4.0484344181578535% |
| `0.9512837921832329` | `4.7564189609161645e-07` | 15 | 4.8716207816767110% |

Thus every pass lies only 3.7--4.9% below the frozen threshold. The gate spec
records `observed_anchor_max = 3.630482550143199e-07` and safety factor `1.25`.
Their unrounded product is `4.53810318767899875e-07`; every one of the 150
observed maxima exceeds it (by 4.8107--6.1419%). Without the registered upward
`next_1_2_5` rounding to `5e-07`, all 150 cells would fail and no cell would be
timing-eligible.

This is a sensitivity finding, not evidence of post-hoc tuning: the fused-v2
spec predates crossed v1r1 and the campaign applied it unchanged. Preserve the
frozen threshold and report this saturation whenever crossed-v1r1 feasibility
or timing eligibility is cited. Any threshold-policy experiment requires a new
gate version, seed namespace, validation split, and campaign binding.

## Evidence bindings

| Artifact | SHA-256 |
|---|---|
| `calibration/gate_spec_fused_v2.json` | `b6c3afaadf61f5d380a97c6c541828a1ecacd1a89a00b73b66207fe22d765e2e` |
| `validation/fused_gate_acceptance_v2.json` | `4907fddefa2cac3f7215033cb93caaaa550cecd4a6c70971f29c5f331cb0485c` |
| `../fused_epilogue_crossed_v1/results/crossed_v1r1/audit_summary.json` | `229b8a34c7a0f6be24668a14a60cfe4606e523f6adc3138c1b60ea7b095ad74f` |
