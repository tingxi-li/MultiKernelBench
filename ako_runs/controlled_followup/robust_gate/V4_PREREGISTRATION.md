# Matmul robust-gate v4 preregistration

Status: preregistered; no v4 calibration or validation output existed when this
file and `manifest_matmul_v4.json` were frozen. GPU execution is a separate
action.

## Why v3 is a failed pilot

The frozen v3 semantic-mixed threshold for `abs_signed_bias` was `0.0005`,
derived from an anchor maximum of `0.00037172289396566457`. The locked native
holdout failed only this metric in `opposing_means`, at validation seed 35
(`0.0005494917254859255`) and seed 46 (`0.0005281476162742614`). All other v3
matmul gates and metrics passed.

This is evidence that 32 input seeds per case were too sparse for a maximum-
based tolerance threshold, not permission to raise the threshold to either
observed holdout value. The two anchors share each input seed and are nearly
paired replicates (`abs_signed_bias` correlation 0.9993 in the v3
`opposing_means` calibration), so their 64 rows represent 32 independent input
draws. The relevant v3 holdout rate is 2/64 for `opposing_means`, not the pooled
2/384 across six heterogeneous cases.

V3 remains a failed pilot and none of its validation measurements may be used
to calculate a v4 threshold.

## Frozen v4 design

Relative to `manifest_matmul_v3.json`, v4 changes only campaign identity, seed
namespace, split sizes, and the validation-count wording in `success_rule`:

- campaign: `controlled-followup-robust-gate-matmul-v4`;
- seed namespace: `MKB-gate-matmul-v4-20260730`;
- calibration: 640 seeds per case;
- validation: 512 seeds per case;
- unchanged threshold rule: observed anchor maximum times 1.25, rounded upward
  to the next 1/2/5 value;
- unchanged cases, metrics, anchors, contracts, shapes, tuning count, and
  performance count.

The sample size is fixed without reference to either failed holdout magnitude.
For one matmul gate there are 6 cases x 4 calibrated metrics = 24 families.
For each case/metric/seed, treat the maximum across the two paired anchors as
one observation. Under the registered independent pseudorandom input model,
the chance that any family sample maximum misses its population 99th
percentile is bounded by

```text
24 * 0.99^640 = 0.0386132 < 0.05.
```

The unchanged safety factor and upward rounding can only enlarge that
nonparametric tolerance bound. This 95% simultaneous statement is per gate;
no pooled 72-family claim across all three gates is made.

V4 validation is quarantined until all calibration records have been collected
and `gate_spec_matmul_v4.json` has been frozen and hashed. The new namespace and
the split label in the domain-separated seed derivation make every v4
validation tensor seed distinct from v3 and from v4 calibration. No validation
output may be evaluated before the gate-spec hash is recorded. With zero
failures among 512 seeds in each of six cases, the Bonferroni-adjusted one-sided
95% upper bound on seed-level failure probability is
`1 - (0.05 / 6)^(1 / 512) = 0.009307` per case. Report validation by case; do
not substitute a pooled 0/3072 rate.

Acceptance remains fail-closed: every metric, case, and seed must pass with
complete coverage. If locked v4 validation fails, retain v4 as another failed
experiment. Any later rule change requires a new campaign ID, namespace, and
untouched validation split.

## Preserved v3 evidence

These are raw-file SHA256 values captured before v4 execution:

| Evidence | SHA256 |
| --- | --- |
| `manifest_matmul_v3.json` | `16d4a38e6a55cd70b605cada88bc88caf4ad89a5609b50c85cb834cebeaa706d` |
| `gate_spec.json` | `48b987bd052b81c20d9b47cdba8b4693c2e1130ae46d4d5ab5147212b1b1ce34` |
| `calibration/raw/matmul_semantic_mixed_gpu1_v3.jsonl` | `867adea0f97d222beada435b5ee395af718222c7b472d74f3f682f93ec8c702b` |
| `calibration/raw/matmul_conformance_mixed_gpu2_v3.jsonl` | `894887046f4927adf881b4ec932ccd60c48e4f3a0ab1b3bd65c7eef8966e1299` |
| `calibration/raw/matmul_semantic_q32_gpu3_v3.jsonl` | `f55569f22f69dbe9292e5b27b205c246b210f626e6d623abed8ac9dbacec9dd0` |
| `validation/raw/matmul_semantic_native_gpu1_v3.jsonl` | `9a4e816dd7ee7bd6165a486316e5d698797042094e186bc247d456a63d426c32` |
| `validation/raw/matmul_conformance_native_gpu2_v3.jsonl` | `b0bbc6c6943ea5b0971003b62622b509ebfe498138b7ffe2d81914ec55dc4925` |
| `validation/raw/matmul_semantic_q32_native_gpu3_v3.jsonl` | `935fd492c0d47f6e7aa95ee941096584beb7766bec5ae5d9b050a8f3e250ec44` |
| `validation/matmul_holdout_summary_v3.json` | `492064ca15ad96c6cdc58657fad5180a243723727b1d0899d0101a4baa7d6c65` |

The canonical JSON hashes are
`8f2329c14b152a71520f9ff28bd3787092154081ebcf7fb97ef571f3bea316b2`
for the v3 manifest and
`bf7516e5da72544a65f4547d5841a0547d8900374fd719980b520b95f7f94c1a`
for the v3 gate spec. Keep all v3 files byte-for-byte unchanged and write v4
outputs only to names ending in `_v4`.
