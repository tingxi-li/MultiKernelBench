# Robust correctness-gate campaign

This subtree is an isolated follow-up to the completed Phase 1 and Phase 2
campaigns. It does not edit or import their result files, and importing this
package does not initialize CUDA or compile a kernel.

The campaign separates two questions:

1. `semantic_q32` / `semantic_mixed`: is the output close to fp64 evaluation of
   the original fp32 inputs at the named quality tier?
2. `conformance_mixed`: does the implementation realize its declared fp16/fp32
   rounding boundaries?

The legacy `1e-4 + 1e-4*abs(ref)` predicate is not used to freeze these gates.
It is inadequate in opposite directions for the studied operations: it is
distribution-dependent for matmul, weak at softmax probability scale, and can
reject correctly rounded fp16 SDPA storage.

## Execution status

The fused-v2 gate and the powered matmul-v4 gate are complete. Matmul v3 is a
preserved failed pilot: its semantic-mixed gate failed 2/64 seeds in the
`opposing_means` case. V4 used a fresh namespace and did not tune from those
holdout magnitudes; it completed 23,040 calibration records and passed all
9,216 locked validation records. See [`V4_PREREGISTRATION.md`](V4_PREREGISTRATION.md),
[`validation/matmul_holdout_summary_v4.json`](validation/matmul_holdout_summary_v4.json),
and [`validation/v4_acceptance_receipt.json`](validation/v4_acceptance_receipt.json).

The post-campaign
[`FUSED_V2_GATE_SATURATION_20260731.md`](FUSED_V2_GATE_SATURATION_20260731.md)
reports that all 150 crossed-v1r1 passes sit within 3.7--4.9% of the frozen
fused-v2 row-sum threshold and would fail at its unrounded calibrated value.
It is a sensitivity report only; no frozen threshold was changed.

## Frozen protocols

- Fused v2 uses 32 domain-separated calibration seeds and 64 locked validation
  seeds per registered distribution. Matmul v4 uses 640 calibration seeds per
  anchor/case and 512 locked validation seeds per case.
- Tuning/smoke: 8 separate seeds.
- Performance inputs: 3 separate seeds; performance statistics are outside
  this package.
- Each gate uses at least two named anchors. Candidate output is never used to
  set a threshold.
- A calibrated threshold is
  `next_1_2_5(1.25 * maximum_anchor_metric)` unless a larger preregistered
  minimum is present.
- Fixed structural thresholds, including non-finite and negative-probability
  counts, are zero.
- A candidate passes only if every metric passes on every registered case and
  every required validation seed. Missing records, duplicate records, wrong seeds,
  collection errors, shape mismatches, empty tensors, non-floating tensors,
  NaN, and Inf fail closed.
- For fused v2, zero failures among 64 independent validation seeds gives a
  one-sided 95% upper bound of about 4.6% per distribution. For matmul v4,
  zero failures among 512 seeds in each of six cases gives the preregistered
  Bonferroni-adjusted one-sided 95% upper bound of 0.009307 per case. The v4
  result is reported case-by-case, not as pooled 0/3,072 evidence. Tensor
  elements are not treated as independent observations.

The operation-specific gate metrics are:

| Operation | Calibrated metrics | Fixed checks |
|---|---|---|
| Matmul | maximum absolute error, product-scale RMS error `||E||F / ||abs(A)@abs(B)||F`, absolute signed bias, componentwise backward error | no non-finite values |
| Fused exact-erf GELU-softmax | maximum absolute error, NRMSE, row-sum error, row total variation, square-root Jensen-Shannon distance, probability-scaled maximum error | no non-finite or negative probabilities |
| SDPA | maximum absolute error, NRMSE, per-query relative L2, cosine distance, per-query scaled maximum error | no non-finite values |

The frozen fused-v2 distribution matrix and contracts remain in
[`manifest.json`](manifest.json). Matmul v3, which replaces undefined
zero-reference NRMSE with product-scale RMS error, is in
[`manifest_matmul_v3.json`](manifest_matmul_v3.json). Its interchange contract and the
generated gate contract are in [`schemas/manifest.schema.json`](schemas/manifest.schema.json)
and [`schemas/gate_spec.schema.json`](schemas/gate_spec.schema.json).
The v3 locked holdout failed and is retained as pilot evidence. The fresh,
larger-sample recovery experiment was preregistered in
[`V4_PREREGISTRATION.md`](V4_PREREGISTRATION.md) and
[`manifest_matmul_v4.json`](manifest_matmul_v4.json), then completed under its
frozen rule; v3 examples below remain plumbing examples, not authorization to
reuse its holdout.

## CPU tests

From the repository root:

```bash
python -m unittest discover \
  -s ako_runs/controlled_followup/robust_gate/tests \
  -p 'test_*.py' -v
```

The tests cover seed golden vectors and domain separation, strict JSON,
distribution invariants, fp16 rounding boundaries, hand-computed matmul and
SDPA oracles, SDPA metamorphic invariants, the softmax total-variation
pathology, fail-closed metric preflight, calibration coverage, and both passing
and deliberately failing validation candidates.

## Resolve and pin seeds

```bash
python -m ako_runs.controlled_followup.robust_gate.seeds \
  --manifest ako_runs/controlled_followup/robust_gate/manifest.json \
  --out /tmp/robust_gate_seeds.json
```

Every seed is derived from the NUL-separated tuple
`(namespace, op, case, split, tensor, index)` with SHA256. Tensor streams are
independent, so changing construction order cannot perturb existing operands.

## Short CPU matmul smoke test

The following intentionally uses only two seeds and `--allow-incomplete`. It
tests plumbing; its gate spec is **not eligible for a campaign freeze**.

```bash
python -m ako_runs.controlled_followup.robust_gate.collect \
  --manifest ako_runs/controlled_followup/robust_gate/manifest_matmul_v3.json \
  --op matmul --gate semantic_q32 --split calibration \
  --candidate anchor --cpu-smoke --max-seeds 2 \
  --out /tmp/robust_gate_calibration.jsonl

python -m ako_runs.controlled_followup.robust_gate.calibrate \
  --manifest ako_runs/controlled_followup/robust_gate/manifest_matmul_v3.json \
  --records /tmp/robust_gate_calibration.jsonl \
  --op matmul --gate semantic_q32 --allow-incomplete \
  --out /tmp/robust_gate_spec.json

python -m ako_runs.controlled_followup.robust_gate.collect \
  --manifest ako_runs/controlled_followup/robust_gate/manifest_matmul_v3.json \
  --op matmul --gate semantic_q32 --split validation \
  --candidate exact --candidate-name cpu-exact \
  --cpu-smoke --max-seeds 2 \
  --out /tmp/robust_gate_validation.jsonl

python -m ako_runs.controlled_followup.robust_gate.validate \
  --manifest ako_runs/controlled_followup/robust_gate/manifest_matmul_v3.json \
  --gate-spec /tmp/robust_gate_spec.json \
  --records /tmp/robust_gate_validation.jsonl \
  --allow-incomplete --out /tmp/robust_gate_summary.json
```

## Legacy v3 calibration example

This illustrates the old command shape only and writes outside the evidence
tree so it cannot overwrite the preserved failed pilot. Omit `--cpu-smoke`,
`--max-seeds`, and `--allow-incomplete`. A full matmul calibration against the
manifest shape starts with:

```bash
python -m ako_runs.controlled_followup.robust_gate.collect \
  --manifest ako_runs/controlled_followup/robust_gate/manifest_matmul_v3.json \
  --op matmul --gate semantic_q32 --split calibration \
  --candidate anchor --device cuda \
  --out /tmp/matmul_semantic_q32_v3_example.jsonl

python -m ako_runs.controlled_followup.robust_gate.calibrate \
  --manifest ako_runs/controlled_followup/robust_gate/manifest_matmul_v3.json \
  --records /tmp/matmul_semantic_q32_v3_example.jsonl \
  --op matmul --gate semantic_q32 \
  --out /tmp/gate_spec_v3_example.json
```

Long collections may be sharded with `--seed-start`, `--max-seeds`, and
`--cases`. Pass `--records` repeatedly to calibration or validation to consume
multiple JSONL shards. Strict calibration verifies every expected
`anchor × case × seed` tuple before emitting a gate.

The retained `gate_spec.json` is the preserved failed-v3 artifact; do not
overwrite or promote it. The accepted v4 gate is
`calibration/gate_spec_matmul_v4.json`. Its reciprocal launch lock binds the
campaign, manifest, calibration evidence, exact gate bytes, successful
validation summary, acceptance receipt, and the raw holdout hashes named by
that receipt.

## Candidate adapters

Built-in candidates (`exact`, `zeros`, `row_reverse`, `native_fp32`, and
`native_mixed`) exist for testing. GPU kernels are loaded only when explicitly
requested:

```bash
python -m ako_runs.controlled_followup.robust_gate.collect \
  --manifest ako_runs/controlled_followup/robust_gate/manifest_matmul_v3.json \
  --op matmul --gate semantic_q32 --split validation \
  --candidate callable --candidate-name tilelang-D-robust \
  --callable package.module:function --device cuda \
  --out candidate.jsonl
```

The callable receives keyword arguments `op`, `inputs`, `case`, and `contract`
and returns one tensor. It must not mutate the supplied inputs. The collector
records its source SHA256, exact seed map, manifest hash, shape, contract,
metrics, output dtype, and collection failures.
