# Archived versus current fused artifacts, v1

This append-only campaign closes recommendation 6 in
`ako_runs/controlled_followup/REVIEW_20260730.md`. It remeasures the archived
`solution_opus48` and current fused artifacts for TileLang and Triton on one
physical GPU in 15 preregistered randomized complete blocks.

The artifact is the estimand. The harness imports each historical file by its
exact frozen path, instantiates its own `Model`, copies the same `x`, `weight`,
and `bias`, and times the unmodified `Model.forward(x)`. Thus, casts, cached or
native weights, fp16 or fp32 intermediates, tiles, pipelines, softmax designs,
and Triton autotuning remain exactly as the checked-in artifact implements
them. The campaign does not retrofit a common adapter that could change those
semantics.

Each subject runs in a fresh process. Within every block, all four subjects run
once in a deterministic frozen random permutation. This gives 15 paired
current/archived observations per DSL while avoiding Python module, compiler,
and autotuner state leaking between subjects. Fixed-time warmup, trial count,
L2 policy, inputs, GPU UUID, source hashes, and inference rules are frozen in
`campaign.json` and `source_receipt.json` before GPU execution.

The performance-input correctness check is intentionally labeled a diagnostic:
it prevents timing an obviously bad execution but does not claim to replace
the full four-case, 64-seed fused-v2 gate.

Typical commands:

```bash
python ako_runs/controlled_followup/archived_current_fused_v1/test_protocol.py
python ako_runs/controlled_followup/archived_current_fused_v1/freeze.py
python ako_runs/controlled_followup/archived_current_fused_v1/launch.py \
  --gpu 0 --tag main_v1
python ako_runs/controlled_followup/archived_current_fused_v1/analyze.py \
  --result ako_runs/controlled_followup/archived_current_fused_v1/results/main_v1 \
  --out-dir ako_runs/controlled_followup/archived_current_fused_v1/analysis/main_v1
python ako_runs/controlled_followup/archived_current_fused_v1/capture_evidence.py \
  build --name main_v1 \
  --include ako_runs/controlled_followup/archived_current_fused_v1/results/main_v1 \
  --include ako_runs/controlled_followup/archived_current_fused_v1/analysis/main_v1
```

