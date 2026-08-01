# Fused same-seed robustness stress v1

This correctness-only follow-up evaluates every frozen v3 frontier candidate
on the same 512 seeded inputs. Indices `0..255` reuse the prior stress seed
namespace (including the known seeds `186` and `197`); indices `256..511` use a
new disjoint namespace. No performance selection, threshold fitting, or gate
mutation is authorized.

Run on an idle RTX 6000 Ada (default physical GPU 2):

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 \
  python -m ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v1.run
```

The output is an append-only JSONL stream. Use `analyze.py` afterward to
report per-seed paired outcomes, effective sample size, and metric/threshold
ratios.
