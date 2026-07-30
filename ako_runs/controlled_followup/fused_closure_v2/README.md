# Fused closure v2

This isolated prospective campaign closes three fused-grid questions without
editing receipt-bound v1 code: a contemporaneous vendor anchor, a strict
all-four common-grid comparison, and estimator-aligned paired inference.

The authoritative design is `PREREGISTRATION.md` plus `campaign.json`.
`source_receipt.json` is deterministic and must verify before execution.

Typical validation and execution sequence:

```bash
python -m unittest ako_runs.controlled_followup.fused_closure_v2.test_closure
python ako_runs/controlled_followup/fused_closure_v2/provenance.py --check
python ako_runs/controlled_followup/fused_closure_v2/gate_adjudicate.py --cpu-smoke
python ako_runs/controlled_followup/fused_closure_v2/gate_adjudicate.py --gpu 0 --tag gate_v1
python ako_runs/controlled_followup/fused_closure_v2/launch.py --gpu 0 --tag performance_v1
python ako_runs/controlled_followup/fused_closure_v2/analyze.py --tag performance_v1
```

Execution is resumable only for missing records with identical bindings. A new
tag is required after any failure or amendment. Generated evidence lives under
`results/` and is deliberately outside the frozen source set.

