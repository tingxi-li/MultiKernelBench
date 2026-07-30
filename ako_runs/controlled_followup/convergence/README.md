# Replicated convergence campaign

Generate the frozen factorial with:

```bash
python make_manifest.py --campaign 20260729_controlled_followup_v1 \
  --model MODEL_REVISION_A --model MODEL_REVISION_B
```

The core manifest contains 120 trajectories: three operations by four DSLs by
two model revisions by five independent replications.  The prompt extension is
48 additional GEMM trajectories; its neutral cells reuse replications 0–2 from
the core manifest.

Generation is not execution.  `manifests/summary.json` lists the prerequisites
that must be resolved before an optimizer trajectory can be validly launched.
