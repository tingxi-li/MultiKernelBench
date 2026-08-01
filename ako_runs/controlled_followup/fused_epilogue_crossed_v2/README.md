# Fused epilogue crossed v2

This hash-bound successor corrects the crossed-v1 feasibility instrument without
editing any sealed v1 file. It requests four strategies × four lanes × nineteen
grid points (304 cells), binds the v1r1 final summary and settled-tail overlay,
and reuses the frozen robust gate and exact statistics.

The added `register_common_postprocess` strategy runs lane-native `GBG` (bias
and exact GELU in the register epilogue) followed by the same checked CUDA
softmax body used by `global_intermediate`, compiled with bias and GELU disabled.
Here “register epilogue” is a source-level placement contract: there is no
explicit fp32 epilogue tile. Per-kernel ptxas register and spill diagnostics are
retained and may not be interpreted as a zero-spill guarantee.
It is called four-lane-common only if the measured CUDA-no-PTX register probe
passes. The two support probes retain source, diagnostics, gate rows, and one
terminal receipt for each of all 19 grids.

Current state: both measured support grids are complete. CUDA-no-PTX register
support resolved supported from 19 gate-passing attempts; Triton explicit-smem
resolved unsupported from 19 identical public-API limitation receipts. The final
campaign lock binds their full evidence closure. Audit launch still requires that
lock's commit on the configured upstream plus a fresh live GPU preflight;
historical availability is never accepted.

The first `crossed_v2` audit launch at commit `416e8c4` stopped before retaining
any cell outcome because its diagnostic margin reducer divided exact-zero gate
thresholds. Its four launch receipts and the adjacent incident receipt are
preserved and bound into the corrective lock. The tested v1r1 reporting helper is
now reused unchanged; it reports exact-zero maxima and violations without using
them as denominators. The corrective result tag is `crossed_v2r1`.

## CPU validation and probe preregistration

```bash
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/make_manifest.py
pytest -q ako_runs/controlled_followup/fused_epilogue_crossed_v2/tests
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/validate.py
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/freeze.py probes --write
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/freeze.py probes
```

Commit and push the complete probe lock before any probe GPU process. Verify the
commit on the configured upstream independently of GPU availability:

```bash
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/validate.py \
  --stage probes --remote-ready
```

When the driver is visible, the same command with `--launch-ready --gpu N`
adds the live identity and occupancy checks. Then run both immutable grids:

```bash
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/support_probes.py \
  --probe-key cuda_noptx_register --tag crossed_v2 --gpu 0
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/support_probes.py \
  --probe-key triton_smem --tag crossed_v2 --gpu 0
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/resolve_support.py \
  --cuda-noptx-index ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2/support_probes/cuda_noptx_register/index.json \
  --triton-smem-index ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2/support_probes/triton_smem/index.json \
  --write
```

`resolve_support.py` re-derives each verdict from its 19 receipt hashes. Missing,
mixed, changed, or API-present-but-unreviewed evidence stays unresolved.

## Final freeze and campaign

After resolution, create the campaign lock, commit the lock plus the entire
probe-evidence closure, push, and remotely verify again before the audit:

```bash
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/freeze.py campaign --write
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/freeze.py campaign
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/validate.py \
  --stage campaign --remote-ready
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/validate.py \
  --stage campaign --launch-ready --gpu 0
```

Audit shards may use the four frozen GPUs; timing is fixed to physical GPU 0:

```bash
for shard in 0 1 2 3; do
  python ako_runs/controlled_followup/fused_epilogue_crossed_v2/campaign_runner.py audit \
    --tag crossed_v2r1 --gpu "$shard" --shard-index "$shard" --shard-count 4
done
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/analyze.py audit \
  --result-root ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1 \
  --out ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1/audit_summary.json
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/campaign_runner.py screen \
  --tag crossed_v2r1 --gpu 0 \
  --eligibility ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1/audit_summary.json
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/analyze.py screen \
  --result-root ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1 \
  --audit-summary ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1/audit_summary.json \
  --out ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1/confirmation_selection.json
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/campaign_runner.py confirmation \
  --tag crossed_v2r1 --gpu 0 \
  --eligibility ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1/confirmation_selection.json
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/analyze.py confirmation \
  --result-root ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1 \
  --selection ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1/confirmation_selection.json \
  --out ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1/final_summary.json
```

Trials 60–99 are controlling. Full-window and first/last-decile drift summaries
are diagnostics. The paired duplicate-label control at g01 defines the largest
absolute log-CI endpoint as the publication floor; an effect is reported only
when its whole interval lies beyond that floor.

Build and verify complete evidence only after the final summary is complete:

```bash
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/capture_evidence.py build \
  --tag crossed_v2r1 \
  --summary ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2r1/final_summary.json \
  --out-prefix ako_runs/controlled_followup/fused_epilogue_crossed_v2/evidence/crossed_v2r1_complete_v1
python ako_runs/controlled_followup/fused_epilogue_crossed_v2/capture_evidence.py verify \
  --index ako_runs/controlled_followup/fused_epilogue_crossed_v2/evidence/crossed_v2r1_complete_v1.index.json
```
