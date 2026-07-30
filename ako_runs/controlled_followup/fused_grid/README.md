# Controlled fused-GBGS equal-grid scaffold

Status: the equal-grid screen, robust correctness tuning/validation, and
five-process confirmation are complete. The execution ledger is
[`../RUN_20260730.md`](../RUN_20260730.md); the bound confirmation analysis is
[`results/fused_gbgs_confirm_robust_v1/summary.json`](results/fused_gbgs_confirm_robust_v1/summary.json).
The 1.448x result is a provisional spread between per-DSL point minima within
the frozen confirmation set, not a strict DSL order or theoretical ceiling.

This subtree schedules the missing full shared-grid experiment for
`matmul_gelu_softmax`. It applies Phase 1's exact 19 valid
`BM/BN/BK/stages/KC` points to Phase 2's full `GBGS` implementation in all four
lanes:

- TileLang
- Triton
- CUDA-WMMA (`cuda_noptx`)
- CUDA-PTX (`cuda_unlimited`)

The controlled fixed factors are `M=1024`, `K=N=8192`, 256 threads,
`KC=2048`, FP16 tensor-core arithmetic, precast activation, cached weight, and
the matched shared-memory epilogue. The row softmax remains Phase 2's fixed
second kernel. There are 19 points × 4 DSLs = 76 jobs; the ranking campaign
uses two fresh processes per point by default (152 process records).

## Manifests

`make_manifest.py` reads and audits
`phase1_matmul/jobs/native_tuned.json`. It fails unless every Phase-1 DSL has
the same ordered 19 points. It then writes a plain Phase-2-driver-compatible
job list plus a content-addressed campaign manifest.

```bash
python ako_runs/controlled_followup/fused_grid/make_manifest.py --check
python ako_runs/controlled_followup/fused_grid/launch.py --validate-only
```

Regenerate only if the Phase-1 source grid intentionally changes:

```bash
python ako_runs/controlled_followup/fused_grid/make_manifest.py
```

## Inspect without running

Both commands are CPU-only and do not run GPU preflight or create results:

```bash
python ako_runs/controlled_followup/fused_grid/launch.py --list
python ako_runs/controlled_followup/fused_grid/launch.py --dry-run
```

Use `--limit 4` to inspect only the first four entries in the deterministically
shuffled plan.

## Launch and resume

The completed ranking launch can be reproduced or content-checked with:

```bash
python ako_runs/controlled_followup/fused_grid/launch.py \
  --gpu 0 --reps 2 --tag fused_gbgs_grid_rank
```

Re-running the identical command resumes it. A record is skipped only when its
campaign, manifest, job-list, job, and repetition hashes/identifiers match.
Failed builds are retained as equal-budget outcomes; pass `--retry-failed` to
retry them explicitly. Foreign or malformed records cause an abort instead of
being silently accepted; prefer a new tag over `--force`.

Outputs live only under `fused_grid/results/<tag>/`. Each raw record contains
the source hashes, manifest/job hashes, commit, grid ID, and repetition. The
launcher delegates actual builds and measurements to Phase 2's
`driver2.run_job` and `runner2.py` interfaces.

## Structural test

```bash
python -m unittest ako_runs/controlled_followup/fused_grid/test_scaffold.py
```
