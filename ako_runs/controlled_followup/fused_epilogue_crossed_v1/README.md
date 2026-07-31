# Fused epilogue crossed v1

This prospective campaign crosses three implementation strategies with four lanes and the frozen 19-point fused grid. Its 228 requested cells remain the feasibility denominator even when a strategy cannot be expressed safely by a checked builder.

The strategies are:

- `register_fused`: bias and exact-erf GELU before the accumulator's first global store, followed by the common row-softmax algorithm.
- `smem_staged`: force the complete accumulator tile through per-block shared memory before bias/GELU, followed by the common row-softmax algorithm.
- `global_intermediate`: a lane-native GEMM writes fp32 global scratch, then one common CUDA kernel performs bias, exact-erf GELU, and row-softmax together.

There is no fallback. The pinned Triton API has no checked explicit user-managed shared accumulator buffer, and `nvcuda::wmma` does not expose a portable register-element mapping. Those 38 requested cells are preregistered `UNSUPPORTED` outcomes; substituting global scratch or inline PTX would change the requested strategy. This makes API reachability visible while limiting any conclusion to these concrete implementation paths.

## Frozen protocol

- Full fused-v2 validation precedes timing for every launchable cell: four cases, validation seeds 0--63, and both mixed gates (512 retained records/cell).
- The non-timing feasibility/build/gate audit may be sharded over GPUs 0--3.
- All timing is on physical GPU 0. The screen has two fresh processes per legal cell.
- Within every strategy x lane stratum, confirmation freezes the top two screen cells plus `g01` when legal, without duplication.
- Confirmation has 15 randomized fresh-process blocks for both the positive input (`rand`, seed 0) and a withheld signed input (`randn`, seed 2026073101), 100 event-timed trials/process after two seconds of warm-up, with L2 flushing.
- Median intervals are exact order-statistic intervals; at n=15 they are `[x4,x12]` with 0.96484375 coverage.

Feasibility is reported over all requested cells. Performance contrasts are restricted to corresponding commonly feasible/selected grids, with g01 serving as the preregistered expression-matched bridge whenever it is legal. Distribution stability reports positive/signed ratios, ranks, and winner changes.

## Lifecycle

Run all commands from the repository root. First regenerate and inspect the deterministic manifest:

```bash
python ako_runs/controlled_followup/fused_epilogue_crossed_v1/make_manifest.py
pytest -q ako_runs/controlled_followup/fused_epilogue_crossed_v1/tests
```

After all campaign sources are final, create the lock, commit it with the sources, and push that commit to the configured upstream:

```bash
python ako_runs/controlled_followup/fused_epilogue_crossed_v1/freeze.py --write
python ako_runs/controlled_followup/fused_epilogue_crossed_v1/validate.py
python ako_runs/controlled_followup/fused_epilogue_crossed_v1/validate.py --launch-ready --gpu 0
```

`--launch-ready` fails closed if a locked file changed, the campaign source/lock is uncommitted, the commit is not on its upstream, the NVIDIA driver is unavailable, the GPU is busy, or the RTX 6000 Ada identity differs from the frozen UUID/model/compute-capability set.

Launch four non-timing audit shards concurrently, using one terminal per command:

```bash
python ako_runs/controlled_followup/fused_epilogue_crossed_v1/audit.py --tag crossed_v1 --gpu 0 --shard-index 0 --shard-count 4
python ako_runs/controlled_followup/fused_epilogue_crossed_v1/audit.py --tag crossed_v1 --gpu 1 --shard-index 1 --shard-count 4
python ako_runs/controlled_followup/fused_epilogue_crossed_v1/audit.py --tag crossed_v1 --gpu 2 --shard-index 2 --shard-count 4
python ako_runs/controlled_followup/fused_epilogue_crossed_v1/audit.py --tag crossed_v1 --gpu 3 --shard-index 3 --shard-count 4
```

Then analyze, screen, freeze selection, confirm, and capture evidence:

```bash
python ako_runs/controlled_followup/fused_epilogue_crossed_v1/analyze.py audit \
  --result-root ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1 \
  --out ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1/audit_summary.json

python ako_runs/controlled_followup/fused_epilogue_crossed_v1/launch.py screen --tag crossed_v1 --gpu 0 \
  --eligibility ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1/audit_summary.json

python ako_runs/controlled_followup/fused_epilogue_crossed_v1/analyze.py screen \
  --result-root ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1 \
  --audit-summary ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1/audit_summary.json \
  --out ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1/confirmation_selection.json

python ako_runs/controlled_followup/fused_epilogue_crossed_v1/launch.py confirmation --tag crossed_v1 --gpu 0 \
  --eligibility ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1/confirmation_selection.json

python ako_runs/controlled_followup/fused_epilogue_crossed_v1/analyze.py confirmation \
  --result-root ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1 \
  --selection ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1/confirmation_selection.json \
  --out ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1/final_summary.json

python ako_runs/controlled_followup/fused_epilogue_crossed_v1/capture_evidence.py \
  --tag crossed_v1 \
  --summary ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1/final_summary.json \
  --out-prefix ako_runs/controlled_followup/provenance/evidence_v3/fused_epilogue_crossed_v1
```

Results are deliberately not gitignored. Partial files, extension caches, binaries, and active locks are excluded from controlling evidence.

