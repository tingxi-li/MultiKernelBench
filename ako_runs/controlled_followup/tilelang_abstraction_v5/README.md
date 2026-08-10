# TileLang abstraction v5: exact-executable fused successor

This campaign preserves v4's single local question: on the matched fused
softmax pair, does TileLang's manual F4c reduction improve efficiency over the
high-level F1 reduction? The F1/F4c implementations, shape, two frozen gate
contracts, 15-block randomized timing design, settled-tail estimator, and sham
controls are unchanged. Matmul, SDPA, and cross-family claims remain outside
the denominator.

V4 is permanently noncontrolling. Both arms passed all 1,024 correctness-gate
records, but admission initialized TileLang against CUDA 13.3 before switching
the environment to the frozen CUDA 13.1 tools. Profiling then rebuilt each arm,
and byte comparison of raw PTX/SASS failed. Those text exports were themselves
fresh recompilations and were never the loaded runtime objects. The bound
incident re-hashes all 24 files in `ada_v4r1`, records zero timing rows, and
forbids reuse or a post-hoc overlay.

V5 fixes only that provenance break. Each arm is freshly compiled once into an
isolated TileLang TVM-FFI cache after CUDA 13.1 and the cache/temp roots are set,
before TileLang is imported. Admission retains and hashes the complete cache,
including generated sources, frontend metadata, and exactly two distinct
`executable.so` objects (held-fixed GEMM and treatment softmax). Correctness
gating must not mutate those bytes.

Profiling and every timing process reopen that exact cache. A load-only guard
fails on a frontend-cache miss, compiler call, cache write, foreign code object,
or before/after hash difference. Its receipt must show exactly two distinct
admitted executable loads. V5 does not export PTX/SASS, invoke the legacy
recompiling profiler target, or invoke the legacy recompiling timing CLI.
Unavoidable legacy top-level modules are imported through path checks so a
same-named module cannot silently shadow the frozen dependency.
The lock also hashes TileLang's environment and JIT adapter modules, every
executed recovery gate-summary module, and the complete NCU dispatcher,
version-wrapper, final-ELF chain, and each legacy module loaded through an
exact-path import. Profiling invokes that pinned final ELF directly.

An eligible pair still produces exactly 120 records: two distributions × 15
fresh-process blocks × F1, F4c, sham A, and sham B. Both sham labels load the
same admitted F1 cache. Trials 60--99 control the effect; full-window and drift
intervals remain diagnostic. A direction is reportable only when its complete
interval clears the sham resolution floor. The claim remains local to this
pair and Ada GPU.

CPU/static validation:

```bash
python -m ako_runs.controlled_followup.tilelang_abstraction_v5.protocol check
python -m unittest \
  ako_runs.controlled_followup.tilelang_abstraction_v5.test_protocol \
  ako_runs.phase2_fused_sdpa.test_fused_tilelang_abstraction_cache
```

After committing and pushing this source/material closure, freeze a new lock on
idle physical GPU 0, commit and push the lock, and verify it remotely:

```bash
python -m ako_runs.controlled_followup.tilelang_abstraction_v5.protocol freeze --gpu 0
git add ako_runs/controlled_followup/tilelang_abstraction_v5/campaign_lock.json
git commit -m "Freeze exact-executable TileLang abstraction study"
git push
python -m ako_runs.controlled_followup.tilelang_abstraction_v5.protocol verify-remote
```

Then run only fresh v5 evidence:

```bash
python -m ako_runs.controlled_followup.tilelang_abstraction_v5.campaign_runner \
  admit --lock ako_runs/controlled_followup/tilelang_abstraction_v5/campaign_lock.json \
  --gpu 0 --tag ada_v5r1

python -m ako_runs.controlled_followup.tilelang_abstraction_v5.protocol manifest \
  --lock ako_runs/controlled_followup/tilelang_abstraction_v5/campaign_lock.json \
  --admission ako_runs/controlled_followup/tilelang_abstraction_v5/results/ada_v5r1/admission/summary.json \
  --output ako_runs/controlled_followup/tilelang_abstraction_v5/results/ada_v5r1/timing_manifest.json

python -m ako_runs.controlled_followup.tilelang_abstraction_v5.campaign_runner \
  timing --lock ako_runs/controlled_followup/tilelang_abstraction_v5/campaign_lock.json \
  --admission ako_runs/controlled_followup/tilelang_abstraction_v5/results/ada_v5r1/admission/summary.json \
  --manifest ako_runs/controlled_followup/tilelang_abstraction_v5/results/ada_v5r1/timing_manifest.json \
  --gpu 0 --tag ada_v5r1
```

All writers are create-only. Any interrupted or failed stage requires a new
result tag; v4 files and partial v5 attempts are never resumed or overwritten.
