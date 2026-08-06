# TileLang abstraction v4: corrected fused-only successor

This campaign asks one local question: for the matched fused-softmax pair,
does TileLang's manual F4c reduction improve efficiency over the high-level F1
reduction? It retains the v3 F1/F4c shape, inputs, gate, generated-artifact
checks, profiler equality checks, timing estimator, and sham controls. Matmul
and SDPA are outside this successor's denominator.

V3 is bound as noncontrolling predecessor evidence. Its fused gate reused a
soft-only scratch value when CUDA recycled an `x_fp16` address. The corrected
implementation retains all three input tensors and invalidates scratch on
tensor identity or mutation-version changes. The v4 material validator also
re-hashes the full v3 admission artifact census, its campaign lock, summary,
and result memo.

Admission builds F1 and F4c in separate processes on physical GPU 0 and runs
the complete frozen fused-v2 validation gate: 512 records per arm, 1,024
total. Only two full gate passes with matching registered work and launch
geometry authorize timing. Generated CUDA, PTX, and SASS must reproduce each
arm's admitted identity during profiling and timing.

An eligible pair produces exactly 120 timing records: two distributions × 15
fresh-process blocks × F1, F4c, sham A, and sham B. Both hidden sham labels use
the byte-identical admitted F1 implementation. Trials 60--99 determine the
settled-tail effect; full-window and drift intervals remain diagnostic. A
direction is reportable only when its full interval clears the measured sham
resolution floor. The claim remains local to this pair and Ada GPU.

CPU validation:

```bash
python -m ako_runs.controlled_followup.tilelang_abstraction_v4.protocol check
python -m unittest \
  ako_runs.controlled_followup.tilelang_abstraction_v4.test_protocol \
  ako_runs.phase2_fused_sdpa.test_fused_tilelang_abstraction_cache
```

Commit and push the source/material closure, then freeze on idle physical GPU
0. Freezing creates a lock but does not authorize execution:

```bash
python -m ako_runs.controlled_followup.tilelang_abstraction_v4.protocol \
  freeze --gpu 0

git add ako_runs/controlled_followup/tilelang_abstraction_v4/campaign_lock.json
git commit -m "Freeze corrected fused TileLang abstraction study"
git push

python -m ako_runs.controlled_followup.tilelang_abstraction_v4.protocol \
  verify-remote \
  --lock ako_runs/controlled_followup/tilelang_abstraction_v4/campaign_lock.json
```

Run admission, project its deterministic timing manifest, and time only if the
pair is eligible:

```bash
python -m ako_runs.controlled_followup.tilelang_abstraction_v4.campaign_runner \
  admit \
  --lock ako_runs/controlled_followup/tilelang_abstraction_v4/campaign_lock.json \
  --gpu 0 --tag ada_v4r1

python -m ako_runs.controlled_followup.tilelang_abstraction_v4.protocol \
  manifest \
  --lock ako_runs/controlled_followup/tilelang_abstraction_v4/campaign_lock.json \
  --admission ako_runs/controlled_followup/tilelang_abstraction_v4/results/ada_v4r1/admission/summary.json \
  --output ako_runs/controlled_followup/tilelang_abstraction_v4/results/ada_v4r1/timing_manifest.json

python -m ako_runs.controlled_followup.tilelang_abstraction_v4.campaign_runner \
  timing \
  --lock ako_runs/controlled_followup/tilelang_abstraction_v4/campaign_lock.json \
  --admission ako_runs/controlled_followup/tilelang_abstraction_v4/results/ada_v4r1/admission/summary.json \
  --manifest ako_runs/controlled_followup/tilelang_abstraction_v4/results/ada_v4r1/timing_manifest.json \
  --gpu 0 --tag ada_v4r1
```

All writers are create-only. An interrupted stage requires a new result tag;
no retained result is resumed or overwritten.
