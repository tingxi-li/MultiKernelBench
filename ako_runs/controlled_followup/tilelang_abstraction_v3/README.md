# TileLang abstraction v3 (Ada successor)

Status: implemented, **not frozen and not launched**. This is a new successor;
it does not modify the launch policy or evidence semantics of
`tilelang_abstraction_v2`, and historical Phase-1/2 timing is never reused as a
controlling result.

The fixed denominator contains one existing H/M pair for each of matmul, fused
softmax, and SDPA. Matmul is admitted only through matmul-v4 and fused softmax
only through fused-v2. SDPA has no accepted current robust gate, so its retained
row stops as `CURRENT_GATE_UNAVAILABLE` before build, profiling, or timing. A
future SDPA gate requires a new registry/lock; it cannot be supplied as an
overlay to a frozen run.

Every executable arm re-hashes the material closure, binds the live RTX 6000
Ada UUID/toolchain to the campaign lock, builds in a fresh process, runs the
complete current validation split, and captures generated CUDA, PTX, and SASS.
The toolchain binding includes the resolved Python executable and hash,
implementation/version, PyTorch and its CUDA version, Triton, TileLang, CUDA
tool binaries/version output, and the imported campaign, builder, registry,
runner, profiler, and common-module files. Stage parents and every executable
child independently rederive it, while every launch and child receipt retains
the exact lock value for analyzer comparison.
Profiling and timing rebuild independently and must reproduce that same-arm
artifact identity; missing capture or any hash drift fails closed. The analyzer
re-hashes every gate JSONL row, re-derives its coordinate census and thresholds,
and validates selected NCU work counters and registered launch geometry before
admitting a pair. Algorithm, dtype, tile, pipeline, instruction-family, and
logical-work descriptions remain visibly labeled preregistered design
assumptions; they are not misreported as empirical equality checks.

Timing is a deterministic admission-bound manifest. Each pair/distribution has
15 fresh-process blocks containing randomized H, M, sham-A, and sham-B rows.
The two shams must reproduce the admitted H CUDA/PTX/SASS identity under
different hidden labels. Trials 60--99 are controlling; all recorded medians
are re-derived from retained trials, while full-window and drift intervals are
diagnostics. Results remain local to admitted pairs because this material
set does not meet the three-families × three-shapes × two-implementers
generalization design.

Both GPU stages hold the same host-global, UUID-keyed physical-GPU0 lock used
by the other current Ada campaigns. Every GPU child must inherit that lock and
retains idle pre/post receipts. Timing additionally retains one immutable
position receipt per manifest row, binding the unique child PID, launch and
completion timestamps, raw hash, and predecessor-receipt hash. The analyzer
requires the exact randomized sequence and exact admission/timing file census;
extra files, missing files, overlap, or reordered direct child calls fail
closed.

CPU-only validation:

```bash
python -m ako_runs.controlled_followup.tilelang_abstraction_v3.protocol check
python -m unittest ako_runs.controlled_followup.tilelang_abstraction_v3.test_protocol
```

After committing and pushing the exact source/material closure, freeze it on
idle physical GPU 0. The cleanliness check is scoped to frozen inputs, so
explicitly non-input result roots from other campaigns do not block it:

```bash
python -m ako_runs.controlled_followup.tilelang_abstraction_v3.protocol \
  freeze --gpu 0
```

Freezing creates the lock but does **not** authorize GPU execution. Commit and
push `campaign_lock.json`, then perform the second-step live verification:

```bash
git add ako_runs/controlled_followup/tilelang_abstraction_v3/campaign_lock.json
git commit -m "Freeze TileLang abstraction v3"
git push
python -m ako_runs.controlled_followup.tilelang_abstraction_v3.protocol \
  verify-remote --lock ako_runs/controlled_followup/tilelang_abstraction_v3/campaign_lock.json
```

Every GPU child repeats this check and refuses an untracked lock, dirty frozen
input, or HEAD that differs from both the cached and live configured upstream.

Then execute the staged workflow. Manifest creation and every timing child
independently re-derive admission from the retained gate/profile evidence; a
caller-edited summary cannot authorize a timing row:

```bash
python -m ako_runs.controlled_followup.tilelang_abstraction_v3.campaign_runner \
  admit --lock ako_runs/controlled_followup/tilelang_abstraction_v3/campaign_lock.json \
  --gpu 0 --tag ada_v3r1

python -m ako_runs.controlled_followup.tilelang_abstraction_v3.protocol \
  manifest \
  --lock ako_runs/controlled_followup/tilelang_abstraction_v3/campaign_lock.json \
  --admission ako_runs/controlled_followup/tilelang_abstraction_v3/results/ada_v3r1/admission/summary.json \
  --output ako_runs/controlled_followup/tilelang_abstraction_v3/results/ada_v3r1/timing_manifest.json

python -m ako_runs.controlled_followup.tilelang_abstraction_v3.campaign_runner \
  timing --lock ako_runs/controlled_followup/tilelang_abstraction_v3/campaign_lock.json \
  --admission ako_runs/controlled_followup/tilelang_abstraction_v3/results/ada_v3r1/admission/summary.json \
  --manifest ako_runs/controlled_followup/tilelang_abstraction_v3/results/ada_v3r1/timing_manifest.json \
  --gpu 0 --tag ada_v3r1
```

All result writers are create-only. Gate/profile failures remain explicit
admission outcomes, but an interrupted/partial stage or failed timing child is
terminal, is never resumed, and requires a wholly new result tag (including a
fresh admission) rather than being overwritten.
