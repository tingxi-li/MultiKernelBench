# Current-Ada two-stage finite-candidate procedure v1

This successor tests one bounded, two-stage candidate-selection procedure for
“Does any DSL have a higher performance ceiling?”:

> After the frozen screen and positive-input selection stages choose one
> candidate per DSL, does the selected TileLang or Triton candidate have lower
> independently confirmed latency on physical RTX 6000 Ada GPU 0?

It is not an exhaustive estimate of the minimum over the registered space and
not a theoretical language ceiling. CUDA lanes are outside the estimand. The
withheld-signed result is a generalization test for the positive-selected pair,
not a signed-distribution frontier.

## Design

The existing `crossed_v2r3` instrument is reused as immutable training
evidence. `import` independently re-derives its 152 requested TileLang/Triton
audit cells and 226 two-process screen records. The screen mechanically admits
the fastest three candidates per DSL to a fresh 15-block selection-confirm
stage. Selection-confirm locks one winner per DSL using only the positive input
distribution. A separate fresh 15-block terminal-confirm stage estimates the
paired TileLang/Triton ratio on both frozen distributions.

Every timing record is a fresh process on the same physical GPU. Each process
uses 2 s warm-up, L2 flushes, 100 trials, and trials 60–99 as the primary
window. Two sham labels bind the same verified cell/config/source fingerprint;
emitted binary bytes are not claimed identical. A selected candidate is called
locally faster only if its entire terminal paired interval clears the terminal
sham floor in the same direction on the positive distribution and its
withheld-signed generalization test.

A host-wide advisory lock keyed to the physical GPU UUID excludes cooperating
GPU 0 launchers, including another checkout, for the whole stage. Every record retains idle-except-self GPU checks
before and after work. Plan positions are frozen and record timestamps must
follow their deterministic randomized order. Each successful record must retain
exactly two kernels and binds the immutable stage-launch receipt; a child may
write only the next missing position after that receipt and its idle preflight.

## CPU validation and freeze

```bash
python -m unittest discover \
  -s ako_runs/controlled_followup/finite_frontier_ada_v1 -p 'test_*.py'
python ako_runs/controlled_followup/finite_frontier_ada_v1/analyze.py import
python ako_runs/controlled_followup/finite_frontier_ada_v1/launch.py freeze \
  --authorization-basis "User request 2026-08-05: implement and run controlled experiments for the four GPU DSL research questions on the currently available GPUs"
```

Commit and push this directory, including `execution_lock.json`, before GPU
work. Then execute selection:

```bash
python ako_runs/controlled_followup/finite_frontier_ada_v1/launch.py ready \
  --stage selection-confirm
python ako_runs/controlled_followup/finite_frontier_ada_v1/launch.py selection-confirm
python ako_runs/controlled_followup/finite_frontier_ada_v1/analyze.py select
```

Before terminal timing, seal the adaptive input and verify it live upstream:

```bash
git add ako_runs/controlled_followup/finite_frontier_ada_v1/results/selection_confirm \
  ako_runs/controlled_followup/finite_frontier_ada_v1/results/selection_lock.json
git commit -m "Seal finite-frontier Ada selection"
git push
python ako_runs/controlled_followup/finite_frontier_ada_v1/launch.py ready \
  --stage terminal-confirm
python ako_runs/controlled_followup/finite_frontier_ada_v1/launch.py terminal-confirm
python ako_runs/controlled_followup/finite_frontier_ada_v1/analyze.py final
python ako_runs/controlled_followup/finite_frontier_ada_v1/analyze.py verify
```

Terminal readiness requires every selection raw record, its receipt and status,
and `selection_lock.json` to be clean, tracked, and at the live configured
upstream head. Launch or analysis fails on source/material drift,
GPU/toolchain mismatch, a held GPU lock, non-idle pre/post checks, record
collision, any missing or extra raw file, plan/timestamp disorder, inconsistent
implementation fingerprints, or a selection sham floor above 5% in log-ratio
units.

A hard crash before the atomic record write can resume at the missing plan
position. A caught failure is retained at its canonical path and is terminal
for this campaign; retry requires a newly frozen successor tag, not deletion or
an in-place rerun.

## Scope intentionally left to broader F1

This successor does not add operators, shapes, input domains, GPU
architectures, or operator-family sampling. It cannot support an average DSL
ranking, a cross-architecture claim, a true minimum over every registered
candidate, or a theoretical/open-ended performance ceiling. Those require the
broader multi-family F1 corpus and clustered inference described in
`../GPU_DSL_CLAIM_EXPERIMENT_DESIGN_20260805_ZH.md`.
