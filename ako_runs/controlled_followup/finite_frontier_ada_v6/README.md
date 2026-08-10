# Current-Ada finite-frontier terminal successor v6

This campaign answers one bounded version of “Does either DSL have a higher
performance ceiling?”:

> Under the frozen screen and positive-selection procedure, does the selected
> TileLang or Triton implementation have lower independently confirmed latency
> for one GEMM+bias+exact-GELU+row-softmax workload on physical RTX 6000 Ada
> GPU 0?

It is not an open-ended language ceiling, an average DSL ranking, or evidence
about CUDA lanes, other operators, shapes, GPUs, or input domains.

## Why v6 exists

V5 completed and sealed 7/7 performance-blind artifact admissions and all
240/240 selection records. It mechanically selected
`register_fused.tilelang.g05` and `register_fused.triton.g05`, with selection
sham floor `0.01686377744677924` log-ratio units and terminal authorization.
Terminal readiness then failed before any terminal timing record because a
deferred bare `analyze` import resolved a foreign module after `sys.path`
shadowing.

`../finite_frontier_ada_v5/INCIDENT_TERMINAL_READINESS_20260806.json` preserves
that attempt as non-controlling. It binds result commit
`d120fa42b9c6ca578f116579736db113a0678be2`, all 376 retained result files,
the 11-file frozen source closure, the exact selection lock, and the observed
7/240/0 admission/selection/terminal census.

V6 independently re-derives the exact v5 selection plan, 240 raw records,
selection statistics, sham floor, winners, and unchanged 120-record terminal
plan. The v5 selection is used only for its preregistered winner-selection
role; its latencies are not reused as terminal performance evidence. V5
executables are never reused.

## Controlled design

V6 freshly admits exactly three artifacts: both sealed winners and one shared
sham implementation. Admission is performance-blind. Each artifact is built,
correctness-gated, content-addressed, and then loaded in a separate process
through read-only native caches. Generated sources and all loadable `.so` and
`.cubin` objects are bound in the admission manifest.

Terminal confirmation preserves the v5 protocol unchanged:

- 15 randomized blocks;
- positive and withheld-signed distributions;
- one fresh process per record;
- two sham labels loading one byte-identical admitted implementation;
- 2 s warm-up, L2 flush, and 100 trials;
- trials 60–99 as the controlling settled-tail estimator;
- 120 exact records: 60 candidate and 60 sham records.

The paired TileLang/Triton interval must clear the terminal sham floor in the
same direction on both distributions. The withheld-signed distribution is a
generalization test for the positive-selected pair, not a second frontier.

All campaign imports and child commands are package-qualified. The launcher
accepts only terminal timing; there is no v6 selection command or selection
results directory. GPU identity, toolchain, upstream commit, deterministic
order, per-record process uniqueness, idle checks, correctness, two-kernel
structure, admitted cache contents, and exact input hashes are fail-closed.

## Preparation

CPU validation and lock generation do not use the GPU:

```bash
python -m unittest ako_runs.controlled_followup.finite_frontier_ada_v6.test_protocol
python -m ako_runs.controlled_followup.finite_frontier_ada_v6.analyze binding
python -m ako_runs.controlled_followup.finite_frontier_ada_v6.launch freeze \
  --authorization-basis "User authorized the main agent to implement and manage the planned Q1 experiment successor on 2026-08-06; v6 permits only fresh admission and terminal confirmation after commit and upstream readiness."
```

Commit and push the complete source, binding, contract, incident, and execution
lock before GPU work. Then run fresh admission:

```bash
python -m ako_runs.controlled_followup.finite_frontier_ada_v6.launch ready \
  --stage artifact-admission
python -m ako_runs.controlled_followup.finite_frontier_ada_v6.admit run
```

Commit and push the complete three-artifact admission closure. Only then run
terminal confirmation:

```bash
python -m ako_runs.controlled_followup.finite_frontier_ada_v6.launch ready \
  --stage terminal-confirm
python -m ako_runs.controlled_followup.finite_frontier_ada_v6.launch terminal-confirm
python -m ako_runs.controlled_followup.finite_frontier_ada_v6.analyze final
python -m ako_runs.controlled_followup.finite_frontier_ada_v6.analyze verify
```

A retained caught failure is terminal for v6 and requires another frozen
successor; it must not be deleted or overlaid. A hard crash before an atomic
record write may resume only at the next missing deterministic plan position.

## Scope left to the broader study

Additional operator families, shapes, input domains, GPU architectures,
operator-family clustered inference, and theoretical or open-ended language
ceilings remain outside this campaign.
