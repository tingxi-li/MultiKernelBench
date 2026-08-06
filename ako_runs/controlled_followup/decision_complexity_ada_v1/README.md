# Ada decision-complexity experiment

Status: **implementation and local execution lock complete; GPU launch remains
blocked until the frozen closure is committed, pushed, and verified against
the live configured upstream**.

This campaign implements the randomized C2 part of the fourth research
question on the current Ada host. It does not revive `convergence_v2` and does
not claim that one fused kernel represents all simple or complex kernels.
It is explicitly a non-controlling decision-space pilot, not evidence that
semantically simpler kernels converge faster than semantically complex ones.

## Bounded estimand

For the frozen `register_fused.tilelang` candidate family, does a fixed uniform
searcher require more attempts or active evaluation time when it must choose
among a larger nested set of optimization decisions?

The shared target is the current-gate-legal
`register_fused.tilelang.g09`, selected before this successor by
`crossed_v2r3`. The candidate grammar is the frozen 19-grid TileLang census.
The three treatment arms retain the same target and progressively open:

- one axis, `stages`: 3 candidates;
- two axes, `stages + BM`: 6 candidates;
- four axes, `stages + BM + BK + BN`: all 19 candidates.

Common random ranks give nested arms the same relative candidate order within
a replicate. Two hidden-label arms execute one byte-identical two-axis search
contract as a negative control. Their candidate/target timing order is seeded
from that shared execution-contract hash, so the hidden label cannot change
conduct. A valid-hint sensitivity arm uses the four-axis space but places the
known target first.

Every attempted candidate is rebuilt in a fresh child process. Its current
implementation hash must equal the hash bound by the frozen `crossed_v2r3`
record; otherwise the trajectory fails closed. Prior 512-record gate evidence
may be reused only after the complete evidence closure and exact candidate
record are re-hashed. Gate, build, launch, and resource failures consume an
attempt. A timeout or hard child exit receives a parent-authored immutable
charged-failure receipt. Every attempt starts with a new empty cache root for
Torch extensions, Triton, TileLang, TVM, CUDA, Numba, TorchInductor, and XDG;
`TMPDIR` is the same attempt-local root and TileLang caching is forced off.
Gate-legal candidates and the target are timed
as a randomized paired block on the withheld-signed distribution; the terminal
event is the first candidate whose paired settled-tail median is within 5% of
the target.

The `trajectory` and `attempt` commands are internal children, not independent
launch interfaces. Each must consume a one-use canonical authorization from an
inherited pipe owned by its immediate parent. The parent retains a write-once
receipt binding the authorization hash, parent/child PIDs, timestamps, frozen
order, raw hash, and idle-GPU evidence before and after the child. Thus a saved
launch receipt or an arbitrary `--output` path cannot authorize either command.
Each four-trajectory wave also uses inherited ready/release pipes. A trajectory
validates its authorization and all four GPU locks, emits a canonical ready
receipt, and blocks. Only after all four receipts arrive does the parent issue
one common release witness; every child completion and parent receipt retains
that witness and the child-side start, release, and end timestamps.
The four live GPU-lock descriptors are inherited through the same
launcher-to-trajectory-to-attempt chain, so the locks remain held even if an
intermediate parent exits.

## Claim boundary

The first launch stage is a non-controlling four-replicate pilot, balanced one
replicate per GPU, used only to estimate trajectory-level variance and active
time. A later controlling successor must freeze its repetition count and
inference lock without reusing pilot outcomes.

The two adjacent treatment contrasts (`open_2 - open_1` and
`open_4 - open_2`) are computed within GPU/replicate for attempts and active
seconds. The pilot reports exact two-sided sign-flip tests and applies Holm
adjustment across those four predeclared tests; these remain non-controlling.
RMST is evaluated only through the smaller of the preregistered horizon and
common observed support, never by extending a surviving tail beyond the data.

Two frozen checks gate interpretation. Within every replicate the label shams
must match exactly on attempts consumed, event status, and first-event attempt;
all four valid-hint trajectories must reach the target at attempt one. The
complete ledger is still retained if either check fails, but the analysis is
marked `pilot_valid: false` and permits no treatment interpretation.

Even a completed controlling successor would identify only the causal burden
of this constructed finite search space under the fixed uniform searcher. It
would not by itself establish that semantically complex kernels, adaptive
optimizers, humans, or model-based searchers converge more slowly in general.

## Execution lifecycle

The runner re-derives the complete crossed-v2 audit and selection before
preparation. Freeze binds the runner, analyzer, tests, 24-row manifest, all 19
candidate records, the gate closure, and the current four Ada UUIDs. GPU work
remains blocked until that lock is committed at the configured upstream and an
idle four-GPU provenance receipt is captured.
The execution lock also freezes the resolved Python executable, explicit CUDA
home and `nvcc`, and the runtime/distribution versions of PyTorch, Triton, and
TileLang. Provenance and every result layer bind that same fingerprint.

```bash
python ako_runs/controlled_followup/decision_complexity_ada_v1/runner.py prepare
python ako_runs/controlled_followup/decision_complexity_ada_v1/runner.py freeze
python -m unittest \
  ako_runs.controlled_followup.decision_complexity_ada_v1.test_protocol \
  ako_runs.controlled_followup.decision_complexity_ada_v1.test_execution

# Only after commit + push:
python ako_runs/controlled_followup/decision_complexity_ada_v1/runner.py provenance
python ako_runs/controlled_followup/decision_complexity_ada_v1/runner.py ready
python ako_runs/controlled_followup/decision_complexity_ada_v1/runner.py execute
python ako_runs/controlled_followup/decision_complexity_ada_v1/analyze.py \
  --out ako_runs/controlled_followup/decision_complexity_ada_v1/results/analysis.json
```

Execution uses six randomized four-trajectory waves. Each attempt is a new
child process with attempt-local cold caches. Candidate and target are rebuilt
and hash-checked, then timed in randomized order on one withheld-signed input;
only trials 60--99 determine the terminal ratio. Every failure consumes its
position. The analysis uses only dependency-free Kaplan--Meier, RMST, and Holm
primitives from `convergence_v2/analyze.py`; it does not reuse that module's
unpaired all-arm tests, retired manifests, or launch paths.
The launcher acquires four host-wide lock files keyed by the sorted frozen GPU
UUIDs before readiness and retains all four locks through final status. Every
trajectory and attempt additionally retains idle occupancy queries immediately
before and after its child interval.
All four trajectories in a wave must retain the same parent release witness,
strictly inside every child-side lifetime. The analyzer re-derives that common
witness from all four ready receipts rather than using the parent's later
`wait()` observation as a completion time. A later wave cannot begin before
every trajectory in the prior wave completes. Execution is deliberately
non-resumable once any result evidence exists: an interruption requires a new
successor campaign/result tag and result root, never a mixture of retained and
new trajectories.
