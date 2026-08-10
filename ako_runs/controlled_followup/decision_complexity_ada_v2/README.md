# Ada decision-complexity experiment v2

Status: **CPU implementation prepared for one-GPU serialization; GPU launch
remains blocked until the closure is frozen, committed, pushed, and verified
against the configured upstream**.

This is the one-GPU successor to the sealed, unexecuted
`decision_complexity_ada_v1`. It keeps the same bounded Q4 pilot: six arms,
four replicate trajectories, the 19-candidate
`register_fused.tilelang` family, fixed uniform search, correctness gate,
attempt timeout, terminal rule, and non-controlling analysis. It does not
revive `convergence_v2` or claim that one fused kernel represents all simple
or complex kernels.

The successor also retains the valid v1 prelaunch provenance receipt at
SHA-256 `872abbd3bbff7f50b888b7678351af9bdfd824e4886593d6979d1d658cbfc150`.
The v2 lock re-hashes that receipt, the v1 lock, and the complete source bundle
named by the v1 lock, while requiring that v1 still has no result evidence.
This preserves the predecessor's sealed prelaunch state; it does not authorize
v1 reuse or GPU execution.

## Bounded estimand

For the frozen candidate family, does the fixed uniform searcher require more
attempts or active evaluation time when it must choose among a larger nested
set of optimization decisions?

The common target is `register_fused.tilelang.g09`. The three treatment arms
open one axis (`stages`, 3 candidates), two axes (`stages + BM`, 6 candidates),
or four axes (`stages + BM + BK + BN`, 19 candidates). Two hidden-label arms
execute the byte-identical two-axis search as a negative control. A valid-hint
arm puts the target first. The v1 arm order, common candidate ranks, and
candidate/target timing order are preserved exactly for every replicate.

Every attempt runs in a fresh child and empty cache root. The implementation
hash and full normalized artifact digest must match the frozen `crossed_v2r3`
record. Correctness, build, launch, resource, timeout, and hard-child failures
all consume their frozen position.
Passing candidate and target implementations are timed as a randomized paired
block on the withheld-signed distribution; only trials 60--99 determine
whether the candidate is within 5% of the target.

## One-GPU execution control

All 24 manifest rows bind physical GPU slot 0 and UUID
`GPU-45af34ad-0c74-74d0-ef3a-652090d837ae`. The launcher takes that UUID's
host-wide lock once and retains it through final status. It executes rows one
at a time in explicit manifest order; there is no ready/release wave barrier.

Each trajectory authorization and parent receipt binds the exact previous
trajectory completion path, SHA-256, trajectory ID, and child-side end
timestamp. The first row binds `null`. Validation requires every later child
launch timestamp to be strictly greater than its bound predecessor's end
timestamp. Parent and child idle-GPU checks bracket every trajectory, and the
same inherited UUID lock continues through each attempt child.

Execution remains non-resumable after any result evidence. An interruption or
caught failure requires another successor rather than mixing retained and new
trajectories.

## Analysis boundary

The pilot retains four replicates per arm. It reports the two adjacent paired
contrasts (`open_2 - open_1` and `open_4 - open_2`) for attempts and active
seconds, exact sign-flip tests, and Holm adjustment across the four planned
tests. RMST stops at the smaller of the preregistered horizon and common
observed support.

Interpretation requires exact sham agreement on attempts, event status, and
first-event attempt within each replicate, plus a first-attempt event in all
four valid-hint trajectories. Failure retains the ledger but marks the pilot
invalid. Even a valid result concerns this finite search space and fixed
uniform searcher only; it does not establish a general relationship between
semantic kernel complexity and convergence.

## Lifecycle

```bash
python ako_runs/controlled_followup/decision_complexity_ada_v2/runner.py prepare
python -m unittest \
  ako_runs.controlled_followup.decision_complexity_ada_v2.test_protocol \
  ako_runs.controlled_followup.decision_complexity_ada_v2.test_execution
python ako_runs/controlled_followup/decision_complexity_ada_v2/runner.py freeze

# Commit and push the complete frozen source closure, then:
python ako_runs/controlled_followup/decision_complexity_ada_v2/runner.py provenance
python ako_runs/controlled_followup/decision_complexity_ada_v2/runner.py ready
python ako_runs/controlled_followup/decision_complexity_ada_v2/runner.py execute
python ako_runs/controlled_followup/decision_complexity_ada_v2/analyze.py \
  --out ako_runs/controlled_followup/decision_complexity_ada_v2/results/analysis.json
```

`provenance` requires a clean upstream commit, a live matching remote head,
the frozen toolchain, the current GPU-0 identity, and an idle occupancy query.
The analyzer independently verifies all 24 completion/parent chains, attempt
hashes, strict serialized intervals, final GPU idle receipt, and the exact
result census before producing the non-controlling summary.
