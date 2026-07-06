# ncu-in-the-loop runbook — the redo discipline

The 12-op pass picked direction from *analytical* rooflines, never live ncu. That
is fine for a "stop at floor" decision (endpoint verifiable post-hoc) but not for
"continue — which lever next?", which shapes the search path. This toolset makes
that decision **measured**. Two disciplines are wired in structurally so the redo
can't repeat the blind-search mistake or regress the validated artifact.

## Discipline 1 — profile at BASELINE and STALLS, not every iteration

ncu is 10–100× a plain bench (serialized replays), worst on many-launch / large
tensors (group_norm 514 launches @ 8.6 GB). So profiling is a **separate path**,
never inside `bench.sh`:

- **Per iteration:** rank by the solution's own `RUNTIME` (`bench.sh` — cheap).
- **At baseline + at each stall** (3 iters, no ≥3% gain): run
  `ako_runs/tools/ncu_profile.sh <op>/<dsl>` to learn *why* and pick the next lever.

## Discipline 2 — steer by DRAM bytes / structural counters, NEVER %peak or ncu-time

`ncu_profile.sh` prints a report whose **actionable** block is bytes → passes,
L2-hit rate, sectors/request (coalescing), achieved occupancy, launch count —
all replay-invariant. Throughput `%peak` and ncu wall-time are shown ONLY under a
"⚠ qualitative — saturated-or-not" banner because ncu locks clocks and serializes
replays (depresses %peak for many-launch/sub-µs kernels; see `../NCU_VALIDATION.md`).
Do not do arithmetic on %peak; do not treat it as the reward. The anchor is:

> **passes = DRAM total / primary tensor.** 2 = copy floor. 3 = an un-reused
> re-read (the layer_norm/group_norm loser signature). A lever "fires" when it
> drops passes toward 2.

Validated against `NCU_VALIDATION.md`: `layer_norm/cuda_noptx` measures 3.04
passes (2.03 GiB read); `layer_norm/cuda_unlimited` measures 1.97 passes (1.06 GiB
read, 130 launches) — the driver reproduces both to 3 sig figs.

## Discipline 3 — the redo of the 12 CANNOT regress the validated artifact

`committed_baseline.csv` holds each committed cell's speedup floor. After a redo
produces a verdict, gate it:

```
python ako_runs/tools/check_gate.py --op <op> --dsl <dsl> --speedup <verdict>
# exit 1 => KEEP COMMITTED (git checkout the prior bytes); do not accept a slower redo
```

The floor seed is from `RESULTS.md` (mixed clock states). Before a redo, refresh
each cell's floor with an **agent-free** re-bench of the committed bytes:

```
python ako_runs/tools/record_committed_baseline.py --op <op> --dsl <dsl> --gpu <g> --write
```

The 28 new ops have no floor — the gate passes them (nothing to protect).

## Per-cell loop (what each agent does during the redo)

1. `bash scripts/bench.sh baseline` → confirm `CORRECT=True`, note baseline speedup.
2. `bash ../../tools/ncu_profile.sh <ws>` (or the agent's cwd) → read passes +
   the bottleneck kernel. Pick iter-1 direction from **bytes/structural**, not %peak.
3. Iterate: edit `solution/` → `bash scripts/bench.sh iter-N` (rank by RUNTIME) →
   log `ITERATIONS.md` → commit. Re-profile with ncu only at a stall.
4. Verdict: full `bash scripts/bench.sh final` → for the 12, run `check_gate.py`;
   FAIL ⇒ restore committed bytes.

## Orchestration (from project memory)

4 GPU lanes via `parallel()` over 4 async thunks, one cell per GPU (no OOM on the
49 GB cards); per-workspace `TORCH_EXTENSIONS_DIR`; agents do NOT git commit
(orchestrator commits after the fan-out). `ncu_profile.sh` and `bench.sh` both
respect a pre-set `CUDA_VISIBLE_DEVICES`, so pin each lane explicitly. ncu holds a
GPU longer per cell than a bench — keep it one-cell-per-GPU and watch `nvidia-smi`
for cross-process grabs (the memory records a 32 GB grab that transient-OOM'd a
cell). Reuse each workspace's `.torch_ext` so ncu doesn't recompile.
