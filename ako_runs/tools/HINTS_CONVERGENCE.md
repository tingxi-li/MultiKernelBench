# Convergence-run addendum (read before optimizing any 6-op-redo cell)

You are optimizing ONE cell (one op × one DSL) on branch `cross-dsl-6op-ncu-redo`.
Use your DSL's native method freely (triton: autotune; cuda: hand-variants; tilelang:
`T.Pipelined`/autotune). The measurement is fixed — obey these rules; full spec in
`ako_runs/CONVERGENCE_PROTOCOL.md`.

## Your only logging duty
Bench every variant through the wrapper, never the raw bench.sh:

```
ako_runs/tools/timed_bench.sh <cell_dir> "<one-line variant_desc>" --gpu3-serial [--ncu-key "passes=3.04"]
```

It times compile+bench, parses the result, and appends a row to
`<cell_dir>/convergence.csv`. Pass a short honest `variant_desc`. That's it — do NOT
hand-edit convergence.csv, do NOT self-report timings (the wrapper owns the clock).

**ALWAYS pass `--gpu3-serial` for memory-bound ops (layer_norm, sum_reduction).** It pins
GPU3 and flocks a shared lock so your benches don't overlap another agent's and corrupt the
speedup ratio (Discipline 4). Lock-wait is excluded from the timer. Do your source edits in
parallel; the benches themselves queue through GPU3. (Compute-bound ops — matmul/attn/conv —
may instead use `--gpu N` on a dedicated card, no lock needed.)

## Rank & keep
- A variant becomes the new best only on a **same-GPU speedup win** (the wrapper sets
  `kept`). Keep the source of your current best in `solution/`; revert losers.
- Never trust an absolute speedup from a concurrent GPU lane. Optimize on whatever GPU,
  but the numbers that matter are these serial per-cell benches.

## Profile at baseline + stalls only (never per-iter)
- Profile the baseline once, and again at a **stall** = 2 consecutive variants within ~3%.
- Use `ako_runs/tools/ncu_profile.sh <cell_dir>`; steer by **bytes / passes /
  TFLOP-fraction / occupancy / launches** — NEVER `%peak` or ncu-time (clock-locked here).
- When a profile drives a decision, pass its one steering number as `--ncu-key`.

## Stop rule (uniform across DSLs)
Stop when EITHER: within 5% of the external ceiling (cuBLAS/cuDNN/torch-SDPA/known
layer_norm), OR 2 consecutive ncu-guided levers each <3% AND ncu confirms the binding
roofline is hit. Write the stop reason in the cell's `ITERATIONS.md`.

## Verdict (leave to the orchestrator)
Final best is re-benched **serial on GPU3** and gated with `tools/check_gate.py`. Don't
run the verdict yourself; just converge and stop cleanly.
