# Convergence protocol — the 6-op cross-DSL ncu redo

Branch: `cross-dsl-6op-ncu-redo`. Ops: `layer_norm` (calibration / known-answer),
`sum_reduction_over_a_dimension`, `standard_matrix_multiplication`,
`matmul_gelu_softmax`, `conv_depthwise_2d_square_input_square_kernel`,
`scaled_dot_product_attention`. Four DSLs each: triton, cuda_noptx, cuda_unlimited,
tilelang → 24 cells.

Goal: measure **how fast each DSL reaches its own ceiling**, under a *relaxed* protocol
where each DSL uses its native optimization method — but with the **measurement unit
frozen** so the comparison is honest. See `CROSS_DSL_FINDINGS.md` §"convergence rate"
for why the old 12-op history cannot answer this.

## The one principle: instrument the measurement, free the method

The per-DSL freedom (triton autotune sweeps, cuda hand-variants, tilelang
`T.Pipelined`/autotune) lives **entirely above the bench call**. Everything at and
below the bench call is fixed: same wrapper, same log schema, same ceiling definition,
same verdict. Method roams; the yardstick does not.

## Split the clock (or you measure the LLM, not the DSL)

Wall-clock has two parts and only one is a DSL property:

- **compute_s** — compile / JIT / autotune / bench seconds. Reproducible and
  DSL-attributable (a triton autotune over N configs *costs* real GPU-seconds; an nvcc
  recompile costs real seconds). **This is the convergence yardstick.**
- **agent_s** — LLM reasoning/edit latency between benches. A function of model+prompt,
  NOT the DSL. Logged for the record, **never** compared across DSLs.
- **ncu_s** — profiling passes. *Our* instrument cost; the DSL converges identically
  whether or not we watch. **Excluded from the convergence clock**; logged separately.

**Primary metric:** cumulative `compute_s` to within 5% of the cell's ceiling.
**Secondary:** distinct variants benched (search breadth).

`compute_s` is captured by the `timed_bench.sh` wrapper (times compile+bench only; the
agent's thinking sits outside the timer), so it is not self-reported and not gameable.

## What logs a row

One `convergence.csv` row **per benched variant** (i.e. per `timed_bench.sh` call),
appended automatically by the wrapper. Schema:

```
iter,cum_compute_s,variant_desc,runtime_ms,speedup,ncu_key,kept,agent_s
```

- `iter` — 1-based, per cell.
- `cum_compute_s` — running sum of `compute_s` for this cell.
- `variant_desc` — the agent's ONLY logging duty: a one-line label ("autotune BLOCK=1024 nw=8", "L2-resident 2-pass", ...).
- `runtime_ms`, `speedup` — parsed from bench stdout by the wrapper.
- `ncu_key` — blank unless this variant was profiled at a stall; then the one steering number (e.g. `passes=3.04` or `tflops=41% cuBLAS`).
- `kept` — `1` iff this variant beat the running RUNTIME-best (same-GPU), else `0`.
- `agent_s` — optional, for the record only.

## Ceiling & the convergence number (computed post-hoc, never circular)

`ceiling = max(cell_final_best, external_ref)` where an external reference exists:

| op | external ceiling reference |
|---|---|
| standard_matrix_multiplication | cuBLAS TFLOP/s (torch.mm) |
| matmul_gelu_softmax | cuBLAS + fused epilogue (torch eager) |
| conv_depthwise_2d | cuDNN (torch conv2d, groups=C) |
| scaled_dot_product_attention | torch SDPA (flash / mem-efficient backend) |
| layer_norm | known committed 1.95–2.16x |
| sum_reduction | HBM 1-read+write roofline |

`ceiling` is fixed only **after** a cell's search stops (= its final best, or the external
ref if higher). Then `time_to_ceiling` = the `cum_compute_s` of the **first** row whose
`speedup ≥ 0.95 × ceiling_speedup`. Reported per cell, summarized per DSL. Because it is
read off the finished curve, it is never circular during the run.

## When ncu fires (baseline + stalls only)

Profile at the **baseline** and at a **stall** = 2 consecutive benched variants within
~3% RUNTIME. Steer by **bytes / passes / TFLOP-fraction / occupancy / launches** — never
`%peak` or ncu-time (clock-locked + serialized here). Per-op replay rule is baked into
`tools/ncu_profile.sh` (layer_norm/group_norm → application replay + `--cache-control
none`; else kernel replay + `--cache-control all`). ncu seconds → `ncu_s`, out of the
convergence clock.

## Stop rule (relaxed, but uniform across DSLs)

Stop a cell when **either**:
1. within 5% of the external ceiling (where one exists), OR
2. 2 consecutive ncu-guided levers each yield <3% AND ncu confirms the binding roofline is
   hit (bytes at N-pass floor / TFLOPs at the cuBLAS/cuDNN fraction).

Log the stop reason in the cell's `ITERATIONS.md`. The stop rule is the same for every DSL;
only the *method* of generating the next variant differs.

## Verdict & gate (unchanged from the committed discipline)

- A variant advances the running best only on a **same-GPU RUNTIME** win (fan-out-safe).
- The **verdict** number is always a **serial-GPU3** re-bench of the final committed bytes —
  never a speedup read from a concurrent lane (memory-P-state contamination; see
  `tools/NCU_RUNBOOK.md` Discipline 4).
- Gate the verdict with `tools/check_gate.py` vs `tools/committed_baseline.csv`. layer_norm
  has committed floors (1.95 / 2.10 / 2.16 / 2.10); the 5 new ops have **no floor → they set
  it** (auto-PASS, and their verdict becomes the new floor).

## layer_norm = the adversarial calibration

layer_norm is run first *because* its answer is known. The scaffold is validated against
**both ends** of its curve before any unknown op is trusted:
- committed winner (pre-reset) must read ~1.95x (noptx) / ~2.1x (others) and gate-PASS;
- identity (post-reset) must read ~1.0x.
If the instrumented pipeline reports anything else at a known point, the scaffold is wrong —
fix it before proceeding.
