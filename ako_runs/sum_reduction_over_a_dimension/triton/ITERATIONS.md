# Iteration Log

<!--
Per-iteration template (copy when adding a new iter entry under "## Iterations"):

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False
  - Runtime: ___ ms (mean), ___ ~ ___ ms (min ~ max)
  - Speedup: ___x (mean), ___ ~ ___x (min ~ max)
- **Analysis:** Why it worked or failed
- **Next:** What to try next

Append one row per iter to the Summary table below.
Status values: improved / no-change / regression / failed.
-->

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | identity (torch.sum) baseline | 1.0000x | 9.84 ms | baseline |
| 2 | triton 2D-tile reduce dim1, autotune BLOCK_M/N,nw | 1.0061x | 9.77 ms | improved (best) |
| 3 | pipelined loads num_stages 3-4, BLOCK_N up to 1024 | 1.0031x | 9.80 ms | regression (revert) |

## Iterations

### Op summary — sum_reduction_over_a_dimension (triton), MEMORY-BOUND

- **Shape:** sum over dim=1 of (128,4096,4096) fp32; input 8.0 GiB, output 2 MB. Pure 1-read+write roofline.
- **Iter 2 (best):** each program owns (batch b, N-tile) and streams all M=4096 rows in BLOCK_M chunks, `tl.sum(axis=0)` accumulate, coalesced BLOCK_N loads. 9.77 ms → 1.0061x (edges past torch.sum's 9.83 ms). Kept.
- **Iter 3:** software-pipelined loads (num_stages 3-4) + bigger BLOCK_N (512/1024) to hide the 16 KB-strided row latency. Regressed to 9.80 ms — the read is already bandwidth-saturated so deeper pipelining only added register/scheduling overhead. Reverted to iter-2 configs.
- **ncu at stall (iter 2/3 within 0.3%):** DRAM total = 8.005 GiB = **1.00 pass** (the anchor — one read, negligible write, no re-reads), occupancy 91.7%, dram saturation ~97.5%. Binding roofline is HIT: bytes are at the 1-pass floor, ~880 GB/s effective (~92% of 960 GB/s theoretical).
- **Stop reason:** stop-rule #2 — 2 consecutive levers <3% AND ncu confirms the binding DRAM roofline (1.00 pass, ~97.5% saturated) is hit. Can't reduce traffic below 1 read. Final best 1.0061x.
- **Detector:** clean (forward is allocate+launch glue only; reduction fully in the `@triton.jit` kernel).

