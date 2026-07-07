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
| 1 | identity baseline (torch.sum) | 1.0000x | 9.79 ms | baseline |
| 2 | tilelang tile-reduce block_C=2048 ns=2 vec=4 | 1.0072x | 9.72 ms | improved / KEPT |

## Iterations

### Op summary (MEMORY-BOUND, at roofline)
- Reduce dim=1 of (128,4096,4096) fp32 = read 8.59 GB once, write tiny -> pure HBM
  streaming. Kernel: one block per (batch, column-tile of 2048); block_C threads (vec4)
  stream the 4096 rows accumulating column-wise in a register fragment. Reads fully
  coalesced, each element read exactly once.
- Local config scan (block_C 256..4096, ns 1..4, vec 2/4) all landed 9.72-9.74 ms
  (~883 GB/s); differences within noise. Picked block_C=2048/ns=2/vec=4.
- **Result: 1.0072x (9.72 vs 9.79 ms). torch.sum already runs at 877 GB/s; we hit
  883 GB/s ~= 92% of ~960 GB/s theoretical peak = the achievable HBM roofline.**
- STOP: within 5% of ceiling (ceiling = HBM 1-read roofline; we ARE at it). Detector-clean.
