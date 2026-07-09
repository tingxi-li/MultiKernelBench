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
| 1 | float4 loads + 8-unroll | 1.006x | 9.73 ms | improved |

## Iterations

### Iter 1 — float4 vectorized loads + 8-step unroll

- **Hypothesis:** The operation is bandwidth-bound. Using float4 loads (128-bit transactions) with 8-step loop unrolling increases ILP and hides memory latency, improving effective bandwidth utilization.
- **Changes:** Replaced identity PyTorch implementation with a custom CUDA kernel using float4 loads over the C dimension, 8-step unrolled loop over D dimension with __ldg hints for read-only cache.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 9.73 ms (mean), 9.72 ~ 9.75 ms (min ~ max)
  - Speedup: 1.006x
- **Analysis:** Marginal improvement. PyTorch's sum is already near the bandwidth roofline. The float4 read pattern helps a little, but PyTorch's kernel is also well-optimized. Need to explore different block/grid configurations or warp-level parallelism.
- **Next:** Try a 2D block strategy where threads collaborate within each row-slice to reduce, plus larger tile sizes and shared memory staging to get better memory access patterns.

