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
| 1 | float4 PTX ld.cs streaming | 1.01x | 9.67 ms | improved |

## Iterations

### Iter 1 — float4 PTX ld.cs streaming loads

- **Hypothesis:** Float4 vectorized streaming loads (PTX ld.cs.v4.f32) should increase memory bandwidth utilization for this column-reduction pattern. ld.cs evicts early, reducing L2 pollution from non-reused data.
- **Changes:** Replaced torch.sum with custom CUDA kernel using float4 reads with PTX cache streaming hint. Each thread handles 4 consecutive K-elements; grid over (K/4, N).
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 9.67 ms
  - Speedup: 1.01x
- **Analysis:** Very small improvement (1%). The kernel is latency-bound on the reduction loop. The M=4096 iterations per output element serialized into a single thread means we're not utilizing ILP or multiple SMs to cover latency. Need to parallelize across the M dimension.
- **Next:** Try two-pass or warp-level reduction: split the M dimension across threads within a block. Use shared memory or warp shuffles to reduce partial sums.

