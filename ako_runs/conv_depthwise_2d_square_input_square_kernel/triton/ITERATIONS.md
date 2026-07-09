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
| 1 | Custom Triton depthwise kernel (autotune) | 1.12x | 3.25 ms | improved |

## Iterations

### Iter 1 — Custom Triton depthwise kernel with autotune

- **Hypothesis:** cuDNN's depthwise conv is suboptimal; a custom Triton kernel that tiles output spatially and uses @triton.autotune should beat the reference.
- **Changes:** Replaced identity solution with a Triton depthwise conv kernel. Grid=(N*C, ceil(H_out/BLOCK_H), ceil(W_out/BLOCK_W)), BLOCK_H/BLOCK_W autotuned across 13 configs. Unrolled KH×KW inner loop. Accumulates in fp32.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 3.25 ms (mean), 3.19 ~ 4.83 ms (min ~ max)
  - Speedup: 1.12x
- **Analysis:** First pass beats cuDNN by 12%. The BLOCK_H=4, BLOCK_W=64 or similar tile configs are likely winning. The kernel is memory-bound (depthwise convs have very low arithmetic intensity). Room to improve by better memory access patterns and coalescing.
- **Next:** Improve memory coalescing — try packing output rows or using wider loads. Consider having one warp handle a full channel row to maximize coalescing. Also try vectorized loads (float4) if the output width aligns.

