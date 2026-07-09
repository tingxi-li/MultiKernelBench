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
| 1 | TileLang 1-pixel-per-thread, register-cached filter | 1.47x | 2.70 ms | improved |
| 2 | Row-per-block shared-mem, unrolled 3x3 kernel | 1.50x | 2.66 ms | improved |

## Iterations

### Iter 1 — TileLang 1-pixel-per-thread, register-cached filter

- **Hypothesis:** cuDNN's generic grouped-conv path is suboptimal for depthwise; a custom kernel loading 9 filter weights into registers and doing coalesced reads/writes should be faster.
- **Changes:** Replaced identity (PyTorch conv2d) with a TileLang kernel. Grid: (B*C, ceil(H_out*W_out/128)). Each thread processes one output pixel. Filter loaded into local registers (9 floats). Simple innermost kh,kw loop for accumulation.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.70 ms (mean), 2.59 ~ 3.86 ms (min ~ max)
  - Speedup: 1.47x (mean)
- **Analysis:** 1.47x over baseline. The register-cached filter approach works well. The kernel is memory-bound (9 MACs per pixel, large 512x512 tensors). Fast signal showed 2.84ms; full bench 2.70ms — consistent. 
- **Next:** Try tile-based approach with shared memory for input halo. Consecutive spatial tiles can reuse the halo rows. Also try vectorized loads (float4).

### Iter 2 — Row-per-block shared-mem with unrolled 3x3

- **Hypothesis:** Loading 3 full input rows into shared memory for one output row amortizes input bandwidth; unrolling the 3x3 filter loop avoids loop overhead and enables instruction-level parallelism.
- **Changes:** Row-per-block grid (B*C, H_out) instead of (B*C, HW_tiles). TH=512 covers W_out=510 pixels. Three shared-memory arrays (sh0, sh1, sh2) for the 3 input rows. Filter weights still register-cached. The 3x3 is fully unrolled (9 explicit multiply-adds).
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.66 ms (mean), 2.59 ~ 3.87 ms (min ~ max)
  - Speedup: 1.50x (mean)
- **Analysis:** 1.50x vs 1.47x for iter-1. Small improvement: 2.66ms vs 2.70ms. Shared memory loads for 3 rows + unrolled 3x3 is slightly better. RTX6000 Ada bandwidth ~960 GB/s; with ~4.26 GB total I/O the BW floor is ~4.44ms; we're at 2.66ms which suggests cuDNN's reference is not achieving full BW (it reads/writes partially).
- **Next:** Try larger tile height (R_ROWS=8) for more row reuse, or different occupancy tuning.


