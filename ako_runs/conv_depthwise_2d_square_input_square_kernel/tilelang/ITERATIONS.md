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


