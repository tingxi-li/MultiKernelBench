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
| 1 | Shared-mem tiled CUDA (32x8, KS=3 specialised) | 1.35x | 3.12 ms | improved |
| 2 | 64x4 tile variant + multi-row variant (32x8x4) | 1.36x | 3.09 ms | improved |

## Iterations

### Iter 1 — Shared-memory tiled CUDA kernel (3x3 specialised)

- **Hypothesis:** cuDNN depthwise is not well-optimised. A custom tiled CUDA kernel that loads a (34x10) input tile into shared memory and computes the 3x3 depthwise MAC in registers should reduce repeated global-memory reads and beat PyTorch eager.
- **Changes:** Replaced identity solution with CUDA kernel `dw_conv_k3s1p0` using TILE_W=32 × TILE_H=8 blocks. Each block covers one (B,C) slice and loads a shared-memory halo tile. 9 weights loaded per channel into registers. Full 3x3 MAC unrolled. Also includes a general fallback for non-standard configs. Extra compiler flags: `-O3 --use_fast_math`.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 3.12 ms (mean), 2.79 ~ 4.93 ms (min ~ max)
  - Speedup: 1.35x (mean)
- **Analysis:** Good improvement. 1.35x speedup over PyTorch eager. Shared-memory tiling avoids redundant global reads for the 3-pixel overlap between adjacent tiles. Weights loaded via `__ldg` cached reads.
- **Next:** Try wider tiles (e.g., 64x4 or vectorised float4 loads) to improve memory coalescing. Consider using register file to hold input rows and slide the window (implicit im2col). Also try increasing occupancy by reducing per-thread register pressure.

### Iter 2 — Wider tile (64x4) + multi-row variant

- **Hypothesis:** A wider tile (64 cols) would improve memory coalescing. Multi-row variant loads larger SM tile but amortises barrier overhead.
- **Changes:** Added v2 kernel with TILE_W=64, TILE_H=4 (same 256 threads). Added v2b kernel with BLKW=32, BLKH=8, OUT_ROWS=4 (each thread computes 4 rows). Using v1 (64x4) by default.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 3.09 ms (mean), 2.72 ~ 4.46 ms (min ~ max)
  - Speedup: 1.36x (mean)
- **Analysis:** Marginal improvement over iter 1 (3.09 vs 3.12ms). The wider tile improves memory coalescing slightly. SM load pattern still dominates. Need to think differently about the bottleneck.
- **Next:** The kernel is likely memory-bandwidth bound. Strategy: reduce SM occupancy to increase L2 cache hits, or try processing multiple channels per block to amortise weight loading. Also try 1D blocks for better warp utilisation.


