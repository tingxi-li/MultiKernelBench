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
| 1 | Shared-mem tiling 3x3 kernel | 1.34x | 2.91 ms | improved |
| 2 | Wide-tile: 4 outputs/thread, 128x8 tile | 1.54x | 2.63 ms | improved |

## Iterations

### Iter 1 — Shared-memory tiling for 3x3 depthwise conv

- **Hypothesis:** Depthwise conv is memory-bound. Each input element is reused for 9 outputs in a 3x3 kernel. Shared memory tiling (32x8 output tiles with 34x10 input halos) should yield ~8-9x data reuse from smem vs L2/HBM, beating PyTorch's cuDNN path.
- **Changes:** Replaced identity nn.Conv2d forward() with a custom CUDA kernel using 32x8 shared memory tiles. Loaded 9 weights into registers. Specialized path for 3x3 kernel; general fallback for others.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.91 ms (mean), 2.73 ~ 4.91 ms (min ~ max)
  - Speedup: 1.34x (mean)
- **Analysis:** 1.34x speedup achieved. The shared memory approach works and reduces data movement. The large std/max hints at clock variance; the min of 2.73ms is encouraging.
- **Next:** Try wider tiles and/or vectorized float4 loads. Also consider larger tile sizes. The 32x8 block = 256 threads per block; possibly increase occupancy or try register-only approach without smem (inline PTX).

### Iter 2 — Wide-tile 4 outputs/thread (128x8 output tile)

- **Hypothesis:** Having each thread compute 4 output columns increases arithmetic intensity, reduces smem bank conflicts (more data per load transaction), and improves ILP.
- **Changes:** 32x8 block with each thread computing 4 output cols. Output tile 128x8. Input smem 130x10. Loads smem with __ldg. Unrolled 3-row computation reusing 6 smem values per row.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.63 ms (mean), 2.58 ~ 3.84 ms (min ~ max)
  - Speedup: 1.54x (mean)
- **Analysis:** 1.54x speedup, improved from iter-1 (1.34x). The wide tile reduces overhead per output and improves instruction-level parallelism. The min of 2.58ms is very close to the theoretical bandwidth limit of ~2.48ms.
- **Next:** Try PTX streaming loads (ld.cs) to reduce cache pressure, or cp.async for latency hiding. Also try larger NVEC (8 per thread).

