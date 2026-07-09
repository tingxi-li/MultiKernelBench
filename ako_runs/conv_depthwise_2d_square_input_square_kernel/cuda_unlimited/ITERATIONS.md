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
| 3 | PTX ld.cs streaming loads + st.cs stores | 1.35x | 2.63 ms | no-change |
| 4 | Y-direction coarsening (NVEC_Y=2, 128x16 tile) | 1.53x | 2.63 ms | no-change |
| 5 | Float4 vectorized smem loads (128x8 tile, NVEC=4) | 1.34x* | 2.62 ms | improved |
| 6 | NVX=4+NVY=2 combined with float4 loads (128x16 tile) | 1.52x | 2.64 ms | no-change |
| 7 (new-1) | NVEC=8 wider tile (256 cols/block), float4 smem loads | 1.56x | 2.59 ms | improved |
| 8 (new-2) | NVEC=8 RPTS=2 (256x16 tile), float4 smem loads | 1.53x | 2.61 ms | regression |
| 9 (new-3) | Dual-channel fusion (2 NC-planes/block), NVEC=8 | 1.53x | 2.60 ms | no-change |
| 10 (new-4) | Register-only, no smem, direct L2 float4 reads | 1.54x | 2.61 ms | no-change |
| 11 (new-5) | ld.cs streaming loads for smem fill + __launch_bounds__ | 1.53x | 2.66 ms | regression |

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

### Iter 3 — PTX ld.cs streaming loads + st.cs stores

- **Hypothesis:** Since each input element is only used by a small number of output elements (in neighboring blocks), streaming loads that bypass L2 might reduce cache pressure and allow more bandwidth.
- **Changes:** Replaced __ldg with PTX ld.cs.global.f32 for smem loading. Added PTX st.cs.global.f32 for output stores.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.63 ms (mean), 2.56 ~ 3.81 ms (min ~ max)
  - Speedup: 1.35x (mean) - ref was lower this run at 3.56ms
- **Analysis:** Same runtime as iter-2 (2.63ms). The ld.cs approach doesn't help - the L2 cache on Ada Lovelace (96MB) is not helping much at this scale anyway, but bypassing it doesn't hurt either. The solution runtime has converged at ~2.63ms. The speedup variation (1.35x vs 1.54x) is due to reference runtime variance.
- **Next:** Try a completely different approach - use cp.async for double-buffered pipeline, or try processing multiple channels per block (channel-fused), or try half-precision intermediate computation.

### Iter 4 — Y-direction coarsening (NVEC_Y=2, 128x16 output tile)

- **Hypothesis:** Having each thread handle 2 output rows (via Y coarsening) doubles the data reuse for each smem row read, reduces the number of smem load loops per block, and amortizes __syncthreads cost.
- **Changes:** 32x8 block, each thread computes NVEC_Y=2 output rows * NVEC_X=4 cols = 8 outputs. Tile 128x16, smem 130x18.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.63 ms (mean), 2.57 ~ 3.83 ms (min ~ max)
  - Speedup: 1.53x (mean)
- **Analysis:** Same 2.63ms as iter-2. Y-coarsening doesn't help. The solution is at the memory bandwidth ceiling. Three consecutive iters at ~2.63ms with different tile sizes.
- **Next:** Re-assess - try a fundamentally different approach. Consider: (1) Warp-level reduction with direct HBM reads (no smem), (2) Persistent kernel with circular buffer, (3) Process 2 NC-planes per block simultaneously to better utilize instruction-level parallelism.

### Iter 5 — Float4 vectorized smem loads (128x8 tile, NVEC=4)

- **Hypothesis:** Float4 vectorized global loads (16 bytes = 4 floats per transaction) reduce the number of memory transactions for smem filling, potentially improving throughput.
- **Changes:** Replaced scalar __ldg loop with explicit float4 loads for the first 128 columns (32 float4 per row), then 2 scalar loads for the trailing columns (128-129). Same NVEC=4 compute.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.62 ms (mean), 2.57 ~ 3.82 ms (min ~ max)
  - Speedup: 1.34x (mean) - *ref was 3.51ms this run; raw runtime is best so far
- **Analysis:** 2.62ms, marginally better than iter-2/3/4 (2.63ms). Float4 vectorized loads give a tiny improvement. The solution is essentially at the memory bandwidth floor (~2.48ms theoretical).
- **Next (iter 6 = last): try combining float4 stores with float4 loads, or revert to the cleaner scalar version. The solution is converged.

### Iter 6 — NVX=4 + NVY=2, 128x16 tile, float4 smem loads (FINAL iter)

- **Hypothesis:** Combining X-coarsening (NVX=4) with Y-coarsening (NVY=2) AND float4 smem loads should give the best of both worlds.
- **Changes:** 32x8 block, each thread computes 2 Y rows x 4 X cols = 8 outputs. 128x16 output tile with smem 130x18. Float4 vectorized smem loads.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.64 ms (mean), 2.59 ~ 3.83 ms (min ~ max)
  - Speedup: 1.52x (mean)
- **Analysis:** 2.64ms - marginally worse than iter-5 (2.62ms). The larger smem (9.5KB vs 5.3KB) slightly reduces occupancy/performance. Iter-5 (float4 loads, NVY=1) is the best.
- **Conclusion:** Solution is at ~94% of theoretical bandwidth floor. Iter-5 is the best at 2.62ms mean.

### Iter 7 (new iter 1) — NVEC=8 wider tile (256 output cols per block)

- **Hypothesis:** Widening the tile from 128 to 256 output columns halves the X-dimension grid size (2 vs 4 blocks per row), reducing kernel launch overhead and improving L2 reuse of input data loaded across the 512-column input. More ILP per thread (8 vs 4 outputs).
- **Changes:** T1_NVEC=8, T1_OUT_W=256, smem 258x10+2=2600 floats (10.4KB). Float4 smem loads (64 float4 per row). Scalar output stores to avoid misalignment (OW=510 is not 16B aligned). NVEC=8 output computation with 30 registers per thread.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.59 ms (mean), 2.54 ~ 3.84 ms (min ~ max)
  - Speedup: 1.56x (mean)
- **Analysis:** 2.59ms vs prior best 2.62ms. Small but real improvement. Wider tiles reduce grid overhead and improve ILP. Min 2.54ms hints at a floor around 2.50ms.
- **Next:** Try NVEC=16 to push even further, or try 2-row coarsening with NVEC=8 (RPTS=2, 16 outputs/thread) with a 256x16 tile.

### Iter 8 (new iter 2) — NVEC=8 x RPTS=2 (256x16 output tile), 256 threads

- **Hypothesis:** Combining 8x wide X tile with 2x Y coarsening further reduces grid size (half Y blocks) and amortizes smem barrier cost. 16 outputs per thread maximizes ILP.
- **Changes:** T2_OUT_H=16 (via RPTS=2), T2_IN_H=18, smem 260x18=18.7KB. Float4 smem loads. Macro COMPUTE8 for both rows.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.61 ms (mean), 2.55 ~ 3.85 ms (min ~ max)
  - Speedup: 1.53x (mean)
- **Analysis:** 2.61ms - slightly worse than iter-1's 2.59ms. The larger smem footprint (18.7KB) limits occupancy: Ada has 100KB smem/SM, but with 256 threads we can fit ~5 blocks; at 18.7KB, only ~5 blocks (OK); however the main issue is more smem load iterations without proportional speedup. Grid is half in Y dimension vs iter-1 but savings are smaller than register/smem overhead.
- **Next:** Try a different optimization: warp-level parallelism where each warp handles all 64 channels for a single spatial position (channel-fused). Or try L1 prefetch with cp.async.

### Iter 9 (new iter 3) — Dual-channel fusion (2 NC-planes per block)

- **Hypothesis:** Processing 2 channels per block halves the grid Z dimension, reducing launch overhead and allowing memory requests for two channels to be coalesced in the same warp. Each thread computes 16 outputs (8 for ch0 + 8 for ch1).
- **Changes:** New kernel processes nc0=blockIdx.z*2 and nc0+1 simultaneously. Uses two smem arrays s0[] and s1[] (total 20.8KB). Loads both channels' float4 data in same loop iteration. Computes 8 outputs per channel sequentially per thread.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.60 ms (mean), 2.55 ~ 3.84 ms (min ~ max)
  - Speedup: 1.53x (mean)
- **Analysis:** 2.60ms - same as prior iters. The dual-channel approach doesn't help: the larger smem (2x = 20.8KB) limits occupancy, and the doubled register pressure (30 weight registers) causes spilling or scheduling issues. We're firmly at 2.59-2.60ms floor.
- **Next (final iter): try a register-only approach without smem at all — direct __ldg loads, warp-level coalescing with NVEC=16 or 32. At this scale, L2 cache should provide good hit rate anyway.

### Iter 10 (new iter 4) — Register-only, no smem, direct L2 float4 reads

- **Hypothesis:** Eliminating smem avoids __syncthreads overhead and maximizes SM occupancy. The L2 cache (96MB on Ada) can handle the working set. With NVEC=8, each thread reads 30 values via __ldg/float4 directly from L2 and computes 8 outputs.
- **Changes:** New kernel v4 with no __shared__ memory. Uses float4 __ldg to read 3 rows x 8 floats + 2 scalar per row = 30 reads. __launch_bounds__(256, 5) hints the compiler.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.61 ms (mean), 2.56 ~ 3.85 ms (min ~ max)
  - Speedup: 1.54x (mean)
- **Analysis:** 2.61ms - same as other approaches. The smem approach and direct L2 reads are essentially equivalent. The bottleneck is pure memory bandwidth to/from HBM, not smem latency or barrier overhead. We're at the hardware bandwidth limit.
- **Next (stall reached): iter-1 (NVEC=8, smem, 256x8 tile) remains the best at 2.59ms. Will restore it for final.

### Iter 11 (new iter 5) — ld.cs streaming loads for smem fill + __launch_bounds__(256,4)

- **Hypothesis:** Using ld.cs (cache streaming) PTX instruction for smem fills avoids evicting other L2-cached data, potentially improving cache efficiency for the weight lookups and adjacent channel accesses.
- **Changes:** Replaced __ldg float4 smem fills with PTX ld.global.cs.v4.f32 instruction. Added __launch_bounds__(256, 4) to hint compiler to limit register spill.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.66 ms (mean), 2.60 ~ 3.86 ms (min ~ max)
  - Speedup: 1.53x (mean)
- **Analysis:** 2.66ms - slightly worse than iter-1 (2.59ms). ld.cs doesn't help because at this scale the L2 cache pressure is not from smem-bound reads competing with computation; the bottleneck is pure HBM bandwidth. The __launch_bounds__(256,4) may be slightly restricting occupancy.
- **Conclusion (STALL):** 5 new iters tried: NVEC=8 wider tile (1), NVEC=8+RPTS=2 (-), dual-channel (-), register-only (-), ld.cs (-). Best is iter-1 (new) at 2.59ms mean / 1.56x speedup. The kernel is at ~96% of theoretical HBM bandwidth floor (~2.5ms). Will restore iter-1 as final.

