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
| 3 | 32x16 tile, 512-thread blocks (v4) | 1.31x | 2.86 ms | improved |
| 4 | 32x32 tile, 1024-thread blocks + warp-row variant | 1.20x | 3.46 ms | regression |
| 5 | Coalesced SM fill: TW=30, SW=32 (power-of-2), maxreg=40 | 1.42x | 2.91 ms | improved |
| 6 | Wide coalesced: TW=62, SW=64, TH=8, block=64x8=512 | 1.38x | 2.69 ms | improved (best abs) |

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

### Iter 3 — Multiple kernel variants (stream, 32x16, multi-channel)

- **Hypothesis:** Larger tiles (32x16=512 threads) would give better occupancy and reduce block-scheduler overhead. Per-channel streaming with __ldg would hit L1 better. Multi-channel blocks would amortise SM load overhead.
- **Changes:** Added v3 (streaming 1-blk-per-channel), v4 (32x16 512-thread), v5 (multi-channel 4ch/block). Using v4 (32x16) as primary.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.86 ms (mean), 2.61 ~ 3.82 ms (min ~ max)
  - Speedup: 1.31x (mean) — lower speedup due to variable ref time
- **Analysis:** Best absolute runtime so far at 2.86ms. Note the ref runtime was only 3.74ms this run (vs 4.2ms before), so speedup ratio looks lower but actual kernel latency improved. The 32x16 tile gives better SM utilisation. Low std (0.135ms) means more stable measurements.
- **Next:** Try pushing further with: (1) reducing register usage for higher occupancy, (2) loop over rows for the stream variant to reduce block count, (3) try direct global memory reads with __ldg and no shared memory to test if SM overhead isn't worth it for this problem size.

### Iter 4 — 32x32 tiles (1024 threads) regression

- **Hypothesis:** 32x32=1024-thread blocks would give maximum occupancy and better SM utilisation.
- **Changes:** Added 32x32 SM tiling (1024 threads), padded SW=36 variant, and warp-row (no SM) variant. Used variant=1 (32x32).
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 3.46 ms (mean), 3.37 ~ 4.94 ms (min ~ max)
  - Speedup: 1.20x
- **Analysis:** Regression vs iter 3. 32x32 tile with 1024 threads is actually slower than 32x16 with 512 threads. The issue: with 1024 threads/block, each SM can host fewer concurrent blocks (register/SM bandwidth limited). Also, 1024 threads = 32 warps, and the SM is only 1156 floats — very little data reuse. The optimal configuration appears to be 32x16 (512 threads).
- **Next:** Restore iter 3's 32x16 config. Then try: (1) varying the tile in Y dimension (e.g. 32x8, 32x12, 32x16 comparison), (2) experiment with #pragma unroll for the inner MAC, (3) try occupancy-driven approach with --maxrregcount=32 to allow more resident blocks.

### Iter 5 — Coalesced SM fill: TW=30, SW=32 (power-of-2), maxreg=40

- **Hypothesis:** Setting SM width = TW+2 = 32 (power of 2) ensures every warp in the SM fill loop covers exactly one SM row, resulting in perfectly coalesced 128-byte global reads. Combined with maxrregcount=40, we get higher SM occupancy (3 blocks instead of 2).
- **Changes:** Redesigned kernel with TW=30, SW=32. Block=32×16=512 threads. Fill loop uses i>>5 / i&31 bit operations. Added wide variant TW=62, SW=64. maxrregcount=40.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.91 ms (mean), 2.71 ~ 4.57 ms (min ~ max)
  - Speedup: 1.42x
- **Analysis:** Best speedup so far (1.42x). The coalesced SM fill strategy helps. 2.91ms is close to iter 3's 2.86ms on an absolute basis but speedup is higher because ref is more stable. The TW=30/SW=32 trick is clearly beneficial.
- **Next:** Test the TW=62, SW=64 (wider) variant which should be even more efficient. Also try reducing TH to 8 (smaller blocks) to increase occupancy further, or TH=32 for more output reuse.

### Iter 6 — Wide coalesced: TW=62, SW=64, TH=8

- **Hypothesis:** A wider tile (TW=62, SW=64) with 512-thread blocks should give better L2 reuse and lower block-scheduling overhead. 64-wide SM row = 2 cache lines. Fewer total blocks for same input.
- **Changes:** Switched to variant=1 (TW=62, SW=64, TH=8, block=64×8=512). Added also TW=126/SW=128 and retained TW=30/SW=32.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.69 ms (mean), 2.52 ~ 3.81 ms (min ~ max)
  - Speedup: 1.38x (ref was faster this run: 3.70ms)
- **Analysis:** Best absolute runtime (2.69ms) — better than all previous. However speedup is 1.38x vs iter 5's 1.42x because ref runtime was lower in this run (3.7ms vs 4.14ms). Note: the ref runtime is noisy across runs due to GPU clock variation. The key metric is the solution's own runtime: 2.69ms is the best so far. Lower std (0.158ms) confirms stability.
- **Conclusion:** Iter 6 is the best absolute performer (2.69ms). This will be the final version.

---
## Final Bench (iter 6 = latest = best)

- COMPILED: True, CORRECT: True
- RUNTIME: 2.88 ms (mean), 2.71 ~ 3.85 ms
- REF_RUNTIME: 4.20 ms
- SPEEDUP: **1.46x**
- Status: **win** (beat PyTorch eager cuDNN depthwise)

Best direction: coalesced SM fill with power-of-2 SM width (SW=64, TW=62).
Each warp covers exactly one SM row → 128-byte coalesced transactions. Combined
with --maxrregcount=40 for higher SM occupancy (3 blocks/SM instead of 2).






