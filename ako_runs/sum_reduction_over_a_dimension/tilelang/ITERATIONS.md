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
| 1 | Serial-per-thread row reduction | 0.98x | 9.99 ms | no-change |
| 2 | 2D block: TH_H threads reduce per column, TH_W columns coalesced | 0.97x | 10.1 ms | no-change |
| 3 | Unrolled 8x serial loop, BLOCK_W=256 | 0.99x | 9.85 ms | no-change |
| 4 | T.Parallel+T.vectorized float4 approach | WRONG | - | failed |

## Iterations

### Iter 1 — Serial-per-thread row reduction

- **Hypothesis:** Assign one (b, w) output element per thread, accumulate serially over h. Coalesced reads across consecutive threads within each h row.
- **Changes:** Replaced torch.sum with TileLang kernel: T.Kernel(B, BW) with 128 threads each reading 4096 h-elements.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 9.99 ms (mean), 9.97 ~ 10.0 ms (min ~ max)
  - Speedup: 0.98x (mean)
- **Analysis:** Essentially same as PyTorch. Per-thread serial loop means 4096 iterations of 1-element reads, sequential within a single thread but the stride in H means each access is 4096*4 = 16384 bytes apart for consecutive h steps in the same (b,w). Cache is not effectively used. PyTorch's cublasReduce path is highly optimized.
- **Next:** Use shared-memory tile: load a tile of (TH_H x TW_W) elements, do intra-block tree reduction, then atomic-add to output. Better cache utilization and fewer global memory transactions.

### Iter 2 — 2D block layout: TH_H reduction workers, TH_W output columns

- **Hypothesis:** 2D block (32×32=1024 threads), TH_H=32 threads share reduction over H, TH_W=32 output columns per block. Shared memory tree-reduction. Coalesced reads along w.
- **Changes:** Replaced flat-thread layout with (th, tw) 2D decomposition. Each thread accumulates H/TH_H elements. Shared-mem tree over TH_H for each tw column.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 10.1 ms (mean), 10.1 ~ 10.2 ms (min ~ max)
  - Speedup: 0.97x (mean)
- **Analysis:** Still slower than baseline. The strided access pattern (each thread reads elements spaced TH_H*W=131KB apart) kills L1/L2 efficiency. The shared memory overhead adds latency without helping bandwidth.
- **Next:** Analysis: 9.79ms is already very close to theoretical bandwidth limit (~9.94ms at 864GB/s for 8.59GB). The only lever is reducing memory traffic (vector loads) or better cache utilization. Try: larger tile approach with float4 loads, or use the layer_norm style single-row-at-a-time to keep data L2-resident.

### Iter 3 — Unrolled 8x serial loop with BLOCK_W=256

- **Hypothesis:** Manual 8x loop unrolling reduces branch overhead, increasing throughput. BLOCK_W=256 increases occupancy.
- **Changes:** Changed BLOCK_W from 128 to 256, added manual 8x unrolling with 8 adds per iteration.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 9.85 ms (mean), 9.82 ~ 9.88 ms (min ~ max)
  - Speedup: 0.99x (mean)
- **Analysis:** Marginally better than prior attempts but still ~1% slower than PyTorch. The unrolling helps slightly but the bottleneck is memory bandwidth which is already at near-ceiling. The strided access pattern (stride W=16KB per h-step) means each thread is reading from separate cache lines.
- **Next:** Try T.Parallel + T.vectorized for float4 coalesced reads.

### Iter 4 — T.Parallel + T.vectorized float4 (failed correctness)

- **Hypothesis:** T.Parallel maps iterations to threads, T.vectorized(4) enables float4 loads; each block tile handles BLK_W=1024 output elements with 4-wide vector loads.
- **Changes:** New kernel using T.Parallel(TH) + T.vectorized(VEC) for vectorized h-loop iteration.
- **Bench:**
  - Compiled: True
  - Correct: False
  - Runtime: - ms
  - Speedup: WRONG
- **Analysis:** Output incorrect. T.vectorized(VEC) inside T.Parallel(TH) does not work as expected - accumulator per-thread/per-vector combination is incorrect. The accumulator needs to be per (thread, vec) but the loop structure may be mixing thread-local and shared state.
- **Next:** Use a simpler approach - process 4 output elements per thread with explicit indexing, avoiding T.vectorized complexity.



