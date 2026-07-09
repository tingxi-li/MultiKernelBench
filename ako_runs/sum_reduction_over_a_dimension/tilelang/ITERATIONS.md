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
| 3 | TBD | TBD | TBD | TBD |

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


