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
| 2 | Coalesced load, shared-mem tree reduction | TBD | TBD | TBD |

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

