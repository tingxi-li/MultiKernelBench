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
| 1 | fp16 T.gemm + GELU epilogue + 2-pass softmax | 1.7853x | 3.40 ms | improved |

## Iterations

### Iter 1 — fp16 T.gemm + GELU epilogue + 2-pass softmax

- **Hypothesis:** Using fp16 tensor cores for the GEMM (x @ W^T) with GELU fused in the epilogue, then a separate row-wise online softmax kernel, should beat fp32 cuBLAS by getting tensor-core throughput and avoiding two intermediate HBM writes.
- **Changes:** Full rewrite. Kernel A: fp16 T.gemm (BM=128, BN=128, BK=64, stages=2) with GELU(erf-exact) fused in T.Parallel epilogue -> fp32 scratch. Kernel B: row-wise 2-pass softmax (1 block per row, 256 threads, tree-reduce max then sum).
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 3.40 ms (mean), 3.34 ~ 3.47 ms (min ~ max)
  - Speedup: 1.7853x (mean)
- **Analysis:** fp16 tensor cores deliver ~1.79x over fp32 cuBLAS+eager. GEMM dominates at 8192x8192, so the fp16 lever is the primary win. Softmax overhead is minimal (row-wise, embarrassingly parallel). The erf-based GELU passes the 1e-4 fp32 gate cleanly.
- **Next:** Try BN=256 (wider tiles) to improve GEMM utilization and reduce softmax kernel launch overhead by fusing it into the GEMM epilogue.

