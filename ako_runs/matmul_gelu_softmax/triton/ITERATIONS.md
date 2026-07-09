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
| 1 | Fused matmul+GELU + separate softmax | 3.55x | 1.76 ms | improved |

## Iterations

### Iter 1 — Fused matmul+GELU kernel + row-wise softmax kernel

- **Hypothesis:** PyTorch eager runs matmul, GELU, softmax as separate ops with intermediate tensor traffic. A fused Triton kernel for matmul+bias+GELU eliminates the intermediate write+read, and a separate row-wise softmax kernel processes the output in L2 cache. The 8192×8192 GEMM is compute-bound so triton.autotune will find good tile configs.
- **Changes:** Replaced PyTorch eager forward with: (1) `_matmul_gelu_kernel`: autotuned Triton GEMM fused with bias add and exact GELU (using tl.erf); (2) `_softmax_kernel`: one CTA per row, loads full 8192-element row into SRAM, computes safe softmax in registers. Used `tl.erf` for exact GELU matching PyTorch default.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 1.76 ms (mean), 1.71 ~ 1.94 ms (min ~ max)
  - Speedup: 3.55x (mean)
- **Analysis:** Large gain from fusion eliminating intermediate tensor writes. The autotune found 128×256 tile with 8 warps as best config. GELU via erf was correct — PyTorch F.gelu defaults to exact erf-based formula.
- **Next:** Try fusing softmax directly into the matmul epilogue (compute per-row max and sum during GEMM epilogue using cross-warp reduction), or try fp16 computation to double tensor core throughput.

