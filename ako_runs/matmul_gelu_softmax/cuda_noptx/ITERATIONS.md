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
| 1 | cuBLAS GEMM + fused bias+GELU+softmax | -1 (INCORRECT) | N/A | failed |

## Iterations

### Iter 1 — cuBLAS GEMM + fused bias+GELU+softmax

- **Hypothesis:** Use cuBLAS SGEMM for the matmul, then fuse bias-add + GELU + row-softmax into a single kernel to avoid 2 extra HBM passes.
- **Changes:** Full rewrite with load_inline: cuBLAS GEMM + custom `fused_bias_gelu_softmax_kernel` (THREADS=256, EPT=32).
- **Bench:**
  - Compiled: True
  - Correct: False (4/5 trials pass, max_diff=0.000308, threshold=1e-4)
  - Runtime: N/A
  - Speedup: N/A
- **Analysis:** `--use_fast_math` makes `erff` slightly less accurate. After softmax normalization, small GELU errors get amplified past the 1e-4 tolerance. Need to remove `--use_fast_math` and use ATen mm to match PyTorch's TF32 GEMM exactly.
- **Next:** Use `at::mm` from C++ (matches PyTorch handle/TF32 settings), remove `--use_fast_math`, keep explicit `__expf` for softmax.

