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
| 1 | Tiled SGEMM (BM=128,BN=128,BK=8,TM=8,TN=8) | 0.68x | 5.98 ms | regression |

## Iterations

### Iter 1 — Tiled SGEMM shared-mem + register blocking (no PTX)

- **Hypothesis:** A well-tuned float32 tiled SGEMM with 128x128 thread-block tiles and 8x8 register accumulators should be competitive with cuBLAS SGEMM on RTX 6000 Ada. Without PTX/mma.sync, we must demonstrate whether any custom kernel can approach cuBLAS.
- **Changes:** Replaced torch.matmul with load_inline CUDA kernel — BM=128, BN=128, BK=8, 16x16 thread block (256 threads), each thread accumulates TM=8 x TN=8 output elements, shared-mem tiles with +1 padding to avoid bank conflicts, -O3 --use_fast_math sm_89.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 5.98 ms (mean), 5.85 ~ 6.44 ms (min ~ max)
  - Speedup: 0.68x (mean)
- **Analysis:** As expected for a FLOOR op, the custom float32 SGEMM without tensor cores (no PTX mma.sync available in cuda_noptx DSL) is slower than cuBLAS which uses TF32 tensor cores on Ada Lovelace. cuBLAS achieves ~4.1 ms while our pure FMA kernel takes ~6.0 ms. Without access to mma.sync or wmma intrinsics (which require PTX or CUDA C++ wmma headers, but still benefit from vectorized memory access), a custom kernel cannot beat cuBLAS on this hardware for large GEMM.
- **Next:** Iter 2: Try WMMA (Warp Matrix Multiply-Accumulate) C++ API headers which are available without inline PTX — these provide tensor core access. This is the primary lever for potentially approaching cuBLAS performance.

