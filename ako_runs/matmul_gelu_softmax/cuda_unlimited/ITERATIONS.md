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
| 1 | TF32 WMMA GEMM + fused GELU + softmax | 0.86x | 7.19 ms | regression |

## Iterations

### Iter 1 — TF32 WMMA GEMM + fused bias/GELU + online softmax

- **Hypothesis:** Use TF32 tensor-core WMMA (sm_89 supported) for the GEMM with FP32 accumulators, then fuse bias+GELU into the epilogue, and do online safe-softmax as a second kernel. This should approach cuBLAS throughput while eliminating the GELU intermediate HBM round-trip.
- **Changes:** Replaced identity PyTorch solution with a 2-kernel CUDA solution: (1) `gemm_gelu_tf32` with BM=64, BN=64, BK=32, 4 warps, WMMA m16n16k8 TF32; (2) `softmax_kernel` with one block per row, 256 threads, online reduce.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 7.19 ms (mean), 7.04~7.30 ms (min~max)
  - Speedup: 0.86x (mean)
- **Analysis:** TF32 WMMA kernel is slower than cuBLAS GEMM (6.15ms ref). The custom GEMM with BK=32 and 128 threads achieves low occupancy and inefficient memory access. The small tile size means too many sync barriers and smem bandwidth is the bottleneck.
- **Next:** Try much larger tiles (BM=128, BN=128) with more warps, or use PTX `mma.sync` directly for finer control. Also consider a split approach: PyTorch mm + fused GELU+softmax kernel.

