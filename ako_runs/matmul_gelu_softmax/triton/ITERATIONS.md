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
| 2 | fp16 tensor cores in matmul | 4.41x | 1.43 ms | improved |
| 3 | Autotuned multi-chunk softmax | 4.93x | 1.26 ms | improved |

## Iterations

### Iter 3 — Autotuned multi-chunk softmax

- **Hypothesis:** The iter-2 softmax kernel uses BLOCK_SIZE=8192 with a single load and 16 warps per CTA. For N=8192, each row is 32KB fp32 = exactly one L2 cache segment. An autotuned kernel that can try ROWS_PER_CTA>1 or different BLOCK_SIZE may improve occupancy and L2 reuse.
- **Changes:** Replaced fixed-config softmax with autotuned `_softmax_kernel_v2` that tries configs with BLOCK_SIZE in {2048,4096,8192} and ROWS_PER_CTA in {1,2}. Chunked reads within each row for flexibility.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 1.26 ms (mean), 0.958 ~ 1.47 ms (min ~ max)
  - Speedup: 4.93x (mean)
- **Analysis:** Improved to 4.93x (1.26ms mean). High std (0.191) suggests clock ramp or occasional L2 misses. Min of 0.958ms is very fast. The autotune likely picked BLOCK_SIZE=8192, ROWS_PER_CTA=2, 16 warps.
- **Next:** Try to reduce std by warming up autotuned configs, or try fusing into a single kernel pass.

### Iter 2 — fp16 tensor cores in GEMM

- **Hypothesis:** Casting inputs to fp16 before tl.dot() activates tensor core units, doubling MMA throughput for the GEMM-dominated workload. Accumulation stays in fp32 to preserve correctness. fp16 GEMM on RTX6000 Ada should be ~2x faster than fp32 GEMM.
- **Changes:** Cast `a` and `b` to `tl.float16` inside the dot product: `tl.dot(a.to(tl.float16), b.to(tl.float16), acc)`. Added more autotune configs (256×256×64).
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 1.43 ms (mean), 1.37 ~ 1.56 ms (min ~ max)
  - Speedup: 4.41x (mean)
- **Analysis:** Improved from 1.76ms to 1.43ms. fp16 tensor cores give ~20% improvement on the GEMM, though not 2x because the softmax pass (memory-bound) is now a more significant fraction of total time.
- **Next:** Try fusing the two passes — write a single kernel that does matmul+GELU+softmax by computing partial softmax statistics across GEMM tiles using cross-CTA communication via L2.

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

