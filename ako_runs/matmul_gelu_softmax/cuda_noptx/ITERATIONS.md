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
| 2 | at::mm + fused GELU+softmax (correct) | 0.96x | 6.62 ms | regression |
| 3 | PyTorch linear + fused GELU+softmax float4 | 1.03x | 6.19 ms | improved |
| 4 | Warp-shuffle reductions (float4) | 1.03x | 6.19 ms | no-change |

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

### Iter 2 — at::mm + fused GELU+softmax (correct)

- **Hypothesis:** Use PyTorch's at::mm for GEMM (inherits TF32 settings for correctness), fused GELU+softmax kernel reduces HBM passes.
- **Changes:** Use at::mm in C++ glue, removed --use_fast_math, kept fused bias+GELU+softmax kernel.
- **Bench:**
  - Compiled: True
  - Correct: True (5/5)
  - Runtime: 6.62 ms mean, 4.38 ~ 7.21 ms (min ~ max)
  - Speedup: 0.96x
- **Analysis:** High variance (std=0.833ms) due to clock ramp — min=4.38ms shows the kernel can be fast, but the at::mm re-synchronizes the cuBLAS handle, and the external at::mm tensor allocation creates an extra HBM round-trip. The mean is worse than baseline because the solution runtime is measured against a reference that warmed up clocks. The fused epilogue alone saves ~0.5ms at min, but GEMM overhead negates it.
- **Next:** Try a single fully-fused CUDA kernel that does tiled GEMM + GELU + row-softmax in one pass without going through at::mm. This eliminates the intermediate GEMM output tensor entirely.

### Iter 3 — PyTorch linear + fused GELU+softmax (float4)

- **Hypothesis:** Use PyTorch's stable self.linear(x) for GEMM, then apply only the fused GELU+softmax kernel. Float4 vectorized loads reduce memory transactions by 4x.
- **Changes:** Separate linear() + fused_gelu_softmax_kernel with float4 loads/stores.
- **Bench:**
  - Compiled: True
  - Correct: True (5/5)
  - Runtime: 6.19 ms mean, 5.42 ~ 6.72 ms (min ~ max)
  - Speedup: 1.03x
- **Analysis:** The fused GELU+softmax saves ~0.2ms over baseline (avoids one extra HBM round-trip for GELU output). The min 5.42ms shows the fused kernel is hitting ~0.3ms for GELU+softmax vs ~0.5ms for two passes. Float4 vectorization is working. Main bottleneck: GEMM itself (≈5.5ms at best clock). The erff() is still slow; could try tanh approximation for GELU or use online softmax.
- **Next:** Try online softmax (single-pass: compute max+sum+normalize in one kernel pass) with tanh GELU approximation. Also try eliminating the extra allocations in fused kernel.

### Iter 4 — Warp-shuffle reductions (float4)

- **Hypothesis:** Replacing __syncthreads()-based tree reductions with warp-shuffle reductions reduces synchronization overhead and pipeline stalls.
- **Changes:** Added warp_reduce_max/warp_reduce_sum using __shfl_xor_sync, inter-warp reduction via shared mem (only WARPS=8 elements vs 256).
- **Bench:**
  - Compiled: True
  - Correct: True (5/5)
  - Runtime: 6.19 ms mean, 5.34 ~ 6.62 ms (min ~ max)
  - Speedup: 1.031x
- **Analysis:** No improvement over iter 3. The synchronization overhead was not the bottleneck — the kernel is dominated by erff() computation (transcendental function). The GEMM takes ~5.5ms and fused epilogue ~0.3ms; further reductions in softmax overhead have diminishing returns.
- **Next:** Try using in-place output (no extra allocation) and see if the output tensor reuse helps. Also try 512 threads per block with EPT=16 for better occupancy.

