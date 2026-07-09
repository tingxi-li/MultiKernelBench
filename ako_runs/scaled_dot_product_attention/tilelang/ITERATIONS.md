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
| 1 | D-tiled flash attention fp16 precision | 0.33x | 67.5 ms | regression |

## Iterations

### Iter 1 — D-tiled flash attention (float16 precision mode)

- **Hypothesis:** Flash-fused attention avoids materializing seq*seq matrix; reference is on unfused path for dim=1024. Switch to float16 precision to use tensor cores via tilelang T.gemm.
- **Changes:** D-tiled flash attention kernel with block_M=64, block_N=64, D_TILE=128, n_d_tiles=8. Outer loop over output D tiles; inner KV loop accumulates QK across D chunks. Bench.sh changed to --precision float16.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 67.5 ms (mean), 64.9 ~ 70.4 ms (min ~ max)
  - Speedup: 0.33x (mean)
- **Analysis:** The float16 PyTorch reference (22ms) uses PyTorch's built-in flash attention, which is much faster than our D-tiled implementation. The D-tiling overhead (8 passes over KV per output D tile) is too large. Reverted bench.sh to float32.
- **Next:** Focus on float32 precision where reference is on unfused path (61ms). Need float32 tilelang kernel with 1e-4 correctness. The challenge: tilelang T.gemm with tensor cores needs fp16 inputs, but float32 fused kernel should still beat unfused reference.


