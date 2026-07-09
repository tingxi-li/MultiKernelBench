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
| 1 | Custom Triton depthwise kernel (autotune) | 1.12x | 3.25 ms | improved |
| 2 | Refined autotune configs, constexpr N/C | 1.35x | 2.68 ms | improved |
| 3 | More configs, num_stages=3/4, tl.fma | 1.30x | 2.79 ms | regression |
| 4 | Channel batching (BLOCK_NC), focused config space | 1.35x | 2.68 ms | no-change |
| 5 | Row-per-CTA: 1D tile along W only | 1.30x | 3.14 ms | regression |
| 6 | 2D tile with NC on grid axis 0 (clean refactor) | 1.37x | 2.67 ms | improved |

## Iterations

### Iter 1 — Custom Triton depthwise kernel with autotune

- **Hypothesis:** cuDNN's depthwise conv is suboptimal; a custom Triton kernel that tiles output spatially and uses @triton.autotune should beat the reference.
- **Changes:** Replaced identity solution with a Triton depthwise conv kernel. Grid=(N*C, ceil(H_out/BLOCK_H), ceil(W_out/BLOCK_W)), BLOCK_H/BLOCK_W autotuned across 13 configs. Unrolled KH×KW inner loop. Accumulates in fp32.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 3.25 ms (mean), 3.19 ~ 4.83 ms (min ~ max)
  - Speedup: 1.12x
- **Analysis:** First pass beats cuDNN by 12%. The BLOCK_H=4, BLOCK_W=64 or similar tile configs are likely winning. The kernel is memory-bound (depthwise convs have very low arithmetic intensity). Room to improve by better memory access patterns and coalescing.
- **Next:** Improve memory coalescing — try packing output rows or using wider loads. Consider having one warp handle a full channel row to maximize coalescing. Also try vectorized loads (float4) if the output width aligns.

### Iter 2 — Refined autotune configs with N, C as constexpr

- **Hypothesis:** Making N and C constexpr allows better register allocation and specialization. Refined tile configs to favor wider OW tiles for better coalescing.
- **Changes:** Refactored kernel with BLOCK_OH/BLOCK_OW naming, made N and C constexpr. Added wider OW configs (256, 512). Cleaner grid computation.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.68 ms (mean), 2.62 ~ 3.79 ms (min ~ max)
  - Speedup: 1.35x
- **Analysis:** Significant improvement (+0.57ms, +0.23x). The constexpr N/C and better configs allowed triton to generate better code. 1.35x > the target "beat PyTorch" threshold.
- **Next:** Try further improvements: (1) vectorized float4 loads along W dimension, (2) preload kernel weights into registers before the spatial loop, (3) try persistent kernels that avoid reloading weights.

### Iter 3 — More autotune configs, num_stages=3/4, tl.fma

- **Hypothesis:** More configs + software pipelining (num_stages=3/4) + tl.fma should improve performance.
- **Changes:** Expanded configs to 17, added num_stages=3/4 variants, used tl.fma instead of multiply-add.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.79 ms (mean), 2.64 ~ 4.89 ms (min ~ max)
  - Speedup: 1.30x
- **Analysis:** Regression from iter-2 (1.35x → 1.30x). Higher num_stages may consume more SRAM and reduce occupancy. More configs slow autotuning and the best config may differ. tl.fma likely has no effect (triton does fused ops anyway). The extra variance (std 0.288) suggests the autotuner picked a worse tile.
- **Next:** Revert to iter-2 config space but try adding nc-level parallelism or try channel batching to reduce launch overhead.

### Iter 4 — Channel batching (BLOCK_NC) with focused config space

- **Hypothesis:** Processing multiple (n,c) slices per CTA (BLOCK_NC=2,4) could improve weight cache reuse and reduce launch overhead. Focused configs around the iter-2 winners.
- **Changes:** Added BLOCK_NC dimension, focused config set, removed num_stages>2, fixed break issue (triton doesn't support break).
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.68 ms (mean), 2.60 ~ 3.82 ms (min ~ max)
  - Speedup: 1.35x
- **Analysis:** Matches iter-2 exactly. Channel batching doesn't help — the sequential loop over BLOCK_NC within a CTA reduces parallelism. The kernel is throughput-limited by memory bandwidth, not by launch overhead.
- **Next:** Try a completely different algorithmic approach: use shared memory to cache the input tile and reuse it across kernel positions. This could significantly reduce L2 traffic.

### Iter 5 — Row-per-CTA: 1D tile along W only

- **Hypothesis:** One CTA per output row maximizes coalescing and reduces register pressure (1D vs 2D tile).
- **Changes:** New row-based kernel with grid (NC, H_out, ceil(W_out/BLOCK_W)). 1D accumulator.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 3.14 ms (mean), 2.71 ~ 4.42 ms (min ~ max)
  - Speedup: 1.30x
- **Analysis:** Worse than iter-2 (1.35x). The grid is NC*H_out = 16*64*510 = ~522K blocks which adds launch overhead. The 2D tiling in iter-2 (BLOCK_OH*BLOCK_OW) is better — it keeps more work per block.
- **Next:** This is iter 5 of 6. Iter-2 is still best at 1.35x. For iter-6, try a fundamentally different approach: process a larger spatial tile to maximize arithmetic intensity, or try using the weight as a constexpr to enable Triton to unroll/specialize more aggressively.

### Iter 6 — 2D tile with NC on grid axis 0 (clean refactor, focused configs)

- **Hypothesis:** Going back to 2D tiling with NC on first grid axis but focused config set based on learnings. The iter-2 approach is the best template; small variations in config set or kernel body may yield marginal gain.
- **Changes:** Clean kernel with NC on grid axis 0, BLOCK_OH/BLOCK_OW 2D tile. Focused on config set with BLOCK_OW ≥ 64, BLOCK_OH in [2, 32]. Static range for KH/KW unrolling.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2.67 ms (mean), 2.60 ~ 3.82 ms (min ~ max)
  - Speedup: 1.37x
- **Analysis:** New best! 1.37x > iter-2's 1.35x. The focused config set allowed the autotuner to find a slightly better tile size. This is the best achieved.
- **Next:** Iter cap reached. Restore iter-6 as best.






