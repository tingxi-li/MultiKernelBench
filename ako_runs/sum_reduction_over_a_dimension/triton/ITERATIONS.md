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
| 1 | Triton autotune streaming reduction | 1.0072x | 9.72 ms | improved |

## Iterations

### Iter 1 — Triton autotune streaming reduction

- **Hypothesis:** Replace PyTorch eager sum with a hand-written Triton kernel that uses @triton.autotune to pick the best BLOCK_C (64–4096) and num_warps/num_stages combination for the (128, 4096, 4096) → (128, 1, 4096) reduction over dim=1. Each program handles BLOCK_C contiguous columns and loops over all 4096 reduction rows.
- **Changes:** Full rewrite from identity (torch.sum) to Triton kernel with autotune over BLOCK_C, num_warps, num_stages. Float32 accumulator for correctness. Row-unrolling by 4 to hide memory latency. Fallback to torch.sum for non-standard dims/shapes.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 9.72 ms (mean), 9.71 ~ 9.72 ms (min ~ max)
  - Speedup: 1.0072x
- **Analysis:** The op is bandwidth-bound at ~884 GB/s = 92% of the 960 GB/s peak on RTX 6000 Ada. Autotune selected BLOCK_C=4096 with appropriate num_warps/num_stages. The Triton kernel is slightly faster than PyTorch's torch.sum (9.72 vs 9.79 ms) because it avoids some library dispatch overhead. Manual exploration showed the floor is ~9.71 ms regardless of BLOCK_C, unrolling, or pipeline depth — all configs converge to ~884 GB/s, very close to hardware peak.
- **Next:** Try alternative tiling: BLOCK_C smaller (more parallelism), split-K two-pass for even more SM saturation, or look at float4/vectorized loads to reduce instruction overhead.

