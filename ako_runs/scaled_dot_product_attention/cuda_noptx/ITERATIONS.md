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
| 1 | Flash Attn FP32 BM=8 BN=8 | 0.14x | 474 ms | regression |
| 2 | 2-pass: all-scores-in-regs then V | — | — | — |

## Iterations

### Iter 1 — Flash Attention FP32 BM=8 BN=8 grid-split

- **Hypothesis:** Flash-attention fused kernel avoids materializing the S×S attention matrix to DRAM, saving ~2GB memory bandwidth vs reference's mem_efficient path.
- **Changes:** Replaced identity (PyTorch SDPA) with custom flash-attention CUDA kernel. BM=8 query rows per block, BN=8 KV rows per smem tile, 64KB smem. Grid (1024, 64), Block (32,8).
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 474 ms (mean), 467~478 ms (min~max)
  - Speedup: 0.14x (mean)
- **Analysis:** The kernel is correct but 7x slower than the reference. Root cause: for D=1024 ≥ S=512, flash attention reads K/V S/BM=64 times per head. Total K+V reads = 64×4MB×1024 heads = 256GB vs reference's ~12GB (tensor-core matmul reads Q/K/V once). The reference uses PyTorch `_efficient_attention_forward` (xformers mem-efficient) backed by tensor-core matmuls (~100 TFLOP/s), while our scalar FP32 kernel achieves ~20 TFLOP/s. The flash-attention optimization (avoiding S×S DRAM) saves ~2GB but costs ~244GB extra K/V reads.
- **Next:** Try 2-pass kernel: compute all S dot products for one Q row, hold scores in registers (16 regs per thread × 32 threads = 512 scores), normalize once, then multiply V. This reads K once (not 64 times), reducing K read traffic by 64×. V is still read 64 times per K tile... no. Actually 2-pass reads K once and V once per Q row. Total: same as reference. But eliminates S×S DRAM materialization.

