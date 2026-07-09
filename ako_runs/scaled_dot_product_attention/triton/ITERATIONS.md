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
| 1 | Flash-Attn2 fp16 BM=16 BN=32 | 1.6541x | 37.0 ms | improved |

## Iterations

### Iter 1 — Flash-Attention 2 with fp16 tiles (BM=16, BN=32)

- **Hypothesis:** HEAD_DIM=1024 exceeds Flash-Attention's D<=256 limit, so PyTorch SDPA falls back to memory-efficient attention (~61ms). A fused Flash-Attention 2 kernel can eliminate the O(N^2) attention matrix materialization. With fp16 inputs, Q+K tile (16+32)*1024*2=96KB fits in shared memory (<101KB hardware limit). tl.dot K-dims: 1024 (QK) and 32 (pV), both >=16.
- **Changes:** Complete rewrite from identity to Flash-Attention 2 Triton kernel. Inputs cast to fp16 for SMEM efficiency; QK accumulation and softmax in fp32; output in fp32. BM=16, BN=32, num_stages=1, num_warps=4.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 37.0 ms (mean), 35.1~39.1 ms (min~max)
  - Speedup: 1.6541x
- **Analysis:** The flash-fusion eliminates materialization of the [32,32,512,512] attention weight tensor (~1GB), cutting HBM bandwidth. fp16 inputs allow larger block sizes than fp32-only would permit. Correct within fp32 tolerance (1e-4) thanks to fp32 softmax arithmetic and fp32 output.
- **Next:** Try larger blocks (BN=64 or BM=32 via smem tricks), more warps, or better pipelining to improve IPC. Try autotune to find optimal BM/BN.

