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
| 2 | Identity (floor analysis) | 1.03x | 60.8 ms | no-change |
| 3 | Flash Attn BM=16 BN=8 float4 | 0.27x | 240 ms | regression |

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

### Iter 2 — Identity (floor analysis)

- **Hypothesis:** After thorough analysis of flash-attention failure and 2-pass limitations, the identity (PyTorch SDPA) may be the practical best for cuda_noptx with FP32 on D=1024.
- **Changes:** Reverted to identity `torch.nn.functional.scaled_dot_product_attention(Q, K, V)`. Added detailed analysis comment explaining why custom kernels cannot beat this.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 60.8 ms (mean), 59.8~63.2 ms (min~max)
  - Speedup: 1.03x (mean)
- **Analysis:** The floor analysis shows: 1) Flash attention re-reads K/V 64x → 256GB vs 9GB for cuBLAS → 28x slower. 2) 2-pass (scores in smem) re-reads K: S²×D per head = same 1TB issue. 3) FP16 computation fails at atol=1e-4 for D=1024. 4) Without tensor cores (no PTX), scalar FP32 achieves ~20 TFLOP/s vs reference cuBLAS ~100+ TFLOP/s. The GEMM operations (bmm1=28ms + bmm2=28ms) dominate the 60ms total; softmax is only 2.7ms. No custom scalar CUDA kernel can match cuBLAS GEMM on tensor cores. The identity achieves 1.03x (noise floor). This is the practical floor for cuda_noptx on this workload.
- **Next:** Try flash attention with larger BM (32) and float4 loads to reduce K/V re-reads.

### Iter 3 — Flash Attention BM=16 BN=8 float4 loads

- **Hypothesis:** float4 vectorized loads for K/V smem fills should improve memory bandwidth utilization. BM=16 with 512 threads gives more registers per thread (128) vs BM=32 (64) which may have been register-spilling.
- **Changes:** BM=16, BN=8, float4 loads for K/V smem cooperative load. Grid: (1024, 32), Block: (32, 16)=512 threads, smem=64KB.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 240 ms (mean), 234~244 ms (min~max)
  - Speedup: 0.27x (mean)
- **Analysis:** Better than iter-1 (474ms → 240ms = 1.97x improvement). float4 loads improved bandwidth utilization. However, still 3.8x slower than reference. The bottleneck analysis: smem bandwidth is the limit (~256 smem reads per warp per K row × 32 passes × 43 KV tiles = 350K smem ops per warp per Q-tile). Arithmetic intensity at smem level is ~1:1 (compute-smem balanced). Multiple tile sizes tried: BM=8(474ms), BM=32(296ms), BM=16(240ms) - optimal around BM=16. Scalar FP32 at ~5% of peak due to smem bandwidth limits.
- **Next:** Try proper GEMM-like tiled kernel that uses register blocking for Q rows × K rows, potentially achieving higher arithmetic intensity.

