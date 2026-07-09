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
| [prior] 1 | Flash-Attn2 fp16 BM=16 BN=32 | 1.6541x | 37.0 ms | improved |
| [prior] 2 | Autotune BM/BN/warps/stages | 1.6240x | 38.3 ms | no-change (regression) |
| [prior] 3 | allow_tf32=True + 8 warps + cache hints | 1.6373x | 37.5 ms | no-change (near BW wall) |
| [prior] 4 | D-tiled D_TILE=256, 4 separate dot calls | 1.7235x | 35.8 ms | improved |
| [prior] 5 | D_TILE=512, 2 chunks (larger K-dim for QKT) | 1.6877x | 36.5 ms | no-change (regression vs iter4) |
| 1 | BN=64 num_stages=2 (OOM) | N/A | N/A | failed (OOM) |
| 2 | BM=32 8warps (OOM) | N/A | N/A | failed (OOM) |

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

### Iter 2 — Autotune BM/BN/warps/stages

- **Hypothesis:** Autotuning over (BM in {8,16}, BN in {16,32}, num_warps in {2,4,8}, num_stages in {1,2}) would find a better config than the hand-picked iter-1 params.
- **Changes:** Added @triton.autotune with 16 configs over BM/BN/nwarps/nstages.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 38.3 ms (mean), 37.8~40.7 ms (min~max)
  - Speedup: 1.6240x
- **Analysis:** Slightly worse than iter 1 (37.0ms). The autotune overhead during warm-up phases and slightly less stable timing accounts for the difference. The best autotune config apparently chose BM=16, BN=32 (same as iter 1) but with different warps/stages that are slightly worse on this hardware. Autotune benchmark results are noisy and may have picked a suboptimal config.
- **Next:** Go back to fixed BM=16, BN=32, but try to optimize the inner loop: use tl.dot with allow_tf32=True for faster tensor cores, add num_stages=2 for better pipelining, or try fp16 accumulation inside the softmax-rescale path.

### Iter 3 — allow_tf32=True + 8 warps + cache hints

- **Hypothesis:** allow_tf32=True enables tensor core acceleration on Ada (TF32 mode). 8 warps doubles SM occupancy for better latency hiding. Cache hints (evict_last for Q, evict_first for K/V) optimize L1/L2 use.
- **Changes:** allow_tf32=True for both QK and pV dots. num_warps=8. eviction hints. Otherwise same as iter 1 (BM=16, BN=32).
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 37.5 ms (mean), 33.9~39.5 ms (min~max)
  - Speedup: 1.6373x
- **Analysis:** Essentially same as iter 1 (37.0ms). The kernel is near the theoretical memory-bandwidth ceiling: K/V read amplification = N/BM * N * D * 2B = 32*512*1024*2 = 33.5 MB per (b,h), total 34 GB at 960 GB/s = ~35ms theoretical minimum. We're at 37ms = ~95% of HBM bandwidth limit. allow_tf32 and 8 warps had negligible impact on this BW-bound workload.
- **Next:** Try a different algorithmic approach: split-K parallel flash attention (Flash-Decoding) where each CTA handles a subset of K/V tiles and partial results are merged. This could improve parallelism across SMs at the cost of a second reduction pass.

### Iter 4 — D-tiled Flash-Attention 2 (D_TILE=256, 4 separate dot calls)

- **Hypothesis:** Loading Q as 4 separate [BM=16, D_TILE=256] tiles (instead of one [16,1024] tile) reduces per-CTA register pressure from ~128 regs/thread to ~32 regs/thread for the Q tile portion. This allows more CTAs per SM (higher occupancy) and better latency hiding for HBM loads.
- **Changes:** Load Q as 4 separate fp16 tiles (q0..q3), each [BM, D_TILE=256]. Similarly tile K and V. QK accumulates sum of 4 partial dot products. Output accumulates in 4 separate [BM, D_TILE] fp32 arrays (a0..a3). Final 4 separate tl.store calls. Otherwise BM=16, BN=32, 4 warps, fp16 inputs.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 35.8 ms (mean), 31.8~37.6 ms (min~max)
  - Speedup: 1.7235x
- **Analysis:** Better than iter 1 (37.0ms) and iter 3 (37.5ms). The D-tiling reduces register pressure, allowing higher SM occupancy. With 4x D_TILE=256 sub-tiles: SMEM per K/V tile = 32*256*2 = 16 KB, total SMEM ≈ (16+16)*256*2=16 KB vs previous 96 KB. This leaves more L1/SMEM for thread context switching. Min latency of 31.8ms is 9% better than iter 1's 35.1ms min.
- **Next:** Try varying D_TILE (512, 128) or tuning BN to see if further improvement is possible. Also try 2 warps to reduce occupancy trade-off.

### Iter 2 (blind run) — BM=32 8 warps (OOM)

- **Hypothesis:** BM=32 doubles Q rows per CTA, halves K/V re-reads, better TC utilization for pV.
- **Changes:** BM=16→32, num_warps=4→8, BN=32, D_TILE=256.
- **Bench:**
  - Compiled: False (OOM)
  - Correct: N/A
  - Runtime: N/A
  - Speedup: N/A
- **Analysis:** SMEM required 133120B vs 101376B HW limit. BM=32 with 4 D-tiles × Q preloaded is too large for SMEM.
- **Next:** Keep BM=16 but try num_stages=2 with the working D_TILE=256 config. Or try Flash-Decoding (split-K over N dim).

### Iter 1 (blind run) — BN=64 num_stages=2 (OOM)

- **Hypothesis:** BN=64 reduces inner loop count from 16 to 8; num_stages=2 pipelining hides K/V load latency.
- **Changes:** BN=32→64, num_stages=1→2.
- **Bench:**
  - Compiled: False (OOM)
  - Correct: N/A
  - Runtime: N/A
  - Speedup: N/A
- **Analysis:** SMEM required 296960B vs 101376B HW limit. BN=64 × 4 D-tiles × fp16 exceeded SMEM budget.
- **Next:** Try BM=32 (larger Q-tile, fewer K/V re-reads per SM) with D_TILE=256 and 8 warps. Or try split-K decoding approach to parallelize over N dimension.

### Iter 5 (prior run) — D_TILE=512 (2 chunks, larger K-dim for QKT)

- **Hypothesis:** Halving the number of D-tiles (2 vs 4) reduces loop overhead for the D iteration. Larger D_TILE=512 means K-dim=512 for QKT dot products → better tensor core utilization. SMEM: 2*(16+32)*512*2 = 96 KB ✓.
- **Changes:** D_TILE=512 (was 256), so 2 separate q0/q1, k0/k1, v0/v1, a0/a1 tensors instead of 4. Otherwise same as iter 4.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 36.5 ms (mean), 34.7~38.7 ms (min~max)
  - Speedup: 1.6877x
- **Analysis:** Slightly worse than iter 4 (35.8ms). With D_TILE=512, the register footprint for q0,q1,a0,a1 increases: 2*16*512 fp16 + 2*16*512 fp32 = 64K reg elements → more register pressure. The 4-tile approach (D_TILE=256) gives better balance between D-loop overhead and register pressure.
- **Next:** Iter 4 approach (D_TILE=256, 4 tiles, BN=32) is the best so far. Try adding BM=32 with D_TILE=128 (8 tiles) to give more Q-rows per CTA while keeping SMEM small.

