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
| 1 | Flash Attn warp-per-row (no smem) | 0.034x | 2520 ms | regression |
| 2 | smem-tiled FA2 (Br=16, Bc=16, DC=64) | ~0.16x | 363 ms (fast) | regression |
| 3 | wmma FA2 (Br=64, Bc=64, 4 warps) | 0.17x | 397 ms | regression |
| 4 | 3-kernel unfused: QKT+softmax+PV (fp32 S) | 0.71x | 84.5 ms | improved |
| 5 | 3-kernel wmma fixed bh offset, fp16 inputs | 0.75x | 82.4 ms | improved |
| 6 | 3-kernel wmma fp16, Bc=64 (64×64 output tile) | 1.12x | 52.7 ms | improved |

## Iterations

### Iter 1 — Flash Attn warp-per-row (no smem caching)

- **Hypothesis:** One warp (32 threads) per Q row, accumulate QKV with online softmax; warp-reduce dot products.
- **Changes:** Replaced identity with a CUDA Flash Attention kernel using 1 warp per Q row, EPL=32 (D/WARP_SIZE), iterating over KV tiles of Bc=64 with global-memory loads each iteration.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 2520 ms (mean), 1510 ~ 3280 ms (min ~ max)
  - Speedup: 0.034x
- **Analysis:** Catastrophically slow. Inner loop loads K and V directly from global memory for each KV row, with no smem caching. For each Q row: N=512 K-row loads + N=512 V-row loads = 1024 * 4KB = 4MB of uncoalesced global reads sequentially. Also huge register pressure (~100 regs/thread). Need proper smem-tiled approach.
- **Next:** Proper Flash Attention 2 with smem K/V tiles. All WPB warps cooperate to load Bc*D K/V rows into smem once per KV tile, then each warp computes its Q row's scores and O update against smem.

### Iter 2 — smem-tiled FA2 (Br=16, Bc=16, D-chunk=64, no mma)

- **Hypothesis:** Cooperative smem loading with D-tiled QK^T and PV accumulation
- **Changes:** Block (Bc,Br)=(16,16) threads, D processed in DC=64 chunks, global output accumulation
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 363 ms (fast signal, 20 trials)
  - Speedup: ~0.16x (fast signal only)
- **Analysis:** Still very slow. The inner O update loop is serial per-warp (lane==0 for normalization), and the D-chunked approach with Bc=16 generates too many K/V tile iterations. Need tensor cores for the QK^T and PV mma steps.
- **Next:** Use wmma (mma.sync) for QK^T and PV. Br=64, Bc=64, 4 warps handle 16 Q rows each.

### Iter 3 — WMMA FA2 (Br=64, Bc=64, 4 warps, serial softmax)

- **Hypothesis:** wmma tensor cores for QK^T and PV will dramatically improve compute throughput
- **Changes:** 4 warps, 2x2 warp tile layout on 64x64 score matrix, wmma 16x16x16 for QK^T, serial softmax per warp, serial O rescale
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 397 ms (mean)
  - Speedup: 0.17x
- **Analysis:** wmma computation IS happening but serial softmax loops (O rescale lane==0, normalize lane==0) serialize most lanes. Also __float2half() per element during K/V load from global is expensive. Several subsequent variants tried: warp-per-row with register O (all slower due to poor occupancy or non-coalesced access or bank conflicts). Best fast-signal result was 363ms for the iter-2 DC=64 smem tiling.
- **Next:** Unfused approach: GEMM for QK^T + pointwise softmax + GEMM for PV. Use 3 separate optimized kernels.

### Iter 4 — 3-kernel unfused: wmma QKT + row softmax + wmma PV (fp32 S)

- **Hypothesis:** Unfused 3-kernel approach using wmma for QKT and PV, fp32 S matrix
- **Changes:** Kernel 1: batched wmma QKT (1 warp/16x16 tile); Kernel 2: row softmax; Kernel 3: wmma PV with on-the-fly P fp32->fp16 conversion via smem
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 84.5 ms (mean), 81.5 ~ 86.6 ms (min ~ max)
  - Speedup: 0.71x
- **Analysis:** Closest to target (57ms) so far. S[BH,N,N] = 1024*512*512*4 = 1GB is the bottleneck - writing and reading 1GB kills bandwidth. fp16 S would cut to 512MB saving ~14ms of bandwidth.
- **Next:** Use fp16 for S matrix (512MB instead of 1GB), saving ~half S bandwidth.

### Iter 5 — 3-kernel wmma fixed bh offset + fp16 inputs

- **Hypothesis:** Fix bh offset bug, use wmma fp16 for QKT/PV, fp32 softmax for correctness
- **Changes:** Added `blockIdx.z` bh offset to both GEMM kernels; 4 warps per block (64x16 output tile)
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 82.4 ms (mean), 67.8 ~ 153 ms (min ~ max)
  - Speedup: 0.75x
- **Analysis:** Best result so far (0.75x). Still below target (1.0x). Main bottlenecks: fp32→fp16 conversion (~22ms) + S matrix bandwidth (2.3ms). Total overhead ~25ms on top of compute. Need fused Flash Attention to eliminate S matrix.
- **Next:** Fused Flash Attention 2 with wmma tiles: Br=Bc=16, D-tiled QKT+PV, online softmax. Avoids writing S to global memory.

### Iter 6 — 3-kernel wmma fp16, Bc=64 (64×64 output tile per block)

- **Hypothesis:** Larger Bc=64 (vs Bc=16 in iter-5) gives each block a 64×64 output tile; 4 warps each cover 16 rows × 64 cols (4 column tiles). Reduces grid size by 4× in K dimension, better SM utilization per block, more wmma ops per smem load.
- **Changes:** QKT grid=(N/64, N/64, BH) with 4 wmma accumulators per warp; PV same. smem: Qs[64][64]+Ks[64][64] fp16 = 16KB. Keeps fp32 S matrix (correct). Same fp32→fp16 conversion overhead for Q/K/V inputs.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 52.7 ms (mean), 52.1 ~ 53.7 ms (min ~ max)
  - Speedup: 1.12x (WINS)
- **Analysis:** Beats PyTorch reference (58.8ms). The 4× larger tiles dramatically reduce kernel launch overhead and improve warp utilization. With Bc=64 each warp does 4 wmma accumulations per smem load instead of 1. The grid is 4× smaller (256 vs 1024 blocks for QKT), reducing scheduler overhead.
- **Next:** This is the best result (1.12x). Proceed to [final].

