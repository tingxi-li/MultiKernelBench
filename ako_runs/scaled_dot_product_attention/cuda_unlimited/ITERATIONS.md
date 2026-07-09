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

