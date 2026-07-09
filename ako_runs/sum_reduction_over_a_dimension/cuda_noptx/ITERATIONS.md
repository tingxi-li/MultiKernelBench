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
| 1 | float4 loads + 8-unroll | 1.006x | 9.73 ms | improved |
| 2 | 2x float4 per thread + 2 accumulators | 1.008x | 9.71 ms | improved |
| 3 | 2 float4/thread + 8-unroll + BLOCK_X=128 | 1.008x | 9.71 ms | no-change |
| 4 | ptr-walk single float4/thread BLOCK_X=256 | 1.008x | 9.71 ms | no-change |

## Iterations

### Iter 1 — float4 vectorized loads + 8-step unroll

- **Hypothesis:** The operation is bandwidth-bound. Using float4 loads (128-bit transactions) with 8-step loop unrolling increases ILP and hides memory latency, improving effective bandwidth utilization.
- **Changes:** Replaced identity PyTorch implementation with a custom CUDA kernel using float4 loads over the C dimension, 8-step unrolled loop over D dimension with __ldg hints for read-only cache.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 9.73 ms (mean), 9.72 ~ 9.75 ms (min ~ max)
  - Speedup: 1.006x
- **Analysis:** Marginal improvement. PyTorch's sum is already near the bandwidth roofline. The float4 read pattern helps a little, but PyTorch's kernel is also well-optimized. Need to explore different block/grid configurations or warp-level parallelism.
- **Next:** Try 2 float4s per thread to improve ILP and reduce grid overhead.

### Iter 2 — Two float4 accumulators per thread + BLOCK_X=256

- **Hypothesis:** Processing 2 consecutive float4 positions per thread doubles work per thread (reduces grid overhead) and has 2 independent accumulator chains for ILP.
- **Changes:** Each thread handles c4_base and c4_base+1 positions, accumulating two independent float4 chains per D-row. Grid halved.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 9.71 ms (mean), 9.71 ~ 9.72 ms (min ~ max)
  - Speedup: 1.008x
- **Analysis:** Marginal improvement over iter-1. We are at the memory bandwidth limit (8.59GB / 900GB/s ≈ 9.54ms). PyTorch and our kernel are both near the ceiling. The float4 coalescing and ILP provide only tiny wins.
- **Next:** Try ptr-walk pattern (pointer+stride increment) to help compiler optimize.

### Iter 3 — 2 float4/thread + 8-unroll + BLOCK_X=128

- **Hypothesis:** Larger 8-step unroll + smaller BLOCK_X (higher occupancy) should allow the GPU to hide more memory latency.
- **Changes:** BLOCK_X=128, explicit 8-step unroll of the D-loop with all 16 loads in flight before adds. Two float4 per thread retained.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 9.71 ms (mean), 9.71 ~ 9.71 ms (min ~ max)
  - Speedup: 1.008x
- **Analysis:** No improvement vs iter-2. We're firmly at the memory bandwidth ceiling. The kernel is already reading data at near-peak bandwidth. RTX 6000 Ada peak: ~900 GB/s, this op reads 8.59 GB -> 9.54 ms minimum. We're at 9.71ms, only 1.8% above theoretical minimum.
- **Next:** Try pointer-walk single float4/thread to minimize register pressure.

### Iter 4 — Pointer-walk single float4/thread, BLOCK_X=256, __launch_bounds__

- **Hypothesis:** Simplest possible kernel: single float4 per thread, pointer walk `ptr += C4`, minimal register usage. Let the hardware memory subsystem handle all ILP.
- **Changes:** Single float4 per thread, `for (; ptr < end; ptr += C4)` loop. __launch_bounds__(256). No unroll pragma.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 9.71 ms (mean), 9.71 ~ 9.72 ms (min ~ max)
  - Speedup: 1.008x
- **Analysis:** Consistent 9.71ms = 1.008x vs 9.79ms ref. We are firmly at the memory bandwidth ceiling (theoretical: 9.54ms at 900GB/s). All kernel variants that correctly implement float4 reads converge to 9.71ms. The 0.08ms gap to theory is due to GPU core latency, L2 cache miss overhead, and memory controller overhead.
- **Next:** Try processing the problem as B parallel matrix-vector ops using cublasSgemv, or try a grid-stride pattern for better SM utilization.




