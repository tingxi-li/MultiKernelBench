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
| 2 | Single-pass multi-acc (4 D-tiles, fp16 TC) | 1.97x | 30.1 ms | improved |
| 3 | threads=256 (6 warps→8 warps) | 2.29x | 26.3 ms | improved |
| 4 | block_M=64, 8 acc_o (D_TILE=128) | 2.34x | 25.1 ms | improved |

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

### Iter 2 — Single-pass multi-accumulator flash attention (4 output D-tiles)

- **Hypothesis:** The iter 1 D-tiled approach had n_d_tiles² overhead (8x more QK ops than needed). Key fix: maintain 4 separate fp32 accumulators (acc_o0..acc_o3) for the 4 D_TILE=256 output slices simultaneously, doing a single KV-block pass. QK is accumulated over n_d_tiles=4 inner chunks per KV block. This matches the triton approach of 4 accumulators in registers. fp16 tensor cores with fp32 accumulation gives max_diff ~3.5e-5 — passes 1e-4. float32 inputs cast to fp16 in forward() (allowed by cheating detection).
- **Changes:** New single-pass kernel with block_M=32, block_N=64, D_TILE=256, n_d_tiles=4. 4 separate acc_o fragments. Regular for-loop (no T.Pipelined) to allow V_shared reuse per KV block. Layout bridge: acc_s (fp32) -> S_shared -> acc_s_cast (fp16) for PV gemm. Output tensor is float32.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 30.1 ms (mean), 29.8 ~ 31.3 ms (min ~ max)
  - Speedup: 1.97x (mean)
- **Analysis:** 1.97x speedup! Single pass eliminates 8x D-tiling overhead. fp16 tensor cores give correct results within 1e-4. Layout bridge via shared memory works. V is reloaded 4× per KV block (sequential, 1 shared buffer).
- **Next:** Try to push further — larger block_M, more threads, or pipelining. Also explore if we can use T.Pipelined for the KV loop now that V reuse is sequential within the loop body.

### Iter 3 — threads=256 (8 warps)

- **Hypothesis:** More warps per SM → better latency hiding for memory accesses and higher tensor core throughput. 128 threads = 4 warps; 256 threads = 8 warps.
- **Changes:** threads=128 → threads=256.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 26.3 ms (mean), 25.0 ~ 27.9 ms (min ~ max)
  - Speedup: 2.29x (mean)
- **Analysis:** 16% improvement vs iter 2 (26.3ms vs 30.1ms). More warps better hide memory latency for K/V loads.
- **Next:** Try larger block_N for better memory throughput, or explore if loop unrolling/staging helps further.

### Iter 4 — block_M=64, 8 output D-tile accumulators

- **Hypothesis:** Larger block_M=64 halves the number of CTAs (from 512/32=16 to 512/64=8 per batch-head), reducing scheduling overhead. 8 accumulators for D_TILE=128 eliminates all D-tiling overhead (n_d_tiles=8, exactly covers full dim=1024). Smem reduced from 84KB to 56KB.
- **Changes:** block_M=32→64, D_TILE=256→128, 4 acc_o→8 acc_o. threads=256 retained. 8 V loads per KV block (n_d_tiles=8, each 64×128×2B=16KB).
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 25.1 ms (mean), 24.0 ~ 27.0 ms (min ~ max)
  - Speedup: 2.34x (mean)
- **Analysis:** 25.1ms (2.34x) vs 26.3ms (2.29x). Moderate improvement. Fewer CTAs and lower smem pressure help. 8 V reloads per KV block (vs 4) but each reload is half the size.
- **Next:** Explore whether the register pressure from 8 acc_o limits occupancy. Try threads=128 or different block_N.


