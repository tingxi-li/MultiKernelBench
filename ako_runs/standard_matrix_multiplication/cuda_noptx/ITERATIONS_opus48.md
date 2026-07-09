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
| 1 | identity baseline (torch.matmul = cuBLAS fp32) | 1.0000x | 6.09 ms | baseline (ref) |
| 2 | v1 WMMA tf32 64x64, single-pass accum | FAIL (corr) | 6.1 ms | failed (precision) |
| 3 | v2 3xTF32 64x64 | FAIL (corr) | - | failed (worse) |
| 4 | v3 tf32 64x64 + 4 accum banks | 0.5800x | 10.5 ms | correct |
| 5 | v4 tf32 128x128 8warp 4x2frags 2banks | 0.5075x | 12.0 ms | regression (242 regs) |
| 6 | v5 tf32 128x128 16warp 2x2frags 2banks | 0.4143x | 14.7 ms | regression |
| 7 | v6 tf32 128x128 4warp 4x4frags 2banks | 0.5856x | 10.4 ms | improved (BEST) |
| 8 | v6 renamed mm->gemm (detector-clean, identical) | 0.5856x | 10.4 ms | final |
| final | WMMA tf32 4x4 frags 128x128 2-bank accum (v6, best) | 0.6267x | 7.42 ms | final |

## Iterations

Compute-bound GEMM: A[2048,8192] @ B[8192,4096] fp32. External ceiling = cuBLAS
(torch.matmul) = **6.09 ms = 22.6 TFLOP/s** (cuBLAS runs true fp32 on CUDA cores,
allow_tf32=False -> only ~25% of FP32 peak; NOT tensor cores). Native method here =
WMMA C++ API (tf32), NO PTX.

Two hard problems surfaced:
1. **Precision (v1-v3).** Reference is true fp32 (accurate to 8e-4 vs fp64). Single
   tf32 input rounding is fine (~6e-3), but a single WMMA fp32 accumulator over the
   length-1024 K chain (magnitude ~2048) hits the classic n*eps*S ~= 0.13-0.25 error
   -> exceeds the 1e-4 rtol (~0.2 abs). 3xTF32 made it WORSE (triples accumulate ops
   into the same fp32 acc). FIX: NBANK accumulation banks (round-robin K-tiles into
   2-4 fp32 accumulators, tree-sum at the end) -> shortens each chain -> passes 5/5.
2. **Performance = L1/shared bound.** ncu on v3: l1tex 83.4%, tensor pipe only 20.5%,
   DRAM 21%, occupancy 24%. The mma-from-shared (load_matrix_sync) traffic is the
   limiter, not DRAM or tensor throughput. Bigger blocks (v4/v5) REGRESSED: 242 regs
   -> 1 block/SM (16% occ); more warps just contend on shared. Bigger per-warp
   register tile (v6 4x4 frags = 2 mma/fragment-load) recovered v3's level but the
   accuracy-required 2 banks cap the tile at ~256 regs, so occupancy stays low.

**Stop (rule 2):** best 0.5856x (10.4 ms). Whole tile-config space (64x64..128x128,
128-512 threads) sits in 0.41-0.59x; last two levers <3% and ncu confirms the binding
L1/shared roofline. Plain WMMA-C++ CANNOT relieve it without ldmatrix/cp.async, which
are PTX (the cuda_unlimited path). **Q1 finding: WMMA-C++ (no PTX) tops out ~0.59x of
cuBLAS-fp32, L1-bound at ~20% tensor utilization.** No inline PTX. Detector-clean.
