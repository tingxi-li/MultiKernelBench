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
| 1 | Tiled register-blocked SGEMM BM=BN=128 BK=16 TM=TN=8 | 0.82x | 5.46 ms | regression |

## Iterations

### Iter 1 — Tiled register-blocked SGEMM (BM=BN=128, BK=16, TM=TN=8)

- **Hypothesis:** A hand-written register-blocked SGEMM with 128x128 block tiles, BK=16, TM=TN=8 per-thread tile, vectorized float4 loads, and shared-memory padding to avoid bank conflicts should approach cuBLAS throughput, confirming the floor.
- **Changes:** Replaced `torch.matmul` identity with `load_inline` CUDA kernel: tiled SGEMM, smA[BM][BK+PAD] and smB[BK][BN+PAD], 256 threads per block (16x16 warp grid), TM=TN=8, vectorized float4 loads for both A and B tiles.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 5.46 ms (mean), 5.05 ~ 5.54 ms (min ~ max)
  - Speedup: 0.82x (REF=4.45 ms)
- **Analysis:** Custom SGEMM is 18% slower than cuBLAS. cuBLAS on RTX 6000 Ada uses highly optimized CUTLASS-based kernels with LDGSTS async loads, double buffering, and optimal instruction scheduling. A naive register-blocked kernel cannot match this. Confirms the FLOOR designation.
- **Next:** Try PTX mma.sync (tensor-core) kernel for iter 2 to see if we can come closer.

