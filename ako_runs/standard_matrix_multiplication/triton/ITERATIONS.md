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
| 1 | identity (torch.matmul / cuBLAS fp32) baseline | 1.0000x | 6.08 ms | baseline (vendor) |
| 2 | triton tl.dot fp32 IEEE, autotune 2D grid | 0.7909x | 7.70 ms | best correct |
| 3 | tf32x3 tensor-core + L2 swizzle | 0.5477x | 11.10 ms | correct but slower (revert) |
| 4 | single-pass tf32 tensor-core | FAIL (max_diff 1.63) | - | incorrect (revert) |
| 5 | fp32 IEEE + L2 swizzle, fp32 tiles | 0.7815x | 7.78 ms | no-change (revert) |

## Iterations

### Op summary — standard_matrix_multiplication (triton), COMPUTE-BOUND (GEMM)

- **Shape:** A(2048,8192) @ B(8192,4096), fp32, tol 1e-4. Vendor ceiling = cuBLAS SGEMM = 6.08 ms (~22.6 TFLOP/s effective, true fp32, no tensor cores).
- **Iter 2 (best correct):** classic triton `tl.dot(..., input_precision='ieee')` GEMM, autotune BLOCK_M/N/K/nw/ns, 2D grid. 7.70 ms → **0.7909x** of cuBLAS. Triton's fp32 FMA dot can't match hand-tuned cuBLAS SGEMM scheduling.
- **Iter 3 (tf32x3):** 3-pass tf32 (recovers ~fp32 accuracy) on tensor cores — CORRECT but 11.1 ms / 0.55x. On these locked clocks tensor-core tf32 is only ~2x fp32-FMA, so 3 passes net *slower* than one fp32 pass.
- **Iter 4 (tf32):** single-pass tf32 tensor cores would be ~2x faster, but **fails the 1e-4 gate** (max_diff 1.63 at |out|~2048 → rel ~8e-4 > 1e-4). The fp32 tolerance forbids the fast tensor-core path.
- **Iter 5 (fp32 + L2 swizzle):** 7.78 ms / 0.7815x — swizzle gave nothing; ncu shows this is not L2-bound.
- **ncu at stall (iter2/iter5 within 1.2%):** DRAM = 2.89 passes, **dram%=5.5 (idle), sm%=60.7, occ 16.7%** → compute/FMA-bound. Binding roofline is the fp32 FMA pipe, and triton sits ~21% below cuBLAS SGEMM there.
- **Stop reason:** stop-rule #2 — 2 consecutive fp32 levers <3% AND ncu confirms compute(FMA)-bound; the only way past cuBLAS (tensor cores) is blocked by the fp32 correctness gate. Honest ceiling gap: triton fp32 = 0.79x vendor.
- **Detector:** clean (forward allocate+launch glue; GEMM fully in `@triton.jit`).

