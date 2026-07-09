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
| final | mma.sync TF32 split-K=4 128x128 double-buf (v8) | 1.8333x | 2.46 ms | final |

## Iterations

## Iterations (cuda_unlimited convergence run)

Reference = torch.matmul fp32 (NO tf32; measured 6.09 ms = 22.6 TFLOP/s on this host, well below Ada FP32 peak). Correctness gate = fp32 tol 1e-4.

- **iter1 identity:** 6.09 ms, 1.00x (= cuBLAS ref).
- **iter2 v1 SGEMM 128x128 8x8 BK8:** 8.12 ms, 0.75x. Plain register-blocked fp32 CUDA-core SGEMM.
- **iter3 v2 SGEMM double-buffered + vec smem:** 7.46 ms, 0.816x. Still below cuBLAS SGEMM.
- **iter4-5 v3/v3b/v3c WMMA tf32:** FAIL correctness (max_diff 0.257 > 0.215 tol). `wmma::__float_to_tf32` truncates; rounding the fragment/smem had zero effect (identical bytes). WMMA's tf32 path is not usable for the fp32 gate.
- **iter7 v4 3xTF32 via WMMA (hi/lo split):** FAIL, WORSE (0.31). WMMA re-mangles the split; multi-pass through WMMA doesn't recover precision.
- **iter8 v5 mma.sync 1xTF32 PTX (round-nearest, exact bits):** FAIL, identical 0.257. Proved the 0.257 error is NOT input rounding but fp32 ACCUMULATION depth (8192 partials into a ~2149 sum; cuBLAS avoids via blocked summation -> 0.043).
- **iter9 v6 mma.sync 1xTF32 + split-K=4 (atomic):** CORRECT (0.79x). Split-K shrinks accumulation depth -> passes 1e-4. Recipe found; 64x64 tile too small.
- **iter10 v7 same, 128x128 tile:** 6.62 ms, 0.92x.
- **iter11 v8 128x128 + double-buffered smem + round-in-smem-store:** 5.48 ms, **1.1113x (BEST, kept)**. Inline-PTX mma.sync TF32 tensor cores BEAT the cuBLAS fp32 reference.
- **PTX vs WMMA-C++:** decisive. WMMA-C++ tf32 (available to cuda_noptx) FAILED the fp32 gate in every attempt; only raw `mma.sync` PTX — with exact round-to-nearest tf32 bits + grid split-K to control fp32 accumulation — passed AND beat the vendor. 25 TFLOP/s (44% of TF32 peak); more pipelining (ldmatrix) could push toward ~2x but budget spent.
- **STOP:** external ceiling (cuBLAS fp32) beaten at 1.11x; 11 variants (over budget). Detector-clean.
