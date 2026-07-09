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
| 1 | identity baseline (torch SDPA) | 1.0000x | 80.6 ms | baseline (ref) |
| 2 | v1 3-kernel QKt + softmax + PV (WMMA-tf32, NBANK=1) | 1.7522x | 46.0 ms | improved (BEST) |
| final | v1 3-kernel QKt + softmax + PV (WMMA-tf32, NBANK=1) | 1.7115x | 36.4 ms | final |

## Iterations

SDPA, Q/K/V = [32,32,512,1024] (BH=1024, S=512, **D=1024**). Reference torch SDPA =
**80.6 ms = only 13.6 TFLOP/s**: D=1024 exceeds the flash backend's supported head
dim, so torch falls back to a slow math/mem-efficient path -> beatable.

Design (3 kernels, batched over BH): (1) `qk_gemm` S = scale * Q@K^T (K loaded
transposed -> col-major matrix_b; scale folded into the epilogue), (2) `softmax_rows`
row softmax over the 512 keys, (3) `pv_gemm` O = P@V (standard). Both GEMMs are the
same 128x128 / 4x4-frag WMMA-tf32 kernel as op3 with **NBANK=1** — accumulation
magnitudes are small (S~256 pre-scale, O~1) so no accuracy banks are needed.

ncu (v1): qk_gemm 23.3 ms (l1tex 84%, tensor 36%), softmax 2.75 ms (DRAM 89%),
pv_gemm 20.1 ms (l1tex 79%, tensor 41%). GEMMs ~26 TFLOP/s combined (slightly above
cuBLAS-fp32) and beat torch's fallback comfortably.

**Stop:** v1 = 1.7522x **beats the vendor reference**, at the cell ceiling. Both
GEMMs confirmed L1-bound at the WMMA-C++ limit (same wall as op2/op3); softmax at its
DRAM roofline. The one remaining lever — a flash-style fusion to drop the 2.75 ms
softmax S round-trip — caps at ~1.87x (the GEMMs stay L1-bound) for large rewrite
risk, so not worth it. No inline PTX. Detector-clean.
