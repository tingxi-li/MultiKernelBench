# Iteration Log — hardsigmoid / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/hardsigmoid/triton/solution/hardsigmoid.py`,
Triton speedup 1.0127x); benched against the same `reference/activation/hardsigmoid.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of hardsigmoid | 0.9877x | 16.2000 ms | 16.0000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** Unary clamp; HBM roofline. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.2000 ms, REF=16.0000 ms, **SPEEDUP=0.9877x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0127x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Re-bench (2026-06-29)
- baseline: SPEEDUP 0.9938x, RUNTIME 16.1ms, REF 16.0ms, CORRECT=True.
- Analysis: pure elementwise hardsigmoid = read N + write N fp32, memory-bound at HBM bandwidth. Runtime == ref within run-to-run noise (std 0.46ms). No arithmetic to optimize; float4 vectorization on this load pattern offers nothing over the compiler's coalesced 256-thread access. At physical floor.
- Verdict: at_floor. No iters attempted (no lever exists).

## Floor-proof re-bench (2026-07-02, GPU 3)
- **Same-GPU baseline:** SPEEDUP 0.9938x, RUNTIME 16.1ms, REF 16.0ms, CORRECT=True, std 0.455ms (~2.8% noise). Fast-signal min 16.7ms.
- **Roofline:** N = 4096×393216 = 1.610e9 fp32 → 12.9 GB read+write. RTX 6000 Ada peak HBM ≈ 960 GB/s → theoretical floor ≈ 13.4 ms. Measured 16.1 ms ≈ 83% of peak — identical to torch's reference (16.0 ms). Both kernels ride the same achievable-bandwidth wall.
- Two GENUINELY DISTINCT levers tested (fast-signal, rank by min runtime), both REVERTED:
  - **Dir A — vectorization width.** Hypothesis: `if idx < N` guard blocks 128-bit coalesced loads; N is an exact multiple of BLK (196608×8192) so the guard is dead code. Specialized a guard-free branch to let tilelang emit vectorized accesses. Result: min 16.7ms — **identical** to baseline. Memory subsystem saturated regardless of access width. REVERT (no gain).
  - **Dir B — occupancy / block config.** Hypothesis: more threads/larger blocks improve latency hiding. TH=1024, BLK=16384 (vs 256/8192). Result: min 17.1ms — **worse**. Occupancy is not the limiter; baseline 256/8192 already saturates HBM. REVERT.
- **Verdict: AT FLOOR.** Baseline restored verbatim (git-diff-clean). Vectorization neutral, occupancy negative, roofline confirms ~83% of peak == the reference. No lever exists. Detector: valid=True, regression_type=None (pass).
- **Final verdict bench:** COMPILED=True, CORRECT=True, RUNTIME 16.1ms, REF 16.0ms, SPEEDUP 0.9938x.
