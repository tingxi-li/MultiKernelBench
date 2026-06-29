# Iteration Log — hardsigmoid / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/hardsigmoid/solution/hardsigmoid.py`,
Triton speedup 1.0127x); benched against the same `reference/activation/hardsigmoid.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of hardsigmoid | 0.9697x | 16.5000 ms | 16.0000 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** Unary clamp; HBM roofline. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.5000 ms, REF=16.0000 ms, **SPEEDUP=0.9697x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0127x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## iter-1: float4 vectorized load/store (+ scalar tail kernel)
- Changed scalar grid-stride to float4 vectorized (4 elems/thread) with a scalar tail kernel for n%4.
- SPEEDUP: 0.9938x  RUNTIME: 16.1ms  CORRECT: True  -> KEEP (best)
- Baseline was 0.9697x/16.5ms. float4 reaches ref parity (HBM bandwidth floor). Memory-bound clamp; no further headroom.
