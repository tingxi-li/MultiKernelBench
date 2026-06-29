# Iteration Log — swish / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/swish/solution/swish.py`,
Triton speedup 2.5253x); benched against the same `reference/activation/swish.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of swish | 2.4321x | 16.2000 ms | 39.4000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** x*sigmoid(x): eager=2 passes, fused=1 -> ~2.5x. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.2000 ms, REF=39.4000 ms, **SPEEDUP=2.4321x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 2.5253x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.
