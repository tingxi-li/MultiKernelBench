# Iteration Log — cumsum / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/cumsum/solution/cumsum.py`,
Triton speedup 1.2264x); benched against the same `reference/math/cumsum.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of cumsum | 1.1944x | 10.8000 ms | 12.9000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** row cumsum dim=1; chunked scan with carry. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=10.8000 ms, REF=12.9000 ms, **SPEEDUP=1.1944x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2264x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.
