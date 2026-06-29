# Iteration Log — relu / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/relu/solution/relu.py`,
Triton speedup 1.0000x); benched against the same `reference/activation/relu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of relu | 0.9877x | 16.2000 ms | 16.0000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** Unary elementwise; HBM-bandwidth roofline. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.2000 ms, REF=16.0000 ms, **SPEEDUP=0.9877x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0000x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## iter-1 (vectorized T.copy)
- Change: VEC=4 inner T.vectorized loop over BLK//4 (float4-style coalesced access)
- SPEEDUP: 1.0323x  RUNTIME: 15.5ms  CORRECT: True
- vs baseline 0.9938x/16.1ms -> ~3.8% faster. KEEP.
