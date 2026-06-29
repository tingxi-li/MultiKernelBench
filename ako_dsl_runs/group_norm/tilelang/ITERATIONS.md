# Iteration Log — group_norm / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/group_norm/solution/group_norm.py`,
Triton speedup 0.9904x); benched against the same `reference/normalization/group_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of group_norm | 0.8470x | 36.6000 ms | 31.0000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** GroupNorm 8 groups; per-(batch,group) reduction (8.6GB). Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=36.6000 ms, REF=31.0000 ms, **SPEEDUP=0.8470x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9904x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.
