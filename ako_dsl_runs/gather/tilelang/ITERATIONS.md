# Iteration Log — gather / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gather/solution/gather.py`,
Triton speedup 1.2217x); benched against the same `reference/index/gather.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of gather | 1.0675x | 0.0252 ms | 0.0269 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** gather dim=1; indexed load (latency-bound, small). Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0252 ms, REF=0.0269 ms, **SPEEDUP=1.0675x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2217x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.
