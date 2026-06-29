# Iteration Log — layer_norm / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/layer_norm/solution/layer_norm.py`,
Triton speedup 1.6050x); benched against the same `reference/normalization/layer_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of layer_norm | 1.6080x | 3.9800 ms | 6.4000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** LayerNorm last 3 dims; split-row reduction + affine. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=3.9800 ms, REF=6.4000 ms, **SPEEDUP=1.6080x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.6050x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.
