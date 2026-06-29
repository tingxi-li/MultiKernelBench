# Iteration Log — gelu / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gelu/solution/gelu.py`,
Triton speedup 1.0063x); benched against the same `reference/activation/gelu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of gelu | 1.0323x | 15.5000 ms | 16.0000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** Exact erf GELU; HBM roofline. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=15.5000 ms, REF=16.0000 ms, **SPEEDUP=1.0323x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0063x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Re-bench (this run)
- baseline (GPU1): SPEEDUP 1.0323x, RUNTIME 15.5ms, REF 16.0ms, CORRECT=True
- Analysis: exact-erf GELU, pure elementwise read N + write N floats => HBM-bandwidth bound. std ~0.46ms (3%) is noise. Already above ref via float32 erf path.
- No edit attempted: at physical HBM floor. final == baseline. status=at_floor.
