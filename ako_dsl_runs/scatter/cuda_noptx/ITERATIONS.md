# Iteration Log — scatter / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/scatter/solution/scatter.py`,
Triton speedup 5.3079x); benched against the same `reference/index/scatter.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of scatter | 6.5580x | 0.0276 ms | 0.1810 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** deterministic last-wins (atomicMax); scored --deterministic. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0276 ms, REF=0.1810 ms, **SPEEDUP=6.5580x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 5.3079x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.
