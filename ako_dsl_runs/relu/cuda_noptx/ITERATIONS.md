# Iteration Log — relu / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/relu/solution/relu.py`,
Triton speedup 1.0000x); benched against the same `reference/activation/relu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of relu | 0.9697x | 16.5000 ms | 16.0000 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** Unary elementwise; HBM-bandwidth roofline. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.5000 ms, REF=16.0000 ms, **SPEEDUP=0.9697x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0000x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.
