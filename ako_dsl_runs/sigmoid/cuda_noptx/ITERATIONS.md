# Iteration Log — sigmoid / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/sigmoid/solution/sigmoid.py`,
Triton speedup 1.0190x); benched against the same `reference/activation/sigmoid.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of sigmoid | 1.0000x | 16.1000 ms | 16.1000 ms | correct |
| AKO iter-1 | float4 vectorized load/store | 1.0000x | 16.0000 ms | 16.0000 ms | correct (kept) |

## AKO4ALL run (GPU3)

- baseline (scalar grid-stride): SPEEDUP=0.9938x, RUNTIME=16.1ms, CORRECT=True.
- iter-1 (float4 vectorize, scalar tail kernel): SPEEDUP=1.0000x, RUNTIME=16.0ms, CORRECT=True.
- **Verdict: AT FLOOR.** Unary elementwise sigmoid is HBM-bandwidth bound. float4
  vectorization changed nothing measurable (<1% = noise); both match ref. Kept iter-1.

## Iter 1 — cuda_noptx port

- **Hypothesis:** Unary elementwise; HBM roofline. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.1000 ms, REF=16.1000 ms, **SPEEDUP=1.0000x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0190x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.
