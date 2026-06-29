# Iteration Log — hardsigmoid / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/hardsigmoid/solution/hardsigmoid.py`,
Triton speedup 1.0127x); benched against the same `reference/activation/hardsigmoid.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of hardsigmoid | 1.0000x | 16.0000 ms | 16.0000 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** Unary clamp; HBM roofline. Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.0000 ms, REF=16.0000 ms, **SPEEDUP=1.0000x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0127x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Re-bench (this session)
- baseline: SPEEDUP 1.0000x, RUNTIME 16.0ms, CORRECT True. Mean==ref 16.0ms.
- Analysis: HBM-bound elementwise; already float4 loads + st.global.cs.v4 streaming store, __launch_bounds__(256,6). Read+write of full tensor = HBM bandwidth floor. ref also hits same bandwidth -> 1.00x is the floor.
- No iteration: at floor. final re-bench 1.0000x CORRECT True.
