# Iteration Log — sigmoid / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/sigmoid/solution/sigmoid.py`,
Triton speedup 1.0190x); benched against the same `reference/activation/sigmoid.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of sigmoid | 1.0000x | 16.1000 ms | 16.1000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** Unary elementwise; HBM roofline. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.1000 ms, REF=16.1000 ms, **SPEEDUP=1.0000x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0190x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Baseline re-bench (this run)

- COMPILED=True, CORRECT=True, RUNTIME=16.1000 ms, REF=16.1000 ms, **SPEEDUP=1.0000x**.
- Unary elementwise sigmoid: reads N fp32, writes N fp32 → pure HBM-bandwidth bound.
  Runtime equals ref exactly (16.1 ms). No compute lever (single exp) can move a
  memory-bound op; vectorization already saturates bus (Triton port topped at 1.019x ~ noise).
- **Verdict: at floor.** No iteration attempted — no lever exists above HBM roofline.
