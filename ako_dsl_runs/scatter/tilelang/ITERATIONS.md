# Iteration Log — scatter / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/scatter/solution/scatter.py`,
Triton speedup 5.3079x); benched against the same `reference/index/scatter.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of scatter | 5.8842x | 0.0311 ms | 0.1830 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** deterministic last-wins (atomicMax); scored --deterministic. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0311 ms, REF=0.1830 ms, **SPEEDUP=5.8842x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 5.3079x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Iter 1 (new) — read int64 indices directly, drop idx.to(int32)

- **Hypothesis (advice):** the `idx.contiguous().to(torch.int32)` cast launches an
  extra kernel; reading int64 IDX directly in pass1 removes one launch.
- **Change:** pass1 IDX tensor dtype int32 -> int64; forward() drops `.to(torch.int32)`.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0286 ms (was 0.0317),
  **SPEEDUP=6.3986x**. ~10% kernel-time reduction (one fewer launch). KEEP.
