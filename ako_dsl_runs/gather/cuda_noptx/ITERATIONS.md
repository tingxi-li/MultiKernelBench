# Iteration Log — gather / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gather/solution/gather.py`,
Triton speedup 1.2217x); benched against the same `reference/index/gather.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of gather | 1.1130x | 0.0239 ms | 0.0266 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** gather dim=1; indexed load (latency-bound, small). Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0239 ms, REF=0.0266 ms, **SPEEDUP=1.1130x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2217x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Iter 1 (AKO4ALL) — 2D grid, drop per-element 64-bit divide

- **Change:** replaced grid-stride loop (`r = i / Cout`, runtime int64 divide per
  element — not strength-reducible) with a 2D grid: `r = blockIdx.y`, `c = block.x*tid`.
  Eliminates the software-emulated 64-bit division.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0236 ms, REF=0.0268 ms, **SPEEDUP=1.1356x**.
- **vs baseline 1.1037x:** +2.9%. Small move → confirms latency-bound, not arithmetic-bound. KEEP.

## Iter 2 (AKO4ALL) — ILP, K=4 columns/thread (strided by blockDim)

- **Change:** each thread handles KPT=4 columns strided by blockDim.x; issue all 4
  idx loads, then all 4 gathers, then all 4 stores → more outstanding memory
  requests per thread to hide latency (the binding constraint).
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0226 ms, REF=0.0266 ms, **SPEEDUP=1.1770x**.
- **vs iter-1 1.1356x:** +3.6%. ILP targets latency directly → real move. KEEP.

## Iter 3 (AKO4ALL) — ILP K=8

- **Change:** KPT 4 → 8 (more outstanding loads per thread).
- **Bench:** CORRECT=True, RUNTIME=0.0220 ms, REF=0.0266 ms, **SPEEDUP=1.2091x**.
- **vs iter-2 1.1770x:** +2.7%. Now matches the Triton port (1.22x). KEEP.

## Iter 4 (AKO4ALL) — ILP K=16 (REVERT)

- **Change:** KPT 8 → 16.
- **Bench:** CORRECT=True, RUNTIME=0.0236 ms, **SPEEDUP=1.1314x**. Regression — only
  1 block/row (128 blocks total) starves occupancy. REVERT to iter-3 (KPT=8).

## FINAL — restored iter-3 (KPT=8, 2D grid)

- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0220 ms, REF=0.0268 ms, **SPEEDUP=1.2182x**.
- **vs same-GPU baseline 1.1037x:** +10.4%. Levers: kill per-element int64 divide (2D grid)
  + ILP K=8 strided columns. Latency-bound; now at/above the Triton port (1.22x).
