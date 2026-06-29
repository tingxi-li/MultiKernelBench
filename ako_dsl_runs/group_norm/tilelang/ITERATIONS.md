# Iteration Log — group_norm / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/group_norm/solution/group_norm.py`,
Triton speedup 0.9904x); benched against the same `reference/normalization/group_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of group_norm | 0.8470x | 36.6000 ms | 31.0000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** GroupNorm 8 groups; per-(batch,group) reduction (8.6GB). Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=36.6000 ms, REF=31.0000 ms, **SPEEDUP=0.8470x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9904x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Iter 1b — float4 vectorization (both passes)

- **Change:** vectorized reduction + affine apply with `T.vectorized(4)` into a
  local float4 (`vec`). Apply hoists `Wt[c]`/`Bs[c]` per float4 (HW%4==0 keeps a
  float4 within one channel). TH=256.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=34.3ms (min 31.4), REF=31.0,
  **SPEEDUP=0.9038x** (up from 0.8493x baseline). KEEP.

## Iter 2/3 — TH sweep (512, 1024)

- 0.9012x / 0.9012x, min 31.4ms — identical to TH=256 within noise. No change. REVERT to TH=256.

## Iter 4 — 4-lane ILP accumulators in reduction

- Broke FP-add dependency chain with 4 accumulator lanes. RUNTIME 34.3ms, min 31.4,
  **0.9038x** — identical to iter-1. Confirms memory-bound; ILP doesn't help. REVERT.

## Verdict: AT FLOOR

- Best = iter-1 (float4 both passes, TH=256): **0.9038x**, min 31.4ms.
- 3-pass memory-bound op: read-reduce + read-apply + write = 25.8GB / ~960GB/s = ~27ms floor.
  Ref min 30.7ms (87% BW); our min 31.4ms = 97.8% of ref. Vectorization closed 0.85x->0.90x;
  TH sweep and ILP gave 0%. At the HBM roofline.
