# Iteration Log — elu / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/elu/solution/elu.py`,
Triton speedup 0.9938x); benched against the same `reference/activation/elu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of elu | 0.9697x | 16.5000 ms | 16.0000 ms | correct |
| 2 | float4 vectorized (16B loads/stores) + scalar tail | 1.0000x | 16.0000 ms | 16.0000 ms | correct (KEEP) |

## Iter 2 — float4 vectorization

- **Hypothesis (ADVICE):** memory-bound; scalar grid-stride leaves a few % of HBM
  bandwidth on the table. Reinterpret as float4 so each thread moves 16B in/16B out,
  halving instruction overhead and saturating HBM.
- **Change:** `elu_k4` processes float4 (n/4 vectors); scalar `elu_k` handles the
  ragged tail. Input n=4096*393216 is div by 4 so the tail kernel is a no-op here.
- **Bench (--num-warmup 200):** COMPILED=True, CORRECT=True (5/5),
  RUNTIME=16.0000 ms == REF 16.0000 ms, **SPEEDUP=1.0000x** (up from 0.9697x).
- **Verdict:** KEEP. Now exactly at the HBM roofline (runtime == ref). at_floor.

## Iter 1 — cuda_noptx port

- **Hypothesis:** Unary (alpha from init); HBM roofline. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.5000 ms, REF=16.0000 ms, **SPEEDUP=0.9697x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9938x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.
