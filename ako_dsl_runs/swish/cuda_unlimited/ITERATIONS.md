# Iteration Log — swish / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/swish/solution/swish.py`,
Triton speedup 2.5253x); benched against the same `reference/activation/swish.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of swish | 2.4472x | 16.1000 ms | 39.4000 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** x*sigmoid(x): eager=2 passes, fused=1 -> ~2.5x. Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.1000 ms, REF=39.4000 ms, **SPEEDUP=2.4472x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 2.5253x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Iter 1 — streaming vectorized load (ld.global.cs.v4.f32)

- **Hypothesis:** input is read-once; ld.global.cs streaming load avoids L2 pollution like the store does.
- **Result:** SPEEDUP=2.3879x, RUNTIME=16.5ms (noisy, max 32.1ms), CORRECT=True. **SLOWER** than __ldg baseline (2.4472x). __ldg's cached path schedules better. **REVERTED.**

## Iter 2 — fast intrinsic __expf

- **Hypothesis:** fewer ALU instrs in actf could help.
- **Result:** SPEEDUP=2.4472x, RUNTIME=16.1ms, CORRECT=True. **Identical** to baseline — compute is fully latency-hidden behind HBM (memory-bound). No win; reverted to expf (more accurate, same speed).

## Conclusion — AT FLOOR

16.1ms == ref/2.45x, HBM-bandwidth-bound elementwise. float4 + streaming store already in place; streaming load regressed; fast expf gave 0%. Kept baseline verbatim as best.
