# Iteration Log — layer_norm / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/layer_norm/solution/layer_norm.py`,
Triton speedup 1.6050x); benched against the same `reference/normalization/layer_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of layer_norm | 1.4862x | 4.3400 ms | 6.4500 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** LayerNorm last 3 dims; split-row reduction + affine. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=4.3400 ms, REF=6.4500 ms, **SPEEDUP=1.4862x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.6050x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Iter 1 (fused) — single kernel, one block per row, fp32, float4

- **Change:** Replaced 3-launch split-row (fp64 atomics + ln_final + ln_apply with
  64-bit div/mod over 268M elems) with ONE kernel, one block per row (M=64 blocks,
  TPB=512). Phase 1: float4 fp32 sum/sumsq + shared-mem tree reduce -> mean/rstd
  (rsqrtf). Phase 2: re-read row float4, write affine. No fp64, no atomics, no
  64-bit div in hot loop (m=blockIdx, col=loop index).
- **Bench:** COMPILED=True, CORRECT=True (5/5), RUNTIME=4.0300 ms, **SPEEDUP=1.5856x**.
- vs baseline 1.4724x / 4.34ms: kept. HBM floor ~3.4ms (~1.88x).

## Iter 2 — block-size sweep TPB=256

- **Change:** TPB 512 -> 256.
- **Bench:** CORRECT=True, RUNTIME=3.9600 ms, **SPEEDUP=1.6162x**. Kept (best).

## Iter 3 — TPB=128

- CORRECT=True, RUNTIME=4.2100 ms, SPEEDUP=1.5202x. Reverted (worse than 256).

## Iter 4 — ILP=4 manual unroll (both passes), TPB=256

- **Change:** each thread processes 4 float4 per stride step (more in-flight loads
  to offset 64-block under-occupancy).
- CORRECT=True, RUNTIME=3.9800 ms, SPEEDUP=1.6080x. Within noise of iter-2; reverted.

## Best = Iter 2 (fused, TPB=256, simple grid-stride): 1.6162x / 3.96ms.
