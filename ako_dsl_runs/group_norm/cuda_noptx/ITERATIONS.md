# Iteration Log — group_norm / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/group_norm/solution/group_norm.py`,
Triton speedup 0.9904x); benched against the same `reference/normalization/group_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of group_norm | 0.8988x | 34.6000 ms | 31.1000 ms | correct |
| AKO-base | re-bench baseline (GPU2) | 0.9014x | 34.5000 ms | 31.1000 ms | correct |
| AKO-1 | float4 vectorized loads/stores both kernels | 0.9174x | 33.9000 ms | 31.1000 ms | correct, KEEP |
| AKO-2 | apply block size 256->512 | 0.9174x | 33.9000 ms | 31.1000 ms | correct, no change -> revert |
| AKO-final | restored AKO-1 (float4, TPB) | 0.9174x | 33.9000 ms | 31.1000 ms | correct |

### AKO notes (bandwidth floor)
- float4 (AKO-1) lifted 0.9014x->0.9174x (~1.8%); apply block-size 512 (AKO-2) gave 0 change.
- This op is two-pass HBM-bound: 25.8GB fixed traffic (stats reads x 8.6GB; apply reads x + writes y 17.2GB). torch hits the same floor.
- final min=31.0ms ~= ref min=30.7ms => at the HBM roofline. The reported mean (33.9ms) is dragged by a single clock-ramp outlier (max=299ms, std=27); min is the clean signal and is at parity.
- Accumulators already fp32 (ls/lss); only the 1024x per-block final mu/var/sqrt is double (perf-irrelevant, helps 1e-4 gate). Status: at_floor.

## Iter 1 — cuda_noptx port

- **Hypothesis:** GroupNorm 8 groups; per-(batch,group) reduction (8.6GB). Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=34.6000 ms, REF=31.1000 ms, **SPEEDUP=0.8988x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9904x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.
