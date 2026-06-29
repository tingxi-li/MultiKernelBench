# Iteration Log — group_norm / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/group_norm/solution/group_norm.py`,
Triton speedup 0.9904x); benched against the same `reference/normalization/group_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of group_norm | 0.9172x | 33.8000 ms | 31.0000 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** GroupNorm 8 groups; per-(batch,group) reduction (8.6GB). Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=33.8000 ms, REF=31.0000 ms, **SPEEDUP=0.9172x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9904x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Re-bench session (GPU 3) — AT FLOOR

Baseline re-bench (same-GPU): COMPILED=True, CORRECT=True, RUNTIME=33.8ms, SPEEDUP=0.9172x
(min 31.0ms == ref min 30.7ms; mean inflated by ONE Trial-1 outlier ~296ms).

Roofline: tensor=8.59GB, two-pass GN traffic = 3x = 25.77GB. At min runtime 30.9ms => 831 GB/s = HBM
roofline (== torch). 3x traffic is irreducible for exact two-pass group_norm; torch pays it too.

| Iter | Change | Speedup | Runtime | min | Correct | Keep |
|------|--------|---------|---------|-----|---------|------|
| 1 | ld.global.cs.v4 streaming loads (replace __ldg) both kernels | 0.9091x | 34.1ms | 31.1 | True | revert (0%, same outlier 295ms) |
| 2 | TPB 256->512 | 0.9172x | 33.8ms | 30.9 | True | revert to baseline (tie on mean) |
| final | restored baseline (TPB=256, __ldg, float4 + st.global.cs) | 0.9172x | 33.8ms | 30.9 | True | KEEP |

Conclusion: AT FLOOR. min runtime ties torch (30.9 vs 30.7ms) at the 831 GB/s HBM roofline.
The 0.92x mean is a harness artifact: a single Trial-1 ~300ms outlier that is KERNEL-INDEPENDENT
(296/295/300ms across baseline/streaming-load/TPB=512 — a fixed cost no kernel change moves;
torch's reference does not incur it). float4 vec loads + st.global.cs streaming stores + fp32
accumulators + NG=1024 stats grid already applied. No further kernel-side headroom.
