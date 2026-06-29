# Iteration Log — cumsum / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/cumsum/solution/cumsum.py`,
Triton speedup 1.2264x); benched against the same `reference/math/cumsum.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of cumsum | 1.1944x | 10.8000 ms | 12.9000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** row cumsum dim=1; chunked scan with carry. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=10.8000 ms, REF=12.9000 ms, **SPEEDUP=1.1944x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2264x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Iter 1 (re-bench) — baseline re-bench: 10.8ms, 1.2130x (C=4096/TH=256)

## Iter 2 — C=8192 / TH=512
- Halves chunks (8→4) and syncs. RUNTIME=10.9ms, SPEEDUP=1.2018x, CORRECT=True.
- Within noise / slightly worse than baseline. REVERT.

## Iter 2b — C=8192 / TH=256: 10.9ms, 1.2018x, CORRECT. Within noise/worse. REVERT.
## Iter 3 — C=4096 / TH=1024: 23.4ms, 0.5598x. Occupancy collapse. REVERT.

## Iter 4 — C=4096 / TH=128: 10.8ms, 1.2037x, CORRECT. Identical to baseline (noise). REVERT.

## Conclusion: AT FLOOR
8.59 GB traffic / 10.8ms = 795 GB/s ≈ 83% of RTX6000-Ada ~960 GB/s peak.
Chunk/thread sweep {C=4096/8192, TH=128/256/512/1024} moved nothing >3% (TH=1024
regressed via occupancy collapse). Scan is HBM-bound. Best = baseline C=4096/TH=256
@ 10.8ms / 1.21x. Restored baseline verbatim.
