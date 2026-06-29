# Iteration Log — cumsum / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/cumsum/solution/cumsum.py`,
Triton speedup 1.2264x); benched against the same `reference/math/cumsum.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of cumsum | 1.2130x | 10.8000 ms | 13.1000 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** row cumsum dim=1; chunked scan with carry. Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=10.8000 ms, REF=13.1000 ms, **SPEEDUP=1.2130x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2264x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Iter 2 — register-resident segment + warp-shuffle block scan
- **Change:** per-thread segment scan kept in registers (shared traffic 32r+32w -> 16r+16w);
  inter-thread Hillis-Steele (16 syncs) replaced by warp `__shfl_up_sync` inclusive scan +
  8-way broadcast combine (2 syncs). Accumulation order byte-identical (fp32 segment-sequential).
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=10.8000 ms, **SPEEDUP=1.2130x**.
- **Verdict:** 0% vs baseline -> scan/shared was NOT the bottleneck. Kernel is DRAM-bound
  (~740 GB/s achieved of ~960 peak). Keep (neutral); pursue occupancy next.

## Iter 3 — occupancy: CHK 4096 -> 2048 (EPT 8)
- **Change:** halve chunk/shared (16KB->8KB buf) to raise resident blocks/SM.
- **Bench:** CORRECT=True, RUNTIME=10.7000 ms, **SPEEDUP=1.2150x**. ~1% (noise) over baseline. KEEP (best).

## Iter 4 — streaming vectorized load (ld.global.cs.v4) instead of __ldg
- **Change:** symmetric evict-first load to match streaming store.
- **Bench:** CORRECT=True, RUNTIME=10.9000 ms, **SPEEDUP=1.2018x**. Slightly WORSE than cached __ldg. REVERT.

## Conclusion — AT FLOOR
DRAM-bound: 8 GB traffic (4GB read + 4GB write) / 10.7ms = ~750 GB/s achieved of ~960 GB/s
peak (~78%, the realistic copy-BW ceiling for AD102). Scan/shared optimizations (register
segment + warp-shuffle) gave 0%, occupancy gave ~1% (noise), streaming load regressed.
Best kept = iter-3 (CHK=2048, register+shuffle scan, cached load, streaming store): 1.2150x.
