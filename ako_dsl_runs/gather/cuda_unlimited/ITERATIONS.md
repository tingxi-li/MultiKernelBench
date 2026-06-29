# Iteration Log — gather / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gather/solution/gather.py`,
Triton speedup 1.2217x); benched against the same `reference/index/gather.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of gather | 1.1203x | 0.0241 ms | 0.0270 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** gather dim=1; indexed load (latency-bound, small). Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0241 ms, REF=0.0270 ms, **SPEEDUP=1.1203x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2217x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Iter 2 — 2D grid, int32 indexing (kill 64-bit divide)
- **Change:** replaced grid-stride 1D loop + `i/Cout` 64-bit divide with a 2D grid (row=blockIdx.y), int32 arithmetic, `if(col<Cout)` guard.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0236 ms, REF=0.0270, **SPEEDUP=1.1441x** (baseline 1.1303x).
- ~1% gain — divide was overlappable, not the bottleneck (dependent idx->x scatter chain dominates). KEEP.

## Iter 3 — K=4 outputs/thread: vector index loads + MLP scatter + v4 streaming store
- **Change:** each thread emits 4 consecutive outputs. 2x `ld.global.nc.v2.s64` index loads, 4 INDEPENDENT scattered `ld.global.nc.f32` loads (memory-level parallelism hides the dependent idx->x latency), one `st.global.cs.v4.f32` streaming store. Cout=4096 divisible by 4 -> all vector ops aligned.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0212 ms, REF=0.0268, **SPEEDUP=1.2642x** (baseline 1.1303x, prev best 1.1441x).
- Beats Triton target (1.2217x). MLP was the real lever. KEEP.

## Iter 4 — K=8 outputs/thread (more MLP)
- **Change:** K=4 -> K=8 (4x v2.s64 loads, 8 scatter loads, 2x v4 store).
- **Bench:** CORRECT=True, RUNTIME=0.0217 ms, **SPEEDUP=1.2396x** — slightly worse than K=4 (0.0212). More registers/less occupancy. REVERT to iter-3 (K=4).

## Iter 5 — K=4 with threads=128 (occupancy)
- **Change:** block size 256 -> 128 threads (K=4 unchanged).
- **Bench:** CORRECT=True, RUNTIME=0.0211 ms, **SPEEDUP=1.2796x** — marginally best (vs 0.0212 @256, within noise). KEEP as best.
