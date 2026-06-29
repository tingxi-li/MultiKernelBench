# Iteration Log — gather / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gather/solution/gather.py`,
Triton speedup 1.2217x); benched against the same `reference/index/gather.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of gather | 1.0675x | 0.0252 ms | 0.0269 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** gather dim=1; indexed load (latency-bound, small). Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0252 ms, REF=0.0269 ms, **SPEEDUP=1.0675x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2217x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Iter 1 — 2D grid, 1 elem/thread (BN=512, 1024 blocks)
- Hypothesis: 128 blocks under one wave on 142 SMs; more blocks to hide scattered-load latency.
- Result: COMPILED=True, CORRECT=True, RUNTIME=0.0265 ms, **SPEEDUP=1.0113x**. WORSE than baseline (killed per-thread ILP). REVERT.

## Iter 2 — 2D grid, COLS=1024 TH=256 (512 blocks, 4 elem/thread)
- Hypothesis: middle ground — more blocks than baseline but keep ILP per thread.
- Result: COMPILED=True, CORRECT=True, RUNTIME=0.0254 ms, **SPEEDUP=1.0512x**. ~baseline (within noise). REVERT.

## Iter 3 — 2D grid, COLS=2048 TH=256 (256 blocks, 8 elem/thread)
- Result: COMPILED=True, CORRECT=True, RUNTIME=0.0253 ms, **SPEEDUP=1.0553x**. ~baseline (noise).

## Iter 4 — 2D grid, COLS=2048 TH=512 (256 blocks, 4 elem/thread, 16 warps/block)
- Result: COMPILED=True, CORRECT=True, RUNTIME=0.0252 ms, **SPEEDUP=1.0635x**. Ties baseline exactly.

## Conclusion — AT FLOOR
Grid/occupancy sweep (1024/512/256 blocks; 1/4/8 elem/thread; TH 256/512) all land within ~3%
noise of baseline 0.0252 ms (1.06x). 1-elem/thread (iter-1) was the only clear loser (-5%, ILP
killed). gather is latency/L2-cache bound, not occupancy bound — scattered X[r,idx] loads hit the
32KB-row cache so adding blocks does not help. Restoring baseline verbatim as best.
