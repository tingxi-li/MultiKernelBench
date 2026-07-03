# Iteration Log — sigmoid / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/sigmoid/triton/solution/sigmoid.py`,
Triton speedup 1.0190x); benched against the same `reference/activation/sigmoid.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of sigmoid | 1.0000x | 16.1000 ms | 16.1000 ms | correct |
| AKO iter-1 | float4 vectorized load/store | 1.0000x | 16.0000 ms | 16.0000 ms | correct (kept) |

## AKO4ALL run (GPU3)

- baseline (scalar grid-stride): SPEEDUP=0.9938x, RUNTIME=16.1ms, CORRECT=True.
- iter-1 (float4 vectorize, scalar tail kernel): SPEEDUP=1.0000x, RUNTIME=16.0ms, CORRECT=True.
- **Verdict: AT FLOOR.** Unary elementwise sigmoid is HBM-bandwidth bound. float4
  vectorization changed nothing measurable (<1% = noise); both match ref. Kept iter-1.

## Iter 1 — cuda_noptx port

- **Hypothesis:** Unary elementwise; HBM roofline. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.1000 ms, REF=16.1000 ms, **SPEEDUP=1.0000x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0190x:** see ako_runs/RESULTS.md for the cross-DSL table.

## AKO4ALL re-run (GPU0, 2026-07-02) — floor re-confirmation

Re-run of an interrupted cell; started from the committed baseline (plain grid-stride
float4, no PTX). Goal per depth policy: confirm the memory floor via 2-3 genuinely
distinct directions, then stop.

- **Baseline on GPU0 (verdict, --num-warmup 200):** SPEEDUP=1.0063x, RUNTIME=16.0ms
  (min 15.8), REF=16.1ms (min 15.7), CORRECT=True.
- **Roofline math:** 4096×393216 = 1.61e9 floats → read 6.44GB + write 6.44GB = 12.9GB.
  RTX6000-Ada HBM ~960GB/s theoretical, but GDDR6 **ECC** costs ~6-12% → achievable
  ~850-900GB/s → achievable floor ~14.3-15.2ms. Our 15.8ms min is ~93-96% of achievable.
- **Floor proof:** the vendor's own `torch.sigmoid` runs at 16.1ms (min 15.7) on this
  pure-bandwidth op; we match/beat it. Matching torch on unary elementwise IS the floor.

### Iter A — streaming cache hints (`__ldcs` load / `__stcs` store)
- **Hypothesis:** output is write-once/never-reread, input read-once; bypassing L2 with
  streaming intrinsics could free a few % of effective bandwidth. (Intrinsics, not inline
  PTX → legal for cuda_noptx; detector-clean.)
- **Change:** `x[i]`→`__ldcs(&x[i])`, `y[i]=v`→`__stcs(&y[i],v)` in the float4 kernel.
- **Fast-signal (--no-ref, 20 trials):** mean 16.9ms, min 16.7ms vs baseline mean 16.7,
  min 16.5 → **no gain (marginally worse, within noise)**. Reproduced on a clean re-run.
- **REVERT.** L2 is not the bottleneck for a streaming elementwise pass at this size.

### Iter B — remove grid-stride block cap (full 1.57M-block grid, ~1 float4/thread)
- **Hypothesis:** the 131072-block cap (grid-stride, ~12 elems/thread) might starve
  memory-level parallelism vs launching the full grid.
- **Change:** raise `blocks` cap so `want`≈1,572,864 blocks launch (one float4 per thread).
- **Fast-signal:** mean 17.1ms, min 16.9ms → **worse.** 142 SMs × ~1536 resident ≈ 218K
  concurrent threads already saturate MLP 150× over; the extra 1.5M blocks only add launch
  overhead. REVERT.

### Skipped (settled analytically, not worth an iteration)
- **Vectorization width:** float4 = 128-bit = the max per-thread global transaction; nothing
  wider exists. Already in the baseline.
- **Launch count / fusion:** single unary op, nothing to fuse.

**Verdict: AT FLOOR (re-confirmed).** Two distinct profile-motivated directions (streaming
hints, block-count/MLP) both land at-or-below baseline within noise; vectorization width is
maxed. Baseline matches vendor `torch.sigmoid` and sits at ~93-96% of ECC-adjusted achievable
HBM bandwidth. Restored the committed baseline verbatim (git-diff-clean, detector regression
None). improved=false, at_floor=true.
