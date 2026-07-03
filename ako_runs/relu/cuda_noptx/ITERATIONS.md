# Iteration Log — relu / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/relu/triton/solution/relu.py`,
Triton speedup 1.0000x); benched against the same `reference/activation/relu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of relu | 0.9697x | 16.5000 ms | 16.0000 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** Unary elementwise; HBM-bandwidth roofline. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.5000 ms, REF=16.0000 ms, **SPEEDUP=0.9697x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0000x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 2 (this run) — float4 vectorized load/store

- **Baseline re-bench:** SPEEDUP=0.9697x, RUNTIME=16.5ms (ref 16.0ms).
- **Change:** float4 vectorized grid-stride kernel (relu_k4) + scalar tail kernel for remainder.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=16.1ms, **SPEEDUP=0.9938x**. KEEP.
- **Verdict:** HBM-bound elementwise. float4 closed the gap from 0.9697x to ~0.9938x (~2.4%, near ref parity). At physical HBM floor; no further headroom.

## Iter 3 (floor-confirm run) — two distinct probes, both neutral, KEEP float4

- **Baseline re-bench (this GPU, full --num-warmup 200):** COMPILED=True, CORRECT=True, RUNTIME=16.1ms, REF=16.0ms, **SPEEDUP=0.9938x**. Fast-signal (--no-ref, 20 trials) = 16.7ms mean (fast-signal clock runs ~0.8ms warmer than full-warmup floor of 15.8ms; used only for relative ranking).
- **Roofline math:** input = 4096×393216 = 1.61e9 floats = 6.44 GB. ReLU streams read+write = 12.88 GB. 12.88 GB / 16.1 ms ≈ 800 GB/s vs RTX 6000 Ada theoretical peak ~960 GB/s → ~83% of peak (higher fraction of the ECC-derated *achievable* peak). Reference is `torch.relu`, itself a saturated bandwidth kernel; we match it to 0.6%, so the realistic ceiling is ~1.00x with no real headroom — only noise. n=1.61e9 is divisible by 4 (dim/4=98304), so the scalar tail kernel never launches; no tail overhead.
- **Probe A — streaming memory hints (`__ldcs`/`__stcs` intrinsics, not PTX asm; allowed):** motivated by avoiding L2 write-allocate/pollution. Fast-signal 16.9ms vs baseline 16.7ms → **no gain (slightly worse, within noise)**. Expected: reads/writes are each single-touch so L2 policy is irrelevant, and GPUs don't do read-for-ownership on full-sector stores. REVERT.
- **Probe B — occupancy/full-grid (threads 256→512, block cap 131072→655360):** more resident threads / less grid-stride looping. Fast-signal 16.5ms vs baseline 16.7ms → 0.2ms apparent, **below the ~0.4ms std / 0.5ms noise threshold and far under the 3% keep bar → noise, not a real win**. REVERT.
- **Distinct axes now logged:** (1) scalar port 0.9697x, (2) float4 vectorization 0.9938x [KEPT], (3) occupancy/block-count/thread-count [neutral], (4) streaming cache hints [neutral]. Four genuinely distinct directions; none beats float4.
- **CONCLUSION — AT FLOOR.** Elementwise ReLU is HBM-bandwidth bound at ~800 GB/s (~83% of theoretical peak, higher of achievable), matching `torch.relu` to within 0.6% (0.9938x). No vectorization-width, occupancy, launch-count, or cache-hint lever moves the number. Kept solution = float4 baseline, verbatim. improved=false, changed_solution=false.
