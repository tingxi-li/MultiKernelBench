# Iteration Log — gelu / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gelu/triton/solution/gelu.py`,
Triton speedup 1.0063x); benched against the same `reference/activation/gelu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of gelu | 1.0323x | 15.5000 ms | 16.0000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** Exact erf GELU; HBM roofline. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=15.5000 ms, REF=16.0000 ms, **SPEEDUP=1.0323x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0063x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Re-bench (this run)
- baseline (GPU1): SPEEDUP 1.0323x, RUNTIME 15.5ms, REF 16.0ms, CORRECT=True
- Analysis: exact-erf GELU, pure elementwise read N + write N floats => HBM-bandwidth bound. std ~0.46ms (3%) is noise. Already above ref via float32 erf path.

### Empirical bandwidth-ceiling proof (this run, GPU1)
- N = 4096*393216 = 1,610,612,736 f32 elements = 6.44 GB. Traffic = read N + write N = **12.9 GB**.
- Measured on this GPU (quick loop, 20 warmup): `torch.copy_` (identical 2N traffic) = **16.01 ms / 804.7 GB/s**; `torch.gelu` = 15.92 ms / 809 GB/s.
- Our kernel (harness, --num-warmup 200): **15.2-15.5 ms => ~843 GB/s effective**. GDDR6 theoretical peak ~960 GB/s => we run at **~85-88% of peak**, which IS the practical GDDR6 ceiling for elementwise.
- Both harnesses agree at ~800-843 GB/s; the two measurements use slightly different warmup, but the conclusion is identical: **the kernel already runs at/above a same-traffic memcpy**, so there is no memory headroom left. Compute (one erf) is fully hidden behind memory.

### Iter 2 — drop dead boundary guard (vectorization lever)
- **Hypothesis:** N % BLK == 0 exactly (1610612736/8192 = 196608), so `if idx < N` is dead code that could inhibit tilelang auto-vectorization of the T.Parallel copy. Removing it may enable wider (128-bit) loads/stores.
- **Change:** removed `if idx < N` guard.
- **Fast-signal (--no-ref, 20 trials):** RUNTIME 16.3ms == baseline 16.3ms. CORRECT=True. No movement (tilelang already lowers the compile-time-constant bound cleanly).
- **REVERT** — no gain; keep the guard for generality/safety. Next: block/thread geometry.

### Iter 3 — block/thread geometry (occupancy lever)
- **Hypothesis:** BLK=8192/TH=256 gives 32 elems/thread; a different geometry (BLK=16384, TH=512, still 32 elems/thread but 2x block work / half the block count) could shift occupancy or memory-coalescing behavior.
- **Change:** BLK=16384, TH=512.
- **Fast-signal (--no-ref, 20 trials):** RUNTIME 16.5ms vs 16.3ms baseline — within noise, slightly slower. CORRECT=True.
- **REVERT** — no gain, as expected at the bandwidth ceiling.

### Conclusion — AT FLOOR
- Two genuinely distinct levers (vectorization-guard, block/thread occupancy) produced no measurable change; the kernel already matches/beats a same-traffic `torch.copy_`. This is the GDDR6 memory roofline. **final == committed baseline (git-clean), status=at_floor, improved=false, changed_solution=false.**
