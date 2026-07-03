# Iteration Log — hardsigmoid / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/hardsigmoid/triton/solution/hardsigmoid.py`,
Triton speedup 1.0127x); benched against the same `reference/activation/hardsigmoid.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of hardsigmoid | 0.9697x | 16.5000 ms | 16.0000 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** Unary clamp; HBM roofline. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.5000 ms, REF=16.0000 ms, **SPEEDUP=0.9697x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0127x:** see ako_runs/RESULTS.md for the cross-DSL table.

## iter-1: float4 vectorized load/store (+ scalar tail kernel)
- Changed scalar grid-stride to float4 vectorized (4 elems/thread) with a scalar tail kernel for n%4.
- SPEEDUP: 0.9938x  RUNTIME: 16.1ms  CORRECT: True  -> KEEP (best)
- Baseline was 0.9697x/16.5ms. float4 reaches ref parity (HBM bandwidth floor). Memory-bound clamp; no further headroom.

## Iter 2 — floor confirmation (2 distinct directions) — AT FLOOR, no change
- **Baseline re-bench (GPU 1, --num-warmup 200):** SPEEDUP 0.9938x, RUNTIME 16.1ms, REF 16.0ms, CORRECT.
- **Roofline / floor-proof:** tensor = 4096×393216 = 1.61e9 f32 = 6.44 GB. Elementwise clamp
  moves read+write = 12.88 GB. 12.88 GB / 16.1 ms ≈ **800 GB/s**, vs RTX 6000 Ada spec peak
  ~960 GB/s → **~83% of peak**. torch reference runs at the *identical* 16.0 ms (805 GB/s), so
  the reference is at the same bandwidth — beating it >3% with a hand-rolled full-line float4
  streaming kernel is not physically available. Fast-signal noise floor std ≈ 0.4 ms ≈ 2.5%,
  below the >3% keep bar → any sub-3% "win" is noise.
- **Direction A — streaming cache hints (`__ldcs` load + `__stcs` store, cuda_noptx-legal intrinsics):**
  hypothesis = cut write-allocate read-for-ownership traffic. Result mean 16.9ms / min 16.7ms
  vs baseline 16.7/16.6 → **no gain** (within noise). Full-cache-line coalesced float4 stores
  (32 thr × 16 B = full line) already elide RFO on Ada; no write-allocate headroom. REVERT.
- **Direction B — occupancy/ILP probe (2× float4 per thread, double-issued loads before stores):**
  hypothesis = more memory-level parallelism improves latency hiding. Result mean 16.6ms /
  min 16.6ms vs baseline 16.7/16.6 → **min unchanged**, no real gain. Grid-stride float4 already
  saturates HBM MLP. REVERT.
- **Verdict:** both genuinely distinct levers (memory-hint + ILP/occupancy) leave min runtime
  inside the ±0.4ms noise band. **AT FLOOR** at 0.9938x (HBM-bandwidth bound, at torch parity).
  Baseline restored verbatim (git-diff-clean); no edit kept. Stopping per depth policy (no padding).
