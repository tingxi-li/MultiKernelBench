# Iteration Log — swish / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/swish/triton/solution/swish.py`,
Triton speedup 2.5253x); benched against the same `reference/activation/swish.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of swish | 2.4472x | 16.1000 ms | 39.4000 ms | correct |
| iter-1 | float4 vectorized load/store + scalar tail | 2.4472x | 16.1000 ms | 39.4000 ms | correct (revert, 0% — HBM floor) |
| rr-2 | float4 + fast-math (__expf/__fdividef) | — | 16.1 / min 15.8 | — | correct, revert (floor) |
| rr-3 | float4 + streaming __ldcs/__stcs | — | 16.3 / min 15.9 | — | correct, revert (worse) |
| rr-4 | float4 ×2 per-thread unroll (more MLP) | — | 16.1 / min 15.7 | — | correct, revert (floor) |
| rr-5 | float4 threads=512 | — | 16.1 / min 15.7 | — | correct, revert (floor) |
| rr-6 | float4 uncapped full grid (no stride loop) | — | 16.3 / min 15.8 | — | correct, revert (worse) |
| FINAL | committed baseline restored verbatim | 2.4472x | 16.1000 ms | 39.4000 ms | correct — at floor |

## Re-run (session-limit re-run) — floor reconfirmed across 5 distinct directions

Baseline re-benched on GPU 0 (RTX 6000 Ada): mean **16.1 ms**, min **15.7 ms**, SPEEDUP **2.4472x**.
Roofline: traffic = 12.885 GB (6.44 read + 6.44 write, irreducible single fused pass);
at 960 GB/s GDDR6 peak the 100%-efficiency wall is 13.42 ms. Baseline min 15.7 ms = **821 GB/s = 85.5%**
of peak — already good-tuning territory for a read+write elementwise op. Measurement noise: std ≈ 0.46 ms,
so any candidate must drop the full-100-trial mean clearly below ~15.7 ms to count (fast-signal deltas < ~0.3 ms are noise).

Fast-signal protocol per candidate: `--no-ref --num-warmup 200 --num-perf-trials 100` (full-quality, ref-free), compare mean/min.

- **rr-2 float4 + fast-math** (`__expf`, `__fdividef`; x∈[0,1) → ≤3 ulp, well inside 1e-4 tol): mean 16.1 / min 15.8.
  The one scenario float4-alone (prior iter-1) and fast-math-alone each miss is instruction-issue binding; combined it still
  does not move → confirms not issue-bound. **Revert.**
- **rr-3 float4 + streaming cache hints** (`__ldcs` read-once / `__stcs` write-once, evict-first L2): mean 16.3 / min 15.9.
  Cache-bypass does not raise achieved DRAM throughput here (array ≫ L2 so L2 is thrashed regardless); marginally worse. **Revert.**
- **rr-4 float4 ×2 per-thread unroll** (2 independent float4 in flight → more memory-level parallelism): mean 16.1 / min 15.7.
  Grid-stride already pipelines enough; no gain. **Revert.**
- **rr-5 float4 threads=512** (occupancy/launch sweep vs 256): mean 16.1 / min 15.7. Launch config is not the limiter. **Revert.**
- **rr-6 float4 uncapped grid** (each thread exactly one float4, no grid-stride loop, max resident-wave parallelism):
  mean 16.3 / min 15.8. 1.57M-block launch overhead makes it marginally worse. **Revert.**

**Conclusion:** every distinct memory-efficiency lever (vectorization width, fast-math, streaming hints, per-thread MLP,
block size, grid sizing) lands at the same ~15.7 ms / 821 GB/s DRAM floor. The op is HBM-bandwidth bound at the practical
GDDR6 ceiling; there is no reachable speedup above the committed baseline. **Restored the committed baseline verbatim**
(git-diff-clean). Final verdict: COMPILED=True, CORRECT=True (5/5), RUNTIME=16.1000 ms, SPEEDUP=2.4472x.
Detector: valid=True, regression_type=None (glue-only forward, pass). **at_floor=true, improved=false.**

## Iter 1 (re-bench, float4) — at floor

- **Hypothesis:** float4 vectorized loads/stores reduce memory transaction count, squeezing the last bit on this memory-bound elementwise op.
- **Edit:** swish_k4 processes float4 (4 elems/thread via reinterpret_cast), scalar swish_tail handles the remainder.
- **Bench:** COMPILED=True, CORRECT=True (5/5), RUNTIME=16.1000 ms, SPEEDUP=2.4472x — bit-identical to the scalar grid-stride baseline.
- **Verdict:** 0% change. Kernel is HBM-bandwidth bound (read x + write y = 2 passes already minimal); vectorization does not help. **Reverted to scalar baseline.** At physical floor.

## Iter 1 — cuda_noptx port

- **Hypothesis:** x*sigmoid(x): eager=2 passes, fused=1 -> ~2.5x. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.1000 ms, REF=39.4000 ms, **SPEEDUP=2.4472x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 2.5253x:** see ako_runs/RESULTS.md for the cross-DSL table.
