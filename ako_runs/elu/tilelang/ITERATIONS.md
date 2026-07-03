# Iteration Log — elu / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/elu/triton/solution/elu.py`,
Triton speedup 0.9938x); benched against the same `reference/activation/elu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of elu | 1.0256x | 15.6000 ms | 16.0000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** Unary (alpha from init); HBM roofline. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=15.6000 ms, REF=16.0000 ms, **SPEEDUP=1.0256x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9938x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Re-bench session (GPU3)
- baseline: SPEEDUP 1.0256x, RUNTIME 15.6ms, CORRECT True
- Analysis: memory-bound elementwise ELU (read N + write N floats), already coalesced contiguous access; at HBM bandwidth floor. ref 16.0ms.
- final: SPEEDUP 1.0256x, RUNTIME 15.6ms, CORRECT True. status=at_floor (no iters attempted; obvious floor).

## Floor-confirmation session (GPU3, 2026-07-02)

**Roofline math.** N = 4096 x 393216 = 1,610,612,736 float32 = 6.44 GB/tensor.
ELU reads N + writes N = 12.88 GB. At 15.6 ms => **825 GB/s**, i.e. **~86% of the
RTX 6000 Ada GDDR6 peak (~960 GB/s, 384-bit)** — the practical ceiling for a
memory-bound kernel. Max realistic upside (90% peak) ~14.9 ms = ~4.5%, below the
baseline noise floor (std 0.465 ms ~= 3%). Note N = 196608 x 8192 exactly, so the
`if idx < N` guard never fails at runtime.

### Iter 2 — guard removal / vectorization
- **Hypothesis:** the always-true `if idx < N` predicate (N % BLK == 0) may block
  the compiler from emitting vectorized (float4) loads/stores; dropping it could
  raise achieved bandwidth.
- **Change:** removed the `if idx < N` guard from the T.Parallel loop.
- **Fast-signal (--no-ref, 20 trials):** mean 16.4 ms, min 15.5 ms, CORRECT True.
- **Verdict: REVERT.** No gain vs baseline (mean 15.6 / min 15.3). Within noise and
  slightly worse mean. tilelang already coalesces the contiguous Parallel access;
  the guard was not the limiter. (Also unsafe for general N — only correct because
  this benchmark's N is BLK-divisible.)
- **Next:** test whether occupancy/block-count is the limiter.

### Iter 3 — occupancy (block-count) sweep
- **Hypothesis:** BLK=8192 (32 elem/thread, 196608 blocks) may under-saturate the
  memory subsystem; more/smaller blocks could improve latency hiding.
- **Change:** BLK 8192 -> 1024 (4 elem/thread, ~1.57M blocks), TH=256 unchanged.
- **Fast-signal (--no-ref, 20 trials):** mean 16.8 ms, min 16.5 ms, CORRECT True.
- **Verdict: REVERT.** Worse than baseline — the kernel is not occupancy-limited;
  more blocks add launch/scheduling overhead without raising bandwidth.
- **Next:** none. Two distinct directions (vectorization, occupancy) both fail to
  beat the baseline.

### Conclusion — AT FLOOR
Baseline (BLK=8192, TH=256, scalar coalesced Parallel) sits at **~86% of GDDR6 peak
bandwidth**, beating PyTorch F.elu (16.0 ms). Neither vectorization/guard-removal
nor occupancy tuning moved runtime above the noise floor. Solution restored to the
committed baseline **verbatim** (md5 f161b21f...). **at_floor=true, improved=false,
changed_solution=false.** Iteration cap (2) respected.
