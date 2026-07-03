# Iteration Log — elu / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/elu/triton/solution/elu.py`,
Triton speedup 0.9938x); benched against the same `reference/activation/elu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of elu | 1.0000x | 16.0000 ms | 16.0000 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** Unary (alpha from init); HBM roofline. Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.0000 ms, REF=16.0000 ms, **SPEEDUP=1.0000x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9938x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Re-bench (baseline, this session) — at floor
- COMPILED=True, CORRECT=True, RUNTIME=16.0000 ms, REF=16.0000 ms, SPEEDUP=1.0000x.
- ELU is memory-bound elementwise; read+write of full tensor = HBM roofline. Already float4-vectorized + st.global.cs streaming store. Math (expf) is free under the bandwidth ceiling. No lever > 3%. Status: at_floor.

## Floor-confirmation session (2026-07-02, GPU 2) — 2 distinct probes, both null

**Baseline re-bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True, RUNTIME=16.0 ms, REF=15.9 ms, **SPEEDUP=0.9938x**.

**Roofline:** input = torch.rand(4096, 393216) = 1.61e9 f32 = 6.44 GB; read+write = 12.88 GB. At 16.0 ms → ~805 GB/s effective ≈ 84% of the RTX 6000 Ada ~960 GB/s HBM peak. Theoretical floor ~13.4 ms. **PyTorch's mature F.elu sits at the identical point (15.9 ms)** — the strongest available floor proof. Note torch.rand ∈ [0,1) so every element takes the `x>0` identity branch; expf is never evaluated → op is a pure bandwidth-bound copy.

Fast-signal (--no-ref --num-perf-trials 20), all three benched under identical warmup regime:

| Direction | mean ms | min ms | std | CORRECT |
|-----------|---------|--------|-----|---------|
| Baseline (float4 __ldg + st.global.cs streaming store, 1 f4/thread) | 16.7 | 16.6 | 0.436 | True |
| **Probe A** — plain float4 store (drop st.global.cs; test write-combining vs evict-first) | 16.7 | 16.6 | 0.442 | True |
| **Probe B** — 2× float4/thread (2 __ldg in flight before stores) for more MLP | 16.7 | 16.6 | 0.427 | True |

- **Probe A (store cache-hint):** identical to baseline. On this chip streaming (cs) and plain float4 stores are indistinguishable for a write-once output — neither pollutes nor benefits at the bandwidth ceiling. **REVERT** (no gain).
- **Probe B (per-thread ILP/MLP):** identical to baseline. The grid-stride loop + __ldg already saturates memory-level parallelism; adding a second in-flight load does nothing. **REVERT** (no gain).
- Skipped block-count/grid-size sweep: a non-lever under a grid-stride loop (depth policy calls it padding).
- **Conclusion: AT FLOOR.** Two genuinely distinct levers (memory cache-hint, per-thread MLP) both land at exactly the baseline (16.7 ms mean / 16.6 ms min), and the baseline already matches vendor F.elu within noise (std ≈ 2.7% ≫ any observed delta). No lever clears the ~3% noise gate. Baseline kept verbatim (git-diff-clean); improved=false.
- **Next:** none. Stop — floor confirmed.
