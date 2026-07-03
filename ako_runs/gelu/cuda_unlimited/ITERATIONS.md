# Iteration Log — gelu / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gelu/triton/solution/gelu.py`,
Triton speedup 1.0063x); benched against the same `reference/activation/gelu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of gelu | 1.0000x | 16.0000 ms | 16.0000 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** Exact erf GELU; HBM roofline. Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.0000 ms, REF=16.0000 ms, **SPEEDUP=1.0000x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0063x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Re-bench (this session)
- baseline: SPEEDUP 1.0000x, RUNTIME 16.0ms (ref 16.0ms), CORRECT=True.
  Solution already float4-vectorized + st.global.cs.v4 streaming store + __launch_bounds__(256,6).
  GELU is pure elementwise: reads N + writes N = 2*numel*4 bytes, HBM-bound. 16.0ms == ref => HBM floor.
  No cheap lever remains (already 128-bit coalesced, streaming write). Status: at_floor.

## Floor-proof session (2026-07-02, GPU 0) — 3 orthogonal directions, all tie baseline
Input 4096x393216 = 1.61e9 floats. Traffic = read N + write N = 12.88 GB.
Baseline (this GPU): verdict RUNTIME 16.0ms / min 15.7ms, SPEEDUP 1.0000x, CORRECT (5/5).
Achieved BW = 12.88GB / 15.8ms = ~815 GB/s = ~85% of RTX-6000-Ada ~960 GB/s peak — the
practical GDDR6 ceiling for a read+write elementwise op. We already exactly match PyTorch's
native `torch.nn.functional.gelu` (ref min 15.7ms) — the vendor kernel is no faster.
Fast-signal method: --num-warmup 200 --num-perf-trials 50, ranked by MIN (mean is clock-ramp
contaminated: first ~25 trials sit at 16.6ms then settle to 15.8ms regardless of warmup).

- **Iter A — store cache hint.** Hypothesis: memory subsystem saturated regardless of write
  policy. Replaced inline-PTX `st.global.cs.v4` with a plain write-back `float4` store.
  Result: min 15.8ms == baseline 15.8ms. COMPILED/CORRECT=True. => streaming hint is neutral;
  memory is saturated either way. REVERT (no gain; keep PTX form as committed). Limiter ruled out: write policy.
- **Iter B — ILP / memory-level parallelism.** Hypothesis: MLP-limited. Unroll grid-stride
  loop by 2 (two independent `__ldg` float4 loads issued before compute). Result: min 15.8ms ==
  baseline. COMPILED/CORRECT=True. => occupancy (1536 thr/SM x 142 SM) already saturates HBM;
  extra per-thread MLP adds nothing. REVERT. Limiter ruled out: memory-level parallelism.
- **Iter C — launch config / block count.** Hypothesis: 131072-block grid-stride cap limits
  throughput. Raised cap so the full grid launches (~1.57M blocks, ~1 float4/thread, no stride
  loop). Result: min 15.9ms ~= baseline (within noise, marginally worse). COMPILED/CORRECT=True.
  REVERT. Limiter ruled out: launch configuration.

**Conclusion: AT FLOOR.** Three orthogonal limiters (write policy, MLP, launch config) all
ruled out; op runs at ~85% of GDDR6 peak and exactly matches the vendor kernel. No >3% headroom
exists. Committed baseline restored verbatim (git-clean, md5 10a63be0). improved=false, at_floor=true.

Note: an external process briefly overwrote solution/gelu.py (and a scratchpad backup) with the
cuda_NOPTX cell's kernel mid-session; restored from the git index (authoritative baseline
md5 10a63be0) and re-verified md5 before the final verdict.
