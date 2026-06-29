# Iteration Log — layer_norm / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/layer_norm/solution/layer_norm.py`,
Triton speedup 1.6050x); benched against the same `reference/normalization/layer_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of layer_norm | 1.4713x | 4.3500 ms | 6.4000 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** LayerNorm last 3 dims; split-row reduction + affine. Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=4.3500 ms, REF=6.4000 ms, **SPEEDUP=1.4713x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.6050x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

## Iter 1 (this run) — per-row-segment apply, float4 + v4 streaming store

- **Change:** rewrote ln_apply to grid=M*S (m,woff from blockIdx, no per-element 64-bit divide),
  hoisted mean/rstd to registers, float4 __ldg on x/w/b + inline-PTX `st.global.cs.v4.f32` store.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=4.7000 ms, REF=6.42, **SPEEDUP=1.3660x**.
- **Verdict: REVERT (regressed vs baseline 4.34ms/1.477x).** float4 + v4 streaming store hurt;
  likely register pressure (3 float4 in flight + v4 store) cut occupancy on this memory-bound pass.

## Iter 2 — per-row-segment scalar apply (divide-free)
- Rewrote ln_apply to grid=M*S, no per-element 64-bit divide, scalar streaming store.
- Bench: CORRECT=True, RUNTIME=4.4400 ms, **SPEEDUP=1.4482x**. Verdict: **REVERT** (ties baseline;
  micro-bench confirmed the per-element divide was NOT the bottleneck — memory latency hides it).

## Iter 3 — column-blocked float4 apply (w/b reuse across rows)  ✅ KEEP
- **Root cause found by micro-bench:** copy floor (x->y, 2GB) = 2.60 ms; baseline apply = 2.92 ms;
  the 0.31 ms gap is redundant w/b reads (same w/b refetched for every one of the M=64 rows).
- **Change:** ln_apply restructured column-blocked — each thread owns 4 cols (float4), loads w/b ONCE,
  sweeps all M rows reusing them from registers; mean/rstd staged in shared; v4 cache-streaming store.
  Apply micro-bench 2.92 -> 2.706 ms (~98% of copy floor).
- **Bench:** COMPILED=True, CORRECT=True (err 1.2e-7), RUNTIME=4.0300 ms, REF=6.43, **SPEEDUP=1.5955x**.
- Detector: valid=True (forward glue-only, kernel via _ext call). Verdict: **KEEP (new best, +8% vs baseline)**.

## Iter 4 — fp32 stats accumulators (rule-5 compliance) ✅ KEEP
- **Change:** sum_acc/sq_acc fp64->fp32, atomics fp32, ln_final in fp32 (rsqrtf). Micro-bench
  confirmed stats is at the read-floor (1.205 ms / 831 GB/s) and fp32-vs-fp64 atomics are perf-identical.
- **Bench:** COMPILED=True, CORRECT=True (still passes 1e-4), RUNTIME=4.0300 ms, **SPEEDUP=1.5955x**.
- Verdict: **KEEP** — same speed as iter-3, now rule-5 compliant (no fp64 accumulators). Detector valid=True.
- **At floor:** total = stats 1.205 + apply 2.706 + final ~0 ≈ 3.91 ms; measured 4.03 ms. Both passes within
  ~2% of their HBM read/copy floors; no further bandwidth headroom on this 3 GB-traffic (x read x2 + y write) op.
