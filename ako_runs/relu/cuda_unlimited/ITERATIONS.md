# Iteration Log — relu / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/relu/triton/solution/relu.py`,
Triton speedup 1.0000x); benched against the same `reference/activation/relu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of relu | 1.0000x | 16.0000 ms | 16.0000 ms | correct |
| re-run | floor-confirm (3 null dirs), baseline kept | 1.0000x | 16.0000 ms | 16.0000 ms | correct, AT FLOOR |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** Unary elementwise; HBM-bandwidth roofline. Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.0000 ms, REF=16.0000 ms, **SPEEDUP=1.0000x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0000x:** see ako_runs/RESULTS.md for the cross-DSL table.

## baseline (re-bench, GPU0)
- unchanged solution: float4 + st.global.cs.v4 streaming store
- SPEEDUP 1.0000x, RUNTIME 16.0ms == REF 16.0ms, CORRECT True
- ReLU is pure read+write elementwise => HBM bandwidth bound. Already vectorized 128-bit + streaming store. At physical floor. No iter attempted.

## Re-run pass (GPU1, interrupted-cell restart) — FLOOR CONFIRMED, baseline kept verbatim

**Baseline re-bench on GPU1 (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
RUNTIME mean 16.0ms (settled **min 15.7–15.8ms**), REF mean 16.0ms (min 15.8ms), **SPEEDUP=1.0000x**.
Ties PyTorch `torch.relu` exactly.

**Roofline:** input 4096×393216 = 1.61e9 fp32 = 6.44 GB; ReLU reads x + writes y => 12.9 GB
HBM traffic. RTX 6000 Ada ≈ 960 GB/s => theoretical floor ≈ 13.4 ms. Measured min 15.8 ms ≈
**85% of peak bandwidth** — the same efficiency PyTorch's own kernel hits. There is no compute
term (one fmaxf/elem), so nothing to overlap; the op is a pure memcpy-with-clamp.

Fast-signal A/B (all `--num-warmup 200 --num-perf-trials 40`, ranked by settled MIN; the ~0.4ms
mean wobble and 18.6ms clock-ramp max are pure noise, min is the reliable metric):

| Direction (distinct lever) | Change | min ms | vs base (15.8) |
|---|---|---|---|
| baseline | float4 __ldg + st.global.cs.v4 | 15.8 | — |
| A · cache/streaming hint | plain `float4` store (drop st.global.cs) | 15.8 | null |
| B · occupancy / launch-count | uncapped grid, 1 float4/thread (no grid-stride loop) | 15.7 | null (1-tick) |
| C · register/occupancy bound | drop `__launch_bounds__(256,6)` | 15.8 | null |

**Conclusion — AT FLOOR.** Three genuinely distinct directions (cache hint, grid/occupancy,
register cap) all land inside the ~0.4ms noise band of baseline; none is a repeatable win. float4
is the max vectorization width and the kernel is already a single coalesced 128-bit read + 128-bit
write per element with a write-once streaming store. Bandwidth-bound at ~85% of peak, tying
PyTorch. No change kept — **committed baseline restored byte-identical (git-diff-clean)**.
improved=false, at_floor=true, changed_solution=false.
