# Iteration Log — elu / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/elu/triton/solution/elu.py`,
Triton speedup 0.9938x); benched against the same `reference/activation/elu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of elu | 0.9697x | 16.5000 ms | 16.0000 ms | correct |
| 2 | float4 vectorized (16B loads/stores) + scalar tail | 1.0000x | 16.0000 ms | 16.0000 ms | correct (KEEP) |
| 3 | FLOOR CONFIRM: occupancy + vector-width sweep | 1.0000x | 16.0000 ms | 16.0000 ms | at floor (no change) |

## Iter 3 — Floor confirmation (re-run baseline, 2 distinct off-baseline directions)

- **Baseline re-bench on GPU 1 (--num-warmup 200):** COMPILED=True, CORRECT=True (5/5),
  RUNTIME=16.0000 ms == REF 16.0000 ms, **SPEEDUP=1.0000x**. Confirms iter-2 KEEP holds.
- **Roofline math:** n = 4096×393216 = 1.611e9 f32; traffic = 2·4·n = 12.88 GB;
  at 16.0 ms → **805 GB/s effective ≈ 84% of the RTX 6000 Ada GDDR6 960 GB/s spec**
  (i.e. right at the achievable copy roofline). NOTE: `torch.rand` inputs are all ≥0,
  so ELU takes only the `x>0` identity branch — this is a **pure copy**, no expf/compute;
  the floor is literally memcpy bandwidth, which torch's F.elu already saturates → 1.00x is the ceiling.
- **Direction A — occupancy / block-count** (float4, threads 256→512, block cap 131072→524288,
  fast-signal --no-ref -n20): min **16.7 ms** vs baseline min 16.6 ms → **no gain (slightly worse)**.
  More resident threads don't help a link already at ~84% of spec.
- **Direction B — vectorization width** (float2, 8B/thread instead of 16B float4,
  fast-signal): min **16.6 ms** → **tie, no gain**. 16B float4 already issues wide enough
  memory transactions; narrowing to 8B doesn't change achieved BW.
- **Verdict:** AT FLOOR. Three genuinely distinct levers (vector width 4→2, occupancy/threads,
  grid/block cap) all land on the same ~16.6-16.7 ms. Bandwidth-bound copy roofline reached;
  no headroom over torch. **KEEP iter-2 float4 solution verbatim.** Stopping (padding trivial
  block tweaks would not change the conclusion).

## Iter 2 — float4 vectorization

- **Hypothesis (ADVICE):** memory-bound; scalar grid-stride leaves a few % of HBM
  bandwidth on the table. Reinterpret as float4 so each thread moves 16B in/16B out,
  halving instruction overhead and saturating HBM.
- **Change:** `elu_k4` processes float4 (n/4 vectors); scalar `elu_k` handles the
  ragged tail. Input n=4096*393216 is div by 4 so the tail kernel is a no-op here.
- **Bench (--num-warmup 200):** COMPILED=True, CORRECT=True (5/5),
  RUNTIME=16.0000 ms == REF 16.0000 ms, **SPEEDUP=1.0000x** (up from 0.9697x).
- **Verdict:** KEEP. Now exactly at the HBM roofline (runtime == ref). at_floor.

## Iter 1 — cuda_noptx port

- **Hypothesis:** Unary (alpha from init); HBM roofline. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.5000 ms, REF=16.0000 ms, **SPEEDUP=0.9697x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9938x:** see ako_runs/RESULTS.md for the cross-DSL table.
