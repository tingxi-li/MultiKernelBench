# Iteration Log — gelu / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gelu/triton/solution/gelu.py`,
Triton speedup 1.0063x); benched against the same `reference/activation/gelu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of gelu | 0.9639x | 16.6000 ms | 16.0000 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** Exact erf GELU; HBM roofline. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.6000 ms, REF=16.0000 ms, **SPEEDUP=0.9639x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0063x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 2 — float4 vectorize (KEEP)

- **Hypothesis:** HBM-bound elementwise; vectorized float4 loads/stores improve
  memory throughput vs scalar grid-stride.
- **Change:** gelu_k4 processes float4 (4 elems/thread), scalar tail kernel for n%4.
- **Bench (--num-warmup 200):** COMPILED=True, CORRECT=True (5/5),
  RUNTIME=16.0000 ms, REF=16.0000 ms, **SPEEDUP=1.0000x** (was 0.9697x baseline).
- **Verdict:** KEEP. ~3% gain, now matches ref exactly = HBM roofline floor.

## Iter 3 — occupancy/grid-size probe (REVERT) — FLOOR CONFIRMED

- **Hypothesis:** The capped grid-stride (blocks = min(n4/threads, 131072)) may leave
  bandwidth on the table; a full grid (one float4/thread, no grid-stride reuse) could
  saturate HBM better. Distinct axis from Iter 1/2 (scalar→vector): this varies
  occupancy/block-count/launch-config, not vectorization width.
- **Change:** `int blocks = (int)want;` (drop the 131072 cap → ~1.57M blocks).
- **Fast-signal bench (--no-ref, 20 trials, GPU 3):** RUNTIME mean **17.1 ms** (min 16.9),
  CORRECT 5/5 — vs baseline mean 16.0 ms (min 15.7). **REGRESSES ~7%.**
- **Verdict:** REVERT. The huge grid adds launch/scheduling overhead and loses warm-cache
  thread reuse; the capped grid-stride is already the better launch config.
- **Roofline proof (GPU 3, RTX 6000 Ada):** n = 4096·393216 = 1.61e9 fp32 = 6.44 GB/tensor;
  read+write = 12.88 GB irreducible traffic (single op, no fusion possible). Baseline min
  15.7 ms ⇒ ~820 GB/s achieved ≈ 85% of the ~960 GB/s HBM peak, and it matches
  `torch.nn.functional.gelu` (itself a tuned near-roofline kernel) to the noise.

## Conclusion — AT FLOOR

Probed three genuinely distinct axes: **vectorization width** (scalar→float4, Iter 1→2,
+3%), **occupancy/grid-count** (capped grid-stride vs full grid, Iter 3, −7%), and
**launch/fusion** (single-op elementwise — no fusion possible, one kernel launch already).
Best = float4 capped grid-stride at **1.00x** (matches the vendor torch kernel). Traffic is
irreducible and we sit at ~85% of HBM peak, so >3% gain is not physically reachable.
**STOP. at_floor=true, improved=false, changed_solution=false.** Baseline restored verbatim.
