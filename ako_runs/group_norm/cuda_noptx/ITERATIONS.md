# Iteration Log — group_norm / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/group_norm/triton/solution/group_norm.py`,
Triton speedup 0.9904x); benched against the same `reference/normalization/group_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of group_norm | 0.8988x | 34.6000 ms | 31.1000 ms | correct |
| AKO-base | re-bench baseline (GPU2) | 0.9014x | 34.5000 ms | 31.1000 ms | correct |
| AKO-1 | float4 vectorized loads/stores both kernels | 0.9174x | 33.9000 ms | 31.1000 ms | correct, KEEP |
| AKO-2 | apply block size 256->512 | 0.9174x | 33.9000 ms | 31.1000 ms | correct, no change -> revert |
| AKO-final | restored AKO-1 (float4, TPB) | 0.9174x | 33.9000 ms | 31.1000 ms | correct |
| AKO2-base | re-bench baseline (GPU0) | 0.9172x | 33.8000 ms | 31.0000 ms | correct, min=31.0ms |
| AKO2-3 | MLP unroll x4 (both kernels) | — | min 31.1 ms | — | correct, no change -> revert |
| AKO2-4 | non-temporal __stwt stores on y | — | min 31.0 ms | — | correct, no change -> revert |
| AKO2-5 | TPB 256->128 (occupancy) | — | min 31.1 ms | — | correct, no change -> revert |
| AKO2-6 | stats 4-way split (4096 blk, atomic+finalize) | — | min 31.0 ms | — | correct, no change -> revert |
| AKO2-final | restored baseline (float4, TPB256) verbatim | 0.9172x | 33.8000 ms | 31.0000 ms | correct, git-clean |

### AKO round 2 notes (floor re-confirmed, GPU0)
- **Measurement:** fast-signal MEAN is worthless here (44.7ms, std ~59) — dragged by a per-run clock-ramp outlier (trial1 ~295ms) that 20 trials can't average out. **min** is the clean signal and is rock-stable at 31.0ms across two baseline runs (run-to-run min variance ≈ 0). Judged every candidate on min; required <=30.6ms to count as a real win. None reached it.
- **Verdict SPEEDUP 0.917x is an environmental artifact, not the kernel.** In the final verdict the SECOND profiling loop ("additional checks") re-measures the identical kernel and settles at **30.6-30.8ms** (min 30.6) — **at parity** with ref (min 30.6, mean 31.0), not below it. The first loop's 0.917x is entirely the clock-ramp outlier on trials 1-7 (max ~295ms); nothing in the kernel can edit it away.
- **Roofline:** 25.77GB fixed traffic (stats reads x 8.59GB; apply reads x + writes y 17.18GB) / 30.8ms = **837 GB/s = 87% of the 960 GB/s GDDR6 peak** — the practical streaming ceiling. Traffic cannot drop below 25.77GB: the 8MB group exceeds shared mem (228KB) and any register budget, so the apply-pass x re-read cannot be cached; a fused-persistent L2-resident scheme would need to throttle to ~12 blocks and lose far more BW than the 33% it saves. torch sits at the same 837 GB/s.
- **5 distinct directions, all null:** (3) MLP unroll — compiler already software-pipelines the independent loop; (4) __stwt streaming stores — write-only y never re-read, no L2 pollution to remove; (5) TPB sweep 128/256/512 — occupancy is not the limiter; (6) stats 4-way split to 4096 blocks — the ~1.2-wave stats grid was NOT tail-limited (even a 160-block tail saturates BW). Status: **at_floor, improved=false, baseline restored verbatim (git-clean).**

### AKO notes (bandwidth floor)
- float4 (AKO-1) lifted 0.9014x->0.9174x (~1.8%); apply block-size 512 (AKO-2) gave 0 change.
- This op is two-pass HBM-bound: 25.8GB fixed traffic (stats reads x 8.6GB; apply reads x + writes y 17.2GB). torch hits the same floor.
- final min=31.0ms ~= ref min=30.7ms => at the HBM roofline. The reported mean (33.9ms) is dragged by a single clock-ramp outlier (max=299ms, std=27); min is the clean signal and is at parity.
- Accumulators already fp32 (ls/lss); only the 1024x per-block final mu/var/sqrt is double (perf-irrelevant, helps 1e-4 gate). Status: at_floor.

## Iter 1 — cuda_noptx port

- **Hypothesis:** GroupNorm 8 groups; per-(batch,group) reduction (8.6GB). Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=34.6000 ms, REF=31.1000 ms, **SPEEDUP=0.8988x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9904x:** see ako_runs/RESULTS.md for the cross-DSL table.
