# Iteration Log — group_norm / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/group_norm/triton/solution/group_norm.py`,
Triton speedup 0.9904x); benched against the same `reference/normalization/group_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of group_norm | 0.9172x | 33.8000 ms | 31.0000 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** GroupNorm 8 groups; per-(batch,group) reduction (8.6GB). Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=33.8000 ms, REF=31.0000 ms, **SPEEDUP=0.9172x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9904x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Re-bench session (GPU 3) — AT FLOOR

Baseline re-bench (same-GPU): COMPILED=True, CORRECT=True, RUNTIME=33.8ms, SPEEDUP=0.9172x
(min 31.0ms == ref min 30.7ms; mean inflated by ONE Trial-1 outlier ~296ms).

Roofline: tensor=8.59GB, two-pass GN traffic = 3x = 25.77GB. At min runtime 30.9ms => 831 GB/s = HBM
roofline (== torch). 3x traffic is irreducible for exact two-pass group_norm; torch pays it too.

| Iter | Change | Speedup | Runtime | min | Correct | Keep |
|------|--------|---------|---------|-----|---------|------|
| 1 | ld.global.cs.v4 streaming loads (replace __ldg) both kernels | 0.9091x | 34.1ms | 31.1 | True | revert (0%, same outlier 295ms) |
| 2 | TPB 256->512 | 0.9172x | 33.8ms | 30.9 | True | revert to baseline (tie on mean) |
| final | restored baseline (TPB=256, __ldg, float4 + st.global.cs) | 0.9172x | 33.8ms | 30.9 | True | KEEP |

Conclusion: AT FLOOR. min runtime ties torch (30.9 vs 30.7ms) at the 831 GB/s HBM roofline.
The 0.92x mean is a harness artifact: a single Trial-1 ~300ms outlier that is KERNEL-INDEPENDENT
(296/295/300ms across baseline/streaming-load/TPB=512 — a fixed cost no kernel change moves;
torch's reference does not incur it). float4 vec loads + st.global.cs streaming stores + fp32
accumulators + NG=1024 stats grid already applied. No further kernel-side headroom.

---

# Re-open (GPU 1) — prior "AT FLOOR" was WRONG: 3x is NOT irreducible

The prior conclusion ("3x traffic irreducible, AT FLOOR") is contradicted by the
**Triton sibling**, which uses an L2-reuse chunk pipeline (3x -> 2x DRAM traffic).

**Diagnostics on GPU 1 (my GPU):**
- Baseline (this naive two-pass CUDA): min **30.8ms**, mean 33.8 (one 300ms outlier), SPEEDUP 0.9172x.
  ref min 30.7, mean 31.0. => baseline is at the 3x roofline (834 GB/s), NOT the real floor.
- Triton sibling (read-only probe): min **20.1ms**, mean 29.5 (292ms outlier). => L2-reuse
  genuinely materializes here (2x traffic, ~855 GB/s). Triton's reported 0.99x is a pure
  mean-artifact hiding a real ~1.5x min win.

Physical floor is now **2x traffic = 17.18GB** -> ~18-20ms (Triton demonstrates 20.1ms min).
Remaining work = tune the chunk pipeline toward roofline (K / SPLIT / SPLITN / TPB / store policy).

## Iter 1 — port L2-reuse chunk pipeline to CUDA (C++-side chunk loop)

- **Hypothesis:** replicate Triton's L2-resident chunking in CUDA. C++ launcher walks the
  tensor K groups at a time; per chunk: cooperative `gn_stats` (SPLIT blocks/group,
  double atomic reduction) then `gn_apply` (GPC*SPLITN blocks, float4 + st.global.cs stream
  store). Chunk (K*8MB) stays in L2 so apply re-reads from L2 => 2x traffic. Launch from C++
  (not Python) to hide the 512 per-chunk launches behind GPU work.
- **Config:** K=4, SPLIT=32, SPLITN=32, TPB=256 (mirror Triton).
- **Bench (verdict, warmup 200, 50 trials):** COMPILED=True, CORRECT=True (5/5),
  min **22.3ms**, mean 27.8 (one 289ms outlier), ref mean 31.2 => **SPEEDUP 1.1223x**.
  Typical (outlier-removed) ~22.5ms => ~1.39x; the 50-trial mean is outlier-dragged.
- **KEEP** — min 30.8 -> 22.3ms confirms L2 reuse fires in CUDA. New best.
- **Next:** tune K / SPLIT / SPLITN / TPB toward the 20ms roofline; min 22.3 vs Triton 20.1
  says there is still ~2ms of occupancy/residency headroom.

## Iter 2 — config sweep (K, SPLIT, SPLITN, TPB, store policy)

Swept via an in-process harness (clean min==med==mean, no outlier) ranking by min runtime.
- **K** (SPLIT=32,SPLITN=32): K=2:33.5, K=3:24.9, **K=4:22.3**, K=6:27.1, K=8:27.4. K=4 optimal
  (32MB chunk = best L2 residency/occupancy tradeoff; matches Triton).
- **SPLIT** (K=4,SPLITN=32): 16:30.9, 32:22.3, **64:21.9**, 128:22.1, 256:26.4. SPLIT=64 best
  (stats grid = K*SPLIT = 256 blocks -> better SM occupancy on 142 SMs).
- **SPLITN** (K=4,SPLIT=32): **32:22.3**, 16:22.4, 64:23.8, 128:30.8. SPLITN=32 best.
- **TPB**: **256:22.3**, 128:25.5, 512:23.6. 256 best.
- **Store policy**: st.global.cs / cg / plain float4 all ~22.3 — write policy immaterial (writes
  are pure DRAM streaming either way; L2 already holds the input chunk).
- **Global optimum: K=4, SPLIT=64, SPLITN=32, TPB=256, cs-store -> min 21.93ms** (clean median).
  maxerr 7.15e-07 vs golden (double atomics well within 1e-4 tol).
- **KEEP** SPLIT 32->64. New best 21.93ms clean (~1.41x vs ref 31 clean of outlier).
- **Next:** 21.93 vs 2x-floor ~20.6ms (17.18GB / 834 GB/s). ~1.3ms gap = inter-kernel bubbles
  across 512 launches. Try a persistent cooperative-grid kernel (grid.sync between stats/apply
  phases, one launch) and a cudaStreamSetAttribute L2-persistence window.

## Iter 3 — persistent cooperative-grid kernel (grid.sync, one launch) — REVERT

- **Hypothesis:** replace the 512 C++ kernel launches with a single cudaLaunchCooperativeKernel;
  loop chunks inside the kernel, grid.sync() between the stats and apply phases. Removes any
  inter-kernel launch bubble.
- **Result:** K4/S64/SN32 -> min **25.67ms** (WORSE than 21.93). S32 variant 42.8ms.
  Correct (maxerr 7.15e-07).
- **REVERT.** Cooperative launch caps resident blocks (occupancy), and 512 device-wide grid.sync
  barriers cost more than the launch gaps they remove. **Conclusion: the C++ multi-launch is
  already launch-hidden** — the residual ~1.3ms gap is genuine BW/phase efficiency, not bubbles.
- **Next:** micro-opt the memory loops (ILP/unroll, apply-side __ldcs streaming read since the
  chunk is not reused after apply) — see if MLP recovers any of the gap.

## Iter 4 — ILP/unroll + apply load policy — REVERT (keep clean __ldg)

- **Hypothesis:** more in-flight loads (UNROLL 2/4) raise MLP; a streaming apply-read
  (ld.global.cs) avoids polluting L2 on the last touch.
- **Result** (K4/S64/SN32/TPB256): UNROLL=1 __ldg **21.95**; UNROLL=2 22.14; UNROLL=4 22.16
  (unroll HURTS — register pressure lowers occupancy). Streaming apply-read (ld.global.cs)
  21.83 — a real but ~0.5% gain, inside run-to-run noise.
- **REVERT / keep clean __ldg SPLIT=64.** 0.5% is below the meaningful margin and not worth an
  extra inline-asm load. Note: the STATS load MUST stay caching (__ldg), never streaming — it
  populates L2 for the apply re-read; a streaming stats load would evict-first and break reuse.
- **Next:** try overlapping stats(c+1) with apply(c) via 2 streams (both chunks fit 96MB L2) to
  fill per-kernel tail waves.

## Iter 5 — 2-stream overlap (stats(c+1) || apply(c)) — REVERT

- **Hypothesis:** run stats and apply on separate streams (event-ordered apply(c) after stats(c))
  so stats(c+1) overlaps apply(c), keeping DRAM always busy and hiding per-kernel tail waves.
- **Result:** K4/S64 -> min **30.9ms** (much WORSE); K6/K8 ~30.5. Correct.
- **REVERT.** Overlap mixes DRAM reads (stats) with writes (apply) -> GDDR read/write turnaround
  penalty, and doubles L2 residency pressure (2 chunks + streaming stores) which breaks the apply
  re-read hit -> regresses toward 3x. **The sequential read-phase/write-phase batching is what
  makes the 2x design fast** — confirmed by this regression.
- **Next:** confirm the reduction epilogue / accumulator precision are not on the critical path
  (warp-shuffle reduction, fp32 vs double accum) — expect no change on a DRAM-bound kernel.

## Iter 6 — warp-shuffle reduction + fp32 accumulators — KEEP (new best)

- **Hypothesis:** the double per-thread accum + double shared-tree reduction + double global
  atomics add a little ALU/shared/occupancy cost to the stats phase; fp32 accumulators (with the
  mean/var/rstd finalize still in double) plus a warp-shuffle reduction lighten it.
- **Result** (K4/S64/SN32/TPB256), confirmed over two runs:
  - double + shared-tree (prev best): 21.91
  - fp32 + shared-tree: 21.54
  - fp32 + warp-shuffle: **21.46 / 21.43** (best); double + warp-shuffle 21.88 (shuffle alone
    is neutral -> the win is the fp32 accumulators). S32/S128/SN16 all worse under fp32 too.
  - maxerr **~3e-6** vs golden (fp32 sum of 2M U[0,1] values, double finalize) -- 30x inside the
    1e-4 tol; worst-case atomic-accumulation error ~2e-6 on the mean.
- **KEEP.** New best 21.46ms clean (~1.44x vs ref 31 clean). Ported to solution: fp32 sumacc/sqacc,
  warp-shuffle stats reduction, double mean/rstd finalize.
- **Floor check:** 21.46 vs 2x-DRAM roofline ~20.6ms = **96% of the 2x floor.** Confirmed near-floor
  via 4 distinct failed/neutral directions (persistent grid.sync, ILP/unroll, 2-stream overlap,
  larger/smaller K/SPLIT). Remaining ~0.9ms is per-kernel tail-wave overhead intrinsic to the
  two-kernel-per-chunk pipeline.

## FINAL VERDICT (GPU 1, scripts/bench.sh final, warmup 200, 100 trials)

- **COMPILED=True, CORRECT=True (5/5), SPEEDUP=1.2810x** (mean 24.2ms, min 21.4; ref mean 31.0,
  min 30.7). trajectory/20260702_173101_final.
- Baseline on same GPU was 0.9172x (mean 33.8, min 30.8). **0.9172x -> 1.2810x** apples-to-apples
  (both carry the identical deterministic Trial-1 ~289ms cudaMalloc(8.59GB) outlier; the ref does
  not — so eating it is the fair choice, not a persistent-buffer hack).
- Clean (outlier-removed) is ~21.5ms mean -> ~1.44x; the reported 1.281x is the mean dragged by
  that one fixed Trial-1 spike (deterministic, re-running only reproduces or worsens it).
- **NOT at floor:** ~96% of the 2x-traffic HBM roofline (~20.6ms); the Triton sibling reached
  20.1ms min (noisy, std 49.5). Prior run's `at_floor=true` was WRONG — it assumed 3x irreducible;
  cutting 3x->2x DRAM traffic via the L2-reuse chunk pipeline is the whole win.
- Detector: valid=True, regression_type=None (forward is allocate/launch glue only).
