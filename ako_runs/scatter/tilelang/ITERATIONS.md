# Iteration Log — scatter / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/scatter/triton/solution/scatter.py`,
Triton speedup 5.3079x); benched against the same `reference/index/scatter.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of scatter | 5.8842x | 0.0311 ms | 0.1830 ms | correct |

### 2026-07-02 pass (GPU 3). Same-GPU baseline (committed) = 0.0286 ms / 6.26x.
### FINAL CHAMPION = partial-fusion + int16 WIN: **0.0223 ms / 8.12x** (COMPILED+CORRECT,
### detector OK). ~1.28x same-GPU over baseline. Floor confirmed from 8 distinct directions.

| # | Direction | Cold RUNTIME | vs prev | Verdict |
|---|-----------|--------------|---------|---------|
| 1 | full fusion (1 launch, shared winner, block/row) | 0.0252 | 0.0286 base | KEEP (superseded) |
| 2 | fused TH sweep | — | tie | no effect |
| 3 | **partial fusion** (shared pass1 flush-row + high-occ pass2) | 0.0234 | 0.0252 | **KEEP** |
| 4 | **int16 global WIN** | 0.0222 | 0.0234 | **KEEP (champion)** |
| 5 | branch pass2 (if/else vs select) | 0.0224 | tie | REVERT |
| 6 | persistent scratch/out buffers | 0.0232 | slower | REVERT |
| 7 | pass1-TH / pass2-WS/TH sweep | 0.0225-0.0233 | tie/worse | REVERT |
| 8 | float4 vectorized pass2 | 0.0221 | tie | REVERT (noise) |

## Iter 1 — tilelang port

- **Hypothesis:** deterministic last-wins (atomicMax); scored --deterministic. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0311 ms, REF=0.1830 ms, **SPEEDUP=5.8842x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 5.3079x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 1 (new) — read int64 indices directly, drop idx.to(int32)

- **Hypothesis (advice):** the `idx.contiguous().to(torch.int32)` cast launches an
  extra kernel; reading int64 IDX directly in pass1 removes one launch.
- **Change:** pass1 IDX tensor dtype int32 -> int64; forward() drops `.to(torch.int32)`.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0286 ms (was 0.0317),
  **SPEEDUP=6.3986x**. ~10% kernel-time reduction (one fewer launch). KEEP.

---

# 2026-07-02 optimization pass (GPU 3). MY-GPU baseline = committed solution above:
# RUNTIME 0.0286 ms, SPEEDUP 6.2587x (num-warmup 200, deterministic). All deltas below
# are vs THIS my-GPU baseline (solution cuda-event RUNTIME), not RESULTS.md numbers.

## Roofline note
- Baseline 28.6us at ~13 MiB traffic = ~476 GB/s effective (RTX6000 Ada peak ~960).
  Only ~50% of peak -> the op is LATENCY/OCCUPANCY/launch bound, NOT bandwidth bound.
  So the levers are: cut launches, raise occupancy, hide latency — not just cut traffic.

## Iter 1 — fully-fused single kernel (one block/row, shared-memory winner)
- **Hypothesis:** 3 launches (torch.full init + pass1 + pass2) cost ~5us each of
  overhead; the global WIN buffer + init are pure traffic. One block per row can
  build the winner map in shared memory (shared atomicMax, ~10x faster than global),
  gather in the same block, and never materialize WIN globally: 3->1 launch,
  ~13 MiB -> ~7 MiB. Shared atomic_max on k stays commutative -> last-wins preserved.
- **Change:** replaced two-pass with `_build_fused` (T.alloc_shared win[W], init -1,
  T.sync, T.Parallel(K) shared atomicMax, T.sync, T.Parallel(W) gather). TH=512.
- **Bench (full, --deterministic, warmup 50):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0263 ms (baseline 0.0286), SPEEDUP=6.88x. **KEEP** (~8% faster).
- **Read:** gain is modest — effective BW ~267 GB/s, even further from peak, so the
  fused kernel is latency/occupancy bound (only 64 blocks on 142 SMs, phase-serialized).
  Next levers: TH sweep (more MLP per block), vectorize gather, partial-fusion variant.
- **Next:** sweep TH (256/1024) on the fused kernel.

## MEASUREMENT NOTE (critical) — bench.py is COLD; warm harnesses mislead
- bench.py clears the L2 (256 MB thrash) before EVERY timed trial -> inputs come
  from DRAM = COLD. Cold, DRAM TRAFFIC dominates the ranking.
- A warm profiler/cuda-event loop (inputs stay L2-resident) ranks by COMPUTE and
  gives the OPPOSITE verdict (it showed baseline < fused). DO NOT trust warm harnesses.
- Authoritative = bench.py RUNTIME/SPEEDUP (--num-warmup 200 --deterministic). Fast
  signal = `bench.py --no-ref --num-warmup 200` (same cold path, skips ref timing).
- SPEEDUP (=REF/solution, both measured same run) is clock-invariant -> use it to A/B
  across invocations. Clean back-to-back: fused 6.88x vs baseline 6.21x -> fused wins.
- Cold, fused = 0.0253 ms for ~7 MiB => ~290 GB/s (peak ~960). The 64-block fused
  kernel can't saturate cold DRAM bandwidth (too few blocks/MLP). THAT is the headroom:
  push the bandwidth-heavy work to HIGH occupancy (partial-fusion / high-occ pass2).

## Iter 2 — TH sweep on fused (256/512/1024)
- **Hypothesis:** more threads/block -> more MLP to hide cold DRAM latency with 64 blocks.
- **Result (warm proxy, unreliable but TH-invariant):** all TH within noise. TH is not
  the lever; occupancy (block count) is. Kept TH=512. REVERT any change (no delta).

## Iter 3 — PARTIAL FUSION (shared-mem pass1 flush-full-row + high-occ pass2)  **CHAMPION**
- **Hypothesis:** the fused kernel is cold-bandwidth-starved (64 blocks -> ~290 GB/s).
  Split so the bandwidth-heavy gather (pass2) runs at HIGH occupancy (Rr*WS=2048 blocks).
  pass1 stays one-block-per-row shared-atomicMax but writes the FULL WIN row to global
  (every column incl. -1 sentinel) -> no torch.full init. 2 launches, ~11 MiB cold.
- **Change:** solution/scatter.py = `_build_pass1` (shared win[W], init -1, sync,
  T.Parallel(K) shared atomicMax, sync, flush win->WIN global) + baseline `_build_pass2`
  (WS=32). forward allocates `win = torch.empty` (pass1 writes all cols).
- **Bench (--no-ref cold, warmup 200, ×2):** CORRECT=True, RUNTIME=0.0234, 0.0234 ms.
  vs fused 0.0251/0.0254, vs committed baseline 0.0286. **KEEP** (~7% over fused, ~18%
  over baseline). Confirms cold ranking is occupancy-gated, not pure traffic.
- **Cold BW:** 11 MiB / 23.4us = ~493 GB/s (peak ~960) -> still headroom.
- **Next:** shrink WIN traffic (int16 global WIN, int32 shared atomics); pass2 vectorize.

## Iter 4 — int16 global WIN buffer  **CHAMPION**
- **Hypothesis:** winner k in [0,4095], sentinel -1 -> fits int16. Halve the WIN
  global round-trip (2 MiB write+read -> 1 MiB) while keeping int32 shared atomics
  (int16 atomics unsupported). Flush casts int32 shared -> int16 global; pass2 casts back.
- **Change:** WIN tensor int32 -> int16 in both prim_funcs; flush `T.Cast("int16",..)`,
  pass2 `wk = T.Cast("int32", WIN[r,c])`; forward `win = torch.empty(..., int16)`.
- **Bench (--no-ref cold, warmup 200, ×3):** CORRECT=True, RUNTIME 0.0229/0.0222/0.0222
  vs int32 partial 0.0235/0.0233/0.0234. **KEEP** (~5% consistent).
- **Next:** vectorize pass2 (float4 on contiguous X/OUT); WS/TH sweep.

## Iter 5 — branched pass2 (if/else instead of if_then_else select)
- **Hypothesis:** `T.if_then_else` is a select that loads BOTH UPD[gather] and X even
  though only one is used -> ~1 MiB wasted X reads on winner elements. A real branch
  (if wk>=0: OUT=UPD else OUT=X) loads only the used operand.
- **Bench (cold ×2):** CORRECT=True, RUNTIME 0.0228/0.0224 vs champion 0.0224/0.0223.
  **REVERT** — branch divergence offsets the traffic saved (net wash).

## Iter 6 — persistent scratch/out buffers (kill --deterministic FillFunc)
- **Hypothesis:** profiler shows 2 `FillFunc` elementwise kernels/forward (~2.8us warm):
  under --deterministic, torch.empty fills "uninitialized" memory. Caching the `win`
  scratch + `out` (pass2 loses out_idx, takes pre-alloc OUT param a la group_norm) and
  reusing across calls removes both fills. Verified safe: bench correctness compares each
  trial's output immediately (no cross-trial aliasing).
- **Bench (cold ×2):** CORRECT=True, RUNTIME 0.0231/0.0233 vs champion 0.0222/0.0222.
  **REVERT** — SLOWER. The fills are hidden/overlapped in the cold-DRAM-bound path;
  out_idx fresh-alloc is actually faster than reusing a persistent buffer here.

## Iter 7 — pass1-TH / pass2-WS / pass2-TH sweep
- **Hypothesis:** raise pass1 warps (TH 1024) to hide cold idx-read latency at 64 blocks;
  retune pass2 grid (WS) / threads.
- **Bench (cold, single-run each):** P1TH 512/1024/256 = 0.0227/0.0228/0.0230;
  WS 16/32/64 = 0.0225/0.0227/0.0231; P2TH 512 = 0.0233. All within noise of champion
  0.0222-0.0227; nothing beats it. **REVERT** (keep 512/32/256). pass1 is block-count-
  bound (64 blocks, shared-winner-per-row) not warp-bound -> TH can't help it.

## Iter 8 — vectorized (float4) pass2 gather
- **Hypothesis:** load/store contiguous X/OUT as float4 (T.vectorized(4)) for wider
  memory transactions; WIN + UPD-gather stay scalar.
- **Bench (cold, ×3 A/B):** vec 0.0219/0.0223/0.0222 vs scalar 0.0226/0.0221/0.0224 —
  statistically TIED (overlapping). **REVERT** — no reliable gain, keep simpler scalar.

## FLOOR VERDICT (structural floor confirmed from 8 distinct directions)
- **Champion = Iter 4 (partial-fusion + int16 WIN), scalar pass2.** Cold RUNTIME ~0.0222 ms
  vs committed baseline 0.0286 ms = **~1.29x same-GPU** improvement.
- Cold DRAM must-move ~7 MiB -> ~7.3us HBM roofline, but we sit at ~22us (3x). The gap
  is STRUCTURAL and irreducible in this DSL/algorithm: (1) deterministic argmax needs a
  cross-block reduction => a global barrier => 2 phase-serialized launches (pass1 idx-read
  phase cannot overlap pass2 out-write phase); (2) shared-winner argmax forces one block
  per row => pass1 is capped at 64 blocks on 142 SMs and cannot saturate cold DRAM.
- Directions that FAILED to close the gap (all measured): full fusion (occupancy), TH
  sweep, WS/P2TH sweep, branch-vs-select, persistent buffers, float4 vectorization.
  Directions that WORKED: partial fusion (occupancy), int16 WIN (traffic). => at floor.
