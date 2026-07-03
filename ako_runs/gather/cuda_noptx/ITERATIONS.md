# Iteration Log — gather / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gather/triton/solution/gather.py`,
Triton speedup 1.2217x); benched against the same `reference/index/gather.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of gather | 1.1130x | 0.0239 ms | 0.0266 ms | correct |
| deepen best | VEC=4 NG=4 (long2 idx + float4 out, ILP=16) | **1.3284x** | 0.0201 ms | 0.0267 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** gather dim=1; indexed load (latency-bound, small). Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0239 ms, REF=0.0266 ms, **SPEEDUP=1.1130x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2217x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 1 (AKO4ALL) — 2D grid, drop per-element 64-bit divide

- **Change:** replaced grid-stride loop (`r = i / Cout`, runtime int64 divide per
  element — not strength-reducible) with a 2D grid: `r = blockIdx.y`, `c = block.x*tid`.
  Eliminates the software-emulated 64-bit division.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0236 ms, REF=0.0268 ms, **SPEEDUP=1.1356x**.
- **vs baseline 1.1037x:** +2.9%. Small move → confirms latency-bound, not arithmetic-bound. KEEP.

## Iter 2 (AKO4ALL) — ILP, K=4 columns/thread (strided by blockDim)

- **Change:** each thread handles KPT=4 columns strided by blockDim.x; issue all 4
  idx loads, then all 4 gathers, then all 4 stores → more outstanding memory
  requests per thread to hide latency (the binding constraint).
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0226 ms, REF=0.0266 ms, **SPEEDUP=1.1770x**.
- **vs iter-1 1.1356x:** +3.6%. ILP targets latency directly → real move. KEEP.

## Iter 3 (AKO4ALL) — ILP K=8

- **Change:** KPT 4 → 8 (more outstanding loads per thread).
- **Bench:** CORRECT=True, RUNTIME=0.0220 ms, REF=0.0266 ms, **SPEEDUP=1.2091x**.
- **vs iter-2 1.1770x:** +2.7%. Now matches the Triton port (1.22x). KEEP.

## Iter 4 (AKO4ALL) — ILP K=16 (REVERT)

- **Change:** KPT 8 → 16.
- **Bench:** CORRECT=True, RUNTIME=0.0236 ms, **SPEEDUP=1.1314x**. Regression — only
  1 block/row (128 blocks total) starves occupancy. REVERT to iter-3 (KPT=8).

## FINAL (prior pass) — restored iter-3 (KPT=8, 2D grid)

- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0220 ms, REF=0.0268 ms, **SPEEDUP=1.2182x**.
- **vs same-GPU baseline 1.1037x:** +10.4%. Levers: kill per-element int64 divide (2D grid)
  + ILP K=8 strided columns. Latency-bound; now at/above the Triton port (1.22x).

---

# AKO deepen pass (GPU 2) — same-GPU baseline of the committed kernel: RUNTIME 0.0220 ms, SPEEDUP 1.2136x

Roofline: traffic ~10 MiB (idx int64 4 MiB coalesced + x 4 MiB touched cold — bench clears L2
before every trial — + out 2 MiB). Realistic floor ~15-18 us because the x gather is *random*
32-B sector reads (poor GDDR6 row-buffer hit rate), not streaming. Latency/occupancy bound.
Fast-signal protocol: `--no-ref --num-perf-trials 30 --num-warmup 100`, rank by min+mean; every
KEEP re-confirmed at `--num-warmup 200`.

## Iter 1 — threads 256->128 (scalar, 512 blocks)
- Hypothesis: more blocks -> less tail effect. Change: threads=128. Bench: min 0.0205, mean flat.
- REVERT. total_threads×KPT = total_elements is fixed, so warps↔ILP split conserves outstanding
  requests (~112/SM either way); occupancy split alone does nothing.

## Iter 2 — shared-memory row caching (one block/row)
- Hypothesis: stage 32 KB row via coalesced load, turn random-DRAM gather into smem gather.
- Change: extern smem srow[Cin], load+`__syncthreads`+gather. Bench: min 0.0276 (WORSE).
- REVERT. 128 blocks underutilize 142 SMs; mandatory full-row load + barrier serialize ahead of
  any gather; L2 already caches the small row during execution, so smem's overhead isn't repaid.

## Iter 3 — int32 index math (narrow int64->int32)
- Hypothesis: max linear index 524288 < INT_MAX; narrower math = fewer int ops + less reg pressure.
- Bench: min 0.0204, mean 0.0214 (~2% on mean, min tied). Neutral-to-marginal. Kept as working base.

## Iter 4 — exact-tile no-guard fast path
- Hypothesis: Cout=4096 = gx·threads·KPT exactly, so `if(ci<lim)` is dead. Dispatch guard-free
  kernel when the grid tiles perfectly (guarded kernel retained for general shapes).
- Bench: min 0.0205, mean 0.0213. Flat. Confirms kernel is memory-latency bound, NOT issue-bound.

## Iter 5 — VECTORIZE: long2 idx loads + float4 stores (per-group, VEC=4 NG=2, ILP=8)  ✅ KEEP
- Hypothesis: contiguous-per-thread layout makes the coalesced idx/out traffic 128-bit transactions
  (LDG.128 / STG.128); random x gather stays scalar. Fewer/wider mem instrs cut tail jitter.
- Fast: min 0.0205, mean 0.0209. **Verdict: RUNTIME 0.0212, SPEEDUP 1.2547x** (+3.4% vs baseline). KEEP.

## Iter 6 — all-idx-first at NG=2 (hoist all long2 idx loads before gathers)
- Bench: min 0.0205, mean 0.0213. Flat, more registers. Compiler already schedules it. REVERT.

## Iter 7 — __ldg on the random x gather
- Bench: min 0.0205, mean 0.0213 (one lucky 0.0195 trial did not reproduce). No-op (pointers already
  const __restrict__ -> nvcc already emits LDG on sm_89). REVERT.

## Iter 8 — threads=128 + vectorized (512 blocks, ILP=8)
- Bench: min 0.0205, mean 0.0210. Flat (occupancy direction). REVERT. Floor 0.0205 confirmed across
  6 distinct directions.

## Iter 9 — contiguous 8-cols/thread layout (adjacent groups vs blockDim-strided)
- Bench: min 0.0205, mean 0.0216 (slightly worse — strided grouping interleaves better across warps).
  REVERT.

## Iter 10 — NG=4 (VEC=4, ILP=16, 128 blocks, 8 warps/block)  ✅ KEEP (new best)
- Hypothesis (bracketing): high-ILP direction. Scalar KPT=16 regressed before, but vectorization cut
  the instruction/register cost (44 regs, 0 spills) so ILP=16 is now viable — 16 outstanding gather
  loads/thread hide the random-read latency far better.
- Fast: min 0.0195, mean 0.0199 (reproduced). **Verdict: RUNTIME 0.0201, SPEEDUP 1.3284x, min 0.0192.**
  +9.5% vs baseline, +5.9% vs iter-5. KEEP. The 0.0205 "floor" was a NG=2 artifact; real floor ~0.0192.

## Iter 11 — NG=8 threads=128 (ILP=32, 4 warps/block)
- Bench: min 0.0225, mean 0.0235 (WORSE). 4 warps/block too few; register/latency over the edge. REVERT.

## Iter 12 — NG=4 threads=128 (ILP=16, 256 blocks, 4 warps/block)
- Bench: min 0.0195, mean 0.0209. Same min, worse mean than threads=256. 4 warps/block hurts more than
  the 14 idle SMs (at 128 blocks) help. REVERT. Exact-tile math forces ILP=16×256-thread => 128 blocks,
  so 8-warp blocks only exist at 128 blocks — that IS the peak.

## Iter 13 — all-idx-first at NG=4
- Bench: min 0.0195, mean 0.0201. Identical to per-group, more registers. Compiler already pipelines. REVERT.

## Iter 14 (diagnostic) — ptxas -v on NG=4 kernel
- 44 registers, 0 bytes spill, 0 barriers. Clean; block-count-limited (not register-limited).

## Iter 15 — threads=512, NG=2 (warps/block axis: 33% occ, ILP=8, still 128 blocks, outstanding=128)
- Bench: min 0.0195, mean 0.0204. Slightly worse than NG=4. REVERT.

## Iter 16 — threads=1024, NG=1 (warps/block axis: 67% occ, ILP=4, still 128 blocks, outstanding=128)
- Bench: min 0.0195, mean 0.0205. Slightly worse than NG=4. REVERT.

### Occupancy-vs-ILP frontier (fixed 128 blocks / fixed outstanding=128 per SM)
| warps/SM (occ) | config | ILP | mean | min |
|---|---|---|---|---|
| 8 (17%)  | NG=4 threads=256 | 16 | **0.0199** | **0.0192** |
| 16 (33%) | NG=2 threads=512 | 8  | 0.0204 | 0.0195 |
| 32 (67%) | NG=1 threads=1024| 4  | 0.0205 | 0.0195 |

Deep per-thread ILP (a long independent memory pipeline the scheduler overlaps) beats raw resident
warps for this latency-bound gather. **NG=4/threads=256/VEC=4 is the confirmed peak across BOTH the
block-count and warps-per-block axes.** Not the coalesced HBM roofline (~11 us) — the random 32-B
sector x-gather leaves it above that — but the optimum of the searched config space.

## FINAL (deepen pass) — VEC=4 NG=4 long2-idx + float4-out, ILP=16
- **bench.sh final (authoritative):** COMPILED=True, CORRECT=True (5/5), RUNTIME 0.0201 ms,
  REF 0.0268 ms, **SPEEDUP 1.3333x**. Detector: valid=True, regression_type=None (pass).
- **vs same-GPU baseline 1.2136x (0.0220 ms): +9.9%** (real, well above the ~3% noise floor).
