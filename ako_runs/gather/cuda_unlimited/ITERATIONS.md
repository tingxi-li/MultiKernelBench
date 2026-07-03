# Iteration Log — gather / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gather/triton/solution/gather.py`,
Triton speedup 1.2217x); benched against the same `reference/index/gather.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of gather | 1.1203x | 0.0241 ms | 0.0270 ms | correct |
| 3 | K=4 MLP scatter + v4 store | 1.2642x | 0.0212 ms | 0.0268 ms | KEEP |
| 5 | K=4 threads=128 (prev committed best) | 1.2594x | 0.0212 ms | 0.0267 ms | baseline |
| 6 | K=2 threads=256 (occupancy) | tie | 0.0214 ms | — | REVERT |
| 7 | idx .cs streaming cache policy | tie | 0.0212 ms | — | REVERT |
| 8 | __ldg compiler-scheduled loads | tie | 0.0212 ms | — | REVERT |
| 10 | **shared-mem row staging** | **1.4725x** | **0.0182 ms** | 0.0268 ms | **KEEP (best)** |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** gather dim=1; indexed load (latency-bound, small). Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0241 ms, REF=0.0270 ms, **SPEEDUP=1.1203x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2217x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 2 — 2D grid, int32 indexing (kill 64-bit divide)
- **Change:** replaced grid-stride 1D loop + `i/Cout` 64-bit divide with a 2D grid (row=blockIdx.y), int32 arithmetic, `if(col<Cout)` guard.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0236 ms, REF=0.0270, **SPEEDUP=1.1441x** (baseline 1.1303x).
- ~1% gain — divide was overlappable, not the bottleneck (dependent idx->x scatter chain dominates). KEEP.

## Iter 3 — K=4 outputs/thread: vector index loads + MLP scatter + v4 streaming store
- **Change:** each thread emits 4 consecutive outputs. 2x `ld.global.nc.v2.s64` index loads, 4 INDEPENDENT scattered `ld.global.nc.f32` loads (memory-level parallelism hides the dependent idx->x latency), one `st.global.cs.v4.f32` streaming store. Cout=4096 divisible by 4 -> all vector ops aligned.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=0.0212 ms, REF=0.0268, **SPEEDUP=1.2642x** (baseline 1.1303x, prev best 1.1441x).
- Beats Triton target (1.2217x). MLP was the real lever. KEEP.

## Iter 4 — K=8 outputs/thread (more MLP)
- **Change:** K=4 -> K=8 (4x v2.s64 loads, 8 scatter loads, 2x v4 store).
- **Bench:** CORRECT=True, RUNTIME=0.0217 ms, **SPEEDUP=1.2396x** — slightly worse than K=4 (0.0212). More registers/less occupancy. REVERT to iter-3 (K=4).

## Iter 5 — K=4 with threads=128 (occupancy)
- **Change:** block size 256 -> 128 threads (K=4 unchanged).
- **Bench:** CORRECT=True, RUNTIME=0.0211 ms, **SPEEDUP=1.2796x** — marginally best (vs 0.0212 @256, within noise). KEEP as best.

---
## Re-open on GPU 3 (2026-07-02) — noise-disciplined pass (rank by min over >=30 trials, keep only >3%)
**GPU-3 baseline (committed iter-5):** mean 0.0212 ms, **min 0.0205 ms**, SPEEDUP 1.2594x.
ptxas: **18 reg/thread, 0 spill, 0 stack.** At 128 thr the resource occupancy cap is 100%, but total work = only **1024 blocks for 142 SMs (~7 blocks/SM, <1 full wave of 1704)** -> actual occupancy ~60%, i.e. latency/occupancy slack, not a resource wall.

## Iter 6 — occupancy: K=2, threads=256 (double the thread count)
- **Hypothesis:** more warps-in-flight fills the <1-wave grid, hides idx->x latency.
- **Bench:** CORRECT=True. First 40-trial run min=0.0195 (outlier); two 50-trial repeats both min=0.0205, mean 0.0214 — **identical to baseline. TIE.** Not occupancy-limited in a way K moves. REVERT.
- **Next:** cache policy for the read-once idx stream.

## Iter 7 — idx cache policy: ld.global.cs (streaming) instead of .nc
- **Hypothesis:** idx is read once, coalesced, no reuse; routing it through the read-only (.nc) cache is pointless and could evict x.
- **Bench:** CORRECT=True, min 0.0205, mean 0.0212 — **TIE** (<1%). At 96MB L2 the 8MB idx+x working set never evicts, so pollution is a non-issue. REVERT.
- **Next:** let the compiler schedule the loads instead of asm-volatile.

## Iter 8 — compiler-scheduled loads: __ldg (longlong2 idx + float x) replacing inline-asm .nc
- **Hypothesis (weak, per advisor):** asm volatile might serialize the 4 independent x loads.
- **Bench:** CORRECT=True, min 0.0205, mean 0.0212 — **TIE.** Confirms the advisor's call: GPU loads are non-blocking and scoreboard on the *consuming* store, so `volatile` never inserted stalls between independent loads. REVERT.

## Iter 9 — FLOOR DIAGNOSTIC (decisive): decompose the cost under bench.py's cold-L2 timing
- bench.py calls `clear_l2_cache()` (256MB thrash) **before every trial** -> the scattered x reads hit **cold HBM**, not L2. Reproduced bench.py exactly in a CUDA-event harness with the same clear-L2-per-trial.
- Same-input, same-launch micro-kernels (cold L2):
  - **full scatter gather: min 0.0204 ms** (== bench.py 0.0205, harness validated)
  - **noscatter** (read idx, but x read *coalesced*): **0.0072 ms**
  - **copy** (coalesced x, no idx): **0.0070 ms**
- **Conclusion:** the scatter alone is 0.0204 vs 0.0072 coalesced = **2.8x**; idx-read and store are ~free. The random reads sustain ~50% of DRAM BW (row-buffer thrashing) — the physics of a random gather. This is why iters 6-8 (occupancy/cache/scheduling) were all ties: none of them touch DRAM row-buffer behavior. **The naive-scatter baseline was at its own floor.**
- **Insight -> Iter 10:** convert scattered HBM reads into a *coalesced* HBM read by staging the x-row in shared memory, then gather on-chip.

## Iter 10 — SHARED-MEMORY ROW STAGING (new algorithm, KEEP — best)
- **Change:** one block per row (gridDim.x = M). Phase 1: coalesced `float4 __ldg` load of the whole 8192-float x-row into 32KB dynamic shared. `__syncthreads`. Phase 2: each thread emits 4 consecutive outputs, reading `longlong2` idx (coalesced) and gathering from **shared memory** (on-chip random reads), streaming `st.global.cs.v4` store. Cin,Cout multiples of 4.
- **Why it works:** turns the DRAM-random x traffic (~50% BW) into sequential DRAM traffic (~90% BW) + cheap shared-mem random reads. HBM now sees idx 4MB + x 4MB + out 2MB, all coalesced.
- **Diagnostic (cold-L2 harness):** min **0.0173 ms** (baseline full 0.0204) = 1.18x.
- **AUTHORITATIVE bench.py (--num-warmup 200):** COMPILED=True, CORRECT=True, RUNTIME=**0.0182 ms**, REF=0.0268, **SPEEDUP=1.4725x** (repeat: min 0.0167, mean 0.0181). Detector: valid=True, regression_type=None. **KEEP as best.**
- **Explored but rejected (distinct directions):**
  - *Vectorized row-load vs scalar row-load*: tie (0.0173 vs 0.0174) — row-load isn't the bottleneck.
  - *Block-size sweep* T=128/256/512/1024: all 0.0173-0.0184; T=256 chosen.
  - *Blocks-per-row 2/4/8 (redundant row-loads for more occupancy)*: worse (0.0184-0.0218) — the extra x traffic cancels the occupancy gain.
  - *Two-kernel L2-prewarm* (coalesced pass to warm the 96MB L2, then scatter from warm L2): **0.0215, worse** — the prewarm pass's 4MB read + extra launch isn't repaid.
- **Floor of the shmem kernel:** coalesced traffic floor ~0.009 ms; we sit at ~0.017 because only **128 blocks (~1/SM)** run, so the shared-only gather phase can't overlap another block's HBM load phase. With only 128 rows this occupancy ceiling is structural (BPR redundant-load fixes cost more than they save). Near the shmem floor; the big scatter->coalesced win is banked.
