# Iteration Log — group_norm / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/group_norm/triton/solution/group_norm.py`,
Triton speedup 0.9904x); benched against the same `reference/normalization/group_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of group_norm | 0.8470x | 36.6000 ms | 31.0000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** GroupNorm 8 groups; per-(batch,group) reduction (8.6GB). Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=36.6000 ms, REF=31.0000 ms, **SPEEDUP=0.8470x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 0.9904x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 1b — float4 vectorization (both passes)

- **Change:** vectorized reduction + affine apply with `T.vectorized(4)` into a
  local float4 (`vec`). Apply hoists `Wt[c]`/`Bs[c]` per float4 (HW%4==0 keeps a
  float4 within one channel). TH=256.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=34.3ms (min 31.4), REF=31.0,
  **SPEEDUP=0.9038x** (up from 0.8493x baseline). KEEP.

## Iter 2/3 — TH sweep (512, 1024)

- 0.9012x / 0.9012x, min 31.4ms — identical to TH=256 within noise. No change. REVERT to TH=256.

## Iter 4 — 4-lane ILP accumulators in reduction

- Broke FP-add dependency chain with 4 accumulator lanes. RUNTIME 34.3ms, min 31.4,
  **0.9038x** — identical to iter-1. Confirms memory-bound; ILP doesn't help. REVERT.

## ~~Verdict: AT FLOOR~~ (OVERTURNED 2026-07-02 — see below)

- Best = iter-1 (float4 both passes, TH=256): **0.9038x**, min 31.4ms.
- ~~3-pass memory-bound op: read-reduce + read-apply + write = 25.8GB / ~960GB/s = ~27ms floor.~~
  **This "3x irreducible traffic" claim was WRONG.** Traffic is reducible to 2x via L2 reuse; see below.

---

# RE-OPENED 2026-07-02 (GPU 2, RTX 6000 Ada) — real headroom found

## Diagnostic A — the 0.90x is a HARNESS artifact, not the kernel

- Baseline verdict on GPU2: mean 34.4ms, **min 31.4ms**, **max 297ms**, std 26.5 → 0.9041x.
  The entire sub-1.0x mean comes from ONE ~295ms outlier that is **always Trial 1**.
- `bench.py` calls `torch.cuda.empty_cache()` right after warmup (line 99), then times. The
  SOLUTION is timed FIRST (line 775), reference SECOND (line 801). Trial-1's first big
  cudaMalloc after the pool release pays a ~260ms one-time CUDA driver page-commit cost.
- **Proof it is kernel-independent:** running the *reference itself* in the `--solution` slot
  also stalls ~291ms on Trial 1. So this +~2.9ms/100-trial mean penalty hits every solution
  equally and is NOT removable by any legitimate kernel change (buffer-caching the output would
  remove it but that games the harness / risks correctness — rejected).
- `out_idx=[3]` vs explicit `torch.empty_like` in forward: identical (both stall). Not the cause.

## Diagnostic B — Triton oracle steady-state is 20.2ms, not ~31ms (headroom is REAL)

- Measured the committed Triton oracle through the same bench on GPU2: **min 20.2ms**
  (17.2GB / 20.2ms ≈ 851 GB/s = genuine **2x** traffic). Its 0.9904x in RESULTS.md is just its
  OWN mean dragged by the same Trial-1 stall. The prior "25.8GB irreducible / at roofline"
  verdict is refuted: Triton's L2-reuse pipeline (chunk kept L2-resident across the
  stats->normalize launches) turns 3x DRAM traffic into 2x.
- tilelang launch overhead measured at **~3.3us/launch** (256 launches = 0.87ms) → a
  multi-launch chunked port is viable (overhead negligible vs ~20ms compute).

## Iter L2-1 — L2-reuse chunked pipeline (port of the Triton algorithm) — **KEEP**

- **Hypothesis:** process K groups per chunk, interleaving a cooperative stats launch and a
  normalize launch per chunk so the chunk stays in the 96MB L2 between them → normalize re-reads
  from L2 (2x DRAM traffic instead of 3x).
- **Change:** two jitted kernels. `_build_stats`: 2D grid (SPLIT, K), SPLIT blocks/group each
  reduce a SEG segment with float4, shared-atomic to a block accumulator, leader (`tid==0`)
  global-atomic-adds to per-group S/Q. `_build_norm`: 2D grid (SPLIT_N, K), reads S/Q, applies
  float4 affine. forward loops `for n in range(N): for h in range(2):` over (sample, half)
  chunks — K=4 < G=8, so each chunk = one weight-half; weight is pre-sliced `w2[h]` so the
  kernel channel index (`by*GPC + pos//HW`) stays local. Detector-clean (no BinOp / forbidden
  calls in forward; loop bodies are kernel-call glue only).
- **Chunk size matters:** K=8 (64MB chunk) → min 29.7ms (norm's 64MB write traffic + 64MB stats
  data > 96MB L2 → stats data evicted, ~3x). K=4 (32MB chunk) → min 23.0ms (32MB + writes fit
  L2 → reuse materializes). Matches Triton's K=4 choice.
- **Bench (verdict, --num-warmup 200, 100 trials):** COMPILED=True, CORRECT=True (5/5),
  RUNTIME=25.9ms, REF=31.1ms, **SPEEDUP=1.2008x** (up from 0.9041x baseline; steady-state
  min 23.0ms vs baseline 31.4ms). Detector: regression_type None (pass). KEEP.
- **Honest framing:** this is a genuine memory-traffic reduction (3x->2x), not a codegen tweak.
  Residual gap to Triton's 20.2ms is chunk/occupancy tuning (next). The remaining ~0.1-0.15x
  vs a no-stall ideal is the kernel-independent Trial-1 harness artifact (Diagnostic A).
- **Params:** K=4, HALVES=2, SPLIT=128, SPLIT_N=128, TH=256.

## Iter L2-2..L2-9 — chunk / SPLIT / TH sweep (rank by steady-state MIN, --no-ref)

Fast-signal min (ms), K=4 unless noted:

| direction | configs tried | best min |
|-----------|---------------|----------|
| chunk K   | K=8 → 29.7, **K=4 → 23.0**, K=2 → 25.2 | K=4 |
| SPLIT/SPLIT_N (TH=256) | 64/32→24.2, 64/64→(nc), 128/128→**23.0**, 128/256→23.1, 256/256→23.2 | 128/128 |
| TH (SPLIT=128) | 512→23.9, 256→23.0, 128→22.5, 96→22.0, 64→21.6, **32→20.9** | 32 |
| SPLIT @ TH=32 | 32/32→21.0, **64/64→20.7**, 64/32→20.7, 128/128→20.9, 256/256→21.0 | 64/64 |

- **Winner: K=4, SPLIT=64, SPLIT_N=64, TH=32** → steady-state min **20.7ms**.
- **Why TH=32 (single warp/block) wins:** the stats block accumulator is a shared atomic; a
  32-thread block has minimal shared-atomic contention and 512 single-warp blocks give ample
  memory-level parallelism to saturate BW. TH sweep was monotonic 256→32.
- **Why K=4 (not smaller):** K=2 (16MB) over-fragments into 1024 launches with lower per-kernel
  efficiency; K=8 (64MB) overflows L2 with the norm write traffic. K=4 (32MB chunk + 32MB write
  ≈ 64MB < 96MB L2) is the sweet spot — same as the Triton oracle.

## Iter L2-2 verdict — **KEEP (final best)**

- **Bench (verdict, --num-warmup 200, 100 trials):** COMPILED=True, CORRECT=True (5/5),
  RUNTIME=23.5ms, REF=31.1ms, **SPEEDUP=1.3234x**. Detector: regression_type None. KEEP.
- **Params:** K=4, HALVES=2, SPLIT=64, SPLIT_N=64, TH=32.

## Verdict: AT THE 2x-TRAFFIC FLOOR (real floor, was mis-called 3x before)

- Journey: baseline **0.9041x** (34.4ms mean / 31.4ms min, 3x traffic) → L2-reuse pipeline
  **1.3234x** (23.5ms mean / **20.7ms min**, 2x traffic).
- GroupNorm's irreducible minimum is 1 read + 1 write = **2x** (stats must read all input before
  normalize; output written once). The L2-reuse pipeline achieves this by keeping each 32MB chunk
  L2-resident across the stats→normalize launch pair. 17.2GB / 20.7ms ≈ **831 GB/s** ≈ 87% of the
  ~960 GB/s AD102 ceiling, matching the Triton oracle's 20.2ms (851 GB/s) and torch's own BW
  efficiency. Going below 2x is physically impossible.
- The reported mean (23.5ms) still carries the fixed, kernel-independent **Trial-1 harness stall**
  (~+2.9ms/100-trial; Diagnostic A) that afflicts whatever kernel is timed first — it caps the
  achievable mean-speedup at ~1.33x even though steady-state (20.7ms) would be ~1.50x on a
  stall-free harness. This is not a kernel deficiency.
