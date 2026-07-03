# Iteration Log — gather / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/gather/triton/solution/gather.py`,
Triton speedup 1.2217x); benched against the same `reference/index/gather.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of gather | 1.0675x | 0.0252 ms | 0.0269 ms | correct |
| — (2026-07-02) int64 idx direct | **1.2947x** | 0.0207 ms | 0.0268 ms | **NEW BEST** |

## Iter 1 — tilelang port

- **Hypothesis:** gather dim=1; indexed load (latency-bound, small). Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0252 ms, REF=0.0269 ms, **SPEEDUP=1.0675x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2217x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 1 — 2D grid, 1 elem/thread (BN=512, 1024 blocks)
- Hypothesis: 128 blocks under one wave on 142 SMs; more blocks to hide scattered-load latency.
- Result: COMPILED=True, CORRECT=True, RUNTIME=0.0265 ms, **SPEEDUP=1.0113x**. WORSE than baseline (killed per-thread ILP). REVERT.

## Iter 2 — 2D grid, COLS=1024 TH=256 (512 blocks, 4 elem/thread)
- Hypothesis: middle ground — more blocks than baseline but keep ILP per thread.
- Result: COMPILED=True, CORRECT=True, RUNTIME=0.0254 ms, **SPEEDUP=1.0512x**. ~baseline (within noise). REVERT.

## Iter 3 — 2D grid, COLS=2048 TH=256 (256 blocks, 8 elem/thread)
- Result: COMPILED=True, CORRECT=True, RUNTIME=0.0253 ms, **SPEEDUP=1.0553x**. ~baseline (noise).

## Iter 4 — 2D grid, COLS=2048 TH=512 (256 blocks, 4 elem/thread, 16 warps/block)
- Result: COMPILED=True, CORRECT=True, RUNTIME=0.0252 ms, **SPEEDUP=1.0635x**. Ties baseline exactly.

## Prior conclusion (SUPERSEDED) — "AT FLOOR" from grid sweep only
Grid/occupancy sweep (1024/512/256 blocks; 1/4/8 elem/thread; TH 256/512) all land within ~3%
noise of baseline 0.0252 ms (1.06x). 1-elem/thread (iter-1) was the only clear loser (-5%, ILP
killed). gather is latency/L2-cache bound, not occupancy bound — scattered X[r,idx] loads hit the
32KB-row cache so adding blocks does not help. Restoring baseline verbatim as best.
**NOTE (2026-07-02 run):** that floor claim was drawn from ONE direction (grid/occupancy). It never
touched the `idx.to(torch.int32)` conversion in forward(), which is a whole extra kernel + traffic
the Triton oracle never pays. New run below finds a genuinely new lever.

---

# 2026-07-02 run (my-GPU baseline = 1.0672x, RUNTIME 0.0253 ms, REF 0.0270 ms)

## Iter 1 — int64 idx direct (drop the `.to(int32)` conversion)  **[KEPT — NEW BEST]**
- **Hypothesis:** `forward()` did `idx.contiguous().to(torch.int32)` — a full extra kernel that reads
  idx (4 MB int64) and writes idx32 (2 MB), plus a launch, all inside the timed region. The Triton
  oracle (1.22x) feeds int64 idx straight into the gather kernel with no conversion. bench.py calls
  `clear_l2_cache()` before each trial, so this is cold-HBM bound; removing ~4 MB of extra traffic +
  one launch should close most of the gap to the oracle.
- **Change:** kernel `IDX` dtype `int32 -> int64`; `forward()` drops `.to(torch.int32)` (keeps
  `.contiguous()`, still glue-only). Index expr `X[r, IDX[r,c]]` unchanged.
- **Fast signal:** RUNTIME 0.0206 ms (vs 0.0253 baseline, -19%), CORRECT 5/5.
- **Verdict (--num-warmup 200):** COMPILED=True, CORRECT=True, RUNTIME=0.0207 ms, REF=0.0268 ms,
  **SPEEDUP=1.2947x**. Beats baseline 1.0672x AND the Triton oracle 1.2217x.
- **KEEP.** This is the discriminating change.
- **Next:** re-sweep grid/ILP on the int64 kernel (optimum may shift now the conversion is gone);
  then vectorize the contiguous OUT store + idx load (X reads stay scattered so expect small).

## Iter 2 — threads/block re-sweep on the int64 kernel  **[no real change — kept TH=256]**
- **Hypothesis:** with the conversion kernel gone, the block-config optimum may have shifted; more
  warps/block could raise memory-level parallelism to hide the scattered-load latency.
- **Change:** TH ∈ {128, 256, 512, 1024} (128 blocks, 1 block/row, 4096/TH elem/thread).
- **Fast signal (mean / min ms):** 128→0.0204/0.0195, 256→0.0206/0.0195, 512→0.0204/0.0195,
  1024→0.0203/0.0195. All within one std (~0.0006) and identical min. CORRECT at every point.
- **Verdict:** no real effect — kernel is bandwidth/latency bound regardless of block config.
  Not chasing sub-1% noise; **kept TH=256** (the iter-1 KEPT config). REVERT-to-256.
- **Next:** distinct direction — 2D grid to occupy all 142 SMs (128 blocks is under one wave).

## Iter 3 — 2D grid: split each row into column tiles  **[REVERT — worse]**
- **Hypothesis:** 128 blocks (1/row) on 142 SMs is under one wave (14 idle SMs). Split each row into
  NT column tiles → NT×128 blocks to fill the machine and add latency-hiding parallelism.
- **Change:** `T.Kernel(NT, M)`; block (bx,r) handles a COLS-wide slice of row r.
- **Fast signal (mean / min ms):** COLS=2048 (256 blocks) → 0.0211/0.0195; COLS=1024 (512 blocks)
  → 0.0215/0.0205. Both CORRECT but slower than iter-1 (0.0206/0.0195); min rose at 512 blocks.
- **Why worse:** indices are random over the whole 8192-wide row, so every block that owns *any*
  slice of a row still touches ≈all 256 cache lines of that row. Splitting a row across N blocks
  duplicates the cold-HBM X fetch of that row ~N× (only partially recovered by shared L2). One
  block per row minimizes redundant X traffic. **REVERT to iter-1 (1D grid, TH=256).**
- **Next:** decouple the idx-load → X-gather dependency (register prefetch) to raise MLP.

## Iter 4 — shared-mem two-phase (decouple idx-load from X-gather)  **[REVERT — worse]**
- **Hypothesis:** in the fused kernel each thread's X gather depends on its just-loaded idx
  (global→global dependency). Stage all idx into shared (coalesced), `T.sync_threads()`, then gather
  X reading idx from fast shared memory — removes the global-idx latency from the gather critical path.
- **Change:** `idx_s = T.alloc_shared((Cout,), "int64")`; loop-1 loads idx_s coalesced; sync; loop-2
  `OUT[r,c] = X[r, idx_s[c]]`.
- **Fast signal:** RUNTIME 0.0233 ms (mean), min 0.0225 — CORRECT but ~13% SLOWER than iter-1.
- **Why worse:** the barrier serializes the idx-read phase and the X-gather phase so they can no
  longer overlap. The fused kernel already overlaps idx reads and X gathers *across warps* (plenty of
  independent work in flight), which beats a serialized load→sync→gather. Shared write+read is pure
  added traffic. **REVERT to iter-1.**
- **Next:** vectorization lever — coalesced float4 OUT store / vectorized idx load.

## Iter 5 — explicit float4-grouped store / idx load  **[REVERT — no gain]**
- **Hypothesis:** group 4 consecutive outputs per inner step so tilelang can emit 128-bit
  (float4 store / vectorized idx load) memory ops for the *contiguous* idx-read and out-write halves
  (the X gather stays scalar/scattered).
- **Change:** `for co in T.Parallel(Cout//4): for k in T.serial(4): OUT[r,co*4+k]=X[r,IDX[r,co*4+k]]`.
- **Fast signal:** see below — mean ≈ iter-1 within noise, min unchanged 0.0195.
- **Why no gain:** in the iter-1 mapping consecutive threads already own consecutive c, so the idx
  load and OUT store are *already fully coalesced* (warp writes 32 consecutive float32 = one 128B
  line at 100% utilization). 128-bit vectorization cannot reduce that traffic; the only inefficiency
  left is the inherent scattered X gather, which vectorization cannot touch. **REVERT to iter-1.**
- **Fast signal (measured):** mean 0.0206 ms, min 0.0195 ms — identical to iter-1. Confirms.

## Conclusion (2026-07-02 run) — NEW BEST 1.2947x, AT FLOOR
- **Best kept: Iter 1 — int64 idx direct.** my-GPU verdict SPEEDUP **1.2947x** (RUNTIME 0.0207 ms,
  REF 0.0268 ms) vs committed baseline **1.0672x** (0.0253 ms). +21% runtime. Also beats the Triton
  oracle (1.2217x). The whole win came from deleting the `idx.to(torch.int32)` conversion kernel that
  the prior "AT FLOOR" grid-only sweep never questioned.
- **Floor confirmed from 5 distinct directions:** (1) int64-direct = the lever; (2) threads/block
  sweep 128–1024 = flat within noise (min 0.0195 everywhere); (3) 2D grid / more blocks = worse
  (duplicates cold-HBM X fetch per row); (4) shared-mem two-phase decouple = worse (barrier kills
  cross-warp overlap); (5) float4 grouping = no gain (contiguous idx/out already fully coalesced).
- **Roofline:** ~10 MB irreducible traffic (idx 4 MB int64 + X ~4 MB scattered + out 2 MB) + one
  irreducible kernel launch. Best 0.0195 ms ⇒ ~510 GB/s effective (~53% of 960 GB/s peak); the gap is
  the inherent scattered X gather + fixed launch/sync overhead on a ~20 µs kernel. No traffic and no
  launch can be removed without changing the op. **This is the physical floor.**
