# Iteration Log

## Summary

| Iter | Title | Runtime mean (ms) | Runtime min (ms) | Status |
|------|-------|-------------------|------------------|--------|
| 0 | Committed baseline (autotuned BLOCK 256-2048 x warps 4/8) | 0.0209 | 0.0195 | baseline |
| 1 | Plain config sweep -> pin BLOCK=4096 w4 (high ILP) | 0.0196 | 0.0184 | improved |
| 2 | 2D grid (uniform row base pointer) | 0.0195 | 0.0195 | revert (no gain) |
| 3 | int32 addressing (col.to(int32)) | 0.0209 | 0.0195 | revert (worse) |
| 4 | eviction_policy: x=evict_last (+ idx=evict_first) | 0.0138 | 0.0123 | improved (big) |
| 5 | eviction ablation -> x=evict_last is the sole driver | 0.0137 | 0.0123 | analysis |
| 6 | config sweep under evict_last -> BLOCK=4096 w8 | 0.0129 | 0.0121 | improved |
| 7 | warps refinement -> BLOCK=4096 w16 | 0.0126 | 0.0113 | improved (best) |
| 8 | cache_modifier (.cg/.cs on x/idx) | n/a | n/a | revert (invalid w/ evict policy) |
| 9 | no-mask (exact divisibility) + contiguity hints | 0.0125 | 0.0113 | revert (noise-level) |
| 10 | floor confirmation (w8/w16/w32, mask/nomask all -> min 0.0113) | 0.0124-0.0127 | 0.0113 | at floor |

Baseline (this GPU, GPU 1): mean 0.0209 ms / min 0.0195 ms, speedup 1.34x (bench.sh full verdict 0.0227/0.0205).
Best (pinned BLOCK=4096, num_warps=16, x=evict_last): mean ~0.0126 ms / min 0.0113 ms.
All numbers are fresh-process bench.py `--no-ref --num-warmup 200 --num-perf-trials 100` (min = least-interrupted estimate).

## Measurement note (critical)
An early in-process multi-config sweep (scripts/sweep.py, focus2.py) produced
physically impossible times (min 0.0072 ms == 1.46 TB/s > 960 GB/s HBM peak):
`clear_l2()` is unreliable when many kernels run back-to-back in one process, so
x stayed warm in L2 and numbers were bimodal for identical kernels. ALL rankings
below therefore come from fresh-process bench.py runs (clean L2 per process),
which is the orchestrator's source of truth. The in-process sweep was used only
to shortlist configs, never to decide a winner.

## Roofline
Per-launch HBM traffic (L2 thrashed cold before each trial): idx (128x4096 int64
= 4 MiB, unavoidable; int32-view saves nothing because int64 lo/hi interleave so
a stride-2 read still touches every 32B sector) + first-touch of x (~3.9 MiB,
coupon-collector over each row's 1024 sectors with 4096 random picks) + out
(2 MiB) ~= 9.9 MiB. At 960 GB/s peak that is ~10.3 us.
- Baseline min 0.0195 ms = ~538 GB/s = 56% of peak (x reuse missed cache).
- Best min 0.0113 ms = ~928 GB/s = 97% of peak; mean 0.0126 ms = ~832 GB/s = 87%.
=> AT the memory roofline. The gap closed by evict_last is consistent with
redundant HBM re-reads of x on its ~4x intra-launch reuse being converted into
cache hits (exact cache level not isolated; see Iter 5).

## Iterations

### Iter 1 — Plain config sweep, pin BLOCK=4096 num_warps=4
- **Hypothesis:** the committed autotune space tops out at BLOCK=2048; a bigger
  BLOCK gives more independent gather loads per thread (ILP) to hide the random-
  gather latency, and pinning removes autotune/process variance.
- **Change:** single pinned kernel, BLOCK=4096 (== 1 block per row so `row` is
  uniform per program and the base pointer math is hoisted), num_warps=4.
- **Result:** mean 0.0196 / min 0.0184 vs baseline 0.0209 / 0.0195. Small but real.
- **KEEP** (interim). **Next:** attack the random-gather cache behavior.

### Iter 2 — 2D grid (row = program_id(0), n-tiles = program_id(1))
- **Hypothesis:** an explicit 2D launch makes the per-row base pointer uniform
  and removes the `// ncol_out` division.
- **Result:** min 0.0195, no better than the 1D BLOCK=4096 (which already gets a
  uniform row for free). **REVERT.**

### Iter 3 — int32 addressing (col cast to int32)
- **Hypothesis:** 32-bit address math lowers register pressure -> occupancy.
- **Result:** min 0.0195, mean 0.0209 — worse; the extra `.to(int32)` op costs
  more than int64 addressing (all offsets < 2^21 either way). **REVERT.**

### Iter 4 — eviction_policy: x load = evict_last, idx load = evict_first
- **Hypothesis:** each row of x (32 KiB) is reused ~4x within a launch (4096
  random picks over 1024 sectors); `evict_last` on x should raise its cache
  retention so the reuse hits cache instead of re-reading HBM.
- **Result:** mean 0.0138 / min 0.0123 vs 0.0196 / 0.0184. ~30% win. CORRECT.
- **KEEP.**

### Iter 5 — eviction ablation (fresh-process)
- x=evict_last alone: 0.0137/0.0123. idx=evict_first alone: 0.0212 (no effect).
  both=evict_first: 0.0213. none: 0.0211. **x=evict_last is the SOLE empirical
  driver; idx's eviction policy has NO measurable effect** (an earlier "idx
  stream evicts x" story is falsified by this ablation — and both x and idx
  (~4MB each) fit the 96MB L2 with huge margin, so an L2-capacity story would not
  hold anyway; the retention effect is most likely at the small per-SM L1 where
  each block's ~4x reuse of its 32KB row lives, but the exact cache level was not
  isolated). Kept idx=evict_first only as a measurement-neutral hint.

### Iter 6 — config sweep under evict_last
- BLOCK x warps grid (fresh-process): B4096 w8 mean 0.0129 / min 0.0121 best;
  B2048 w8 0.0138; B8192 w8 0.0143 (only 64 blocks -> SM under-utilization);
  B512 w4 0.0140. **KEEP B4096 w8.**

### Iter 7 — warp refinement -> num_warps=16
- B4096 w16: mean 0.0126 / min 0.0113, reproducible over 3 runs (0.0126 each,
  min 0.0113 each) vs w8 0.0128-0.0129 / 0.0123. Real, outside noise. **KEEP.**

### Iter 8 — cache_modifier (.cg/.cs) on x/idx
- Combining cache_modifier with eviction_policy errored in this Triton build; no
  usable result and evict_last already captures the L2/L1 win. **REVERT.**

### Iter 9 — no-mask (BLOCK divides n_out exactly) + contiguity hints
- 128*4096 == n_out so mask is provably always-true; also tried
  tl.max_contiguous/tl.multiple_of on offs. Both -> mean 0.0125 / min 0.0113,
  i.e. ~0.0001 over the masked kernel = within noise. **REVERT** — kept the mask
  (correctness-safe, no measurable cost).

### Iter 10 — floor confirmation
- w8/w16/w32 x mask/nomask x hints all converge to min 0.0113 ms (~97% HBM
  peak), mean 0.0124-0.0127. Multiple distinct directions bottom out at the same
  floor => confirmed at the memory roofline. **STOP.**

## Final
Pinned BLOCK=4096, num_warps=16, num_stages=1, x load eviction_policy=evict_last
(idx evict_first). Full verdict recorded in trajectory/*_final/output.txt.
