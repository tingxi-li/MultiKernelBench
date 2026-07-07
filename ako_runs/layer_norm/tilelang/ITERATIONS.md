# Iteration Log — layer_norm / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/layer_norm/triton/solution/layer_norm.py`,
Triton speedup 1.6050x); benched against the same `reference/normalization/layer_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 0 (baseline) | per-row block, X read twice | 1.6015x | 3.9900 ms | 6.3900 ms | correct |
| 1 | persistent cooperative-grid, X read ONCE | ~1.79x | 3.5600 ms | 6.39 (prev) | correct |
| 2 | tune block count / threads (G=780, TH=64) | ~2.0-2.1x | 3.1400 ms (min 3.00) | 6.39 (prev) | correct |
| **FINAL** | **persistent single-read, G=780 TH=64 (verdict)** | **2.1053x** | **3.0400 ms (min 2.99)** | **6.40 ms** | **correct** |

## Re-run context (my GPU baseline)

Committed baseline re-benched on GPU 0 (this pass): COMPILED=True, CORRECT=5/5,
**RUNTIME=3.99 ms (min 3.90), REF=6.39 ms, SPEEDUP=1.6015x**. All deltas below are
measured against THIS GPU-0 baseline (absolute ms carry clock noise).

### Roofline (the load-bearing fact)
M=64 rows, N=4,194,304/row. X=1 GiB, Y=1 GiB. The baseline per-row block cannot
cache the 16 MB row in shared mem, so it reads X TWICE (stats pass + apply pass)
+ writes Y once = **~3.22 GB DRAM traffic**. 3.22 GB / 3.99 ms = **807 GB/s ≈ 84%**
of the RTX 6000 Ada 960 GB/s peak. So the baseline is AT the 3-pass roofline; the
only sub-roofline win is to cut traffic, i.e. read X once (→ 2 GB). Confirmed with
advisor: micro-opts (float4, split-row, fp64) are spent negatives from prior work.

## Iter 1 — persistent single-read kernel (KEEP, new best)

- **Hypothesis:** Cut DRAM traffic 3*N→2*N by reading X only once. A single
  cooperative grid (`T.sync_grid`, auto `cudaLaunchCooperativeKernel`) processes
  ONE row at a time: all G blocks stream row m for the reduction (fills L2 with
  just 16 MB, fits the 96 MB AD102 L2) → grid-sync → all blocks RE-READ row m
  (now an L2 hit) and write Y. X hits DRAM once. Cross-block sum/sumsq via
  `T.alloc_global` scratch + one global atomic/block. Per-row Acc[M,2] slots
  zeroed once so no reset barrier; a post-apply grid-sync keeps X[m] L2-resident
  (stops the next row's stats reads from evicting it mid-apply). 2 barriers/row.
- **Change:** rewrote the kernel; forward() unchanged (still glue-only,
  subscript-dispatch `_B[0]`). G=396, TH=256.
- **Bench (--no-ref, warmup 200, 30 trials):** COMPILED=True, CORRECT=5/5,
  **RUNTIME mean 3.56 ms / min 3.47 ms** vs baseline 3.99 / 3.90.
  Effective ~2.1 GB / 3.56 ms ≈ 605 GB/s (traffic cut wins despite the barrier
  bubbles + read/write phase separation dropping bandwidth from 807).
- **Speedup ≈ 6.39 / 3.56 = 1.79x** (verdict pending).
- **KEEP.** Next: tune G (block count) and TH; then chase the lost bandwidth
  (streaming stores, vectorization, row pipelining).

## Iter 2 — block-count / thread-count sweep (KEEP, new best)

- **Hypothesis:** 605 GB/s << 807 baseline -> the grid is under-saturating DRAM.
  More, smaller blocks add SMs/waves (better bandwidth) and cut per-block
  shared-atomic contention. Cooperative-launch caps the grid at the resident
  capacity (grid too big -> CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE), which
  grows as TH shrinks.
- **Change:** swept (G, TH) via host-timed harness (median of 30):
  TH=256 best G=416 -> 3.37 ms; TH=128 best G=456 -> 3.12 ms;
  **TH=64 best G=710-780 -> 2.94 ms**; TH=32 regresses (3.10+, too few
  warps/block, 2x barrier participants + global atomics). Picked G=780, TH=64.
- **Bench (--no-ref, warmup 200, 30 trials):** COMPILED=True, CORRECT=5/5,
  **RUNTIME mean 3.14 ms / min 3.00 ms** (mean inflated by one 4.92 warmup
  outlier). ~2.147 GB / 3.0 ms ≈ 716 GB/s. **Speedup ≈ 6.39/3.14 = 2.03x**
  (min-based 2.13x). KEEP. Next: recover the last ~90 GB/s vs the 807 roofline.

## Iters 3-6 — bandwidth-recovery attempts (all REVERT; floor confirmed)

Ideal for the single-read structure ≈ 2.48 ms (stats-read 1 GB + apply-write 1 GB
at ~807 GB/s). We sit at ~2.94 ms host / 3.0 ms min real (~730 GB/s); the ~0.46 ms
gap is grid-barrier bubbles + the pure-read-then-pure-write phase separation that
L2 residency *requires*. Every attempt to close it failed — the L2-residency
constraint (keep the 16 MB row resident stats→apply) dominates:

- **Iter 3 — float4 vectorization** (T.vectorized 128-bit loads/stores, stats +
  apply): 2.936 ms vs 2.940 scalar = NO gain. Memory transactions already
  coalesce; transaction size isn't the limiter. **REVERT** (keep scalar, simpler).
- **Iter 4 — software pipeline** (fuse apply(m) with stats(m+1) in one loop to mix
  DRAM reads/writes + halve barriers): **3.59 ms, REGRESS.** Streaming X[m+1]
  during apply(m) evicts X[m] from L2 → X re-read from DRAM (traffic 2→~3 GB).
  The barrier phase-separation is what protects residency; mixing R/W defeats it.
- **Iter 5 — batch B rows per barrier phase** (stats B rows → barrier → apply B
  rows; B rows = B·16 MB fit L2): B=2 → 3.71, B=4 → 4.41, B=8 → 4.59 ms, all
  **REGRESS.** More concurrent resident rows + W/B(32 MB) + Y write-allocate
  exceed the residency headroom and evict X. B=1 (min L2 footprint) is optimal.
- **Iter 6 — K-way atomic accumulators** (spread G=780 blocks' global atomics
  across K columns to cut single-address L2-atomic contention): K=1 → 2.926,
  K=8 → 2.945, K=32 → 3.075 ms. NO gain — the ~780 global atomics/row are not a
  bottleneck. Also TH=96 (3.02) > TH=64 (2.94). **REVERT** (keep K=1, TH=64).

**Floor confirmed** via 7 distinct directions (barrier-count 1/2/3, vectorization,
pipelining, batching, K-atomics, TH sweep, G sweep). Best kept: persistent
single-read, TH=64, G=780, 2 barriers/row, scalar, single accumulator.

## FINAL VERDICT (bench.sh final, --num-warmup 200, 100 trials)

- COMPILED=True, CORRECT=5/5, **RUNTIME=3.04 ms (min 2.99, std 0.199)**,
  REF=6.40 ms, **SPEEDUP=2.1053x** (cuda_event). Detector: valid=True,
  regression_type=None (forward() glue-only, subscript-dispatch `_B[0]`).
- **1.6015x baseline -> 2.1053x** (3.99 -> 3.04 ms, -24% runtime). The win is
  purely the traffic cut: the baseline reads X twice (3.22 GB @ 807 GB/s = 3.98 ms
  roofline); the persistent cooperative-grid kernel reads X ONCE by keeping each
  16 MB row L2-resident across its stats->apply passes (2.15 GB @ ~707 GB/s).
  Micro-opts on top of the single-read structure are exhausted (see iters 3-6).

---

# Convergence redo (branch cross-dsl-6op-ncu-redo) — rewritten from roofline

Fresh run from the IDENTITY reset baseline (0.9969x). Kernel written from scratch
via the roofline argument below; benched ONLY through `tools/timed_bench.sh
... --gpu3-serial` (frozen yardstick, GPU3 serial lock). convergence.csv is the
authoritative log; this is the narrative.

### Roofline (the lever)
x = (M=64 rows, N=4,194,304 fp32). One row = 16.78 MB; whole tensor = 1.07 GB;
L2 = 96 MB. A naive kernel reads x for stats, reads x AGAIN to normalize, writes y
= 3 HBM passes. ONE row (16.78 MB) fits L2, so a 2-pass design keeps the row
L2-resident across stats->apply (2nd read hits L2). weight/bias (16 MB each) are
reused every row and stay L2-resident too. TileLang's mechanism: a SINGLE
cooperative-grid launch (`T.sync_grid` = `cooperative_groups::this_grid().sync()`).
All G blocks process ONE row at a time -> only that row's 16 MB is in flight ->
stays L2-hot. Per row: grid-stride stats -> [grid barrier A: make cross-block
atomic reduction visible] -> compute mean/rstd -> grid-stride apply -> [grid
barrier B: keep row m resident until every block finishes apply]. All reductions
fp32 (AD102 runs fp64 at 1/64 rate); short per-thread chains (~iters=N/(G*TH)).

## Iter 2 — L2-resident cooperative single-read, TH=64 G=640 (KEEP, new best)
- **Hypothesis:** cut 3 HBM passes -> 2 by reading x once (L2 residency via the
  cooperative single-row loop). Per-thread fp32 sum/sumsq, shared-mem tree
  reduction (log2(TH) barriers), one global atomic/block into a per-row [M,2]
  scratch (zeroed once in glue), `T.sync_grid` between phases.
- **Result:** COMPILED=True, CORRECT=5/5, **RUNTIME 2.92 ms, SPEEDUP 2.1918x**
  (ref 6.40 ms). max_abs_diff 9.9e-6 (tol 1e-4). forward() glue-only (detector
  valid=True, subscript-dispatch `_KB[0]`). **KEEP.** Already above the ≈2.10x
  ceiling on the first custom variant.

## Iter 3 — grid-count push, TH=64 G=768 (near cooperative cap) (REVERT, tie)
- **Hypothesis:** more blocks -> better DRAM saturation. GPU0 launch/timing sweep
  (idle-clock, so only relative) showed all (G,TH) in {32,64,128}x{256..1300}
  launch cooperatively, correct, and flat within 0.4% -> config is insensitive.
- **Result:** **RUNTIME 2.92 ms, SPEEDUP 2.1918x** — identical to G=640 (0% delta).
  kept=0 (tie, didn't beat). **REVERT to G=640.** -> a 2-variant stall.

## Iter 4 — final confirm + ncu, TH=64 G=640 (REVERT, tie; ncu-anchored)
- Re-bench of the kept best carrying the ncu steering key. **2.1884x** (2.92 ms),
  5/5 correct — 3rd bench within 0.16% of the others (stable).
- **ncu (application replay, cache-control none):** DRAM total **2.057 GiB =
  2.06 passes** (2 = copy floor, 3 = un-reused re-read); DRAM read 1.046 GiB =
  **1.05x tensor** (x read ONCE), DRAM write 1.012 GiB = 1.0x (y written once);
  single `kernel_kernel` launch; L2 hit 79.4% (apply re-read hits L2). The
  weight+bias 32 MB fold into the 0.05x read overhead -> also L2-resident.
  **ncu_key = passes=2.06 -> the 2-pass binding roofline is confirmed hit.**

## STOP
Stopped after 3 custom variants (cum_compute_s = 50.82 s). **Reason:** stop-rule
branch 1 met — 2.1918x is ABOVE the ≈2.10x calibration ceiling (+4.3%, well
within 5%). Independently, branch 2 is also satisfied: 2 consecutive variants at
2.1918x (0% delta) = a stall, AND ncu confirms the 2-pass floor (2.06 passes,
read 1.05x / write 1.0x). Traffic ~2.06 GB @ 2.92 ms ~= 720 GB/s; the ~0.2 ms gap
to the ~2.7 ms bandwidth ideal is the grid-barrier bubbles + read-then-write phase
separation that L2 residency *requires* — the intrinsic floor of this structure
(matches the pre-reset winner's iters 3-6 analysis). Best variant left in
`solution/`: cooperative single-read, TH=64, G=640, fp32, 2 grid barriers/row.
