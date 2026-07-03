# Iteration Log

> **Final kept result: 2.10x** — the L2-residency breakthrough (see the "Re-run … L2-residency
> breakthrough" section below). The older "stop at 1.60x" conclusion further down is **superseded**;
> this log is not strictly chronological (a later re-run section was inserted above it).

<!--
Per-iteration template (copy when adding a new iter entry under "## Iterations"):

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False
  - Runtime: ___ ms (mean), ___ ~ ___ ms (min ~ max)
  - Speedup: ___x (mean), ___ ~ ___x (min ~ max)
- **Analysis:** Why it worked or failed
- **Next:** What to try next

Append one row per iter to the Summary table below.
Status values: improved / no-change / regression / failed.
-->

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Split-row two-pass Triton | 1.5686x | 4.08 ms | improved |
| 2 | Tune S/BLOCK/warps/stages | 1.5725x | 4.07 ms | improved (marginal) |
| 3 | L2-fusion (row groups) | 1.7230x* | 4.26 ms | regression (*ref-noise) |

## Re-run (2026-07-02, GPU 2) — L2-residency breakthrough

Prior run concluded "3-pass roofline, done". That was WRONG about the floor: a
single row (16.78 MB) + shared w/b (33.5 MB) + its out (16.78 MB) = 67 MB FITS
the 96 MB L2, so processing ONE row per launch lets the apply pass re-read x from
L2 instead of HBM. That drops HBM traffic from 3 passes (3.22 GB, 4.0 ms) to 2
passes (2.15 GB, 2.69 ms floor). Baseline on THIS gpu: mean 4.05 / min 3.95 ms,
1.5877x. New best lands at min ~2.99 ms (bench) / ~2.86 ms (event probe) = ~2.15x.

| RIter | Title | Runtime(min) | Status |
|------|-------|--------------|--------|
| R1 | L2-resident per-row, fused reduce+apply, evict hints | ~2.99 ms bench / 2.86 probe | improved (big) |
| R2 | Two-stream overlap stats[r+1] || apply[r] | 4.29 ms probe | REVERT (breaks L2 residency) |
| R3 | 1D grid + param re-sweep (S/BLOCK/NS/NW) | 2.85 ms probe | no-change (floor confirmed) |

## Iterations (re-run)

### RIter 1 — L2-resident per-row pipeline (breakthrough: 3-pass -> 2-pass)

- **Hypothesis:** The prior "3-pass roofline" assumed no cross-pass x reuse. But one row is 16.78 MB and L2 is 96 MB. If we process ONE row per launch (stats then fused-reduce+apply), that row stays hot in L2 between the two passes, so the apply RE-READS x from L2, not HBM -> traffic 3.22 GB -> 2.15 GB -> floor 2.69 ms.
- **Changes:** (1) `_stats_kernel` gains a runtime `ROWBASE` arg; grid=(1,S); x-loads use `eviction_policy="evict_last"`. (2) new `_apply_kernel` folds the per-row reduce (tree-reduce of the S partials) IN, so no separate reduce launch and no mean/rstd global round-trip; out-stores use `eviction_policy="evict_first"` so streaming writes don't evict hot x/w/b. forward() loops `for rb in range(M): stats; apply` — no BinOp, no rb-indexed subscript, so it passes the detector (verified: valid=True). Swept G in {1,2,4}: only G=1 fits L2 (G>=2 -> back to ~4.09 ms). Swept S/BLOCK/NS/NW: S=512, BLOCK=4096, NS=2, NW=4 best.
- **Bench:** Compiled True; Correct 5/5; fast-signal min 2.99 / mean 3.11 ms; event-probe 2.856 ms. Baseline on this gpu was min 3.95 / mean 4.05 ms.
- **Analysis:** L2 reuse materializes cleanly at G=1 (67 MB working set < 96 MB). Landing at 94% of the hard 2-pass copy-bandwidth floor (2.69 ms = measured 1R+1W of 1 GB). iter-3's old L2-fusion failed only because G=8 (134 MB) overflowed L2 and it used torch combines; per-row + eviction hints fixes it.
- **Next:** Try to close the last ~0.17 ms (two-stream overlap of stats[r+1]||apply[r]; 1D grid; cache-modifier tweaks). Expect small/none since HBM read+write share one 798 GB/s bus (overlap can't exceed aggregate BW).


### RIter 2 — Two-stream overlap of stats[r+1] with apply[r] (close the launch gap)

- **Hypothesis:** 128 launches (2/row) leave ~0.16 ms of launch-gap between the 2.86 ms achieved and the 2.69 ms 2-pass floor. Overlapping the (read-bound) stats of the next row with the (write-bound) apply of the current row on two CUDA streams could hide those gaps.
- **Changes:** Pipeline on streams sA/sB with events: prime stats[0] on sB; each iter apply[r] on sA (after waiting stats[r]) while stats[r+1] runs on sB.
- **Bench:** Correct; probe 4.29 ms — a large REGRESSION.
- **Analysis:** stats[r+1] streams x[r+1] (16.78 MB) into L2 concurrently with apply[r] still re-reading x[r] from L2. That evicts x[r] mid-apply, so the reuse collapses back toward the 3-pass roofline, plus read/write contention. Confirms the L2 residency is only safe with STRICTLY SEQUENTIAL per-row execution (single stream). REVERT. Also confirms the 2-pass floor is real: HBM read+write share one 798 GB/s bus, so overlap cannot beat aggregate BW anyway.
- **Next:** Micro-variants only.

### RIter 3 — 1D grid + parameter re-sweep at the optimum

- **Hypothesis:** removing the size-1 program_id(0) dimension (grid=(S,) with the row passed directly) and re-sweeping S/BLOCK/NS/NW might shave the last few %.
- **Changes:** 1D-grid kernels; swept S in {256,512}, BLOCK in {4096,8192}, NS=2, NW in {4,8}.
- **Bench:** 1D 2.852 ms vs base 2.872 ms; NW8 2.861; B8192 2.870; S256 3.182. All within run-to-run noise of the deployed base except S256 (fewer programs -> underutilized).
- **Analysis:** Flat plateau at ~2.85-2.87 ms regardless of grid shape or knobs = fully bandwidth-bound at 94% of the hard 2-pass copy floor (2.69 ms). The 1D edge (0.7%) is inside noise and not worth changing the already-verified deployed solution. KEEP base (S=512/BLOCK=4096/NS=2/NW=4). Floor confirmed across 6 distinct directions (roofline math, G-fit sweep, fused-vs-separate reduce, two-stream, grid shape, knob sweep). Stop.
- **Next:** Final verdict on the deployed best.


### Iter 3 — L2-fusion: process rows in groups (stretch, break the 3GB floor)

- **Hypothesis:** The 3-pass floor (3.22GB) assumes no cross-pass L2 reuse. If we process row-groups whose x fits in the 96MB L2, the apply pass could re-read x from L2 instead of HBM, cutting traffic toward ~2GB (-> ~2.5ms ideal, ~2.5x).
- **Changes:** Added a ROWBASE arg to both kernels; loop over groups of G=8 rows, launching stats+combine+apply per group. Offline-swept G in {2,4,8}, S in {64,128,256}: best was G=8/S=64/B=2048 at 4.20ms; smaller groups (G=2,4) were slower despite fitting L2 better.
- **Bench:**
  - Compiled: True
  - Correct: True (5/5)
  - Runtime: 4.26 ms (mean), 4.21 ~ 5.94 ms (min ~ max)
  - Speedup: 1.7230x (mean) — BUT this run's REF measured 7.34ms vs the stable 6.40ms; an apples-to-apples iter2 re-check immediately after gave REF 6.40ms / sol 4.06ms. The solution RUNTIME (4.26ms) is clearly WORSE than iter2 (4.06ms). The inflated speedup is reference-timing noise, not a real gain.
- **Analysis:** L2 reuse does not materialize. At G=4, 64MB of x + 32MB of w/b exactly saturate the 96MB L2, and output write-allocate + w/b reads evict x before the apply pass reuses it; smaller groups also starve parallelism, and the extra per-group kernel launches + tiny torch combines add overhead. The monolithic 3-pass already lets the hardware cache the (small, shared) w/b in L2 — which is why it sits at the 3GB roofline. Grouping only adds overhead. Direction abandoned.
- **Next:** Roofline confirmed across 3 distinct directions (split-row, param tuning, L2-fusion). Restore iter2 (the genuine best by solution runtime) as final. Stop.


### Iter 2 — Tune S / BLOCK / num_warps / num_stages

- **Hypothesis:** iter 1 (4.08ms) is at the 3-pass roofline, but a parameter sweep might shave a few % by improving occupancy / load pipelining.
- **Changes:** Offline-swept S in {8,16,32,64}, BLOCK in {2048,4096,8192}, num_warps in {4,8}, num_stages in {2,3,4} (72 configs). All landed in 3.99-4.06ms (1.5% spread) -> fully bandwidth-bound. Picked the best: S=64, BLOCK=2048, num_warps=4, num_stages=3.
- **Bench:**
  - Compiled: True
  - Correct: True (5/5)
  - Runtime: 4.07 ms (mean), 3.97 ~ 5.99 ms (min ~ max)
  - Speedup: 1.5725x (mean); REF 6.40 ms
- **Analysis:** Within noise of iter 1. Confirms the kernel is HBM-bound: tuning cannot move it because we already saturate the ~805 GB/s the GPU streams. min=3.97ms == 3.22GB/805GB-s floor.
- **Next:** Try the L2-fusion stretch (iter 3) to reduce HBM traffic below 3GB; if it does not beat 4.0ms, the roofline is confirmed and we stop.


### Iter 1 — Split-row two-pass Triton (stats + apply)

- **Hypothesis:** torch native_layer_norm hits only ~507 GB/s here (6.36ms for 3.22GB), well below the ~805 GB/s this GPU streams (measured via x.add/copy). Cause: M=64 rows < 142 SMs, so a per-row kernel underutilizes. Splitting each row into S chunks across many programs should saturate HBM and approach the 3-pass floor of ~4.0ms.
- **Changes:** Two Triton kernels. (1) `_stats_kernel` grid=(M,S): each program sums x and x^2 over a CHUNK=N/S slice using a vector accumulator + single tl.sum (tree reduction). Partials combined in fp64 in torch -> per-row mean/rstd. (2) `_apply_kernel` grid=(M,S): out=(x-mean)*rstd*w+b, w/b indexed by within-row offset. S=16, BLOCK=4096, num_warps=4, num_stages=3. Reads self.ln.weight/bias/eps.
- **Bench:**
  - Compiled: True
  - Correct: True (3/3 fast, 5/5 verdict)
  - Runtime: 4.08 ms (mean), 3.97 ~ 5.98 ms (min ~ max)
  - Speedup: 1.5686x (mean); REF 6.40 ms
- **Analysis:** Landed at 4.08ms mean / 3.97ms min, essentially the 3.22GB / 805 GB/s = 4.0ms roofline for a clean 3-pass (2 reads + 1 write of the 1GB tensor). The split-row design fixed the SM underutilization that throttles torch. This is the clean 3-pass ceiling.
- **Next:** Tune S/BLOCK/num_warps/num_stages to push mean toward min (iter 2). Then evaluate the L2-fusion stretch (process row-groups <=96MB so the apply pass re-reads x from L2, cutting HBM traffic from 3GB to ~2GB) vs the parallelism it sacrifices.


### Iter 2 — Pass MultiKernelBench utils/cheating_detection.py (stats were in torch)

- **Hypothesis:** Iter-1 was fast/correct but did not pass MultiKernelBench's own anti-hack check; make all compute live in custom kernels without regressing.
- **Problem:** Iter-1 computed mean/var/rstd in torch between two Triton kernels — the detector (regression_type=3) flagged torch.rsqrt + tensor arithmetic in forward.
- **Changes:** Added a Triton `_reduce_kernel` (one program/row, tree-reduces the S partial sums into mean/var/rstd); moved CHUNK/N derivation into __init__. forward() is now allocate+reshape+launch only. fp32 tree-reduce holds correctness (<1e-4). Detector: OK. Speedup retained (1.5975x, CORRECT).
- **Bench:** Compiled: True; Correct: True; Runtime 4.00 ms; Reference 6.39 ms; Speedup 1.60x.
- **Anti-hack:** `utils/cheating_detection.py` -> OK (was regression_type=3 in iter 1).
- **Next:** Compliant and at/above the iter-1 speedup — stop.
