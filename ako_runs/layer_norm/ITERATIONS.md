# Iteration Log

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

## Iterations

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
