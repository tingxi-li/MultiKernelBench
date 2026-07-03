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
| 1 | Single-pass Triton row scan (B1024/w16/s3) | 1.2243x | 10.7 ms | improved |
| 2 | Multi-row-per-program scheduling (fast-signal probe) | ~1.22x (tie/regress) | 10.49-10.67 ms | no-change |
| 3 | Cache eviction-policy streaming hints (fast-signal probe) | ~1.22x (regress) | 10.52 ms | regression |

## Iterations

### Iter 1 — Single-pass Triton row scan with running carry

- **Hypothesis:** cumsum is memory-bound (min traffic = read 4.3GB + write 4.3GB = 8.59GB). A single Triton program per row that scans the row in BLOCK chunks with a running scalar carry reads/writes each element exactly once = HBM-roofline traffic. torch.cumsum runs at only 663 GB/s, leaving headroom up to copy bandwidth (~807 GB/s, ~1.22x).
- **Changes:** Wrote `_cumsum_rows_kernel` (grid=(M,), one program/row, `tl.cumsum(axis=0)` inclusive intra-chunk + `carry += tl.sum(chunk)`). Tuned via `--no-ref`-style sweep over BLOCK×num_warps×num_stages; best = BLOCK=1024, num_warps=16, num_stages=3. fp32 carry. Fallback to torch.cumsum for non-(2D/dim1/fp32/cuda) inputs.
- **Bench:**
  - Compiled: True
  - Correct: True (5/5 trials; max abs diff ~0.07 vs ~1.6 tolerance at tail)
  - Runtime: 10.7 ms (mean), 10.5 ~ 12.2 ms (min ~ max)
  - Speedup: 1.2243x (mean)
- **Analysis:** Hit the HBM roofline on the first try. Effective BW = 8.59GB / 10.5ms = 820 GB/s, which *exceeds* the measured `tensor.copy_` bandwidth (807 GB/s). num_stages had ~0 effect (already bandwidth-saturated, not latency-bound). num_warps=16 was the strongest lever (more in-flight loads); 32 warps regressed (occupancy drop). The op cannot do less traffic than a copy, so ~1.22x is the hard ceiling.
- **Next:** Confirm no structural headroom via two distinct directions (multi-row-per-program scheduling; store eviction-policy hint). Both expected to match/regress since we're already at copy bandwidth.

### Iter 2 — Multiple rows per program (scheduling/launch overhead)

> Note: iters 2 & 3 are fast-signal direction probes (standalone cuda-event
> script, one-shot diff for correctness). solution/cumsum.py was not changed
> and `scripts/bench.sh` was not re-run for them, since both regress vs iter-1
> and the roofline was already established by iter-1's full verdict.

- **Hypothesis:** With M=32768 programs over 142 SMs there could be scheduler/tail overhead; having each program loop over R rows (grid=M/R) reduces program count and might raise effective bandwidth.
- **Changes:** Variant kernel with an outer `for r in range(R)` over rows, R in {2,4,8}. Same inner chunked carry scan. Measured with identical cuda-event methodology (warmup + L2-independent 4.3GB tensor).
- **Bench:**
  - Compiled: True
  - Correct: True (max abs diff 0.070, unchanged)
  - Runtime: R=2 10.49ms (820.8 GB/s), R=4 10.50ms (819.8 GB/s), R=8 10.67ms (810.6 GB/s) vs iter-1 10.47ms (822.2 GB/s)
  - Speedup: tie at R=2/4, regresses at R=8
- **Analysis:** No gain — iter-1 already saturates HBM, so fewer/larger programs cannot help; R=8 hurts (less inter-program overlap). Reverted; kept iter-1.
- **Next:** One more distinct direction (cache hints), then stop.

### Iter 3 — Cache eviction-policy streaming hints

- **Hypothesis:** Inputs/outputs are each touched once (streaming); `eviction_policy='evict_first'` on load+store could reduce L2 pollution and lift bandwidth.
- **Changes:** Added `eviction_policy='evict_first'` to both `tl.load` and `tl.store` in the iter-1 kernel.
- **Bench:**
  - Compiled: True
  - Correct: True (max abs diff 0.070, unchanged)
  - Runtime: 10.52 ms (819.2 GB/s) vs iter-1 10.47 ms (822.2 GB/s)
  - Speedup: slight regression
- **Analysis:** No gain. The 4.3GB input already vastly exceeds the 96MB L2, so the data is effectively streamed regardless; the explicit hint only adds a hair of overhead. Reverted; kept iter-1.
- **Next:** STOP. Three distinct directions exhausted (pipeline tuning, multi-row scheduling, cache hints). iter-1 effective bandwidth (822 GB/s) exceeds measured copy bandwidth (807 GB/s) = hard HBM roofline; cumsum cannot do less traffic than a copy. Best = iter-1 at 1.2243x, which is also current HEAD.

### Iter 2 — Pass cheating_detection.py (torch.cumsum fallback + N%BLOCK)

- **Hypothesis:** Iter-1 was fast/correct but did not pass MultiKernelBench's own anti-hack check; make all compute live in custom kernels without regressing.
- **Problem:** Iter-1 kept a `torch.cumsum` fallback path and an `N % BLOCK` guard — both flagged.
- **Changes:** Made `_cumsum_rows_kernel` mask-aware (masked load/store on the tail chunk) so it handles any N, then deleted the fallback and the `%` guard. Single Triton path. Detector: OK. Speedup retained (1.2358x, CORRECT).
- **Bench:** Compiled: True; Correct: True; Runtime 10.6 ms; Reference 13.1 ms; Speedup 1.24x.
- **Anti-hack:** `utils/cheating_detection.py` -> OK (was regression_type=3 in iter 1).
- **Next:** Compliant and at/above the iter-1 speedup — stop.

## GPU-3 re-run (floor re-confirmation on assigned GPU)

> Re-baselined the committed solution on GPU 3 (bench.sh hardcodes GPU 2 and
> HINTS says GPU 1 — both stale for this run; sibling agents own the other
> GPUs so bench.py was invoked directly with CUDA_VISIBLE_DEVICES=3). GPU-3
> baseline of the committed kernel: SPEEDUP 1.2056x, RUNTIME 10.7 ms mean /
> 10.5 ms min, CORRECT (5/5). Two additional distinct floor-confirmation
> directions below.

### Iter 4 — Physical roofline probe (copy_ / add on 4.3 GB)

- **Hypothesis:** cumsum's irreducible traffic = read all input + write all output = 8.59 GB. If the committed kernel already matches a pure elementwise copy/add of the same tensor, there is no headroom by definition (the scan is hidden under memory traffic).
- **Changes:** Standalone cuda-event probe on the real 32768×32768 fp32 tensor (>> 96 MB L2, so uncached/streamed). Measured `out.copy_(x)`, `torch.add(x,1,out=out)`, `torch.cumsum`, vs the committed kernel.
- **Bench:**
  - `out.copy_(x)`  = 10.637 ms → 808 GB/s
  - `add(x,1)`      = 10.504 ms → 818 GB/s
  - `torch.cumsum`  = 12.840 ms → 669 GB/s (the reference)
  - committed kernel = 10.5 ms min / 10.7 mean → 803–818 GB/s
- **Analysis:** The committed kernel's min (10.5 ms) equals the `add` floor (10.504 ms) and beats the `copy_` floor (10.637 ms). cumsum physically cannot move less than 8.59 GB, and that traffic alone costs ~10.5 ms at this GPU's peak achievable bandwidth. Pure memory roofline — no headroom. KEEP baseline.
- **Next:** One more direction — re-tune the launch config on GPU 3 (prior tuning was on a different GPU) to rule out an on-GPU-3 config win.

### Iter 5 — Full launch-config sweep on GPU 3 (BLOCK × num_warps × num_stages)

- **Hypothesis:** The committed 1024/w16/s3 config was tuned on a different GPU (bench.sh=GPU2, HINTS=GPU1). A different config could be faster on GPU 3, tightening the mean (10.7) toward the floor (10.5).
- **Changes:** Standalone sweep importing the committed `_cumsum_rows_kernel`; 45 configs = BLOCK∈{512,1024,2048,4096,8192} × num_warps∈{8,16,32} × num_stages∈{2,3,4}; cuda-event timing on the 4.3 GB tensor; correctness of the winner checked vs torch.cumsum.
- **Bench:** Top configs (ms / GB/s):
  - B=1024 w=16 s=4 → 10.448 / 822 GB/s (winner)
  - B=1024 w=16 s=3 → 10.449 / 822 GB/s (**committed baseline**, rank 2)
  - B=1024 w=16 s=2 → 10.449 / 822 GB/s
  - B=4096 w=16 s=3 → 10.484 / 819 GB/s
  - winner max abs err vs torch = 0.072 (well within tolerance)
- **Analysis:** The best config beats the committed baseline by 0.001 ms = 0.01% — pure measurement noise, far below the 3% keep threshold. num_warps=16 dominates every BLOCK; num_stages is inert (bandwidth-saturated, not latency-bound). The committed 1024/w16/s3 is already at the top cluster (822 GB/s = the ceiling seen for `add`). No real win exists on GPU 3. KEEP baseline verbatim.
- **Next:** STOP. Floor independently confirmed via 6 distinct directions across all runs: (1) pipeline/num_stages tuning, (2) multi-row scheduling, (3) cache eviction hints, (4) copy_ physical floor, (5) add physical floor, (6) 45-config GPU-3 sweep. Committed kernel min (10.45 ms, 822 GB/s) ≥ copy/add floors. cumsum cannot do less traffic than a copy → hard HBM roofline. Best = committed baseline, at_floor=true, improved=false, solution unchanged.

## Summary (GPU-3 confirmation)

| Iter | Title | Result (ms / GB/s) | Status |
|------|-------|--------------------|--------|
| 4 | Physical roofline probe (copy_/add) | copy 10.637/808, add 10.504/818, kernel 10.5min | floor-confirmed |
| 5 | 45-config sweep on GPU 3 | best 10.448/822 vs baseline 10.449/822 (0.01% = noise) | floor-confirmed |
