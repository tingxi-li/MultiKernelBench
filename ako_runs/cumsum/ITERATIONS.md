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
