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
| 1 | identity baseline (torch.sum) | 1.0000x | 9.83 ms | baseline |
| 2 | v1 scalar coalesced col-sum (thread/col, __ldg) | 0.9919x | 9.91 ms | regression |
| 3 | v2 float4 4-col/thread, 4 accums | 1.0072x | 9.76 ms | improved (BEST) |
| 4 | v3 float4 + 4-deep row unroll | 1.0072x | 9.76 ms | no-change |
| final | v2 float4 4-col/thread (best) | 1.0072x | 9.72 ms | final |

## Iterations

Op is MEMORY-BOUND: sum over dim=1 of (128,4096,4096) fp32 = read 8.6 GB once,
write 2 MB. Pure-read HBM roofline ~= 8.96 ms (@960 GB/s theoretical peak);
torch.sum reference = 9.83 ms (875 GB/s, ~91% of peak). This is a column-sum of a
row-major matrix (reduce the stride-D2 axis), so a thread-per-output-column layout
gives fully coalesced loads.

- v1 (scalar, thread/col): 0.9919x — coalesced and correct but one dependent add
  chain per thread; marginally under torch.
- v2 (float4, 4 cols/thread, 4 independent accumulators): 1.0072x, 9.76 ms
  (881 GB/s) — vectorized 16 B loads + 4 accumulators hide the load-use latency of
  the long reduction; BEAT torch by 0.7%. **BEST — kept.**
- v3 (v2 + 4-deep manual row unroll): identical 9.76 ms — already bandwidth-bound,
  extra ILP does nothing. Reverted to v2.

**Stop:** memory-bound, at the practical HBM ceiling (~880 GB/s; torch itself only
sustains 875). Best 1.0072x > torch; 2 consecutive variants (v2,v3) within <3%
(tie). No inline PTX (plain float4 __ldg only). Detector-clean (forward = launch glue).
