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
| final | PTX ld.global.cs.v4 streaming float4 | 1.0124x | 9.67 ms | final |

## Iterations

## Iterations (cuda_unlimited convergence run)

- **iter1 identity (torch.sum):** 9.84 ms, 1.00x baseline. sum dim=1 of (128,4096,4096) fp32, reads 8.6 GB.
- **iter2 v1 1-thread-per-col unroll8 scalar:** 9.79 ms, 1.0051x. One thread per (outer,inner) output, loop over reduce axis; adjacent threads = adjacent inner idx -> coalesced.
- **iter3 v2 float4 (4 cols/thread):** 9.77 ms, 1.0072x. Vectorized coalesced loads, 4 accumulators/thread.
- **iter4 v3 PTX ld.global.cs.v4 streaming:** 9.71 ms, 1.0134x (BEST). Cache-streaming (evict-first) vectorized load avoids L2 pollution on a pure-streaming read.
- **ncu (at v2 stall):** DRAM total = 1.00 passes (8.01 GiB read, exactly one read of the 8 GiB tensor), dram% = 97.3 (saturated), occ 59.8%. Binding roofline = 1-read HBM floor, HIT.
- **PTX finding:** inline PTX `.cs` cache hint gave a small but real ~0.6% unique win over the plain float4 intrinsic — the streaming/evict-first cache policy is not expressible with a default vectorized load. (Contrasts the prior study's "PTX = 0 wins on memory-bound".)
- **STOP:** at 1-read roofline (ncu 1.00 passes, 97% DRAM); v2->v3 and v1->v2 both <3% levers. Final 1.0134x, detector-clean.
