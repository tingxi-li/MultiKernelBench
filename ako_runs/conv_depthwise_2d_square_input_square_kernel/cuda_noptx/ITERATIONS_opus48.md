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
| 1 | identity baseline (cuDNN conv2d groups=C) | 1.0000x | 5.84 ms | baseline (ref) |
| 2 | v1 dw3x3s1 thread/out, coalesced, __ldg | 2.0709x | 2.82 ms | improved |
| 3 | v2 dw3x3s1 RPT=4 OH row-reuse in regs | 2.2375x | 2.61 ms | improved (BEST) |
| 4 | v3 RPT=8 | 2.2121x | 2.64 ms | no-change |
| final | v2 dw3x3s1 RPT=4 OH row-reuse in regs | 1.5909x | 2.64 ms | final |

## Iterations

Depthwise 3x3 s1 p0, x=[16,64,512,512] -> [16,64,510,510]. Reference = cuDNN =
**5.84 ms**, which is ~2.5x above the memory roofline (input 1.07 GB + output
1.02 GB = ~2.1 GB, ~2.4 ms @ peak). cuDNN's generic depthwise path leaves lots on
the table -> big win available.

- v1: specialized unrolled 3x3, one thread per output pixel, coalesced over OW,
  9 taps via __ldg, weights in registers. 2.82 ms, **2.07x** (L2 captures the
  halo reuse).
- v2 (BEST): each thread computes RPT=4 outputs down OH, reusing the 3 overlapping
  input rows in registers -> (RPT+2) row-reads for RPT outputs, halving load
  instructions. 2.61 ms, **2.24x**.
- v3 RPT=8: more registers, fewer blocks, no gain (L2 already handles cross-tile
  reuse) -> reverted to RPT=4.

**Stop:** ncu on v2 = **DRAM 91.1% of peak**, dram__bytes 2.09 GB (= input read
exactly once + output), l1tex 41%, occupancy 90%. This is the memory roofline; the
remaining ~9% to 100% DRAM is unreachable in practice. Beats cuDNN 2.24x. No inline
PTX (plain __ldg). Detector-clean (forward reads conv2d.weight, never calls it).
