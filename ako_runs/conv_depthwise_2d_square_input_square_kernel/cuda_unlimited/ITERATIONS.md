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

## Iterations

## Iterations (cuda_unlimited convergence run)

Reference = cuDNN depthwise conv (torch conv2d groups=C), 5.84 ms. Memory-bound: 1 GiB in + 0.95 GiB out; roofline ~2.2 ms. cuDNN is 2.7x off roofline here (depthwise is a weak cuDNN case).

- **iter1 identity:** 5.84 ms, 1.00x.
- **iter2 v1 shared-tiled 32x8 halo tile:** 3.69 ms, 1.5827x. Each block does a THxTW output tile for one (n,c); (TH+2)x(TW+2) halo in shared.
- **ncu (v1):** DRAM = 1.95 passes (1.0 GiB read + 0.95 GiB write) = byte floor already hit (L2 absorbs halo). But dram% 63, sm% 63 -> NOT bandwidth-saturated (529 GB/s). Lever = throughput, not bytes.
- **iter3 v2 full-width row-strip + float4 coalesced loads (TH8):** 2.69 ms, 2.1710x. One block = full-width strip of TH rows; contiguous float4 input loads -> 725 GB/s (75% peak).
- **iter4 v3 row-strip TH4 (higher occupancy):** 2.68 ms, **2.1791x (BEST, kept)**. Tied with v2 (<0.4%).
- **PTX finding:** none needed — pure memory-bound op, no tensor cores / no inline PTX. Plain float4 coalescing + shared-mem halo is the whole game (matches op1: PTX irrelevant for bandwidth-bound).
- **STOP:** byte floor hit (ncu 1.95 passes), v2/v3 within 0.4% (bandwidth-saturated stall). Final 2.18x over cuDNN. Detector-clean. 4 variants.
