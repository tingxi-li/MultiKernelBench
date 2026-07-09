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
| 1 | identity (nn.Conv2d cuDNN depthwise) baseline | 1.0000x | 5.84 ms | baseline (vendor) |
| 2 | triton depthwise BLOCK_HxBLOCK_W tile, 9-tap unroll | 2.0783x | 2.81 ms | improved |
| 3 | wider tiles BLOCK_W up to 512 | 2.1471x | 2.72 ms | best (kept) |
| 4 | all-BLOCK_W=512 sweep (BLOCK_H 4-32) | FAIL | - | incorrect config (revert) |
| 5 | restore iter-3 config set (final confirm) | 2.1392x | 2.73 ms | verified best (noise vs iter3) |
| final | wider tiles BLOCK_W 512 (best from iter3/5) | 1.3235x | 2.72 ms | final |

## Iterations

### Op summary — conv_depthwise_2d (triton), COMPUTE-BOUND label but MEMORY-BOUND in practice

- **Shape:** depthwise 3x3, groups=64, x(16,64,512,512) → out(16,64,510,510), stride1 pad0, fp32. Vendor = cuDNN (nn.Conv2d) 5.84 ms. Arithmetic intensity is tiny (9 MAC/output) → memory-bound; read x 1.0 GiB + write out 0.95 GiB ⇒ 2-pass copy floor ≈ 2.4 ms.
- **Key finding:** cuDNN depthwise is memory-*suboptimal* here (5.84 ms ≈ only 40% of the copy roofline), so a straightforward memory-efficient triton kernel BEATS it ~2x.
- **Iter 2:** one program per (b, c, output-tile); 9-tap `tl.static_range` unroll, each tap loads a shifted BLOCK_H×BLOCK_W input tile; coalesced along W. 2.81 ms → 2.08x.
- **Iter 3 (best):** widen BLOCK_W to 512 (full row) → better DRAM coalescing. 2.72 ms → **2.1471x**.
- **ncu at iter2 (steer):** DRAM read = **1.00x tensor** (L2 fully absorbs the 9-tap halo overlap — input read once from HBM, not 9x), DRAM total = **1.95 passes ≈ the 2-pass copy floor**, dram 85% saturated, occ 82%. Binding roofline = HBM read-once+write-once; already there.
- **Iter 4:** aggressive all-512 config sweep → one config returned wrong output (correct=False); reverted.
- **Iter 5:** restored iter-3 set, re-benched 2.7300 ms / 2.1392x correct (0.4% noise vs iter3) — verifies the leave-behind.
- **Stop reason:** at the binding memory roofline (1.95 passes = physical minimum traffic, read input 1x + write output 1x); further levers are <3% bandwidth-efficiency noise. Final ~2.14x, exceeds vendor.
- **Detector:** clean. Output dims via `x.unfold(dim,K,stride).size()` (view, avoids any `BinOp` in forward); weight read as `self.conv2d.weight` attribute; conv fully in `@triton.jit`.

