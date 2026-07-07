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
| 1 | identity baseline (torch SDPA math backend) | 1.04x | 57.7 ms | baseline |
| 2 | 3-kernel fp16-TC attention (batched QK^T + softmax + PV) | 3.44x | 17.0 ms | improved / KEPT |

## Iterations

### Op summary (COMPUTE-BOUND attention, win tier)
- B32 H32 S512 d1024 fp32. head_dim=1024 > flash-256 -> reference runs the slow UNFUSED
  math backend (materialises 1GB scores) = 57.7 ms. fp16 tensor cores pass the 1e-4 tol
  because softmax normalises (maxabs 4.7e-5).
- **First tried a single fused flash kernel:** works & is accurate but only 1.55x (41.8
  ms). head_dim=1024 forces tiny tiles on Ada (99KB shared cap; the (bM,1024) O-accumulator
  blows the register file) -> QK^T/PV gemms run at ~26 TFLOP/s. Also, tiling the PV output-D
  into an Of fragment SLICE triggers a TileLang "layout infer conflict" -> can't grow tiles.
- **Winning design = 3 efficient large BATCHED kernels:** (1) scale*Q@K^T with fp32-accum
  fp16 TC -> scores(fp32, materialised, needed for tol: fp16 scores overflow the tol), (2)
  row-softmax over keys -> P(fp16), (3) P@V fp16 TC -> O. Gemms hit 71 (QK) / 85 (PV)
  TFLOP/s; softmax ~0.5 ms.
- **Result: 3.44x (17.0 vs 58.5 ms).** Biggest expressibility limit: at head_dim=1024
  TileLang cannot express an *efficient* fused flash kernel (shared cap + fragment-output-
  slice layout conflict), so materialising scores to global (the 3-kernel form) is the
  fastest expressible path.
- STOP: 3.44x, correct, detector-clean; the fused-flash ceiling is blocked by the tiling
  limits above. Ref (slow math backend) is the only external anchor and we are 3.4x past it.
