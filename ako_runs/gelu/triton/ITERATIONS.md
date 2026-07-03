# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Autotuned Triton exact GELU via erf | 1.00x | 16.0 ms | baseline (== torch) |
| 2 | Grid-stride loop (ITERS/program) | — | 16.7 ms | REVERT (no gain) |
| 3 | No-mask (n divisible by BLOCK_SIZE) | — | 16.6 ms | REVERT (no gain) |

## Iterations

### Iter 1 — Autotuned Triton exact GELU via erf

- **Hypothesis:** Bandwidth-bound; must use erf (tanh approx would exceed 1e-4). tl.math.erf matches torch to 5e-7.
- **Changes:** Replaced the identity baseline with the optimized solution in `solution/gelu.py`.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 15.8 ms (mean); Reference: 16.0 ms
  - Speedup: 1.01x (mean)
- **Analysis:** 1.01x == roofline. erf is cheap vs the 12.8GB memory traffic; memory-bound. Floor reached.
- **Next:** At roofline — stop.

### Iter 2 — Grid-stride loop (fewer program instances, ITERS tiles each)

- **Hypothesis:** If launch/scheduling overhead were non-negligible, having each program process ITERS consecutive tiles (fewer, longer-lived programs) would cut it.
- **Change:** Added `ITERS` constexpr + `tl.static_range` loop; grid = cdiv(n, BLOCK_SIZE*ITERS). Autotuned {BLOCK 4096–16384}×{ITERS 4–8}.
- **Bench (GPU 2, --num-warmup 200, 30 trials, fast-signal):** mean 16.7 ms / min 15.8 ms, correct 5/5.
- **Verdict:** REVERT — not faster than baseline (16.0 mean / 15.7 min). Launch overhead is not the bottleneck; a bandwidth-bound elementwise kernel does not benefit from fewer programs.

### Iter 3 — No-mask variant (n = 3·2^29 is divisible by every BLOCK_SIZE)

- **Hypothesis:** Total elements 1,610,612,736 = 3·2^29 divides evenly by all autotuned block sizes (2048–16384), so the boundary mask never fires. Dropping `mask=` removes the compare + predicated load/store; if predication cost mattered this would show.
- **Change:** Removed `mask` from load/store (unconditional, safe here because no tail).
- **Bench (GPU 2, --num-warmup 200, 30 trials, fast-signal):** mean 16.6 ms / min 15.8 ms, correct 5/5.
- **Verdict:** REVERT — indistinguishable from baseline. Mask/predication cost is hidden under memory latency, as expected for a memory-bound op. (Also unsafe to keep generally: masks off only because this exact shape has no tail.)

## Floor Conclusion — AT HBM COPY CEILING

- **Traffic:** 4096×393216 = 1.61e9 elems × 4 B × 2 (read x + write y) = **12.88 GB**.
- **Measured:** baseline 16.0 ms on GPU 2 → **12.88 GB / 16.0 ms ≈ 805 GB/s ≈ 84% of ~960 GB/s peak** (RTX 6000 Ada). This is the practical copy-bandwidth ceiling; ~13.4 ms would be 100% peak, unreachable in practice.
- **Solution == torch:** ref_runtime and solution runtime both 16.0 ms → torch's own `F.gelu` is at the same ceiling. No headroom to beat it.
- **Coverage of distinct directions:** (a) block-size/warps/occupancy — swept by the autotuner; (b) launch count / work-per-program — Iter 2 grid-stride, no gain; (c) predication/index overhead — Iter 3 no-mask, no gain; (d) precision — must use exact erf (tanh approx fails 1e-4), and erf is compute-cheap vs the 12.9 GB traffic anyway. All roads lead to the same runtime.
- **Decision:** Keep the committed baseline verbatim (git-diff-clean). at_floor=true, improved=false. Stopping — further iterations would be block-size padding, which the depth policy forbids.
