# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Autotuned Triton elementwise sigmoid | 1.01x | 15.9 ms | improved |
| 2 | Streaming eviction_policy hint (probe) | — | 16.6 ms | reverted (no gain) |
| 3 | Grid-stride + num_stages pipelining (probe) | — | 15.8 ms min | reverted (tie) |

## Iterations

### Iter 1 — Autotuned Triton elementwise sigmoid

- **Hypothesis:** Bandwidth-bound unary op; tuned Triton matches torch at the HBM roofline.
- **Changes:** Replaced the identity baseline with the optimized solution in `solution/sigmoid.py`.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 15.9 ms (mean); Reference: 16.1 ms
  - Speedup: 1.01x (mean)
- **Analysis:** 1.01x == roofline. Memory-bound (read+write 6.4GB). Floor reached.
- **Next:** At roofline — stop.

### Iter 2 — Streaming `eviction_policy='evict_first'` hint (probe)

- **Hypothesis (distinct lever: cache/streaming hints):** Tag load+store as streaming so the
  hardware does not retain x/y in L2, freeing L2 for higher effective bandwidth.
- **Change:** Added `eviction_policy='evict_first'` to `tl.load` and `tl.store`.
- **Bench (GPU 2, fast-signal, --num-warmup 200, 2 reruns):** min 15.9 / 15.9 ms; mean 16.6-16.7 ms. Correct 5/5.
- **Result:** No gain vs baseline (min 15.7 ms). As expected: 12.9 GB streams through a 96 MB L2
  with zero reuse, so cache hints have nothing to bite on. **REVERTED.**
- **Next:** Try memory-level-parallelism lever.

### Iter 3 — Grid-stride loop + `num_stages` software pipelining (probe)

- **Hypothesis (distinct lever: MLP / launch count):** Have each program process ITERS tiles in a
  `tl.range(..., num_stages=N)` loop so loads of the next tile overlap stores of the current one,
  raising memory-level parallelism and cutting total block count.
- **Change:** Rewrote kernel as a pipelined ITERS-tile grid-stride loop; autotuned BLOCK_SIZE/ITERS/
  num_warps/num_stages.
- **Bench (GPU 2, fast-signal, --num-warmup 200, 2 reruns):** min 15.8 / 15.8 ms (tie with baseline
  15.7 ms); mean worse (19.7 ms, high variance from recompute/outliers). Correct 5/5.
- **Result:** No repeatable margin — within noise at best. Autotune already sweeps num_warps to 16 on
  large blocks, so MLP is already saturated; extra loop machinery only adds overhead. **REVERTED.**
- **Next:** Two distinct levers exhausted — stop.

## Floor Conclusion

Committed baseline (Iter 1) kept verbatim; solution/ is git-diff-clean.

- Traffic = 1.61e9 fp32 elements x 4 B x 2 (read+write) = **12.88 GB**.
- Baseline 15.9 ms mean (min 15.7) → **~810 GB/s effective**; torch 16.1 ms → **~800 GB/s**. Both sit
  at **~84% of the 960 GB/s HBM peak** — the realistic ceiling for a pure streaming kernel (real
  elementwise tops out ~85-90%, never 100%).
- **Parity with the mature vendor path (torch) at the same effective bandwidth is the floor proof.**
  Two genuinely distinct optimization levers (streaming eviction hints; grid-stride + pipelined MLP)
  both landed inside measurement noise (sub-1%, < the 15.9↔16.5 ms run-to-run jitter on identical code).
- **AT FLOOR. improved=false.** Keeping the known-good committed baseline (1.01x, correct) is the
  honest result; no noise-manufactured "win".
