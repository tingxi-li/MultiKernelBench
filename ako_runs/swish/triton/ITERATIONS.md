# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | FUSED single-pass x*sigmoid(x) | 2.49x | 15.9 ms | improved |
| 2 | Streaming eviction_policy=evict_first | ~2.46x | 16.5 ms | reverted (noise) |
| 3 | Expanded block/warp autotune space | ~2.46x | 16.3 ms | reverted (noise) |
| 4 | Grid-stride persistent + num_stages | <2.46x | 16.6 ms | reverted (worse) |
| 5 | No-mask divisible fast path | ~2.46x | 16.3 ms | reverted (noise) |
| 6 | 2-wide manual ILP unroll | ~2.46x | 16.3 ms | reverted (noise) |

**Verdict: at bandwidth floor.** Re-run baseline on GPU 1 (this pass): min 15.6ms, mean 16.0-16.5ms, std 0.44 (~3%). Six distinct directions all land at min 15.6-16.0ms — none clears the ~3% measurement noise. `solution/swish.py` restored byte-identical to the committed baseline.

## Iterations

### Iter 1 — FUSED single-pass x*sigmoid(x)

- **Hypothesis:** Eager x*torch.sigmoid(x) launches TWO kernels (sigmoid pass writes a 6.4GB temp, then mul reads x+temp writes out) ~32GB traffic. A fused Triton kernel does one read + one write = 12.8GB.
- **Changes:** Replaced the identity baseline with the optimized solution in `solution/swish.py`.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 15.9 ms (mean); Reference: 39.6 ms
  - Speedup: 2.49x (mean)
- **Analysis:** 2.49x and CORRECT, no reward-hack flag. REF 39.6ms -> 15.9ms exactly matches the 2-pass->1-pass traffic reduction. At the fused-op roofline (same 15.9ms as a single elementwise pass). HEADLINE WIN.
- **Next:** At roofline — stop.

### Iter 2 — Streaming eviction_policy=evict_first

- **Hypothesis:** Read-once/write-once streaming data should not occupy L2; `eviction_policy="evict_first"` on load+store frees cache lines sooner and may raise effective HBM BW.
- **Change:** Added `eviction_policy="evict_first"` to `tl.load` and `tl.store`.
- **Bench (fast-signal, --num-warmup 200, 50 trials):** min 15.8, mean 16.5, std 0.451. CORRECT.
- **KEEP/REVERT:** REVERT — min 15.8 vs baseline 15.6-15.9, within the ~3% noise band. No effect (kernel is BW-bound, not L2-capacity-bound).
- **Next:** Sweep block/warp configs.

### Iter 3 — Expanded block/warp autotune space

- **Hypothesis:** A better BLOCK_SIZE/num_warps operating point (1024..65536, num_warps up to 32) might raise bandwidth utilization above the current tuner's pick.
- **Change:** Grew autotune configs to 11 entries spanning BLOCK_SIZE 1024-65536 and num_warps 4-32.
- **Bench:** min 15.6, mean 16.3, std 0.551. CORRECT.
- **Interleaved A/B (baseline vs candidate, 3 rounds back-to-back):** both = min 15.6-15.7. IDENTICAL.
- **KEEP/REVERT:** REVERT — no improvement; the baseline's 4 configs already sit at the same operating point. This is the strongest floor evidence: two different config sets converge to min 15.6ms.
- **Next:** Try software pipelining via a loop (num_stages only pipelines loops).

### Iter 4 — Grid-stride persistent kernel + num_stages

- **Hypothesis:** A persistent kernel (grid = SM*32) with a strided tile loop lets `num_stages=2-4` software-pipeline loads/stores across iterations, hiding memory latency better.
- **Change:** Rewrote kernel as a grid-stride loop over `num_tiles`, fixed grid to `SM_count*32`, autotuned num_stages 2-4.
- **Bench:** min 16.0, mean 16.6, std 0.443. CORRECT.
- **KEEP/REVERT:** REVERT — slightly WORSE. Persistent grid lowers concurrent-block occupancy; on a BW-saturated kernel the extra MLP from many resident one-shot programs already hides latency, so pipelining a smaller resident set only hurts.
- **Next:** Remove mask overhead.

### Iter 5 — No-mask divisible fast path

- **Hypothesis:** n = 4096*393216 = 3·2^29 is divisible by every power-of-two BLOCK_SIZE, so the boundary mask is dead work; dropping it removes predication overhead.
- **Change:** Removed `mask=` from load/store (safe only because n is divisible).
- **Bench:** min 15.6, mean 16.3, std 0.528. CORRECT.
- **KEEP/REVERT:** REVERT — min 15.6 = baseline; mask predication is free on a BW-bound kernel, and dropping it sacrifices generality (OOB on non-divisible shapes). Keep the robust masked baseline.
- **Next:** Increase per-thread MLP via manual unroll.

### Iter 6 — 2-wide manual ILP unroll

- **Hypothesis:** Issuing two independent loads per program before compute raises in-flight memory requests per thread (memory-level parallelism), potentially closing the gap to peak BW.
- **Change:** Each program processes two BLOCK_SIZE tiles (offs0/offs1), two loads then two stores; grid halved.
- **Bench:** min 15.7, mean 16.3, std 0.515. CORRECT.
- **KEEP/REVERT:** REVERT — min 15.7 = baseline. Triton already schedules enough concurrent programs to saturate MLP; manual unroll adds nothing.
- **Next:** None — floor confirmed.

### FLOOR CONFIRMATION

- **Roofline:** out-of-place elementwise has a hard traffic floor of read-once + write-once = 2·6.4GB = 12.8GB. min 15.6ms => 12.8GB/15.6ms = **821 GB/s**.
- **Independent cross-check:** eager reference moves ~32GB in 39.4ms = **812 GB/s**. Two unrelated kernels parked at ~815 GB/s (~85% of the 960 GB/s GDDR6 theoretical, the normal practical ceiling on Ada) is decisive evidence this is the physical floor, not a tuning gap.
- **Result:** kept the committed baseline (2.46x on GPU 1 this pass). `at_floor=true, improved=false`. Restored byte-identical.
