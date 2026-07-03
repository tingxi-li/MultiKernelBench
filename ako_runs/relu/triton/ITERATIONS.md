# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Autotuned Triton elementwise max(x,0) | 1.01x | 16.0 ms | improved |
| 2 | Grid-stride (occupancy/launch-count probe) | 0.99x | 16.9 ms | revert (floor) |

## Iterations

### Iter 1 — Autotuned Triton elementwise max(x,0)

- **Hypothesis:** Bandwidth-bound unary op; a tuned Triton elementwise kernel should hit the same HBM roofline as torch.
- **Changes:** Replaced the identity baseline with the optimized solution in `solution/relu.py`.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 16.0 ms (mean); Reference: 16.1 ms
  - Speedup: 1.01x (mean)
- **Analysis:** 1.01x == HBM roofline. Single read + single write of 6.4GB each; torch's eager relu is already at peak bandwidth, so matching it IS optimal. Physical floor reached.
- **Next:** At roofline — stop.

### Iter 2 — Grid-stride loop (occupancy / launch-count direction)

- **Hypothesis:** A distinct axis from iter-1's block-size autotune: give each program CHUNKS×BLOCK_SIZE of contiguous work (fewer, larger blocks) to test whether reducing launch/block count or improving L2 locality beats the one-shot kernel.
- **Change:** Temporary variant (`static_range(CHUNKS)` inner loop, autotuned over BLOCK_SIZE 4096–16384 × CHUNKS 4–8); tested in a scratch file — `solution/relu.py` left untouched.
- **Bench (fast-signal, --no-ref 20 trials, same clock state):**
  - Grid-stride: 16.9 ms (mean)
  - Baseline (one-shot autotuned): 16.7 ms (mean)
  - Correct: 5/5
- **Analysis:** ~1% slower. For a pure streaming op, fewer/larger blocks only reduce the in-flight parallelism that hides HBM latency; there is no data reuse for L2 to exploit (each element touched once). No win.
- **Roofline math:** 4096×393216 = 1.61e9 fp32 elems → 6.44 GB read + 6.44 GB write = 12.88 GB traffic. At 16.0 ms that is ~805 GB/s ≈ 84% of the RTX 6000 Ada's ~960 GB/s theoretical peak — the practical achievable ceiling for a copy-class kernel. torch eager relu sits at the same point.
- **Directions probed → floor confirmed:** (1) vectorization width / block-size + warps [iter 1, autotuned], (2) occupancy / launch-count [iter 2, grid-stride]. Both at the same ~800 GB/s wall. Memory traffic (1 read + 1 write/elem) is irreducible for elementwise relu.
- **Decision:** REVERT — keep the iter-1 autotuned baseline verbatim. AT FLOOR; stop (iteration cap reached).
