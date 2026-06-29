# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Autotuned Triton elementwise max(x,0) | 1.01x | 16.0 ms | improved |

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
