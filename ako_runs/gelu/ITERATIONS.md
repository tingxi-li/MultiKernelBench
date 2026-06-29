# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Autotuned Triton exact GELU via erf | 1.01x | 15.8 ms | improved |

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
