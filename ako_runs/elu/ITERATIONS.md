# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Autotuned Triton ELU (alpha from init) | 1.01x | 15.9 ms | improved |

## Iterations

### Iter 1 — Autotuned Triton ELU (alpha from init)

- **Hypothesis:** Bandwidth-bound unary op; exp only on the negative branch (untaken for rand>=0 inputs) but correct by construction.
- **Changes:** Replaced the identity baseline with the optimized solution in `solution/elu.py`.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 15.9 ms (mean); Reference: 16.0 ms
  - Speedup: 1.01x (mean)
- **Analysis:** 1.01x == roofline. Memory-bound. Floor reached.
- **Next:** At roofline — stop.
