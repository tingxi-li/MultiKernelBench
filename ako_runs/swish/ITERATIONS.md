# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | FUSED single-pass x*sigmoid(x) | 2.49x | 15.9 ms | improved |

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
