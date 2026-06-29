# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Triton gather along dim=1 | 1.26x | 0.0205 ms | improved |

## Iterations

### Iter 1 — Triton gather along dim=1

- **Hypothesis:** torch.gather is launch/latency dominated at this small size (out 128x4096); a tight Triton kernel with autotuned block can shave overhead.
- **Changes:** Replaced the identity baseline with the optimized solution in `solution/gather.py`.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 0.0205 ms (mean); Reference: 0.0259 ms
  - Speedup: 1.26x (mean)
- **Analysis:** 1.26x and CORRECT. Each thread loads idx then does a dependent gathered load from x; output 2MB. Small-size win over torch.gather.
- **Next:** At roofline — stop.
