# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Triton scatter (semantically faithful) — UNWINNABLE under harness | N/A (CORRECT=False) | - ms | blocked |

## Iterations

### Iter 1 — Triton scatter (semantically faithful) — UNWINNABLE under harness

- **Hypothesis:** scatter-overwrite with random duplicate indices (~1024 collisions/row for idx in [0,8192) over 4096 cols) is ORDER-NONDETERMINISTIC in torch.
- **Changes:** Replaced the identity baseline with the optimized solution in `solution/scatter.py`.
- **Bench:**
  - Compiled: True
  - Correct: False
  - Runtime: - ms (mean); Reference: 0.0268 ms
  - Speedup: N/A (CORRECT=False) (mean)
- **Analysis:** CORRECT=False is unavoidable: even an identity copy (ModelNew=torch.scatter) fails the bench 3/3 times because the reference disagrees with itself across trials. No kernel can match torch's racy duplicate-index result to 1e-4. A deterministic kernel also wouldn't match torch's nondeterministic result. Documented as a harness/op incompatibility, not a kernel bug.
- **Next:** Op is unwinnable under this harness (torch nondeterminism); stop.
