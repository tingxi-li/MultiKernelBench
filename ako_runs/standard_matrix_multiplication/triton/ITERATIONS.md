# Iteration Log

<!--
Per-iteration template (copy when adding a new iter entry under "## Iterations"):

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False
  - Runtime: ___ ms (mean), ___ ~ ___ ms (min ~ max)
  - Speedup: ___x (mean), ___ ~ ___x (min ~ max)
- **Analysis:** Why it worked or failed
- **Next:** What to try next

Append one row per iter to the Summary table below.
Status values: improved / no-change / regression / failed.
-->

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Triton autotune matmul (3-arg tl.dot) | INCORRECT | N/A | failed |

## Iterations

### Iter 1 — Triton autotune matmul (3-arg tl.dot)

- **Hypothesis:** A standard Triton matmul kernel with @triton.autotune over block sizes should match or beat cuBLAS for large fp32 matrices (M=2048, K=8192, N=4096).
- **Changes:** Replaced identity torch.matmul with a @triton.autotune Triton kernel using grouped tiling, tl.dot accumulation with 3-arg form `tl.dot(a, b, acc)`.
- **Bench:**
  - Compiled: True
  - Correct: False
  - Runtime: N/A (correctness failed)
  - Speedup: N/A
- **Analysis:** Large errors: max ~1.63, avg ~1.54. The 3-arg `tl.dot(a, b, acc)` form may use TF32 precision by default on Ada Lovelace GPUs, producing ~1e-1 error vs the float32 reference (cuBLAS SGEMM at 1e-4 tolerance). The 3-arg accumulator form is also not the canonical way to accumulate.
- **Next:** Switch to `acc += tl.dot(a, b, allow_tf32=False)` to disable TF32 and get IEEE float32 precision.

