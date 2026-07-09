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
| 1 | Triton autotune matmul | INCORRECT | N/A | failed |
| 2 | Triton matmul fp16 path fix | TBD | TBD | TBD |

## Iterations

### Iter 1 — Triton autotune matmul

- **Hypothesis:** A standard Triton matmul kernel with autotune over block sizes should match or beat cuBLAS on large matrices (M=2048, K=8192, N=4096).
- **Changes:** Replaced identity (torch.matmul) with a full @triton.autotune matmul kernel using grouped tiling and tl.dot accumulation. Used `tl.dot(a, b, accumulator)` 3-arg form.
- **Bench:**
  - Compiled: True
  - Correct: False
  - Runtime: N/A (correctness failed)
  - Speedup: N/A
- **Analysis:** The kernel compiled but produced wrong results. Max difference ~1.63, avg ~1.54 — these are very large errors (not floating point noise). The 3-arg form of `tl.dot(a, b, accumulator)` is not properly accumulating. The issue is that `tl.dot` in recent Triton requires 2-arg form and we must do `accumulator += tl.dot(a, b)`. Also the input dtype is float32 and we should cast inputs to float16/bf16 for tl.dot if needed, or keep float32 accumulation correctly.
- **Next:** Fix the accumulation: use `accumulator += tl.dot(a, b)` form. Also ensure correct handling of float32 inputs for tl.dot (cast to tf32 or keep fp32).

