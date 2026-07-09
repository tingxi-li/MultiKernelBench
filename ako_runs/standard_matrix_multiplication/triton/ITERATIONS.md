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
| 2 | Triton autotune matmul (allow_tf32=False) | 0.98x | 4.63 ms | floor |

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

### Iter 2 — Triton autotune matmul (allow_tf32=False)

- **Hypothesis:** Using `acc += tl.dot(a, b, allow_tf32=False)` should produce IEEE float32 precision matching the cuBLAS SGEMM reference within 1e-4 tolerance, and the @triton.autotune kernel may perform close to cuBLAS.
- **Changes:** Changed `tl.dot(a, b, acc)` (3-arg) to `acc += tl.dot(a, b, allow_tf32=False)`, and cleaned up the code to write `acc` directly to output.
- **Bench:**
  - Compiled: True
  - Correct: True (5/5)
  - Runtime: 4.63 ms (mean), 4.46 ~ 4.86 ms (min ~ max)
  - Speedup: 0.98x (mean)
- **Analysis:** Correctness is now PASS. Speedup is 0.98x — essentially at parity with the cuBLAS reference (4.54 ms). This is the expected FLOOR result: torch.matmul already dispatches to cuBLAS SGEMM which is near-optimal, and a pure Triton kernel without tensor cores (which can't be used for fp32 precision matching) cannot beat it. The iteration cap of 2 is reached.
- **Next:** Iteration cap reached. Best result is iter-2 (0.98x, CORRECT). This is a confirmed FLOOR op. Restore iter-2 as final.

## Final

Best iter: 2 (0.98x mean speedup, 4.63 ms vs 4.54 ms ref on measurement run; final bench showed 0.93x on a noisier run).
FLOOR confirmed: torch.matmul dispatches to cuBLAS SGEMM (near-optimal). Triton fp32 matmul without TF32 cannot beat it due to the lack of tensor core acceleration for IEEE fp32.

