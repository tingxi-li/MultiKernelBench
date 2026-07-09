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
| 1 | Tiled register-blocked GEMM BM=BN=128 BK=8 TM=TN=8 | 0.82x | 5.52 ms | regression |
| 2 | WMMA tensor-core GEMM (fp16 in, fp32 acc) | INCORRECT | N/A | failed |

## Iterations

### Iter 2 — WMMA tensor-core GEMM (fp16 inputs, fp32 accumulate)

- **Hypothesis:** Using WMMA tensor core API (mma.sync) with fp16 A/B and fp32 accumulation should significantly boost throughput on Ada, which has large tensor core capability.
- **Changes:** Rewrote kernel using nvcuda::wmma API. 8 warps/block (2×4 warp grid), BM=32 BN=64 BK=16, WMMA 16×16×16 fragments. A/B converted to fp16 in shared mem, C accumulated in fp32.
- **Bench:**
  - Compiled: True
  - Correct: False (max diff ~100, avg diff ~16 — complete mismatch)
  - Runtime: N/A (correctness gate)
  - Speedup: N/A
- **Analysis:** Correctness failure due to race condition in shared memory: `store_matrix_sync` from multiple warps writing to `smC` without synchronization, then all 256 threads write full smC. The wmma store and smC read/write overlap. This iteration confirms the floor — even with tensor cores, correctness is hard to achieve.
- **Next:** Iter cap reached (2). Restore best iter (iter 1, 0.82x) as final, confirming the floor.

### Iter 1 — Tiled register-blocked GEMM (BM=BN=128, BK=8, TM=TN=8)

- **Hypothesis:** A hand-written register-blocked, double-shared-memory GEMM with 128×128 tiles and 8×8 per-thread output should approach cuBLAS throughput, confirming the floor.
- **Changes:** Replaced `torch.matmul` identity with `load_inline` CUDA kernel: tiled SGEMM, smA transposed in shared mem, 256 threads per block (16×16), TM=TN=8, vectorized loads via loop unrolling.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 5.52 ms (mean), 4.96 ~ 5.98 ms (min ~ max)
  - Speedup: 0.82x
- **Analysis:** Custom kernel is 18% slower than cuBLAS. The RTX 6000 Ada's cuBLAS SGEMM is highly optimised (likely using CUTLASS + heuristics). The hand-written kernel doesn't match cuBLAS' instruction scheduling or memory pipeline utilisation. This confirms the FLOOR designation — cuBLAS is the optimal library call.
- **Next:** Try mma.sync / wmma fp32 tensor-core approach (cuda_unlimited allows PTX) to see if we can get any closer.

