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

## Iterations

## Iterations (cuda_unlimited convergence run)

Reference = torch F.scaled_dot_product_attention, fp32 (uses the slow math/efficient fp32 backend): 80.6 ms. Q,K,V (32,32,512,1024). Output ~0.5 magnitude, atol 1e-4 -> softmax makes the gate relatively loose.

- **iter1 identity:** 80.6 ms, 1.00x. (fp32 SDPA is ~13 TFLOP/s effective — two fp32 cuBLAS matmuls + 1 GiB score materialization + softmax.)
- **iter2 v1 3-kernel tensor-core attention:** 47.8 ms, **1.6862x (BEST, kept)**, CORRECT.
  - K1 gemm_qkt: batched S = scale*(Q @ K^T) via inline mma.sync TF32 (round-nearest); K read in its natural (key,d) layout so the b-fragment forms K^T with no transpose. scale=1/sqrt(D) folded into the epilogue.
  - K2 softmax_rows: numerically-stable row softmax over the 512 keys (one block/row).
  - K3 gemm_av: batched O = A @ V via the standard mma.sync TF32 GEMM.
  - tf32 (no split-K) PASSES the 1e-4 gate here: D=1024 accumulation is shallow and softmax + ~0.5 output magnitude absorb tf32 error.
- **PTX vs constrained:** the two matmuls ride the same mma.sync TF32 path that beat cuBLAS in op2; WMMA-C++ tf32 would fail the gate as in op2. Attention-specific: the K^T contraction is handled by re-indexing K's smem tile in the b-fragment (no explicit transpose).
- **Residual:** matmuls run at ~23 TFLOP/s (my mma ceiling), vs ~57 TFLOP/s cuBLAS-tf32; a flash-style fused kernel + ldmatrix pipelining could reach ~3-4x but is a large additional build. External ceiling (torch SDPA) beaten -> STOP. Detector-clean. 2 variants.
