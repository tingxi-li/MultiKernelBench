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
| 1 | identity baseline (Linear+gelu+softmax) | 1.0000x | 6.88 ms | baseline (ref) |
| 2 | v1 WMMA-tf32 gemm(x@Wt) + fused bias-gelu-softmax | 1.0472x | 6.57 ms | improved (BEST) |
| final | v1 WMMA-tf32 gemm(x@Wt) + fused bias-gelu-softmax | 1.6799x | 3.78 ms | final |

## Iterations

Fused op: `softmax(gelu(x@W^T + b), dim=1)`, x=[1024,8192], W=[8192,8192].
Reference = cuBLAS GEMM (~6.09 ms) + gelu + softmax = **6.88 ms**.

Design (2 kernels): (1) WMMA-tf32 GEMM Y=x@W^T — W (nn.Linear.weight, [N,K]) is
loaded transposed into shared with coalesced-over-k reads, then read as a
**col-major matrix_b** (ld=BK), so the same 128x128 / 4x4-frag kernel as op2 works
with the [out,in] weight layout. (2) one-block-per-row kernel stages the 8192-wide
row in shared, applies bias+gelu, then a max/exp/sum/normalize softmax (warp-shuffle
+ shared block reduce). W is zero-mean so Y is O(1) -> **a single fp32 accumulator
(NBANK=1) passes correctness** — no accuracy banks, unlike op2.

ncu (v1): gemm_wt 6.47 ms (l1tex 75%, tensor 34%, occ 15%); softmax 66 us (DRAM
56%). NBANK=1 (128 regs) vs op2's NBANK=2 (256 regs) makes the GEMM ~1.6x faster
(6.47 vs 10.4 ms) and near cuBLAS; the fused epilogue then wins the extra passes.

**Stop:** v1 = 1.0472x **beats the vendor reference** and is at the cell ceiling
(= its own best). GEMM confirmed L1-bound at the WMMA-C++ limit (4x4 frags is the
max before register spill; more warps just saturate L1). No inline PTX. Detector-clean.
