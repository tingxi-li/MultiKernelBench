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
| 1 | identity (torch F.scaled_dot_product_attention) | 1.0000x | 80.6 ms | baseline (vendor=math backend) |
| 2 | flash-attn fp32, BLOCK_D=1024, num_stages=2 | FAIL (smem OOM) | - | infeasible |
| 3 | flash fp32 num_stages=1, small blocks | FAIL (smem OOM) | - | infeasible |
| 4 | 3-kernel QK^T + softmax + PV, fp32 | 1.0113x | 79.7 ms | correct |
| 5 | 3-kernel all tf32 | FAIL (max_diff 4.1e-4) | - | incorrect |
| 6 | QK ieee + PV tf32 | FAIL (max_diff 4.1e-4) | - | incorrect (PV is the error source) |
| 7 | QK tf32 + PV ieee | 1.2653x | 63.7 ms | best (kept) |
| 8 | QK tf32 + PV tf32x3 | 1.1941x | 67.5 ms | correct but slower (revert) |
| final | QK tf32 + PV ieee (iter 7 best) | 1.0268x | 59.7 ms | final |

## Iterations

### Op summary — scaled_dot_product_attention (triton), COMPUTE-BOUND (attention)

- **Shape:** Q,K,V (32,32,512,1024), head_dim **D=1024**, tol 1e-4. Vendor = torch SDPA, but head_dim 1024 exceeds every flash/mem-efficient backend (cap 256) so torch falls back to the **fp32 math backend = 80.6 ms** (materializes S×S, no tensor cores). Huge apparent headroom.
- **Flash attempt (iters 2-3) FAILED — the key DSL-expressibility limit:** a fused flash kernel must stage the full-D Q and K/V tiles in shared memory ≈ `(BLOCK_M+BLOCK_N)*D*4` bytes. With D=1024 that caps `BLOCK_M+BLOCK_N ≤ 24` (99 KB smem), i.e. unusably tiny tensor-core tiles; larger tiles OOM. Triton can't express an efficient large-head-dim flash kernel here. Fell back to a 3-kernel materialized design (QK^T → row-softmax → PV) that tiles D as a normal GEMM contraction.
- **Iter 4 (fp32):** 79.7 ms / 1.01x — matches the torch math backend (both fp32 FMA, compute-bound, ~14 TFLOP/s).
- **Precision hunt (iters 5-7):** the 1e-4 gate is brutal for output magnitudes ~0.5. all-tf32 and QK-ieee+PV-tf32 both fail at max_diff ~4.1e-4 → the error source is the **PV dot** (rounding V~0.5 to tf32 gives ~2.4e-4 directly). QK, being softmaxed, tolerates tf32. **Iter 7 (QK tf32 + PV ieee) = 63.7 ms / 1.2653x, correct** — the fastest correct combo.
- **Iter 8:** PV tf32x3 (fp32-accurate tensor cores) = 67.5 ms / 1.19x — slower than fp32 FMA (3 passes), reverted.
- **Stop reason:** ~8-variant budget reached; the output-producing PV dot is pinned to fp32 by the 1e-4 gate (tf32 fails, tf32x3 slower), so it caps the speedup at 1.27x. Exceeds the (slow, fp32) vendor.
- **Detector:** clean. forward reshapes to (B*H,S,D) via `.reshape(-1,S,D)` (no BinOp), scale computed in-kernel (`1/tl.sqrt(D)`); all three dots/softmax in `@triton.jit`.

