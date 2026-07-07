# Compute-bound frontier — the cross-DSL capability ceiling

The 12-op study (`CROSS_DSL_FINDINGS.md`) covered only memory-bound / index /
elementwise / cuDNN ops and found **no capability ceiling** — every DSL reached the
roofline, PTX was a red herring, and the residual gaps were ergonomic. It flagged the
**compute-bound tensor-core ops as the open frontier** where a real ceiling could appear.

This is that frontier. Five ops × 4 DSLs (triton, cuda_noptx = plain CUDA no PTX,
cuda_unlimited = CUDA + inline PTX, tilelang), branch `cross-dsl-6op-ncu-redo`, RTX 6000
Ada, each DSL using its native method from an identity baseline, **no lever hints** (unlike
the layer_norm calibration). Every number below was **independently re-benched** (≥2×,
median-ref), is **CORRECT** (harness fp32 1e-4 oracle, 5/5 seeds), passes the cheating
detector, and no cuda_noptx solution contains inline PTX.

## Verified results

| op | triton | cuda_noptx | cuda_unlimited | tilelang | regime |
|---|---|---|---|---|---|
| sum_reduction | 1.01 | 1.01 | 1.01 | 1.01 | **tie** — HBM 1-read roofline |
| conv_depthwise_2d | 1.41 | 1.46 | 1.46 | **1.53** | **tie** — memory-bound tiling |
| **standard_matmul** | 0.79 | **0.59** | 1.11 | **3.86** | **wide** — tensor-core |
| **matmul_gelu_softmax** | 2.36 | 1.05 | 1.24 | **4.71** | **wide** — tensor-core + fusion |
| **scaled_dot_product_attention** | 1.27 | 1.73 | 1.71 | **3.41** | **wide** — tensor-core (flash) |

(speedup vs the same PyTorch golden; conv row re-benched all-four-on-GPU3 against one
cuDNN reference — see caveat C4.)

## The answer: two regimes

**Memory-bound / low-arithmetic-intensity ops (sum_reduction, conv_depthwise): dead ties.**
Every DSL reaches the roofline (reduction: 1-read HBM floor at ~92–97% peak; conv: all four
solutions ~2.6 ms, a modest ~1.4–1.5× over cuDNN's memory-suboptimal depthwise). No
capability ceiling — this **extends the 12-op finding** to two more op families. On these,
inline PTX again bought ≤0.6% (reconfirmed on the unlimited lane): still a red herring.

**Tensor-core ops (matmul, matmul_gelu_softmax, sdpa): a real, wide capability ceiling —
and the compiler DSL (tilelang) owns it.** The GEMM ordering 0.59 → 0.79 → 1.11 → 3.86 is
set by two independent factors:

1. **Precision-managed tensor cores under the 1e-4 gate.** `torch.matmul` at fp32 runs
   cuBLAS on CUDA cores (`allow_tf32=False`, ~30 TFLOP/s), *not* tensor cores. Tensor cores
   are the obvious lever, but the 1e-4 gate blocks naive use: over the K=8192 contraction,
   tf32/fp16 accumulation-depth error (n·ε·S) exceeds tolerance. The fix is **split-K into
   fp32 accumulators** (+ round-to-nearest tf32 bits). Who can express that ranks the DSLs:
   - **cuda_noptx (WMMA C++): structurally can't.** `load_matrix_sync` truncates and gives
     no accumulation control → every tf32/3×tf32 attempt failed the gate → stuck at fp32
     WMMA, L1/shared-bandwidth bound (tensor pipe ~20% busy) → **0.59×**.
   - **triton:** tf32 `tl.dot` failed the gate; never found fp16+split-K → fp32 **0.79×**.
   - **cuda_unlimited:** raw `mma.sync` PTX gives exact tf32-bit + split-K control → clears
     the gate, **1.11×** (beats cuBLAS-fp32) but only ~25 TFLOP/s (44% of tf32 peak).
   - **tilelang:** fp16 `T.gemm` + split-K → **3.86×** at ~118 TFLOP/s.
2. **Pipelining for free vs by hand.** tilelang's compiler auto-emits the cp.async / ldmatrix
   software pipeline that feeds the tensor cores; it also chose fp16 over tf32. That is why a
   **compiler DSL beat the hand-PTX lane ~3×** — the unlimited lane explicitly ran out of
   budget before hand-building that pipeline (its `mma.sync` kernel is un-pipelined).

### PTX's role splits by op class (refines the 12-op "PTX-null" thesis)
- **Memory-bound:** red herring (≤0.6%). Unchanged.
- **Tensor-core GEMM:** **decisive for noptx→parity** (WMMA C++ can't clear the gate, raw
  `mma.sync` can) — but **not sufficient for the frontier**: tilelang reached ~3× the
  hand-PTX result with **zero PTX**, via its compiler's fp16 + auto-pipelining. So the
  frontier belongs to *compiler-emitted, precision-managed, pipelined tensor cores*, not to
  hand-PTX per se.

### Where each DSL beat the vendor, and why (not always a GEMM win)
- **matmul_gelu_softmax / sdpa** wins are partly **weak-vendor**: `torch` runs the fused op
  eagerly (extra launches/HBM roundtrips), and at head_dim=1024 `torch` SDPA exceeds flash's
  head-dim cap and falls back to a slow fp32 math path (80.6 ms) — so even modest custom
  fusion beats it. tilelang's large multipliers there still ride its fp16-TC GEMM.
- **conv_depthwise**: cuDNN's generic depthwise runs well above the memory roofline; a
  coalesced tiled kernel that reads the input once (L2 absorbs the 3×3 halo) beats it —
  equally in every DSL.

## Caveats (stated, not hidden)
- **C1 — precision discovery is uneven.** triton and noptx never found fp16+split-K; had
  they, their matmul numbers would likely rise. Part of the spread is search luck, not pure
  capability. (noptx's WMMA-C++ ceiling is a genuine structural limit, though.)
- **C2 — effort is uneven.** cuda_unlimited didn't have budget to hand-build the cp.async
  pipeline; its 1.11× is a floor for the hand-PTX path, not its ceiling.
- **C3 — fp16/tf32 asymmetry.** On the fused op, triton went tf32 (2.36×) while noptx stayed
  fp32 WMMA (1.05×); same correctness bar, different precision exploited. The comparison
  measures "which DSL best exploits reduced precision within the tolerance," which is a real
  and legitimate axis but is not pure algorithmic capability.
- **C4 — conv reference is unstable.** `conv_depthwise` is effectively memory-bound (low AI),
  so it was mis-classified as compute-bound and benched per-GPU; the cuDNN ref rode the
  GPU-0/1/2 slow-clock state (5.84 ms) vs GPU3 (3.9 ms), inflating the first triton/noptx/
  unlimited readings to ~2.1–2.2×. The table row above is the corrected all-on-GPU3 number
  (~1.4–1.5×, a tie). Even GPU3's cuDNN ref may be mildly elevated; the robust statement is
  "all four land ~2.6 ms, a modest tie over cuDNN."
- **C5 — convergence-time for tilelang is not yet comparable.** The tilelang lane autotuned
  *off-wrapper* (only 2 logged benches/op), so its `compute_s` understates its true search
  cost. Its *ceilings* above are verified-solid; its *convergence rate* is being re-measured
  with every config logged (see the Convergence section, being finalized).

## Convergence (compute_s → within 5% of each cell's ceiling)
Wrapper-logged lanes (comparable): per-op `cum_compute_s` and variant counts are in each
`<op>/<dsl>/convergence.csv`. triton logged 26 variants across the 5 ops, cuda_noptx 20,
cuda_unlimited 23. **tilelang is being re-run with full logging** to make its curve
comparable; this section is finalized once that completes.
