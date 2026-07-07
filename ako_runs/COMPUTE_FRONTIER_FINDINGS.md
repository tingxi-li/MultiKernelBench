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
| **standard_matmul** | 0.79 | **0.59** | 1.11 | **4.13** | **wide** — tensor-core |
| **matmul_gelu_softmax** | 2.36 | 1.05 | 1.24 | **5.04** | **wide** — tensor-core + fusion |
| **scaled_dot_product_attention** | 1.27 | 1.73 | 1.71 | **3.41** | **wide** — tensor-core (flash) |

(speedup vs the same PyTorch golden; conv row re-benched all-four-on-GPU3 against one
cuDNN reference — see caveat C4. tilelang standard_matmul/matmul_gelu_softmax reflect the
logged convergence re-run, which improved on the first run's 3.86/4.71 — its in-block
split-K flush beat the earlier grid-z atomic; independent 2× re-bench 4.0–4.25 / 4.9–5.15.)

## The answer: two regimes

**Memory-bound / low-arithmetic-intensity ops (sum_reduction, conv_depthwise): dead ties.**
Every DSL reaches the roofline (reduction: 1-read HBM floor at ~92–97% peak; conv: all four
solutions ~2.6 ms, a modest ~1.4–1.5× over cuDNN's memory-suboptimal depthwise). No
capability ceiling — this **extends the 12-op finding** to two more op families. On these,
inline PTX again bought ≤0.6% (reconfirmed on the unlimited lane): still a red herring.

**Tensor-core ops (matmul, matmul_gelu_softmax, sdpa): a real, wide capability ceiling —
and the compiler DSL (tilelang) owns it.** The GEMM ordering 0.59 → 0.79 → 1.11 → 4.13 is
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
   - **tilelang:** fp16 `T.gemm` + in-block split-K flush → **4.13×** at ~120 TFLOP/s.
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
- **C5 — RESOLVED.** The first tilelang lane autotuned *off-wrapper* (2 logged benches/op),
  so a logged re-run from identity was done with every config through the wrapper (34 logged
  variants, ~1157 s). It re-reached the two ties and *improved* matmul (3.86→4.13) and the
  fused op (4.71→5.04) via an in-block split-K flush; sdpa's fresh search landed ~6% short
  (3.12 vs 3.44), so the committed 3.44 kernel is kept as the shipped sdpa while its
  convergence curve is the logged run's (to 3.12). See the Convergence section.

## Convergence — all four lanes wrapper-logged (compute_s = compile+bench, DSL-attributable)
Every benched config went through `timed_bench.sh` (the tilelang re-run closes C5). Per-op
cumulative `compute_s` to each cell's ceiling / logged-variant count:

| op | triton | cuda_noptx | cuda_unlimited | tilelang |
|---|---|---|---|---|
| sum_reduction | 193s / 3 | 349s / 4 | 354s / 4 | 325s / 5 |
| standard_matmul | 55s / 5 | 296s / 8 | 407s / 11 | 85s / 10 |
| matmul_gelu_softmax | 58s / 5 | 48s / 2 | 48s / 2 | 41s / 5 |
| conv_depthwise | 60s / 5 | 150s / 4 | 153s / 4 | 65s / 5 |
| sdpa | 779s / 8 | 212s / 2 | 210s / 2 | 641s / 9 |
| **total** | **~1145s / 26** | **~1055s / 20** | **~1172s / 23** | **~1157s / 34** |

Reading it:
- **Per-variant cost is the DSL convergence story.** Compiler DSLs (triton, tilelang) recompile
  a config in seconds (JIT); the nvcc lanes (noptx, unlimited) pay ~30–50 s/variant. On
  `standard_matmul`, tilelang explored 10 configs in **85 s** and reached **4.13×**, while
  unlimited spent **407 s** over 11 configs to reach only **1.11×** — the compiler DSL converged
  to a *far higher* ceiling in ~5× less compute. This is the practical convergence advantage the
  12-op study hypothesized, now measured with all lanes logged.
- **compute_s is reference-dominated on slow-ref ops.** sdpa's 641–779 s totals are the ~60–80 ms
  torch reference timed every bench, not DSL search — compare variant counts there, not seconds.
- **Still a *practical/native-workflow* read, not a controlled one (C1–C3):** variant counts,
  stopping points, and discovery differ per lane. But it is now comparable in *kind* — every
  config is wrapper-logged. (tilelang sdpa's curve is the logged re-run to 3.12×; the shipped
  sdpa kernel is the committed 3.44× from a luckier prior run.)
