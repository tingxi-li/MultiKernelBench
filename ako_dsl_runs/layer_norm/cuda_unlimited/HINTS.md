# Hints

<!-- User-supplied behavior directives. The skill reads this at session start
     and respects any constraint named here. Examples:
     - Optimization constraints or focus areas (e.g., "Prefer Triton over raw CUDA")
     - Strategies to try or avoid     (e.g., "Avoid shared memory")
     - Agent behavior controls         (e.g., "Stop after 5 iterations")
     - Dependency policies             (e.g., "Do not install new packages")
     - Environment constraints         (e.g., "ncu is unavailable on this host",
                                             "8GB device memory limit")

     The skill's own protocol (iteration steps, stall handling, ncu fallback,
     stopping rules) lives in SKILL.md — do not duplicate it here.
-->

## Workspace directives (op: layer_norm, dsl: cuda_unlimited)
- **Target DSL: cuda_unlimited.** CUDA via load_inline, EVERYTHING allowed: inline PTX `asm volatile`, cp.async, vectorized ld/st, warp intrinsics, cache hints. No cuBLAS/cuDNN library offload.
- This is a cross-DSL port of the already-optimized Triton kernel at
  `/home/lxt230026/MultiKernelBench/ako_runs/layer_norm/solution/layer_norm.py` — use it as the correctness oracle
  and its `ako_runs/RESULTS.md` speedup as the perf bar / stop criterion.
- `forward()` must stay glue-only (allocate/reshape/launch); all compute in the
  kernel. Verify with `utils/cheating_detection.py` (bench.py does NOT run it).
- `ncu` unavailable -> proceed analytically from runtime stats.
- GPU **3** default in `scripts/bench.sh` (overridable via CUDA_VISIBLE_DEVICES).
- Reference: `/home/lxt230026/MultiKernelBench/reference/normalization/layer_norm.py`. Device memory 47GB.
- Tier: **win**. Iteration cap: **6**. LayerNorm last 3 dims, affine. Fused reduction.
- Fast signal: `--no-ref --num-perf-trials 20`; full verdict (real SPEEDUP): `--num-warmup 200` (in bench.sh).
