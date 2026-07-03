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

## Workspace directives (op: relu, dsl: cuda_noptx)
- **Target DSL: cuda_noptx.** Plain CUDA C++ via torch.utils.cpp_extension.load_inline. NO inline PTX `asm(...)`. Intrinsics (__expf/erff/float4/__ldg/__shfl) OK.
- This is a cross-DSL port of the already-optimized Triton kernel at
  `/home/lxt230026/MultiKernelBench/ako_runs/relu/triton/solution/relu.py` — use it as the correctness oracle
  and its `ako_runs/RESULTS.md` speedup as the perf bar / stop criterion.
- `forward()` must stay glue-only (allocate/reshape/launch); all compute in the
  kernel. Verify with `utils/cheating_detection.py` (bench.py does NOT run it).
- `ncu` unavailable -> proceed analytically from runtime stats.
- GPU **0** default in `scripts/bench.sh` (overridable via CUDA_VISIBLE_DEVICES).
- Reference: `/home/lxt230026/MultiKernelBench/reference/activation/relu.py`. Device memory 47GB.
- Tier: **floor**. Iteration cap: **2**. Elementwise max(x,0); HBM-bandwidth bound -> ~1x roofline.
- Fast signal: `--no-ref --num-perf-trials 20`; full verdict (real SPEEDUP): `--num-warmup 200` (in bench.sh).
