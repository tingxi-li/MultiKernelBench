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

    ## Workspace directives (op: sum_reduction_over_a_dimension, dsl: cuda_unlimited)
    - **Target DSL: cuda_unlimited.** CUDA via load_inline, EVERYTHING allowed: inline PTX `asm volatile`, cp.async, vectorized ld/st, warp intrinsics, cache hints. No cuBLAS/cuDNN library offload.
    - **Correctness oracle:** the PyTorch reference `/home/lxt230026/MultiKernelBench/reference/reduce/sum_reduction_over_a_dimension.py` is the SOLE oracle — this op has no
      prior DSL solution. bench.py renames your `Model`->`ModelNew` and checks output vs the
      reference within tolerance.
    - **Perf bar / stop criterion:** beat PyTorch eager. Tier **win**, iteration cap **6**. sum dim=1 of (128,4096,4096); bandwidth-bound; x=8.6GB (mem watch); near roofline -> modest win
    - `forward()` must stay glue-only (allocate/reshape/launch); all compute in the kernel.
      Verify with `utils/cheating_detection.py` (bench.py does NOT run it).
- **Memory watch (~8.6GB):** device has 49GB but bench holds ref+solution inputs resident simultaneously. The identity baseline MUST bench green (no OOM) before optimizing; if it OOMs, trim the reference input shape and note it.
    - `ncu` unavailable -> proceed analytically from runtime stats.
    - GPU **1** default in `scripts/bench.sh` (overridable via CUDA_VISIBLE_DEVICES). Device memory 49GB.
    - Fast signal: `--no-ref --num-perf-trials 20`; full verdict (real SPEEDUP): `--num-warmup 200` (in bench.sh).
