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

    ## Workspace directives (op: scaled_dot_product_attention, dsl: cuda_noptx)
    - **Target DSL: cuda_noptx.** Plain CUDA C++ via torch.utils.cpp_extension.load_inline. NO inline PTX `asm(...)`. Intrinsics (__expf/erff/float4/__ldg/__shfl) OK.
    - **Correctness oracle:** the PyTorch reference `/home/lxt230026/MultiKernelBench/reference/attention/scaled_dot_product_attention.py` is the SOLE oracle — this op has no
      prior DSL solution. bench.py renames your `Model`->`ModelNew` and checks output vs the
      reference within tolerance.
    - **Perf bar / stop criterion:** beat PyTorch eager. Tier **win**, iteration cap **6**. head_dim=1024 EXCEEDS flash-256 limit -> reference on UNFUSED math/mem-efficient path -> flash-style fusion is a real win; QKV 6.4GB
    - `forward()` must stay glue-only (allocate/reshape/launch); all compute in the kernel.
      Verify with `utils/cheating_detection.py` (bench.py does NOT run it).
- **Memory watch (~6.4GB):** device has 49GB but bench holds ref+solution inputs resident simultaneously. The identity baseline MUST bench green (no OOM) before optimizing; if it OOMs, trim the reference input shape and note it.
    - `ncu` unavailable -> proceed analytically from runtime stats.
    - GPU **0** default in `scripts/bench.sh` (overridable via CUDA_VISIBLE_DEVICES). Device memory 49GB.
    - Fast signal: `--no-ref --num-perf-trials 20`; full verdict (real SPEEDUP): `--num-warmup 200` (in bench.sh).
