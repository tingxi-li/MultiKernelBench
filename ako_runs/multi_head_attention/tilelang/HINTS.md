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

    ## Workspace directives (op: multi_head_attention, dsl: tilelang)
    - **Target DSL: tilelang.** TileLang DSL (import tilelang). JIT-compiled tile kernels.
    - **Correctness oracle:** the PyTorch reference `/home/lxt230026/MultiKernelBench/reference/attention/multi_head_attention.py` is the SOLE oracle — this op has no
      prior DSL solution. bench.py renames your `Model`->`ModelNew` and checks output vs the
      reference within tolerance.
    - **Perf bar / stop criterion:** beat PyTorch eager. Tier **win**, iteration cap **6**. nn.MultiheadAttention batch16 seq256 d512 h8; fuse projections+attention
    - `forward()` must stay glue-only (allocate/reshape/launch); all compute in the kernel.
      Verify with `utils/cheating_detection.py` (bench.py does NOT run it).
- **Param-bearing op:** the reference builds learnable `nn.*` layers with SEEDED init. `ModelNew.__init__` MUST construct the same layers (same types, same args, same order) so the seeded weights match — otherwise correctness fails and looks like a kernel bug. Keep the layers as attributes; do compute in the kernel, but read weights from those layers.
    - `ncu` unavailable -> proceed analytically from runtime stats.
    - GPU **2** default in `scripts/bench.sh` (overridable via CUDA_VISIBLE_DEVICES). Device memory 49GB.
    - Fast signal: `--no-ref --num-perf-trials 20`; full verdict (real SPEEDUP): `--num-warmup 200` (in bench.sh).
