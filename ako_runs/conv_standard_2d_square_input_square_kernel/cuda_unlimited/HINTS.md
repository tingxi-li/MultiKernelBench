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

    ## Workspace directives (op: conv_standard_2d_square_input_square_kernel, dsl: cuda_unlimited)
    - **Target DSL: cuda_unlimited.** CUDA via load_inline, EVERYTHING allowed: inline PTX `asm volatile`, cp.async, vectorized ld/st, warp intrinsics, cache hints. No cuBLAS/cuDNN library offload.
    - **Correctness oracle:** the PyTorch reference `/home/lxt230026/MultiKernelBench/reference/convolution/conv_standard_2d_square_input_square_kernel.py` is the SOLE oracle — this op has no
      prior DSL solution. bench.py renames your `Model`->`ModelNew` and checks output vs the
      reference within tolerance.
    - **Perf bar / stop criterion:** beat PyTorch eager. Tier **floor**, iteration cap **2**. nn.Conv2d 3->96 k11 s4 (AlexNet conv1); cuDNN near-optimal -> ~1x
- **FLOOR op:** eager already dispatches to cuBLAS/cuDNN (library-optimal). ~1x is the physical ceiling. Confirm the floor within the cap and STOP; do not chase a speedup that isn't there.
    - `forward()` must stay glue-only (allocate/reshape/launch); all compute in the kernel.
      Verify with `utils/cheating_detection.py` (bench.py does NOT run it).
- **Param-bearing op:** the reference builds learnable `nn.*` layers with SEEDED init. `ModelNew.__init__` MUST construct the same layers (same types, same args, same order) so the seeded weights match — otherwise correctness fails and looks like a kernel bug. Keep the layers as attributes; do compute in the kernel, but read weights from those layers.
    - `ncu` unavailable -> proceed analytically from runtime stats.
    - GPU **1** default in `scripts/bench.sh` (overridable via CUDA_VISIBLE_DEVICES). Device memory 49GB.
    - Fast signal: `--no-ref --num-perf-trials 20`; full verdict (real SPEEDUP): `--num-warmup 200` (in bench.sh).
