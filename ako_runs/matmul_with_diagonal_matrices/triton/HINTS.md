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

## Workspace directives (op: matmul_with_diagonal_matrices, dsl: triton)
- **Target DSL: triton.** Triton DSL (import triton, triton.language as tl). @triton.jit kernels; triton.autotune allowed. No torch-op offload in the hot path.
- **Correctness oracle:** the PyTorch reference `/home/lxt230026/MultiKernelBench/reference/matmul/matmul_with_diagonal_matrices.py` is the SOLE oracle — this op has no
  prior DSL solution. bench.py renames your `Model`->`ModelNew` and checks output vs the
  reference within tolerance.
- **Perf bar / stop criterion:** beat PyTorch eager. Tier **win**, iteration cap **6**. reference MATERIALIZES diag(A) 4096^2 then full GEMM; semantically rowwise scale A[:,None]*B -> large win (avoid N^2 work)
- `forward()` must stay glue-only (allocate/reshape/launch); all compute in the kernel.
  Verify with `utils/cheating_detection.py` (bench.py does NOT run it).
- `ncu` unavailable -> proceed analytically from runtime stats.
- GPU **3** default in `scripts/bench.sh` (overridable via CUDA_VISIBLE_DEVICES). Device memory 49GB.
- Fast signal: `--no-ref --num-perf-trials 20`; full verdict (real SPEEDUP): `--num-warmup 200` (in bench.sh).
