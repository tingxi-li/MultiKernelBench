# MultiKernelBench × AKO4ALL — Cross-DSL Kernel Port + Optimization

For each of the 12 NPUKernelBench-matched ops (optimized in **Triton** under `ako_runs/`), this directory ports the kernel to **three more DSLs** and runs the **AKO4ALL optimization loop** on each, benched on NVIDIA RTX 6000 Ada (nvcc 13.1, torch 2.10+cu128, TileLang 0.1.11) against the same `reference/<cat>/<op>.py` golden:

- **cuda_noptx** — plain CUDA C++ via `cpp_extension.load_inline`, no inline PTX `asm`.
- **cuda_unlimited** — CUDA with inline PTX (float4 128-bit vec, `st.global.cs` streaming stores, `red.global.max.s32` reduction-atomics).
- **tilelang** — the TileLang DSL (JIT-compiled tile kernels).

Verdict runs use `--num-warmup 200` (GPUs idle at 210MHz). All `forward()` bodies are allocate/launch glue only and pass `utils/cheating_detection.py`.

## Cross-DSL speedup (vs the same PyTorch golden)

Cells modified by the optimization pass show **bold** fresh post-opt verdicts; unmodified solutions retain their prior verdict (identical code).

| Op | Cat | Triton | cuda_noptx | cuda_unlimited | tilelang |
|---|---|---|---|---|---|
| **relu** | activation | 1.0000x | **0.9938x** | 1.0000x | **1.0323x** |
| **sigmoid** | activation | 1.0190x | **1.0063x** | 1.0063x | 1.0000x |
| **hardsigmoid** | activation | 1.0127x | **0.9938x** | 1.0000x | 0.9877x |
| **elu** | activation | 0.9938x | **1.0000x** | 1.0000x | 1.0256x |
| **gelu** | activation | 1.0063x | **1.0000x** | 1.0000x | 1.0323x |
| **swish** | activation | 2.5253x | 2.4472x | 2.4472x | 2.4321x |
| **layer_norm** | normalization | 1.6050x | **1.6121x** | **1.5955x** | 1.6080x |
| **group_norm** | normalization | 0.9904x | **0.9174x** | 0.9172x | **0.9038x** |
| **gather** | index | 1.2217x | **1.2182x** | **1.2607x** | 1.0675x |
| **scatter** | index | 5.3079x | 6.5580x | 6.4621x | **6.0204x** |
| **cumsum** | math | 1.2264x | **1.2130x** | **1.2243x** | 1.1944x |
| **lstm** | arch | 1.0000x | 1.0000x | 0.9742x | 0.9869x |

**Correctness: 36/36 ported kernels pass** (COMPILED+CORRECT vs the reference at fp32 1e-4) and **36/36 pass the cheating detector**. The optimization pass modified 15 solutions.

## Per-op runtime (ms, verdict)

| Op | cuda_noptx | cuda_unlimited | tilelang | ref |
|---|---|---|---|---|
| relu | 16.1 | 16 | 15.5 | 16 |
| sigmoid | 16 | 16 | 16.1 | 16.1 |
| hardsigmoid | 16.1 | 16 | 16.2 | 16 |
| elu | 16 | 16 | 15.6 | 16 |
| gelu | 16 | 16 | 15.5 | 16 |
| swish | 16.1 | 16.1 | 16.2 | 39.4 |
| layer_norm | 3.97 | 4.03 | 3.98 | 6.4 |
| group_norm | 33.9 | 33.8 | 34.3 | 31 |
| gather | 0.022 | 0.0211 | 0.0252 | 0.0269 |
| scatter | 0.0276 | 0.0277 | 0.0294 | 0.177 |
| cumsum | 10.8 | 10.7 | 10.8 | 12.9 |
| lstm | 15.1 | 15.5 | 15.3 | 15.1 |

## Optimization pass (AKO4ALL loop)

Each of the 36 kernels was run through the AKO profile→edit→bench→log loop (`--num-warmup 200`; `--deterministic` for scatter), depth proportional to headroom, with the floor as a legitimate stop. Improvements are validated by each agent's **same-GPU baseline→final delta** (controls for clock state); absolute table numbers carry run-to-run clock noise.

**9 kernels improved >3%:**
- **layer_norm/cuda_noptx 1.49→1.61x** and **cuda_unlimited 1.47→1.60x** — both reached the same ~3.9 ms HBM-read floor via **opposite** structural routes. **cuda_noptx fused** the 3-launch split-row reduction into ONE per-row kernel (one block per row, **fp32** accumulators, shared-mem tree reduce, float4 apply), dropping the `ln_final` launch + global mean/rstd round-trip. **cuda_unlimited kept the split-row layout** (more blocks → better stats-pass occupancy; the one-block-per-row fusion was profiled and *ruled out* as occupancy-starved on 142 SMs) and won instead by **column-blocking the apply** so each thread owns 4 columns and fetches w/b once, reusing them across all 64 rows from registers. Both now match the TileLang port (~1.61x), the fastest of all four abstractions. The split-vs-fused divergence shows the per-row mapping is regime-dependent, not universally best (see `GAP_ANALYSIS.md`).
- **gather/cuda_unlimited 1.12→1.26x** and **cuda_noptx 1.11→1.22x** — the kernel was latency/MLP-bound, not at the HBM floor; the win was a 2D grid (kills the emulated int64 divide) + K outputs/thread issuing independent scattered loads to hide the dependent idx→x latency. Both now meet/beat the Triton port (1.22x).
- **group_norm/tilelang 0.85→0.90x** — float4-vectorized both the reduce and apply passes.
- **scatter/tilelang 5.88→6.02x** (same-GPU 5.33→6.02, +13%) — dropped the `idx.to(int32)` cast and read int64 indices directly in-kernel, removing one launch.
- **relu/tilelang 0.99→1.03x, elu/cuda_noptx & gelu/cuda_noptx 0.97→1.00x** — float4 vectorization lifted the scalar ports to the HBM roofline.

**The rest were confirmed at a physical floor** (a legitimate AKO stop, not a shortfall):
- **Memory-bound elementwise** (relu/sigmoid/hardsigmoid/elu/gelu/swish): ~1.0x (2.45x for swish's 2-pass→1-pass fusion) **is** the HBM-bandwidth floor — they match torch, already at peak bandwidth. float4 closed the remaining cuda_noptx scalar gaps to ~1.0x; further levels (streaming loads, fast-`expf`) gave 0% because ALU is fully latency-hidden.
- **group_norm** (all DSLs ~0.90–0.92x): the exact 3-pass op moves 25.8 GB of irreducible traffic; min runtime (≈31 ms) ties torch's own min — at the roofline. The sub-1.0x *mean* is a harness clock-ramp outlier (one ~300 ms trial-1), kernel-independent.
- **cumsum** (~1.21–1.22x): 8.6 GB R+W at ~750–795 GB/s ≈ 78–83% of the AD102 mixed-BW ceiling; register/warp-shuffle scan gave 0% (scan was never the bottleneck).
- **scatter cuda tracks** (~6.4–6.9x, unchanged): already well above Triton (5.31x), near the deterministic-scatter floor.
- **lstm** (~1.0x): cuDNN's fused multi-layer LSTM is the floor; only the projection GEMM is a custom kernel.
- **layer_norm/tilelang** (1.61x, unchanged): already the fastest of the four — see `GAP_ANALYSIS.md`.

Full per-kernel iteration logs (hypothesis → bench → keep/revert, with roofline evidence at each stop) are in each `<op>/<dsl>/ITERATIONS.md`; raw bench outputs in `trajectory/`.
