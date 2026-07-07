# MultiKernelBench × AKO4ALL — Cross-DSL Kernel Port + Optimization

For each of the 12 NPUKernelBench-matched ops (optimized in **Triton** under `ako_runs/`), this directory ports the kernel to **three more DSLs** and runs the **AKO4ALL optimization loop** on each, benched on NVIDIA RTX 6000 Ada (nvcc 13.1, torch 2.10+cu128, TileLang 0.1.11) against the same `reference/<cat>/<op>.py` golden:

- **cuda_noptx** — plain CUDA C++ via `cpp_extension.load_inline`, no inline PTX `asm`.
- **cuda_unlimited** — CUDA with inline PTX (float4 128-bit vec, `st.global.cs` streaming stores, `red.global.max.s32` reduction-atomics).
- **tilelang** — the TileLang DSL (JIT-compiled tile kernels).

Verdict runs use `--num-warmup 200` (GPUs idle at 210 MHz) and `--deterministic` for scatter. All `forward()` bodies are allocate/launch glue only and pass `utils/cheating_detection.py`.

## Cross-DSL speedup (vs the same PyTorch golden)

**Bold** = a solution modified by the deeper optimization pass; its number is a **fresh single-batch re-bench of the exact committed bytes** (all 16 changed cells re-run agent-free on the same host, 4 pinned GPUs, one controlled batch — `COMPILED=CORRECT=True` for every one). Non-bold cells were byte-unchanged and retain their prior verdict (re-benching them would only inject clock noise).

> **2026-07-07 — `layer_norm` re-optimized (6-op cross-DSL convergence redo, branch `cross-dsl-6op-ncu-redo`).** All four `layer_norm` cells were re-run from an identity baseline through the ncu-in-loop convergence scaffold, each DSL using its native method. **All four beat their prior committed floors** (triton 2.10→**2.17**, cuda_noptx 1.95→**2.15**, cuda_unlimited 2.16→**2.29**, tilelang 2.10→**2.19x**), independently re-benched serial-GPU3, gate-passed, detector-clean, noptx PTX-free. The `layer_norm` rows below and the "+1.9% parity" note for `layer_norm/cuda_noptx` are superseded by these. The +10% noptx jump nearly closes the layer_norm-noptx residual that was the one gap surviving in `CROSS_DSL_FINDINGS.md`. Convergence curves in each `layer_norm/<dsl>/convergence.csv`.

| Op | Cat | Triton | cuda_noptx | cuda_unlimited | tilelang |
|---|---|---|---|---|---|
| relu | activation | 1.0000x | 0.9938x | 1.0000x | 1.0323x |
| sigmoid | activation | 1.0190x | 1.0063x | 1.0063x | 1.0000x |
| hardsigmoid | activation | 1.0127x | 0.9938x | 1.0000x | 0.9877x |
| elu | activation | 0.9938x | 1.0000x | 1.0000x | 1.0256x |
| gelu | activation | 1.0063x | 1.0000x | 1.0000x | 1.0323x |
| swish | activation | 2.5253x | 2.4472x | **2.5256x** | 2.4321x |
| **layer_norm** | normalization | **2.1729x** | **2.1549x** | **2.2857x** | **2.1918x** |
| **group_norm** | normalization | **1.3537x** | **1.2863x** | **1.2810x** | **1.3234x** |
| **gather** | index | **1.5000x** | **1.3267x** | **1.5084x** | **1.3107x** |
| **scatter** | index | **6.7925x** | **10.6471x** | **10.0000x** | **7.6339x** |
| cumsum | math | 1.2264x | 1.2130x | **1.2404x** | 1.1944x |
| lstm | arch | 1.0000x | 1.0000x | 0.9742x | 0.9869x |

**Correctness: 48/48 cells pass** (COMPILED+CORRECT vs the reference at fp32 1e-4 — the 16 changed cells re-verified this pass, the 32 unchanged retain their prior verdicts) and **48/48 pass the cheating detector**; **0/12 `cuda_noptx` cells contain inline PTX** (contract holds). The deeper pass modified **16 solutions**: **13 substantial structural wins** (gather ×4, group_norm ×3, layer_norm ×3, scatter ×3), and **3 marginal changes** kept and reported as parity, not wins (`cumsum/cuda_unlimited` +3.7%, `swish/cuda_unlimited` +3.2%, `layer_norm/cuda_noptx` +1.9% same-GPU).

## Per-op runtime (ms, verdict)

Changed rows carry the fresh re-bench runtime; unchanged rows retain the prior number (absolute ms carry run-to-run clock noise — the speedup column, judged same-GPU baseline→final, is the reliable metric).

| Op | cuda_noptx | cuda_unlimited | tilelang | ref |
|---|---|---|---|---|
| relu | 16.1 | 16 | 15.5 | 16 |
| sigmoid | 16 | 16 | 16.1 | 16.1 |
| hardsigmoid | 16.1 | 16 | 16.2 | 16 |
| elu | 16 | 16 | 15.6 | 16 |
| gelu | 16 | 16 | 15.5 | 16 |
| swish | 16.1 | **15.6** | 16.2 | 39.4 |
| layer_norm | **2.97** | **2.80** | **2.92** | 6.4 |
| group_norm | **24.1** | **24.2** | **23.5** | 31 |
| gather | **0.0202** | **0.0179** | **0.0206** | 0.0269 |
| scatter | **0.0170** | **0.0171** | **0.0224** | 0.177 |
| cumsum | 10.8 | **10.4** | 10.8 | 12.9 |
| lstm | 15.1 | 15.5 | 15.3 | 15.1 |

## Optimization pass (AKO4ALL loop)

Each cell was driven through the AKO profile→edit→bench→log loop toward ~10 iterations where headroom existed, stopping early with a floor-proof where the memory roofline was reached (no ±block-size padding). Depth and the "keep only on a same-GPU baseline→final win" rule are logged per cell in `<op>/<dsl>/ITERATIONS.md`; raw verdicts in `trajectory/`.

**The headline result: two "physical floors" claimed by the first pass were actually _algorithm_ floors.** Both `layer_norm` and `group_norm` are 3-pass ops (read the input for statistics, read it again to normalize, write the output). The first pass measured their 3-pass HBM traffic and called it irreducible. It is not: when one unit of work (a `layer_norm` row = 16 MB, or a `group_norm` chunk of K=4 groups = 32 MB) fits inside the 96 MB L2, a cooperative/pipelined kernel can serve the **second read from L2 instead of HBM**, cutting HBM traffic from 3 passes to 2. That single lever is what moved both ops off their supposed floors.

**Big structural wins (same-GPU baseline→final):**

- **layer_norm — ~1.6x → ~2.1x on 3 of 4 DSLs.** Triton (1.60→**2.10x**), cuda_unlimited (1.60→**2.16x**), and tilelang (1.61→**2.10x**) all landed the **L2-residency 2-pass kernel**: the stats pass streams `x` into L2, then the apply pass re-reads that 16 MB row from L2 (a hit) rather than HBM, so HBM sees ~2 GB (read `x` once + write `y`) instead of ~3 GB. This is **at the 2-pass-traffic floor, not peak** — the winning `cuda_unlimited` log measures ~715 GB/s (~85–88% of achievable HBM BW) and attributes the residual ~0.6 ms to read↔write bus turnaround plus launch/drain bubbles, "neither removable without losing L2 reuse." `cuda_noptx` **also lands the L2-residency 2-pass kernel (1.61→1.95x)**: a host-side per-row loop keeps one 16 MB row L2-resident across its split-block stats→apply so the apply re-reads from L2, no PTX (`__ldg` caching loads). *(Built during the P2 lever tests and promoted post-hoc — the original deeper-pass noptx search had stopped at the 3-pass 1.64x floor; P2 showed that was search depth, not a ceiling. A ~14% schedule/pipelining gap to the fastest winner remains, left to the ncu-in-loop redo.)*
- **group_norm — ~0.9x → ~1.3x on all 4 DSLs.** Triton (0.99→**1.35x**), tilelang (0.90→**1.32x**), and cuda_unlimited (0.92→**1.28x**) each landed an **L2-reuse chunked pipeline**: process K=4 contiguous groups (a 32 MB chunk that stays L2-resident) through a stats→normalize kernel pair, so the normalize pass re-reads the chunk from L2 — again 3-pass→2-pass. All three report sitting at ~87–96% of the ~960 GB/s ceiling for the now-minimal 2-pass traffic. **Caveat preserved from the logs:** the reported *mean* speedup (~1.28–1.35x) is dragged below the steady-state (~1.50x) by a fixed, kernel-independent Trial-1 harness cudaMalloc stall; it is eaten fairly (the reference eats it too). `group_norm/cuda_noptx` **also lands it now (0.92→1.29x)**: the same K=4 chunk pipeline with the `__stcs` streaming store (no PTX), byte-for-byte the unlimited sibling — built in the P2 lever tests (gn_v2) and promoted post-hoc, closing the last of the four "phantom" cells.
- **scatter — up to ~10x.** cuda_unlimited (6.46→**10.0x**) fused the 3-launch design (init + winner-select + gather) into **one kernel** using a packed 64-bit `atomicMax` shared-memory winner slab (high bits = write index for deterministic last-wins, low bits = value bits) and a branchless coalesced copy. tilelang (6.02→**7.63x**) and triton (5.31→**6.79x**) won by element-tiling the inner dim and **dropping the `idx.to(int32)` cast** to read int64 indices directly in-kernel (one fewer launch) — the residual the first pass had "identified but not chased." The **>10x flag on `scatter/cuda_unlimited` is a threshold warning, not reward-hacking**: the reference is torch's *deterministic* scatter (~16x slower than its racy default), so ~10x over it is legitimate.
- **gather — ~1.1–1.26x → ~1.31–1.51x on all 4 DSLs.** This op was *not* "launch noise" (as the first pass concluded) — there was real headroom. cuda_unlimited (1.26→**1.51x**) and cuda_noptx (1.22→**1.33x**) restructured to **shared-memory row staging**: a decisive cold-L2 diagnostic showed the naive scattered `x` read sustains only ~50% of DRAM BW (row-buffer thrashing), so the kernel now does one *coalesced* row load into shared memory and gathers on-chip. Triton (1.22→**1.50x**) and tilelang (1.07→**1.31x**) won via eviction-policy + block/warp/vectorization tuning. Near the shared-mem floor; capped by only ~128 blocks (≈1/SM) at this batch size.

**Modest / near-floor changes kept (correct, but not headlined):**
- **cumsum/cuda_unlimited** (1.22→**1.24x**, +3.7% same-GPU): a real but small gain; the op is ~78–83% of the AD102 mixed-BW ceiling and the scan was never the bottleneck.
- **swish/cuda_unlimited** (2.45→**2.53x**, +3.2%) and **layer_norm/cuda_noptx** (+1.9%): correct restructures right at the noise floor — kept because they don't regress, but reported as parity, not improvement.

**Confirmed at a genuine physical floor** (a legitimate AKO stop):
- **Memory-bound elementwise** (relu/sigmoid/hardsigmoid/elu/gelu; swish's 2.45x is its 2-pass→1-pass fusion): ~1.0x **is** the HBM copy-bandwidth floor — they match torch. float4 closed the residual scalar gaps; streaming loads / fast-`expf` gave 0% (ALU fully latency-hidden).
- **cumsum** (~1.19–1.24x): bandwidth-roofline; register/warp-shuffle scan gave 0%.
- **scatter/cuda_noptx** — *reclassified: no longer a floor.* The deeper pass left it at a 2-kernel 6.56x design; the P2d test showed the fused single-kernel packed-atomic winner-slab (the cuda_unlimited mechanism, which itself uses **no PTX**) ports directly to no-PTX CUDA → **10.65x**, matching the unlimited sibling. Promoted post-hoc; see `P2_LEVER_TESTS.md` §P2d.
- **lstm** (~1.0x): cuDNN's fused multi-layer LSTM is the floor; only the projection GEMM is custom.

The revised traffic analysis for `layer_norm` and `group_norm` — and why the first pass's "irreducible floor" was an over-attribution — is written up in `GAP_ANALYSIS.md` (§ "Floors revised"). Full per-kernel iteration logs (hypothesis → bench → keep/revert, with roofline evidence at each stop) are in each `<op>/<dsl>/ITERATIONS.md`.
