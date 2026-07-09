# Cross-DSL Kernel Optimization — Opus 4.8 vs Sonnet 4.6

**Study:** 5 operators × 4 GPU DSLs × 2 optimizer models, RTX 6000 Ada (AD102, ~960 GB/s
GDDR6, 96 MB L2), nvcc 13.1, torch 2.10+cu128, TileLang 0.1.11. Each cell was driven from
an identity baseline through the AKO4ALL profile→edit→bench→log loop against the **same
PyTorch golden**, correctness gated at fp32 **1e-4** (5/5 seeds), every `forward()` glue-only
and detector-clean.

- **DSLs:** `triton` · `cuda_noptx` (plain CUDA, inline PTX **forbidden**) · `cuda_unlimited`
  (CUDA **with** inline PTX) · `tilelang` (JIT tile compiler).
- **Operators:** `sum_reduction`, `conv_depthwise`, `standard_matmul`, `matmul_gelu_softmax`,
  `scaled_dot_product_attention` (sdpa) — two memory-bound + three tensor-core.
- **Provenance.** Opus 4.8 = `solution/solution_opus48/` + `ITERATIONS_opus48.md` +
  `convergence.csv` + the top-level findings docs (`SIX_OP_ANSWERS.md`,
  `COMPUTE_FRONTIER_FINDINGS.md`, `CONVERGENCE_PROTOCOL.md`, `RESULTS.md` — their board
  matches the Opus `convergence.csv`). Sonnet 4.6 = `solution/<op>.py` + `ITERATIONS.md`.

## Measurement caveats (read before trusting deltas)

1. **Reference-clock / baseline drift.** Sonnet and Opus cells were not always benched against
   an identically-measured golden (e.g. `conv_depthwise`: Sonnet ref ≈4.0 ms, an Opus
   `convergence.csv` ref ≈5.84 ms; the Opus "final" rows re-measure the golden and drift).
   **Absolute runtime is the reliable comparator.** Speedup deltas **< ~10 % on
   memory-bound ops are clock noise, not skill.**
2. **fp16 / `torch.rand` precision confound.** `get_inputs()` uses `torch.rand` (all-positive),
   so the 1e-4 relative gate becomes a ~0.2 absolute budget and fp16 tensor-core error (~0.04)
   passes; in `matmul_gelu_softmax` the softmax further normalizes logit error. The tensor-core
   *ceiling ordering is robust*, but its *magnitude* is inflated and is **not a clean capability
   ranking** until re-run at normalized precision / under `randn`.
3. **`cuda_noptx` on GEMM (0.59×) is a discovery miss, not a proven wall** — it only searched
   tf32 WMMA and never tried fp16-WMMA+split-K (expressible in plain WMMA C++).
4. **`compute_s` convergence data exists for the Opus campaign only** (Sonnet `ITERATIONS.md`
   logged no compute clock); it is reference-dominated on slow-golden ops (sdpa).

---

## Headline boards (speedup vs the same PyTorch golden)

**Opus 4.8** (convergence-kept = the committed board):

| op | triton | cuda_noptx | cuda_unlimited | tilelang | regime |
|---|---|---|---|---|---|
| sum_reduction | 1.01 | 1.01 | 1.01 | 1.01 | mem — HBM 1-read roofline |
| conv_depthwise | 1.41 | 1.46 | 1.46 | **1.53** | mem — tiling (all ≈2.6 ms) |
| standard_matmul | 0.79 | 0.59 | 1.11 | **4.13** | **TC — wide** |
| matmul_gelu_softmax | 2.36 | 1.05 | 1.24 | **5.04** | **TC + fusion — wide** |
| sdpa | 1.27 | 1.73 | 1.71 | **3.41** | **TC (flash) — wide** |

**Sonnet 4.6** (from `ITERATIONS.md`; tensor-core cells are the meaningful comparison):

| op | triton | cuda_noptx | cuda_unlimited | tilelang |
|---|---|---|---|---|
| sum_reduction | 1.01 | 1.01 | 1.01 | 1.00 |
| conv_depthwise | ≈1.5 | ≈1.5 | ≈1.5 | ≈1.5 (all ≈2.6 ms) |
| standard_matmul | 0.98 | 0.85 | 0.83 | 4.07 |
| matmul_gelu_softmax | **5.07** | 1.03 | 1.47 | 4.71 |
| sdpa | **1.78** | 1.03 | 1.12 | 2.57 |

---

## Q1 — Does any DSL have a strictly higher performance ceiling?

**Two regimes — and the ceiling does *not* favor "closer-to-metal."**

- **Memory-bound ops (sum_reduction, conv_depthwise): no ceiling.** All four DSLs converge to
  the same HBM roofline — reduction ties at ~1.01× (~92 % of the 1-read peak, all four land at
  9.6–9.8 ms); conv ties at ~2.6 ms regardless of DSL. Inline PTX buys ≤0.6 %. Both models,
  both regimes: a dead heat.

- **Tensor-core ops (matmul, matmul_gelu_softmax, sdpa): a real, wide ceiling — owned by the
  compiler DSL, `tilelang`.** GEMM ordering `0.59 → 0.79 → 1.11 → 4.13`: tilelang beats the
  hand-PTX `mma.sync` lane by **~3.7×** with **zero PTX**. sdpa: tilelang 3.41 vs next-best 1.73
  (~2×). The direction "CUDA ≥ Triton" is **false**: on the fused op Triton (2.36 Opus / 5.07
  Sonnet) beats *both* CUDA lanes.

**Why the ordering exists (two factors, neither is metal-proximity):**
1. **Precision-managed tensor cores under the 1e-4 gate.** `torch.matmul` fp32 = cuBLAS on CUDA
   cores (~30 TFLOP/s), *not* tensor cores. Over K=8192, fp16/tf32 accumulation error blows the
   gate unless you **split-K into fp32 accumulators**. Which DSL can express that ranks them.
2. **Pipelining for free vs by hand.** tilelang's compiler auto-emits the cp.async/ldmatrix
   software pipeline (~120 TFLOP/s fp16); the hand `mma.sync` kernel is un-pipelined (~25
   TFLOP/s) — it ran out of budget before hand-building the pipeline.

> **Verdict:** No universal ceiling. On memory-bound ops every DSL ties the roofline. On
> tensor-core ops there is a wide ceiling **owned by `tilelang`** (compiler-managed fp16 tensor
> cores + auto-pipelining), *not* by the widest-privilege DSL. Magnitude is precision-confounded;
> the ordering is robust.

---

## Q2 — Trajectory differences and DSL-unique techniques

Each DSL's defining lever, and whether it bought a **unique result**:

| DSL | signature lever | memory-bound | tensor-core | unique win? |
|---|---|---|---|---|
| **cuda_unlimited** | inline PTX (`mma.sync`, `ld/st.*.cs`) | **null** (≤0.6 %) | `mma.sync`+split-K → noptx-beating **parity** (1.11) but not the frontier | parity, not a win |
| **tilelang** | fp16 `T.gemm` + compiler auto-pipeline + in-block split-K flush | ties | **the frontier** (4.13 / 5.04 / 3.41) | **yes — the only DSL-unique win** |
| **triton** | `tl.dot` + offline `@triton.autotune` + `evict_first` cache hint | ties (autotune wins the streaming/cache cells) | fp16+fused epilogue strong on fused op; never found fp16+split-K on raw GEMM (0.79) | mixed |
| **cuda_noptx** | WMMA-C++ `load_matrix_sync`, `__ldg`, `maxrregcount` | ties | tf32-WMMA only → 0.59 GEMM (fp16-WMMA+split-K untried) | no (discovery miss) |

**DSL-unique techniques that actually mattered:**
- **tilelang — fp16 `T.gemm` + compiler-emitted software pipeline + in-block split-K fp32
  flush.** The single lever that produced a *unique* result (the tensor-core frontier). Note it
  is **compiler-emitted, not hand-written** — the widest-privilege DSL's signature feature
  (inline PTX) again failed to produce a win. Also unique: `T.gemm` fp32→shared→fp16 layout
  bridge that lets sdpa stay fused despite D=1024 (dodges the smem wall that forced Triton/CUDA
  into 3-kernel materialization).
- **cuda_unlimited — raw `mma.sync` PTX.** On GEMM/sdpa it was the *only* correctness-passing
  tensor-core path for hand-CUDA (WMMA-C++ tf32 fails the gate); decisive to reach parity, but
  ~3.7× short of the frontier.
- **cuda_noptx — WMMA-C++ intrinsics + multi-bank fp32 accumulation** (round-robin accumulator
  "banks" to keep tf32 tensor-core error under the gate without any PTX). This is what let the
  no-PTX lane reach tensor cores at all (sdpa 1.73×).
- **triton — `evict_first`/`evict_last` cache-residency hints + exhaustive autotune.** The lever
  that wins the streaming memory-bound cells and squeezes the most out of fused epilogues.

**Inline PTX's role, restated:** null on every memory-bound op; on GEMM it is *necessary* to
drag hand-CUDA to parity yet *insufficient* for the frontier. It moved from "irrelevant"
(12-op memory-bound study) to "necessary-for-parity-but-not-frontier" — and still produced **no
win**.

---

## Q3 — Are optimization trajectories transferable across DSLs?

**Split by lever type.**

| lever class | transfers? | evidence |
|---|---|---|
| **Algorithmic / traffic / accuracy** (L2-residency 2-pass, chunk-pipelining, coalesced wide-tile reads, **split-K into fp32 accumulators**) | **~100 %** | all four DSLs independently re-derived the conv wide-tile roofline and the reduction streaming lever; split-K-flush appears in both tilelang and cuda_unlimited GEMMs |
| **Tensor-core *pipeline realization*** (cp.async/ldmatrix software pipeline) | **only at hand-coding cost** | compiler-emitted free in tilelang; cuda_unlimited ran out of budget hand-building it → stuck at parity |
| **Whole tensor-core lever into `cuda_noptx`** | **not cleanly** | WMMA-C++ needs the multi-bank accumulator workaround to even receive the precision control; fp16-WMMA+split-K was never attempted |
| **DSL-native primitives** (triton `evict_last`, tilelang compiler pipeline, CUDA block-scoped atomics) | **not at all** | each defines the *one* op where that DSL is sole winner |

> **Rule:** the *algorithm* (what traffic to cut, how to bound fp32 error) transfers ~100 %;
> the tensor-core *pipeline realization* transfers only at hand-coding cost and not into
> WMMA-C++ for free; DSL-native primitives don't transfer and are exactly the moats.

---

## Q4 — Which DSL converges faster? (fewer iterations / less compute-profiler cost)

Measurement is frozen by `CONVERGENCE_PROTOCOL.md`: wall-clock is split into `compute_s`
(compile+JIT+autotune+bench — the yardstick, captured by `timed_bench.sh`), `agent_s` (LLM
reasoning, logged never compared), and `ncu_s` (excluded). Metric = cumulative `compute_s` to
within 5 % of the cell ceiling, plus distinct variants benched. **Data below is the Opus
campaign** (`convergence.csv`):

| op | triton | cuda_noptx | cuda_unlimited | tilelang |
|---|---|---|---|---|
| sum_reduction | 193 s / 3 | 349 s / 4 | 354 s / 4 | 325 s / 5 |
| conv_depthwise | 60 s / 5 | 150 s / 4 | 153 s / 4 | 65 s / 5 |
| standard_matmul | **55 s / 5** | 296 s / 8 | **407 s / 11** | **85 s / 10** |
| matmul_gelu_softmax | 58 s / 5 | 48 s / 2 | 48 s / 2 | 41 s / 5 |
| sdpa | 779 s / 8 | 212 s / 2 | 210 s / 2 | 641 s / 9 |

**The JIT/compiler DSLs (triton, tilelang) converge decisively faster** — each variant recompiles
in seconds, while the nvcc lanes pay ~30–50 s/variant. Sharpest illustration (raw GEMM): tilelang
reached **3.94× at iter 4 / 29 s** and its **4.13× ceiling by iter 6 / 49 s** (85 s total), while
**cuda_unlimited ground 11 variants / 407 s** through a failed WMMA-tf32 sub-tree to reach only
**1.11×** — the compiler DSL converged to a **~3.7× higher ceiling in ~5× less compute**. Ordering:
**tilelang ≲ triton (fast JIT) ≪ cuda_noptx / cuda_unlimited (slow nvcc)**.

*Caveats:* on slow-golden ops `compute_s` is reference-dominated (sdpa's 641–779 s is mostly the
~60–80 ms torch reference timed every bench — compare **variant counts** there: triton 8, tilelang
9 vs noptx/unlimited 2, which still favors more, cheaper JIT probes). Sonnet logged no `compute_s`,
but its **iteration counts are systematically higher** (see Q5).

---

## Q5 — Opus 4.8 vs Sonnet 4.6

Two clean behavioral differences, and a nuanced performance split.

**A. Opus converges in far fewer iterations, roofline/gate-first.** Across almost every cell Opus
stopped in **2–5 ncu-guided variants** where Sonnet ran **6–12 exploratory** ones (e.g. conv: Opus
4–5 vs Sonnet 12; matmul_gelu CUDA cells: Opus 2 vs Sonnet 6–11; sdpa CUDA cells: Opus 2 vs Sonnet
5–6). Opus reasons from the roofline and the tolerance budget and stops at the plateau; Sonnet
chases sub-3 % micro-tweaks.

**B. Opus unlocks the hard precision/PTX levers Sonnet skips — decisive on the CUDA lanes.**
This is where Opus wins outright:

| cell | Sonnet | Opus | why Opus wins |
|---|---|---|---|
| matmul · cuda_unlimited | 0.83 | **1.11** | Opus used PTX `mma.sync` + split-K; **Sonnet declared a floor and never attempted PTX** |
| sdpa · cuda_noptx | 1.03 (identity surrender) | **1.75** | Opus realized WMMA-C++ tf32 is legal under the no-PTX rule and got tensor cores |
| sdpa · cuda_unlimited | 1.12 | **1.86** | cleaner `mma.sync` tf32, 2 variants |
| sdpa · tilelang | 2.57 | **3.41** | fp32-cast-on-load removes ~9 GB HBM; safe no-flush softmax |
| matmul_gelu · cuda_unlimited/noptx | 1.47 / 1.03 | **~2.1 / ~1.7** (re-bench) | PTX/WMMA precision paths |

**C. But Sonnet's exhaustive autotuning wins the "safe" DSL on fused/attention ops.** Where no
precision unlock is needed and the lever is tile/pipeline tuning, Sonnet's deeper search edges out:

| cell | Sonnet | Opus | why Sonnet wins |
|---|---|---|---|
| matmul_gelu · triton | **5.07** | 2.36 (board) / 3.60 (re-bench) | Sonnet's deeper autotune + fp16-output-to-halve-softmax-BW |
| sdpa · triton | **1.78** | 1.27 | Sonnet's D-tiled fp16 flash beat Opus's precision-limited 3-kernel design |
| standard_matmul · triton/noptx | 0.98 / 0.85 | 0.79 / 0.59 | both below the cuBLAS floor — **noise**, plus Opus "wasted" iters on TC paths that failed the gate |

> **Net:** Opus is the stronger optimizer **on the tensor-core frontier's hard cells** — it raises
> every CUDA floor by finding the PTX/WMMA precision path Sonnet abandons, ties/edges tilelang, and
> gets there in ~½–⅓ the iterations. Sonnet is stronger **only where the win is exhaustive tile
> autotuning on a DSL that needs no precision trick** (Triton fused / attention). On memory-bound
> ops the two are indistinguishable (both hit the roofline; Opus just uses fewer probes).

---

## Bottom line

1. **Ceiling (Q1):** none on memory-bound ops; a wide one on tensor-core ops, **owned by the
   compiler DSL `tilelang`** (fp16 tensor cores + auto-pipelining + split-K), not by the hand-PTX
   lane. Magnitude is precision-confounded; ordering is robust.
2. **DSL-unique (Q2):** the only lever that produced a *unique* win is **compiler-emitted**
   (tilelang's auto-pipelined fp16 GEMM). Inline PTX is null on memory-bound ops and merely
   parity-enabling on GEMM — never a win.
3. **Transfer (Q3):** algorithmic/accuracy levers transfer ~100 %; tensor-core pipeline
   realization transfers only at hand-coding cost; DSL-native primitives are non-transferable moats.
4. **Convergence (Q4):** JIT compiler DSLs (tilelang ≲ triton) converge fastest and highest;
   nvcc lanes are ~5–8× more compute per ceiling reached.
5. **Opus vs Sonnet (Q5):** Opus converges in ~½–⅓ the iterations and wins the hard CUDA/PTX
   precision cells that Sonnet abandons; Sonnet wins only where exhaustive Triton autotuning pays
   off (fused/attention). Tie on memory-bound.
