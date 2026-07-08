# The four questions, answered — 6-op cross-DSL ncu redo

Branch `cross-dsl-6op-ncu-redo`. Six ops × 4 DSLs (triton, cuda_noptx = plain CUDA
no inline PTX, cuda_unlimited = CUDA + inline PTX, tilelang) on RTX 6000 Ada, each
DSL using its **native** optimization method from an identity baseline, measurement
unit frozen (`timed_bench.sh` → `compute_s`). Every number is independently re-benched
(≥2×, median-ref), CORRECT under the harness fp32 1e-4 oracle (5/5 seeds), passes the
cheating detector, and no cuda_noptx solution contains inline PTX.

**Why this run exists:** the 12-op study (`CROSS_DSL_FINDINGS.md`) was *all memory-bound*,
so it found no ceiling anywhere and could not measure convergence rate. This redo adds the
tensor-core ops (matmul, matmul_gelu_softmax, sdpa) to force the ceiling question, and
freezes the clock to make the convergence question answerable. Sources for every claim
below: `COMPUTE_FRONTIER_FINDINGS.md`, `CONVERGENCE_PROTOCOL.md`, the per-cell
`convergence.csv` curves, and the layer_norm calibration (`CROSS_DSL_FINDINGS.md` §2026-07-07).

## The verified board

| op | triton | cuda_noptx | cuda_unlimited | tilelang | regime |
|---|---|---|---|---|---|
| sum_reduction | 1.01 | 1.01 | 1.01 | 1.01 | tie — HBM 1-read roofline |
| layer_norm | 2.17 | 2.15 | **2.29** | 2.19 | ~tie (±6%) — L2-resident 2-pass |
| conv_depthwise | 1.41 | 1.46 | 1.46 | **1.53** | tie — memory-bound tiling |
| standard_matmul | 0.79 | 0.59 | 1.11 | **4.13** | **wide** — tensor-core |
| matmul_gelu_softmax | 2.36 | 1.05 | 1.24 | **5.04** | **wide** — tensor-core + fusion |
| sdpa | 1.27 | 1.73 | 1.71 | **3.41** | **wide** — tensor-core (flash) |

The board splits into two regimes: **memory-bound (top 3 rows) tie**; **tensor-core
(bottom 3 rows) diverge wide**. Every answer below turns on that split.

---

## Q1 — Does any DSL have a higher performance ceiling? **Yes, but only on tensor-core ops, and not in the "closer-to-metal wins" direction.**

**Memory-bound half — no ceiling.** sum_reduction, layer_norm, and conv_depthwise all land
within noise at the roofline (reduction ~92–97% of HBM peak; layer_norm 2.15–2.29 band;
conv all four ~2.6 ms). This **extends the 12-op no-ceiling finding** to two more op
families. Inline PTX reconfirmed null here (≤0.6% on the unlimited lane).

**Tensor-core half — a real, wide ceiling exists.** GEMM orders the DSLs
`0.59 → 0.79 → 1.11 → 4.13`. But the ordering **refutes the naive hypothesis** that a
lower-level DSL is an upper bound on a higher-level one:

- The winner is **tilelang — a compiler DSL with zero PTX** — beating the hand-`mma.sync`
  PTX lane (cuda_unlimited) by **~3.7×**.
- On the fused op, **triton (2.36) beats both CUDA lanes** (1.05 / 1.24). So "CUDA ≥ triton"
  is false even inside the compute-bound regime.
- **No DSL is universally on top:** tilelang wins 4 of 6 and ties 1, but *loses* layer_norm
  to cuda_unlimited (2.19 vs 2.29). The closest thing to a universal ceiling advantage is
  tilelang **on the tensor-core sub-board specifically**, where it is uniformly and widely on top.

**cuda_noptx's 0.59× is probably *not* a structural wall — it is a precision-discovery
miss.** All seven of its logged variants are tf32/3×tf32 WMMA (`load_matrix_sync` truncates
inputs to tf32 and gives no accumulation control, so they fail the 1e-4 gate and it settles
at a correct-but-slow tf32 config — L1/shared-bound, tensor pipe ~20% busy). But it **never
tried fp16 WMMA + split-K**, and that *is* expressible in plain WMMA C++ (`half` input
fragments accumulate in `float` — the canonical WMMA recipe, exactly tilelang's approach).
So 0.59× is most likely the *same* precision/search miss as triton's 0.79×, not a "cannot
express it at any effort" ceiling. The only genuinely structural bit is narrower — *tf32*
WMMA has no accumulation control — which is an argument against the tf32 path, not a wall
against WMMA-C++ reaching the frontier. (COMPUTE_FRONTIER C1 asserts a genuine structural
limit here; that is not backed by a tried-and-failed fp16-WMMA experiment and should be
read as **unsettled** — see the fairness caveat and the precision-normalized re-run below.)

**What sets the tensor-core ordering** is two independent factors, neither of which is
"closeness to metal":
1. **Precision-managed tensor cores under the 1e-4 gate.** `torch.matmul` at fp32 runs
   cuBLAS on CUDA cores (~30 TFLOP/s), not tensor cores. Tensor cores are the lever, but
   over K=8192 the tf32/fp16 accumulation error exceeds tolerance unless you **split-K into
   fp32 accumulators**. Who can express that ranks the DSLs.
2. **Pipelining for free vs by hand.** tilelang's compiler auto-emits the cp.async/ldmatrix
   software pipeline feeding the tensor cores (and chose fp16 over tf32). The unlimited lane
   ran out of budget before hand-building that pipeline — its `mma.sync` kernel is
   un-pipelined, ~25 TFLOP/s (44% of tf32 peak).

### Fairness caveat — how much of the ceiling is *precision*, and is it a harness artifact?

The GEMM ordering blends two axes, and only one is "capability":

- **The fp16 pass is enabled by the benchmark's inputs, not just the kernel.** `get_inputs`
  uses `torch.rand` (uniform [0,1], **all-positive**), so each output ≈ 8192×0.25 ≈ 2048 with
  no cancellation. The 1e-4 *relative* gate is then a ~0.2 *absolute* budget, and the fp16
  kernel's error is ~0.04 → **0% of elements fail**. **Under `randn` inputs, 79% of elements
  fail and fp16 is disqualified** (verified empirically). So the fp16 lever rides a tolerance
  budget the benchmark's input distribution creates. This is legitimate under KernelBench's
  rules — it is an output-correctness, precision-agnostic benchmark, and reduced-precision
  tensor cores are what real fast GEMMs do — but for a *capability* study it is a **confound**:
  the `0.59→0.79→1.11→4.13` ordering partly measures "which DSL best exploited the permitted
  precision," not pure algorithmic/pipelining capability. (The split-K→fp32 flush in the
  kernel is still a real, *needed* fix — it fights the fp16 *accumulator* bias, ~-0.19 at
  K=8192, right at the 0.2 budget, cutting it to -0.02. That part is not a shortcut.)
- **So the other lanes' floors are softer than they look.** cuda_noptx (0.59) and triton
  (0.79) are stuck mainly because they searched only the tf32 family and never found
  fp16+split-K — a precision-*discovery* gap, not a hard capability floor. cuda_unlimited's
  1.11 is tf32 (`mma.sync`, un-pipelined); tilelang's 4.13 is fp16 (auto-pipelined). Part of
  that 3.7× is fp16's higher raw tensor-core throughput vs tf32 on Ada, part is pipelining —
  the two are **not separated** in these numbers.
- **The detector does not inspect any DSL kernel body.** All four lanes hide the kernel from
  the Python-AST cheating detector (tilelang via a subscript-dispatch idiom; CUDA via an
  opaque C++ string), so the detector's "pass" rests on `forward()` being glue-only +
  correctness, *not* on reading the kernel. Verified: tilelang's three solutions are
  glue-only and correct, and the idiom suppresses a false positive on integer index math
  inside the kernel, not real compute. Not cheating — but "detector passed" carries no
  independent weight for any DSL here.

**To convert the confound into a clean capability result**, re-bench `standard_matmul` with
either (a) all four lanes at the *same* precision, or (b) `randn` inputs / a tighter gate. If
tilelang still wins wide, the ceiling is real capability; if the gap collapses (noptx/triton
rising via fp16-WMMA / fp16-`tl.dot` + split-K), much of it was precision latitude. **This is
not yet done** — every number above is the native-precision, `torch.rand` result.

## Q2 — Trajectory differences / DSL-unique levers, and did they produce unique results?

Each DSL's defining lever, and whether it actually bought anything:

| DSL | defining lever | memory-bound | tensor-core | unique *result*? |
|---|---|---|---|---|
| cuda_unlimited | inline PTX (`mma.sync`, `st.global.cs`) | **null** (≤0.6%, red herring) | `mma.sync` = **decisive for noptx→parity** (1.11×) but **not sufficient for frontier** | parity, not a win |
| tilelang | fp16 `T.gemm` + compiler auto-pipeline + in-block split-K flush | ties | **the frontier** (4.13 / 5.04 / 3.41) | **yes — the only DSL-unique win** |
| triton | `tl.dot` + offline autotune | ties | tf32 `tl.dot` failed the gate; never found fp16+split-K → 0.79 on GEMM, but won the fused op (2.36) | mixed (fusion win, GEMM miss) |
| cuda_noptx | WMMA C++ `load_matrix_sync` | ties | 0.59 — searched only tf32 WMMA; never tried fp16-WMMA+split-K (expressible) → precision-discovery miss, *not* a proven wall | no (unsettled) |

Two headline reads:

- **PTX's role changed shape but still didn't produce the win.** In the 12-op study inline
  PTX was a pure red herring (null on every op). Here it **splits by op class**: still a red
  herring on memory-bound, but on GEMM the raw `mma.sync` precision control is *necessary* to
  drag the hand-CUDA lane up to parity (WMMA C++ can't, `mma.sync` can) — yet *insufficient*
  for the frontier, since tilelang tripled it with **zero PTX**. So PTX moved from "irrelevant"
  to "necessary-for-parity-but-not-frontier," and the actual frontier win still came from
  somewhere else.
- **The lever that produced a unique result was compiler-emitted, not hand-written.**
  tilelang's fp16 + auto-pipelining + split-K flush is the single DSL-unique lever on the
  6-op board that yields a unique *result* (the wide tensor-core wins). The "defining feature"
  of the widest-lever-set DSL (inline PTX) again failed to produce a win — exactly the 12-op
  pattern, now reconfirmed on compute-bound ops.

## Q3 — Are trajectories transferable? **Split, and it sharpens the 12-op rule.**

**Algorithmic levers transfer ~100% (reinforced by the layer_norm calibration).** All four
DSLs independently re-derived the L2-residency 2-pass lever from the roofline and all four
beat their prior floors (2.15–2.29 band). The PTX-null finding reproduced (unlimited's inline
`st.global.cs.v4` tied `__stcs` byte-for-byte). This is the clean, ~100%-transfer case.

**Tensor-core levers transfer only partially, and split into two sub-parts:**

| sub-lever | transferable? | why |
|---|---|---|
| split-K into fp32 accumulators (the *accuracy* algorithm) | **yes, in principle** | DSL-agnostic; it is *why* hand-`mma.sync` reached parity once coded |
| the cp.async / ldmatrix software pipeline (the *speed* realization) | **only at hand-coding cost** | compiler-emitted in tilelang; unlimited ran out of budget hand-building it |
| the whole lever into cuda_noptx | **not at all** | WMMA C++ can't express the precision control to receive it |

**The rule (12-op rule + compute-bound clause):**
> Traffic/accuracy **algorithm** levers (L2-residency, chunk-pipelining, split-K accuracy
> recovery) transfer ~100% across all four DSLs. The tensor-core **pipeline realization** is a
> DSL-native capability that transfers only at hand-coding cost, and does not transfer into
> WMMA-C++ at all. DSL-native primitives (triton `evict_last`, CUDA arbitrary shared atomics,
> tilelang's compiler pipeline) don't transfer — they define the op where that DSL wins.

## Q4 — Which DSL converges faster? **The JIT/compiler DSLs, decisively — now measured, not hypothesized.**

The 12-op study explicitly could not answer this (uncontrolled effort, no frozen clock). The
redo froze the measurement unit — `compute_s` = compile+bench seconds via `timed_bench.sh`,
with agent thinking excluded and ncu passes excluded — so it is now answerable. Per-op
cumulative `compute_s` to ceiling / logged-variant count:

| op | triton | cuda_noptx | cuda_unlimited | tilelang |
|---|---|---|---|---|
| sum_reduction | 193s / 3 | 349s / 4 | 354s / 4 | 325s / 5 |
| standard_matmul | 55s / 5 | 296s / 8 | 407s / 11 | **85s / 10** |
| matmul_gelu_softmax | 58s / 5 | 48s / 2 | 48s / 2 | 41s / 5 |
| conv_depthwise | 60s / 5 | 150s / 4 | 153s / 4 | 65s / 5 |
| sdpa | 779s / 8 | 212s / 2 | 210s / 2 | 641s / 9 |
| **total** | **~1145s / 26** | **~1055s / 20** | **~1172s / 23** | **~1157s / 34** |

**The mechanism is per-variant cost.** JIT DSLs (triton, tilelang) recompile a config in
seconds; the nvcc lanes (noptx, unlimited) pay ~30–50 s/variant. The sharpest illustration is
raw matmul, straight off the `convergence.csv` curves:

- **tilelang** reached **3.94× at iteration 4 / 29 s**, and its 4.13× ceiling by iter 6 / 49 s
  (10 variants, 85 s total).
- **cuda_unlimited** ground through **11 variants / 407 s** — a failed WMMA-tf32 sub-tree, then
  `mma.sync`, then split-K — to reach only **1.11×**.

So the compiler DSL converged to a **~3.7× higher ceiling in ~5× less compute**. Ordering:
**tilelang ≲ triton (fast JIT) ≪ cuda_noptx / cuda_unlimited (slow nvcc)**. This is the
practical convergence advantage the 12-op study hypothesized, now measured with every config
wrapper-logged.

**Three caveats that stay attached to this answer:**
- **`compute_s` is reference-dominated on slow-ref ops.** sdpa's 641–779 s totals are the
  ~60–80 ms torch reference timed every bench, not DSL search — compare *variant counts* there
  (triton 8, tilelang 9 vs noptx/unlimited 2), not seconds.
- **Still native-workflow, not fully controlled (C1–C3).** Discovery and effort are uneven:
  triton and noptx never found fp16+split-K; unlimited didn't have budget to hand-build the
  pipeline. Part of the matmul spread is search luck layered on real capability, not pure
  capability.
- **layer_norm's convergence was *hinted*** (it is the calibration op): tilelang 25 s < triton
  40 s < unlimited 57 s < noptx 69 s. That ordering measures realize-cost (nvcc ~40 s/variant
  vs JIT), **not** discovery-cost. The five un-hinted ops are the real convergence read — and
  they say the same thing.

---

## One-line synthesis

A real cross-DSL ceiling **does** exist but surfaces **only on tensor-core ops**; there it is
owned by the **compiler DSL (tilelang)** via precision-managed, auto-pipelined tensor cores —
**not** by hand-PTX — and that same compiler DSL also **converges fastest and highest**, while
inline PTX flips from "irrelevant" (memory-bound) to "necessary-for-parity-but-insufficient-
for-frontier" (tensor-core). On memory-bound ops all four DSLs remain tied at the roofline,
and algorithmic levers transfer ~100% across all of them.

**Read the tensor-core ceiling with the fairness caveat attached:** its magnitude is inflated
by a precision confound (fp16 rides `torch.rand`'s all-positive tolerance budget; would fail
under `randn`) and the other lanes' floors are precision-discovery misses, not proven walls.
The *qualitative* finding — a compiler DSL reaches the tensor-core frontier that hand lanes
did not, in less compute — is robust; the *quantitative* 0.59→0.79→1.11→4.13 spread is not a
clean capability ranking until re-benched at normalized precision (or under `randn`).
