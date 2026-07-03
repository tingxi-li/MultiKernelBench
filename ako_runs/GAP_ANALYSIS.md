# Why the cross-DSL speedups diverge — root-cause analysis of the two large gaps

Across the 12 ops × 4 abstractions (Triton / cuda_noptx / cuda_unlimited / tilelang),
most ops agree to within ~3–15% between DSLs. **Two ops had large cross-abstraction
gaps, both with TileLang as the laggard:**

| Op | Triton | cuda_noptx | cuda_unlimited | tilelang | max/min spread |
|---|---|---|---|---|---|
| **layer_norm** | 1.61x | 1.49x | 1.47x | **0.65x** → fixed **1.61x** | 2.28× → ~1.0× |
| **scatter** | 5.31x | 6.56x | 6.46x | **3.80x** → fixed **5.88x** | 1.73× → 1.10× |
| group_norm | 0.99x | 0.90x | 0.92x | 0.85x | 1.17× |
| gather | 1.22x | 1.11x | 1.12x | 1.07x | 1.14× |
| (cumsum / activations / lstm) | — | — | — | — | ≤1.05× |

> **Note (post-hoc):** the "fixed" numbers in this table are the *port-parity* fixes (bringing the
> TileLang laggard up to the CUDA/Triton level). A later deeper pass then moved several of these ops
> *past* that level — layer_norm ~1.6x → **~2.1x**, group_norm ~0.9x → **~1.3x**, gather ~1.1x →
> **~1.3–1.5x**, scatter → up to **~10x** — by cutting HBM traffic 3-pass→2-pass. See § **"Floors
> revised (deeper AKO pass)"** at the end. The gap analysis below is unchanged and still correct for
> the question it answers (*why TileLang lagged*); the floor claims are what got revised.

Both gaps surface only because of the benchmark's small batch dimension (M = Rr = 64) —
but they have **two independent causes**: a grid/occupancy mistake (scatter) and a dtype
mistake (layer_norm). The tempting single-root-cause story ("the per-row mapping starves
the GPU") is **refuted by layer_norm's own data below** — there the per-row mapping is
*optimal*. Each cause was isolated with a controlled single-variable experiment
(`--num-warmup 200`, same reference golden), not assumed.

---

## Gap 1 — layer_norm: the cause was **fp64**, not occupancy

Input `(64, 64, 256, 256)`, LayerNorm over the last 3 dims ⇒ **M = 64 rows of
N = 4,194,304**. The shipped TileLang kernel used one block per row (`T.Kernel(M)` = 64
blocks) with **fp64** accumulators (it reduces a whole 4.19M-element row per block, so I
reached for fp64 to be safe against the long accumulation chain). It scored 0.65x while
CUDA hit 1.49x — a 2.28× gap.

The intuitive hypothesis is **occupancy**: 64 blocks on a 142-SM GPU is ~6%. To test it,
I varied precision and grid independently:

| Variant | Grid (blocks) | Accum precision | Speedup | Runtime | Correct |
|---|---|---|---|---|---|
| **A** per-row (shipped) | 64 | **fp64** | 0.65x | 9.88 ms | ✓ |
| **B** per-row | 64 *(same!)* | **fp32** | **1.61x** | 3.97 ms | ✓ (3/3) |
| **C** split-row | 8192 | fp32 | 1.31x | 4.88 ms | ✓ |
| CUDA noptx | 8192 | fp32 partials + fp64 final | 1.49x | 4.34 ms | ✓ |
| Triton | — | — | 1.60x | — | ✓ |

**The hypothesis is refuted by the data:**

- **B holds occupancy fixed at 64 blocks and only swaps fp64→fp32 — yet jumps
  0.65x → 1.61x (2.5×).** So the gap was the precision, not the block count.
- **C adds 128× more blocks (8192) but is *slower* than B** (1.31x vs 1.61x). Adding
  parallelism *hurt*. So occupancy was never the bottleneck.

**Mechanism.** AD102 (RTX 6000 Ada) runs fp64 at **1/64 the fp32 rate** (consumer-class
FP64). The per-row reduction performs `N/TH ≈ 16384` fp64 mul-adds per thread plus
`TH = 256` fp64 atomic-adds to shared memory per block — and that fp64 arithmetic, not
memory, dominated the 9.88 ms. CUDA never paid this: it accumulates thread-locals in
**fp32** and uses fp64 only for the single per-block global `atomicAdd`. Once TileLang
also uses fp32, the per-row kernel is **memory-bound and already near roofline**: each
thread issues ~16384 *independent* loads, giving enough memory-level parallelism to hide
HBM latency even with only 64 resident blocks — which is exactly why splitting the row
(C) only added atomic contention (128-way into 64 accumulators) and a second kernel
launch, making it slower.

**Why per-row fp32 (1.61x) even beats CUDA's split-row (1.49x):** with a 4.19M-element
row, one block per row still has abundant work and perfectly streaming, fully-coalesced
access, while paying **zero** atomic / multi-kernel overhead. The split strategy is the
right call for *short* rows (where one block can't fill the GPU); here the rows are huge,
so the simplest mapping wins.

**Correctness note.** The earlier belief that "fp32 fails layer_norm" was a
**misdiagnosis** — that failure was an unrelated ndim bug (the 3D `weight` wasn't
flattened), not precision. fp32 holds the 1e-4 relative tolerance here (verified 3/3
runs) because LayerNorm inputs are well-conditioned (O(1)). For pathologically large or
ill-conditioned N, variant **C** (split-row fp32, short ~128-element chains) is the
robust fallback at 1.31x.

**Fix shipped:** `layer_norm/tilelang` fp64 → fp32 (one-line change) ⇒ **0.65x → 1.61x**,
now the fastest layer_norm of all four abstractions.

**Follow-up (AKO optimization pass).** The two CUDA layer_norm tracks were still at ~1.47–1.49x
(the 3-launch split-row design). The optimization loop lifted both to ~1.6x — but, instructively,
via **two opposite structural routes**, not one shared lever:

- `cuda_noptx` **1.49 → 1.61x** by **fusing** to one per-row **fp32** kernel (one block per row,
  block-reduce mean/rstd in shared memory, apply in the same launch), dropping the separate
  `ln_final` launch and the global mean/rstd round-trip.
- `cuda_unlimited` **1.47 → 1.60x** by **keeping the split-row layout** (more blocks → better
  stats-pass occupancy; the one-block-per-row fusion was profiled and *ruled out* as
  occupancy-starved — 64 blocks underutilize 142 SMs and still read `x` twice) and instead
  **column-blocking the apply pass** so each thread owns 4 columns and fetches `w`/`b` once,
  reusing them across all 64 rows from registers.

So at that stage all four abstractions agreed at ~1.6x, and I attributed it to a shared **irreducible
~3.9 ms HBM read floor**. **That attribution was wrong — see § "Floors revised (deeper AKO pass)" below.**
The ~3.9 ms assumed a *fixed 3-pass* traffic model (read `x` twice + write `y`); a later pass cut the
second `x`-read from HBM to L2 (per-row L2 residency) and reached **~2.1x** on three of the four DSLs.
What *does* survive from this analysis is the constant lever — fp32 (never the 1/64-rate fp64)
accumulation — and that the fused-vs-split structural choice is **regime-dependent**. See § "Floors
revised" and the optimization-pass section of `RESULTS.md`.

---

## Gap 2 — scatter: the cause **is** occupancy (with a small launch/codegen residual)

Deterministic last-wins scatter (`dim=1`): `x(64, 8192)`, `idx(64, 4096)`,
`updates(64, 4096)` ⇒ **Rr = 64 rows, W = 8192, K = 4096**. The shipped TileLang kernel
mapped one block per row (`T.Kernel(Rr)` = 64 blocks) for both the atomicMax winner pass
and the gather pass. It scored 3.80x vs CUDA's 6.46x.

Unlike layer_norm there is **no fp64 here** (int32 atomics + fp32 copy), so I tested the
occupancy hypothesis directly by tiling the inner dimension across blocks:

| Variant | pass1 / pass2 grid | Speedup | Runtime |
|---|---|---|---|
| per-row (shipped) | 64 / 64 blocks | 3.80x | 0.0445 ms |
| **element-tiled** | **1024 / 2048 blocks** | **5.88x** | 0.0311 ms |
| CUDA unlimited | 1024 / 2048 (grid-stride) | 6.46x | 0.0277 ms |
| Triton | — | 5.31x | — |

**Confirmed.** Tiling K by 16 and W by 32 (matching CUDA's element-parallel grid of
1024/2048 blocks) lifts TileLang **3.80x → 5.88x**, closing the gap from 1.73× to 1.10×.
Here scatter is **compute-light** (one atomicMax or one indexed copy per element), so it
genuinely needs thousands of blocks to fill the SMs — 64 blocks touch <half the machine
and stall. This is the opposite regime from layer_norm, where each block already had
enough independent work.

**The ~12% residual** (0.0311 vs 0.0277 ms) is the launch/codegen tax the small kernel
size makes visible: three TileLang kernel launches (WIN `full(-1)` init + pass1 + pass2)
plus TileLang's `T.Parallel(chunk)` codegen and the `if_then_else` gather, versus CUDA's
leaner `red.global.max.s32` (a no-return reduction-atomic) and grid-stride loop. At a
~31 µs kernel, fixed per-launch overhead is a real fraction — this residual is *not*
cleanly separable without a profiler (`ncu` is unavailable on this host), so we report it
as "launch + codegen" rather than over-attributing.

**Fix shipped:** `scatter/tilelang` element-tiled ⇒ **3.80x → 5.88x**. *(Shape caveat: the
tiled kernel hardcodes KS=16 / WS=32 and indexes `K//KS`, `W//WS`, so it assumes K and W
divide evenly by the tile — true here (4096/16, 8192/32) and the kernels are shape-cached, but
unlike the original per-row `T.Parallel(K)` it is not general for arbitrary shapes.)*

---

## Two distinct pitfalls, one shared context

It is tempting to unify these as "the per-row mapping starves a 142-SM GPU" — but the
**layer_norm data refutes that**: variant B is per-row *and* the fastest of all variants
(1.61x, beating CUDA's split-row). For layer_norm the per-row mapping is **optimal**. So the
two gaps are **not** one root cause; they are independent bugs that merely share a context
(this benchmark's small batch = 64, which is what made each one *visible*):

> - **scatter — a grid/occupancy mistake.** Compute-light, so it genuinely needs thousands
>   of blocks; mapping the 64-row batch to the grid starved the SMs. The *mapping* was wrong;
>   element-tiling the inner dim fixes it (3.80x → 5.88x).
> - **layer_norm — a dtype mistake.** Memory-bound with huge rows, so one block per row is
>   the *right* mapping (a single fused launch, no global atomics; each thread's ~16384
>   independent loads saturate HBM even at 64 blocks). The gap was an unrelated **precision**
>   choice — fp64 accumulators at 1/64 rate. fp32 per-row works and is fastest, so the
>   per-row layout never "forced" fp64; that was an independent — and misdiagnosed — decision.

Two separate bugs ⇒ two separate lessons for porting to a tile DSL on this hardware:
(1) **don't map a small batch dim to the grid for compute-light ops** — tile the inner dim;
(2) **keep reduction accumulators in fp32** on AD102 unless a *short-chain* precision
argument truly demands fp64. The diagnostic that unifies them is the *method*, not the
cause: occupancy and precision both present as "TileLang is slow" until you vary one factor
at a time.

### What did NOT have a large gap (and why)
- **group_norm** (1.17× spread at the time): I called all four "~roofline against torch on 8.6 GB
  tensors ... the residual is codegen, not structural." **That was wrong — see § "Floors revised."**
  The "roofline" assumed a *fixed 3-pass* traffic; an L2-reuse chunk pipeline (K=4 groups kept
  L2-resident) cut it 3-pass→2-pass, lifting three of the four DSLs to **~1.3x**. The headroom was
  **structural**, not codegen.
- **gather** (1.14× spread at the time): I called it "a 24 µs latency-bound indexed load; differences
  are launch noise." **Also headroom, not noise** — a cold-L2 diagnostic later showed the scattered
  read sustains only ~50% of DRAM BW. Shared-memory row staging (CUDA: coalesced load → on-chip
  gather) and eviction-policy + vectorization tuning (Triton, TileLang) lifted all four DSLs to
  **~1.31–1.51x**. See § "Floors revised."
- **cumsum / activations / lstm** (≤1.05×): bandwidth-roofline or cuDNN-floored — all DSLs converge.

---

## Independent review (partial)

Two independent reviewers re-derived each conclusion from the source + grid arithmetic
(no shared reasoning) and both returned **supports**, each sharpening a residual (below).
**Caveat on what this covers:** these two were *residual-explainers* — prompted to assume the
headline claim and account for the leftover — so they are confirmatory by construction, not
adversarial. The two genuinely adversarial passes I queued (a refuter tasked to break each
claim, and a completeness/thesis critic) **stalled on a hung web-search and never returned**,
so the adversarial check did not complete. The empirical backing for the two causes is the
**controlled single-variable experiments** in Gaps 1–2, not this review.

- **layer_norm (fp64, not occupancy) — confirmed.** LayerNorm here is **DRAM-bound**:
  ~3.0 GiB of unavoidable traffic / 960 GB/s ⇒ a ~3.3 ms floor; per-row fp32 (B) hits
  3.97 ms = **~83% of peak bandwidth**. The decisive point: B is the **lowest-occupancy**
  variant (64 blocks) yet the **fastest** — that alone refutes occupancy. B also wins by
  being a **single fused launch** (stats + apply in one `T.Kernel`, only intra-block shared
  atomics), whereas CUDA/split-row pay a kernel boundary + global atomics with **128-way
  contention** into 64 accumulators (8192 blocks / 64 rows), plus (C) a `torch.zeros(M)`
  every forward. Ordering 3.97 (B) < 4.34 (CUDA) < 4.88 (C) tracks fused-1-launch <
  split-C++ < split-TileLang+allocs.
- **scatter (occupancy primary; residual = launch/codegen) — confirmed.** At the tiled
  point the grids are **byte-identical** to CUDA (1024 / 2048 blocks, 1 element/thread), so
  the ~12% residual is host-side, not occupancy. Concrete residual components found:
  (1) the TileLang `Model` casts `idx.to(int32)` → an **extra 262144-element cast kernel**
  (4 launches vs CUDA's 3, which reads `int64` directly); (2) TileLang's per-call JIT
  wrapper vs `load_inline`'s thin pybind; (3) `atomicMax` (returns old) vs the guaranteed
  fire-and-forget `red.global.max.s32` — though `-O3` *may* downgrade the unused-return
  `atomicMax` to a `RED` in SASS, so this component is compiler-dependent. Identified but
  **not** chased (sub-µs, scope): dropping the `int32` cast (read `int64` in-kernel) would
  remove one launch. The review also caught a stale comment in `scatter/cuda_unlimited`
  ("vectorized .nc loads") — the gather pass is scalar; comment corrected.

Both reviewers' caveats agree: without a profiler (`ncu` unavailable) the residual
*components* are attributed by static code+grid+arithmetic analysis, not per-kernel measured
durations — so the residual is reported as a "launch + codegen" bucket, not over-split.

## Method
Each cause was isolated with a **single-variable controlled experiment** through the same
`AKO4ALL/bench/kernelbench/bench.py` harness (`--num-warmup 200`, relative-1e-4 correctness,
`--deterministic` for scatter) on one dedicated GPU. Decomposition variants live in the
session tmp dir; the two fixes are shipped into the live `tilelang` solutions and re-benched
(`layer_norm/tilelang` 1.61x, `scatter/tilelang` 5.88x). The proof is the controlled
experiments, not opinion; two independent residual-explainer reviews concurred (the
adversarial pass stalled — see "Independent review (partial)"). See `RESULTS.md` for the
full table.

---

## Floors revised (deeper AKO pass)

A later, deeper optimization pass (toward ~10 iterations/cell) **falsified two "floors" this
document asserted.** Both were reasoned from roofline arithmetic — and both were wrong for the
*same reason*: they measured a **3-pass** HBM traffic model and called it irreducible, when a unit
of work small enough to stay resident in the 96 MB L2 lets a cooperative/pipelined kernel serve the
**second read from L2 instead of HBM**, cutting traffic 3-pass → 2-pass.

| Op | This doc claimed | Actually | Mechanism |
|---|---|---|---|
| **layer_norm** | "irreducible ~3.9 ms / ~1.6x floor" (§ Gap 1) | **~2.1x** (triton 2.10, cuda_unlimited 2.16, tilelang 2.10) | per-row L2 residency: a 16 MB row stays in 96 MB L2, so the apply pass re-reads it from L2 → 3-pass→2-pass |
| **group_norm** | "~roofline 0.9x, residual is codegen not structural" (§ "What did NOT have a large gap") | **~1.3x** (triton 1.35, cuda_unlimited 1.28, tilelang 1.32) | L2-reuse chunk pipeline: K=4 groups (32 MB) stay L2-resident across the stats→normalize pair → 3-pass→2-pass |
| **gather** | "24 µs latency-bound, differences are launch noise" | **~1.31–1.51x** (all four DSLs) | shared-mem row staging (CUDA: coalesced load → on-chip gather) *or* eviction-policy + vectorization tuning (Triton, TileLang); a cold-L2 probe showed the scattered read sustains only ~50% DRAM BW |
| **scatter** residual | "int32-cast removal … identified but **not** chased" (§ Independent review) | chased | tilelang/triton drop the `idx.to(int32)` cast (read int64 in-kernel, one fewer launch) → 6.02→7.63x / 5.31→6.79x; cuda_unlimited fuses to a single kernel → 6.46→**10.0x** |

**The new 2-pass floors are still just reasoned floors — state them with the humility the prior ones
earned.** This document has now called an *algorithm* floor a *physical* floor **more than once**
(layer_norm's 3.9 ms; group_norm's "roofline"; and the original scatter framing implicitly assumed a
fixed traffic model too). So the current claims — layer_norm/group_norm are "at the **2-pass** floor,
~85–96% of the ~960 GB/s AD102 ceiling" — are the *best current model*, not proven physics. `ncu`
is still unavailable on this host; the 2-pass floors rest on the same bench.py controlled experiments
and roofline arithmetic as the (now-revised) 3-pass ones did. If a future pass finds a way below 2
passes (it would have to avoid reading the input twice at all — e.g. a Welford single-pass streaming
variant that trades numerical margin), these too should be revised.

**Honest asymmetry:** the L2-residency lever landed on **3 of 4 DSLs** for each of layer_norm and
group_norm. `cuda_noptx` did **not** find either L2 pipeline in its iterations — `layer_norm/cuda_noptx`
stayed at the old 3-pass floor (~1.64x, +1.9%) and `group_norm/cuda_noptx` never beat its 0.917x
baseline and was restored verbatim. That the *same* lever was available in plain CUDA but wasn't
discovered there is a property of the search, not the hardware — a caution against reading a
per-DSL number as a per-DSL *ceiling*.

The empirical backing, as before, is the controlled single-variable experiments in each changed cell's
`ITERATIONS.md` (and an agent-free re-bench of the exact committed bytes, one 4-GPU batch,
`COMPILED=CORRECT=True` for all 16 changed cells), not this narrative.
