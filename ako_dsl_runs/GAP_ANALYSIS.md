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

Both gaps trace to **one structural choice — the idiomatic TileLang mapping of "one
block per batch-row"** — but they fail through **two different mechanisms**, which the
benchmark's small batch dimension (M = Rr = 64) exposes. Each cause was isolated with a
controlled single-variable experiment (`--num-warmup 200`, same reference golden), not
assumed.

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

**Fix shipped:** `scatter/tilelang` element-tiled ⇒ **3.80x → 5.88x**.

---

## The unifying thesis

> **Idiomatic TileLang maps the batch dimension to the grid (one block per row); idiomatic
> CUDA grid-strides over flattened elements.** When the batch is small (64 here), the
> per-row mapping under-uses a 142-SM GPU — but the *symptom* depends on the op's arithmetic
> intensity:
> - **scatter** (compute-light): 64 blocks literally starve the SMs → **occupancy** gap.
> - **layer_norm** (memory-heavy, huge rows): 64 blocks are *fine* (per-thread MLP hides
>   latency), but the per-row mapping forced a whole-row reduction → long accumulation
>   chains → an **fp64** choice → a 1/64-rate arithmetic tax.

The same root structural choice; two distinct failure modes; both invisible until you vary
one factor at a time. The lesson for porting to a tile DSL on this hardware: **don't map a
small batch dim to the grid**, and **keep reduction accumulators in fp32** unless a
*short-chain* numerical argument demands otherwise.

### What did NOT have a large gap (and why)
- **group_norm** (1.17× spread): all four are ~roofline against torch's own tuned
  GroupNorm on 8.6 GB tensors; TileLang's slightly-lower 0.85x is the same per-(batch,group)
  mapping but here NG = 1024 blocks, so occupancy is fine — the residual is codegen, not structural.
- **gather** (1.14× spread): a 24 µs latency-bound indexed load; differences are launch noise.
- **cumsum / activations / lstm** (≤1.05×): bandwidth-roofline or cuDNN-floored — all DSLs converge.

---

## Method
Each cause was isolated with a **single-variable controlled experiment** through the same
`AKO4ALL/bench/kernelbench/bench.py` harness (`--num-warmup 200`, relative-1e-4 correctness,
`--deterministic` for scatter) on one dedicated GPU. Decomposition variants live in the
session tmp dir; the two fixes are shipped into the live `tilelang` solutions and re-benched
(`layer_norm/tilelang` 1.61x, `scatter/tilelang` 5.88x). See `RESULTS.md` for the full table.
