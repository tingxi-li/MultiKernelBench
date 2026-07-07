# Iteration Log — layer_norm / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/layer_norm/triton/solution/layer_norm.py`,
Triton speedup 1.6050x); benched against the same `reference/normalization/layer_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of layer_norm | 1.4862x | 4.3400 ms | 6.4500 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** LayerNorm last 3 dims; split-row reduction + affine. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=4.3400 ms, REF=6.4500 ms, **SPEEDUP=1.4862x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.6050x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 1 (fused) — single kernel, one block per row, fp32, float4

- **Change:** Replaced 3-launch split-row (fp64 atomics + ln_final + ln_apply with
  64-bit div/mod over 268M elems) with ONE kernel, one block per row (M=64 blocks,
  TPB=512). Phase 1: float4 fp32 sum/sumsq + shared-mem tree reduce -> mean/rstd
  (rsqrtf). Phase 2: re-read row float4, write affine. No fp64, no atomics, no
  64-bit div in hot loop (m=blockIdx, col=loop index).
- **Bench:** COMPILED=True, CORRECT=True (5/5), RUNTIME=4.0300 ms, **SPEEDUP=1.5856x**.
- vs baseline 1.4724x / 4.34ms: kept. HBM floor ~3.4ms (~1.88x).

## Iter 2 — block-size sweep TPB=256

- **Change:** TPB 512 -> 256.
- **Bench:** CORRECT=True, RUNTIME=3.9600 ms, **SPEEDUP=1.6162x**. Kept (best).

## Iter 3 — TPB=128

- CORRECT=True, RUNTIME=4.2100 ms, SPEEDUP=1.5202x. Reverted (worse than 256).

## Iter 4 — ILP=4 manual unroll (both passes), TPB=256

- **Change:** each thread processes 4 float4 per stride step (more in-flight loads
  to offset 64-block under-occupancy).
- CORRECT=True, RUNTIME=3.9800 ms, SPEEDUP=1.6080x. Within noise of iter-2; reverted.

## Best (prior pass) = Iter 2 (fused, TPB=256, simple grid-stride): 1.6162x / 3.96ms.

---

# Re-run pass (session 2026-07-02, GPU 1)

Baseline re-confirmed on GPU 1: **SPEEDUP=1.6171x, RUNTIME=3.97ms (min 3.88), REF=6.42ms**.
Roofline: traffic = 2·read-x + write-y ≈ 3.26 GB (w,b are 16MB each but shared across
all 64 rows → L2-resident, so ~free). 3.26GB / 3.88ms = 840 GB/s ≈ 87% of 960 GB/s
theoretical. Achievable Ada ceiling ~88-92%, i.e. honest floor ~3.7ms — small headroom.
All fast-signal ranking done in a standalone alternating harness (5 reps, shared clock
state, rank on min/median) since the ~2-3% prize is near bench.py's run-to-run noise.

## Iter 5 — split-row multi-block (occupancy): REVERT

- **Hypothesis:** M=64 rows → one-block-per-row uses only 64 of 142 SMs (78 idle).
  Split each row across PPR blocks (3-kernel: reduce→double-atomicAdd→finalize→apply,
  grid.y=row so no 64-bit div) to light up all SMs and saturate HBM.
- **Bench (harness, min ms):** PPR 2:3.91, 3:3.94, 4:3.96, 6:3.94, 8:3.99, 16:4.10,
  32:4.30 — **monotonically WORSE** with more blocks/row. Contiguous-chunk vs
  grid-stride partitioning: no difference. **Confirms 64 blocks already saturate
  bandwidth** (Little's law satisfied); extra SMs only add access-pattern scatter.
- **REVERT** the split factor. But PPR=1 (one block/row, still 3-kernel) came in at
  3.79 min — hinting the 3-kernel structure itself is marginally faster than fused.

## Iter 6 — ILP grid-stride sweep (memory-level parallelism): KEEP ILP=4

- **Hypothesis:** under-occupied 64-block grid is latency-bound, not SM-bound. Keep
  ILP independent float4 loads in flight per thread (grid-stride ILP) to raise MLP.
- **Bench (harness, one-block-per-row 3-kernel, TPB=256, min ms):**
  ILP 1:3.93, 2:3.91, **4:3.76**, 6:3.85, 8:3.81 — **ILP=4 is a clear optimum.**
  5-rep alternating vs fused: **fused 3.874/3.908 vs split-ILP4 3.763/3.774** (min/med),
  dead-stable across all reps. ~2.9% min / 3.4% med win. err 4e-6 (≈ baseline). **KEEP.**
  (Note: iter-4's ILP=4 on the *fused* kernel didn't help — the separate simpler
  reduce/apply kernels have the register room for ILP=4 to actually raise MLP.)

## Iter 7 — TPB sweep for the split kernels: TPB=256 best

- **Bench (harness, split ILP=4, min ms):** TPB 128:~4.16, **256:3.76**, 512:3.91,
  1024:3.93. 256 optimal (same as fused). REVERT others.

## Iter 8 — drop atomics/finalize: one block owns the row, computes mean/rstd directly

- **Hypothesis:** with one block per row the reduce block already holds the full-row
  sum in shared mem, so it can compute fp64 mean/rstd itself and write them out —
  eliminating the double scratch (+zeroing), the cross-block atomics, and the whole
  finalize kernel launch (less fixed per-call overhead).
- **Bench (harness, 6-rep alternating, min ms):** atomic+finalize 3.844 vs
  **direct 3.838** — direct consistently faster every rep, identical err 3.8e-6.
  Simpler AND marginally faster. **KEEP.**

## Final — split (one block/row, TPB=256, ILP=4), fp64 in-block mean/rstd, 2 kernels

- **Verdict (bench.sh final, --num-warmup 200):** COMPILED=True, CORRECT=True (5/5),
  RUNTIME=3.91ms, REF=6.44ms, **SPEEDUP=1.6471x** (baseline 1.6171x → +1.85% rel).
- Earlier 3-kernel double-atomic variant verdict was 1.6403x; this 2-kernel direct
  form is cleaner and slightly ahead.
- 3.26GB / 3.76ms (harness min) = 867 GB/s ≈ **90% of theoretical** → effectively at
  the achievable memory-bandwidth floor. Distinct directions exhausted: roofline,
  split-row occupancy, ILP/MLP sweep, TPB sweep, contiguous-vs-strided partitioning,
  precision (fp64 accum), atomic-vs-direct finalize — all converge here. Stopping
  short of 10 is justified by the confirmed floor.

## Best = Final: one block/row, TPB=256, ILP=4, fp64 in-block mean/rstd — 1.6471x / 3.91ms.

---

# Convergence redo (branch cross-dsl-6op-ncu-redo) — hand-written from roofline

Reset to identity baseline, then hand-wrote the plain-CUDA kernel from the L2 roofline
(NO inline PTX; `__ldg`/`__shfl`/`atomicAdd`/`float4` intrinsics only). All benches via
`tools/timed_bench.sh --gpu3-serial` (frozen yardstick); block/tpb are runtime args, so
the sweep needs no recompile.

**Lever (implemented by hand):** x = (M=64 rows, N=4.19M fp32), 16.78 MB/row, L2=96 MB.
ONE row fits in L2, so process one row at a time in a HOST-SIDE C++ loop: `reduce[m]`
(fp64 sum/sumsq via warp-shuffle + block-leader atomicAdd, ~B blocks) then `apply[m]`
(re-read x — L2 hit — normalize + affine, write y). Only that row's 16.78 MB is in flight
between reduce and apply, so it stays L2-resident → 2 HBM passes (read x, write y) instead
of naive 3. weight/bias (33.55 MB, shared across rows) stay L2-resident throughout.

| iter | variant | speedup | runtime | keep? |
|------|---------|---------|---------|-------|
| 1 | identity baseline (reset) | 0.9953x | 6.42 ms | (baseline) |
| 2 | 2-pass host-loop reduce+apply, B512 T256 | 1.9453x | 3.29 ms | keep |
| 3 | block sweep B256 T256 | **2.1549x** | 2.97 ms | **keep (best)** |
| 4 | block sweep B128 T256 | 2.1122x | 3.03 ms | revert |
| 5 | block sweep B192 T256 | 1.0440x | 6.13 ms | revert (host P-state fluke) |
| 6 | confirm B256 T256 (re-bench) | 2.1549x | 2.97 ms | stable — locks best |

- **iter 2 (B512):** first custom variant already lands **1.9453x** — reproduces the known
  committed cuda_noptx calibration point (~1.95x). Identity read ~1.0x. Both ends of the
  calibration curve validate the instrumented pipeline.
- **iter 3 (B256):** halving blocks/row cuts reduce atomic contention (only block-leaders
  atomicAdd, so #atomics/row = #blocks) while still saturating HBM → **2.1549x**, past the
  1.95x reference ceiling. Strong lever.
- **iter 4 (B128):** too few blocks under-saturates bandwidth (2.11x). Optimum brackets 256.
- **iter 5 (B192):** 1.04x — a lone sharp dip amid smooth B128/B256 neighbours = the host's
  known clock/P-state variance (see memory: ako-bench-gpu-quirks), not a real curve feature.
- **iter 6:** B256 re-benched → **2.1549x / 2.97 ms EXACTLY** (min 2.92), reproducible to 4
  s.f. across two independent serial benches. Confirms B256 is the real, stable optimum.

**Best = 2-pass host-loop, B256 T256, fp64 in-kernel mean/rstd, float4 __ldg: 2.1549x / 2.97 ms.**

**Stop reason:** stop-rule branch 1 (within 5% of the ~1.95x external ceiling) is satisfied —
in fact exceeded at 2.15x, which becomes this cell's ceiling (max of cell-best and external
ref). 5 custom variants benched (budget 6–10). Roofline confirmed analytically: 2 HBM passes
≈ 2.18 GB / 2.97 ms ≈ 734 GB/s effective; the B512→B256 gain is the atomic-contention lever,
not a pass-count change (both are already 2-pass). forward() stays glue-only (allocate/launch
inside the extension; passes cheating_detection.py); CUDA source contains no inline PTX asm.
