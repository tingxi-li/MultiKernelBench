# Iteration Log — layer_norm / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/layer_norm/triton/solution/layer_norm.py`,
Triton speedup 1.6050x); benched against the same `reference/normalization/layer_norm.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of layer_norm | 1.4713x | 4.3500 ms | 6.4000 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** LayerNorm last 3 dims; split-row reduction + affine. Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=4.3500 ms, REF=6.4000 ms, **SPEEDUP=1.4713x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.6050x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 1 (this run) — per-row-segment apply, float4 + v4 streaming store

- **Change:** rewrote ln_apply to grid=M*S (m,woff from blockIdx, no per-element 64-bit divide),
  hoisted mean/rstd to registers, float4 __ldg on x/w/b + inline-PTX `st.global.cs.v4.f32` store.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=4.7000 ms, REF=6.42, **SPEEDUP=1.3660x**.
- **Verdict: REVERT (regressed vs baseline 4.34ms/1.477x).** float4 + v4 streaming store hurt;
  likely register pressure (3 float4 in flight + v4 store) cut occupancy on this memory-bound pass.

## Iter 2 — per-row-segment scalar apply (divide-free)
- Rewrote ln_apply to grid=M*S, no per-element 64-bit divide, scalar streaming store.
- Bench: CORRECT=True, RUNTIME=4.4400 ms, **SPEEDUP=1.4482x**. Verdict: **REVERT** (ties baseline;
  micro-bench confirmed the per-element divide was NOT the bottleneck — memory latency hides it).

## Iter 3 — column-blocked float4 apply (w/b reuse across rows)  ✅ KEEP
- **Root cause found by micro-bench:** copy floor (x->y, 2GB) = 2.60 ms; baseline apply = 2.92 ms;
  the 0.31 ms gap is redundant w/b reads (same w/b refetched for every one of the M=64 rows).
- **Change:** ln_apply restructured column-blocked — each thread owns 4 cols (float4), loads w/b ONCE,
  sweeps all M rows reusing them from registers; mean/rstd staged in shared; v4 cache-streaming store.
  Apply micro-bench 2.92 -> 2.706 ms (~98% of copy floor).
- **Bench:** COMPILED=True, CORRECT=True (err 1.2e-7), RUNTIME=4.0300 ms, REF=6.43, **SPEEDUP=1.5955x**.
- Detector: valid=True (forward glue-only, kernel via _ext call). Verdict: **KEEP (new best, +8% vs baseline)**.

## Iter 4 — fp32 stats accumulators (rule-5 compliance) ✅ KEEP
- **Change:** sum_acc/sq_acc fp64->fp32, atomics fp32, ln_final in fp32 (rsqrtf). Micro-bench
  confirmed stats is at the read-floor (1.205 ms / 831 GB/s) and fp32-vs-fp64 atomics are perf-identical.
- **Bench:** COMPILED=True, CORRECT=True (still passes 1e-4), RUNTIME=4.0300 ms, **SPEEDUP=1.5955x**.
- Verdict: **KEEP** — same speed as iter-3, now rule-5 compliant (no fp64 accumulators). Detector valid=True.
- **At floor:** total = stats 1.205 + apply 2.706 + final ~0 ≈ 3.91 ms; measured 4.03 ms. Both passes within
  ~2% of their HBM read/copy floors; no further bandwidth headroom on this 3 GB-traffic (x read x2 + y write) op.

---
## NEW RUN (2026-07-02) — the "3 GB floor" was wrong: L2-resident row-blocking cuts x-read to 1×

**Insight the prior run missed:** the 3 GB model (x read ×2 + y write) treats the second x-read as
mandatory HBM traffic. But one row = 4.19M f32 = **16 MB**, and this GPU's L2 = **96 MB**. If we
process **one row at a time**, ln_stats fills that row into L2 and ln_apply re-reads it *from L2*,
not HBM. True HBM floor = x-once (1 GB) + y-once (1 GB) + w/b (32 MB) = **~2.03 GB, not 3 GB**.
Baseline re-benched on GPU-3 this run: **min 3.94 ms / mean 4.03 ms / 1.5856x** (the KEEP to beat).

### Iter A — row-blocked L2 reuse, K=3 rows/block  ❌ REVERT
- Restructured host to a per-row-block loop (stats→final→apply per block of K rows, all in C++ so
  forward() stays glue-only). K=3 → working set 48MB x + 32MB w/b = 80MB.
- Fast bench: min **4.10 ms** (no better than baseline). Reuse NOT materializing at 80 MB.
- REVERT. Next: shrink working set — 80MB may be over L2's effective retention.

### Iter B — K=1 row/block (16MB x + 32MB w/b = 48MB)  ✅ KEEP (physics confirmed)
- Fast bench: min **2.92 ms / mean 3.15 ms**. Big jump — **L2 reuse is real** at a 48 MB footprint.
- KEEP. The x-from-L2 reuse cuts HBM traffic ~3GB→2GB exactly as predicted.

### Iter C — K=2 row/block (64MB)  ❌ REVERT
- Fast bench: min **4.18 ms** — reuse LOST again. Sharp L2-retention cliff: 48MB retains, ≥64MB evicts
  before apply runs (L2 replacement can't hold 64MB of this stride pattern alongside the y-write +
  w/b traffic). K=1 is the sweet spot. REVERT to K=1.

### Iter D — K=1, fold ln_final into ln_apply (drop a kernel/row)  ✅ KEEP (new best)
- ln_apply now finalizes mean/rstd from the partial sums inline (shared), removing the per-row
  ln_final launch: 192→128 launches over the full op.
- Fast bench: min **2.84 ms / mean 2.95 ms**, CORRECT. New best (~1.37× vs baseline 4.03 ms).
- Next: sweep S (stats block count — 128 blocks < 142 SMs underfills a row's stats), TPB, and
  consider a cooperative single-kernel to kill the remaining 128 launches.

### Iter E — S sweep (stats blocks/row): 128 vs 256 vs 512  ❌ REVERT (128 stays best)
- S=512: min **2.91 ms**; S=128 (incumbent): min **2.84 ms**. More/shorter stats blocks don't help
  — 128 blocks already saturate HBM read BW for a 16MB row; extra blocks add atomic/tail cost.
- REVERT to S=128.

### Iter F — cooperative single-kernel (grid.sync, one launch)  ❌ REVERT
- Rewrote as a persistent cg::grid_group kernel: per row grid-stride stats -> grid.sync ->
  grid-stride apply (x from L2) -> grid.sync. Eliminates all 128 launches; double-atomic reduce.
- Fast bench: COMPILED/CORRECT=True, min **3.24 ms** — REGRESSES vs 2.84 ms. The 192 grid.sync
  barriers + occupancy-capped grid (~1136 blocks vs 4096 for apply) cost more than the launches
  they remove; per-phase work between syncs is too small to amortize the barrier. REVERT.
- Verdict: the 128 stream launches are cheaper than grid-sync here; keep the multi-launch design.

### Iter G — TPB=512  ❌ REVERT
- Fast bench: min **2.94 ms** vs 2.84. Larger blocks (longer reduction, lower block count) slightly
  worse. REVERT.

### Iter H — TPB=128  ❌ REVERT
- Fast bench: min **2.98 ms** vs 2.84. Smaller blocks add reduction/atomic overhead. TPB=256 best.

### Iter I — apply grid: 1024 persistent blocks (grid-stride, ~4 cols/thread) vs 4096 (1 col/thread)  ❌ REVERT
- Fast bench: min **2.87 ms** — ties best within noise. No gain. REVERT to apply_blocks = N4/TPB.

## FLOOR CONFIRMATION
Best = **min 2.84 ms** (K=1 + folded-final). HBM traffic with L2 reuse = x-once (1GB) + y-write
(1GB) + w/b (32MB) ≈ 2.03 GB. At 2.84 ms that is **~715 GB/s** effective (~85–88% of this card's
achievable HBM BW). Five distinct directions — K-sweep (K=1 vs 2,3), S-sweep (128 vs 256,512),
TPB-sweep (128,256,512), cooperative single-kernel, and apply-grid sizing — all fail to beat 2.84 ms,
so this is the practical floor for the 2 GB-traffic (L2-reuse) model. The remaining ~0.6 ms vs the
2.25 ms ideal is read↔write bus turnaround (64 pure-read stats → pure-write apply transitions) +
128 launch/drain bubbles, neither removable without losing L2 reuse (overlap would evict x).

## FINAL VERDICT (best kept)
- **Config:** K=1 row-blocked, S=128 stats blocks/row, TPB=256, ln_final folded into ln_apply,
  float4 __ldg loads (x populates L2 in stats, reused from L2 in apply), inline-PTX st.global.cs.v4
  streaming store for y.
- **bench.sh final (--num-warmup 200, 100 trials):** COMPILED=True, CORRECT=True (5/5),
  RUNTIME=**3.00 ms** (mean; min 2.87 ms), REF=6.39 ms, **SPEEDUP=2.13x**.
- **vs committed baseline** (this run, same GPU: 1.5856x / 4.03 ms): **+34% faster**, and clears the
  Triton bar (1.605x) by a wide margin.
- Detector: valid=True, regression_type=None (forward() glue-only; compute in _ext kernel).
- The prior "at floor" claim assumed a fixed 3 GB traffic model; the real lever was L2 residency of
  a single 16 MB row inside 96 MB L2, cutting the second x-read from HBM to L2 (3 GB -> 2 GB).

---
## NEW RUN (2026-07-07) — cross-DSL 6-op ncu redo (from identity baseline, frozen timed_bench yardstick)

Started from the reset IDENTITY passthrough (convergence.csv iter 1 = 0.9938x). Kernel
re-derived from the roofline (NOT copied from prior/committed solution). Ceiling for this
calibration op ≈ 2.16x.

### Roofline (the lever)
x = (M=64 rows, N=4.19M fp32). One row = 16.78 MB; L2 = 96 MB. Reference does x-read×2 +
y-write + w-read + b-read ≈ 5.35 GB (torch keeps nothing resident: 2nd x-read from HBM,
w/b refetched once per row = 64 passes over the 16.78 MB arrays). The win: process ONE row
at a time so (a) the row stays L2-resident stats→apply (2nd x-read hits L2) AND (b) the
per-element affine w/b (16.78 MB each) stay L2-resident ACROSS all 64 rows. Working set per
row = x[m]+w+b = 50 MB < 96 MB → HBM floor ≈ x-once (1.07) + y-once (1.07) + w+b-once
(0.034) = 2.18 GB → ~2.1–2.3x.

### V1 (iter 2) — L2-resident 2-pass K=1, float4 __ldg + __stcs streaming store  ✅ KEEP (best)
- **Design:** per-row loop in the C++ launcher (128 launches = 64 stats + 64 apply, same
  stream → serialized so no overlap evicts the resident set). Stats: 128 blocks×256 t,
  grid-stride float4 __ldg, warp+shared reduce, atomicAdd → d_sum[m]/d_sq[m] (fp32).
  Apply: 4096 blocks×256 t, each block re-derives mean/rstd from the 2 scalars (ln_final
  folded in), float4 __ldg on x(from L2)/w/b(resident), **__stcs** streaming store for y
  (bypasses L2 so the 16.78 MB y-write doesn't evict the 50 MB working set).
- **Bench (timed_bench, --gpu3-serial, warmup 200):** COMPILED=True, CORRECT=True (5/5,
  fp32 tol 1e-4), RUNTIME=2.80 ms (min 2.76), REF=6.40 ms, **SPEEDUP=2.2857x**. KEEP (new best).
- **ncu (baseline+floor profile):** DRAM total = **1.692 GiB = 1.69 passes** (≤ 2-pass copy
  floor — actually beats it). ln_stats reads 1.001 GiB (x once from HBM); ln_apply reads only
  0.036 GiB from HBM at **99.1% L2 hit** (x re-read + w/b fully resident). 2-pass roofline
  CONFIRMED hit. ncu_key: passes=1.69.

### V2 (iter 3) — inline-PTX st.global.cs.v4.f32 store (vs __stcs intrinsic)  ❌ REVERT (tie)
- **Only** change vs V1: swap `__stcs` → hand-written `asm volatile("st.global.cs.v4.f32 ...")`.
  Everything else identical, isolating the store as a clean PTX-vs-intrinsic A/B.
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=2.80 ms (min 2.76), **SPEEDUP=2.2857x** —
  **byte-for-byte identical** to V1 (kept=0). REVERT to the simpler intrinsic.
- **PTX finding (deliverable):** inline PTX gave **ZERO** measurable win over intrinsics.
  `__stcs` already emits exactly `st.global.cs.v4.f32`, so the PTX store is redundant.
  Confirms the earlier study's finding. (Note: inline `ld.global.nc.v4` was NOT attempted —
  it hangs ptxas on this host; `__ldg` already emits the read-only-cache load anyway.)

### STOP — both stop conditions met
1. **Within 5% of ceiling:** 2.29x EXCEEDS the ≈2.16x ceiling (+6%). This cell is the
   fastest known winner and V1 reproduces/beats it.
2. **Roofline confirmed + stall:** ncu shows 1.69 passes (≤ 2-pass floor); apply is 99.1%
   L2-resident; V1 and V2 tie exactly (<3%, two consecutive). No bandwidth headroom left.
- An async/overlap software-pipeline was deliberately NOT pursued: overlapping row m+1's
  stats with row m's apply would pull x[m+1] into L2 and evict the resident x[m]/w/b set —
  destroying the very residency that IS the win. The serial per-row schedule is optimal here.
- **Final best:** V1 (float4 __ldg + __stcs streaming store, K=1 L2-resident 2-pass).
  forward() is glue-only (single _ext call; detector valid=True).
