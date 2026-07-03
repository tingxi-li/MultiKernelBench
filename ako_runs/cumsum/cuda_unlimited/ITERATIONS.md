# Iteration Log — cumsum / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/cumsum/triton/solution/cumsum.py`,
Triton speedup 1.2264x); benched against the same `reference/math/cumsum.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of cumsum | 1.2130x | 10.8000 ms | 13.1000 ms | correct |

## Iter 1 — cuda_unlimited port

- **Hypothesis:** row cumsum dim=1; chunked scan with carry. Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=10.8000 ms, REF=13.1000 ms, **SPEEDUP=1.2130x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2264x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 2 — register-resident segment + warp-shuffle block scan
- **Change:** per-thread segment scan kept in registers (shared traffic 32r+32w -> 16r+16w);
  inter-thread Hillis-Steele (16 syncs) replaced by warp `__shfl_up_sync` inclusive scan +
  8-way broadcast combine (2 syncs). Accumulation order byte-identical (fp32 segment-sequential).
- **Bench:** COMPILED=True, CORRECT=True, RUNTIME=10.8000 ms, **SPEEDUP=1.2130x**.
- **Verdict:** 0% vs baseline -> scan/shared was NOT the bottleneck. Kernel is DRAM-bound
  (~740 GB/s achieved of ~960 peak). Keep (neutral); pursue occupancy next.

## Iter 3 — occupancy: CHK 4096 -> 2048 (EPT 8)
- **Change:** halve chunk/shared (16KB->8KB buf) to raise resident blocks/SM.
- **Bench:** CORRECT=True, RUNTIME=10.7000 ms, **SPEEDUP=1.2150x**. ~1% (noise) over baseline. KEEP (best).

## Iter 4 — streaming vectorized load (ld.global.cs.v4) instead of __ldg
- **Change:** symmetric evict-first load to match streaming store.
- **Bench:** CORRECT=True, RUNTIME=10.9000 ms, **SPEEDUP=1.2018x**. Slightly WORSE than cached __ldg. REVERT.

## Iter 4b conclusion (superseded) — thought AT FLOOR
Prior run stopped at 1.2150x believing DRAM-bound at ~750-800 GB/s was the floor.
Re-opened below: the shared round-trip was leaving ~3% on the table.

---

## RUN 2 (2026-07-02, GPU1) — reopened; found real headroom

My-GPU baseline (committed iter-3 kernel, bench.sh, warmup 200): **RUNTIME 10.7ms mean
(10.5 min), SPEEDUP 1.2056x**. Local harness (warmup 200, 50 trials) reads it as **10.51ms
(±0.003), 817 GB/s** — used for fast ranking; deltas are same-harness so clock noise cancels.

### Iter 5 — establish the true BW ceiling (2 distinct copies)
- **Hypothesis:** if a pure copy can't beat our cumsum, we're at the memory floor.
- **grid-stride float4 copy** (best of a block/thread sweep): 32768 blk x 512 thr = **813 GB/s (10.57ms)**.
- **structure-matched row-copy** (1 block/row, CHK=2048, EPT=8, float4 load+streaming store, NO scan):
  **823 GB/s (10.43ms)**. So the row layout's copy ceiling (EPT=8) ~= our cumsum baseline — scan cost ~0.
- **Verdict:** memory-bound confirmed, BUT the EPT=8 copy ceiling is layout-dependent (see iter 7).

### Iter 6 — register-direct: DROP the shared-memory data round-trip
- **Hypothesis:** the baseline stages every element load->shared->scan->shared->store (2 extra
  syncs + shared BW). If each thread instead loads its CONTIGUOUS segment straight into registers
  as float4 and stores from registers, shared holds only the 4 warp-totals — fewer syncs, and
  4 independent 128-bit loads/thread expose more memory-level parallelism.
- **Change:** register-direct load/store; shared = wsum[TT/32] only.
- **Bench (EPT=8, TT=256):** 828-830 GB/s (10.36ms) vs 817 baseline. **~1.4% faster. KEEP direction.**

### Iter 7 — elements-per-thread (ILP) sweep — the real lever
- **Hypothesis:** more float4 loads in flight per thread -> higher achieved BW, until register
  pressure kills occupancy. CHK must divide N=32768 -> EPT must be a power of 2.
- **Bench (TT=256, streaming store):** EPT 4=815, 8=829, **16=840**, 32=831, 64=771 GB/s.
  EPT=16 (CHK=4096) is the peak; EPT>=32 spills registers and regresses.
- **Verdict:** EPT=16 register-direct = **840 GB/s / 10.223ms** (3x repeats, ±0.001). **NEW BEST.**

### Iter 8 — block-width sweep at EPT=16
- **Bench (EPT=16, streaming, 3x each):** TT=128 -> **10.211ms (841 GB/s)**, TT=256 -> 10.223ms,
  TT=512 -> 10.281ms, TT=1024 (EPT8) -> 10.46ms. TT=128 marginally best (fewer threads/block ->
  more resident blocks/SM). KEEP TT=128.

### Iter 9 — streaming (st.global.cs.v4) vs cached (plain float4) store at EPT=16
- **Bench:** cs=10.211ms vs cg=10.215ms — statistically equal (output never re-read). Keep the
  documented **streaming** store (st.global.cs.v4) per the DSL gotchas.

### Iter 10 — __launch_bounds__(128) on the winner
- **Bench:** 10.242ms — no help (slightly worse). REVERT. Occupancy is not the limiter.

## Conclusion — NEW BEST, near-floor
Best kept = **register-direct, TT=128, EPT=16 (CHK=2048), warp-shuffle scan, streaming float4 store:
10.211ms / 841 GB/s** vs committed baseline 10.51ms / 817 GB/s (same harness) = **+2.9%**.
841 GB/s is **87.6% of the 960 GB/s HBM peak** and *exceeds* both the grid-stride (813) and the
EPT=8 row-copy (823) ceilings — the ILP win lifts a copy-class kernel above a naive copy. Remaining
gap to peak is the irreducible scan/sync + strided-segment L2 overhead. Cumsum reads all N + writes
all N (8.59 GB fixed traffic); no traffic-reducing trick exists. Confirmed at floor via copy ceiling,
ILP saturation (EPT>=32 regresses), width sweep, and store-mode equivalence.
