# Iteration Log — cumsum / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/cumsum/triton/solution/cumsum.py`,
Triton speedup 1.2264x); benched against the same `reference/math/cumsum.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of cumsum | 1.1944x | 10.8000 ms | 12.9000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** row cumsum dim=1; chunked scan with carry. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=10.8000 ms, REF=12.9000 ms, **SPEEDUP=1.1944x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2264x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 1 (re-bench) — baseline re-bench: 10.8ms, 1.2130x (C=4096/TH=256)

## Iter 2 — C=8192 / TH=512
- Halves chunks (8→4) and syncs. RUNTIME=10.9ms, SPEEDUP=1.2018x, CORRECT=True.
- Within noise / slightly worse than baseline. REVERT.

## Iter 2b — C=8192 / TH=256: 10.9ms, 1.2018x, CORRECT. Within noise/worse. REVERT.
## Iter 3 — C=4096 / TH=1024: 23.4ms, 0.5598x. Occupancy collapse. REVERT.

## Iter 4 — C=4096 / TH=128: 10.8ms, 1.2037x, CORRECT. Identical to baseline (noise). REVERT.

## Conclusion (prior run): AT FLOOR
8.59 GB traffic / 10.8ms = 795 GB/s ≈ 83% of RTX6000-Ada ~960 GB/s peak.
Chunk/thread sweep {C=4096/8192, TH=128/256/512/1024} moved nothing >3% (TH=1024
regressed via occupancy collapse). Scan is HBM-bound. Best = baseline C=4096/TH=256
@ 10.8ms / 1.21x. Restored baseline verbatim.

---
# Re-audit run (GPU 2) — deeper floor confirmation via NEW distinct directions

Baseline re-benched on GPU 2 (full verdict, --num-warmup 200): RUNTIME=10.8 ms,
REF=13.1 ms, **SPEEDUP=1.2130x**, CORRECT=True. This is the bar to beat.

**Diagnostic (generated CUDA dump, `k.get_kernel_source()`):** the baseline already
emits **float4 (128-bit) vectorized loads AND stores** for both the X→shared copy and
the shared→Y write. The chunk loop is a plain serial loop with `__launch_bounds__(256,1)`,
no cp.async, no double-buffer. So vectorization has NO remaining lever.

**Occupancy (ptxas -v on the dumped kernel):** 29 registers, 0 spills, 1 barrier,
16.4 KB dynamic shared. On sm_89 (1536 thr/SM, 64K regs/SM, 100 KB smem/SM):
threads cap = 1536/256 = **6 blocks/SM**; registers allow 8; shared allows 6. Kernel is
already at **max occupancy (6 blocks/SM, 100% thread occupancy)**. Occupancy is NOT a lever.

## Iter 5 — T.Pipelined chunk loop (num_stages=2/3), the one untried lever
- **Hypothesis:** overlap the (carry-independent) load of chunk N+1 with the scan/store
  of chunk N, mirroring Triton's num_stages=3, to raise MLP toward peak BW.
- **Change:** `for c0 in T.serial(...)` → `for ci in T.Pipelined(0, N//C, num_stages=2)`.
  In-place `T.cumsum(buf)` + `T.copy`→buf made the planner reject overlapping writes to
  `buf`; split into `sbuf` (loaded, double-buffered) + `obuf` (scanned) via
  `T.cumsum(sbuf, obuf)` and inlined the `carry[0]` scalar (let-binding got mangled
  across stages, "identifier base undefined").
- **Result:** COMPILED=True but **CORRECT=False** (max_diff ~16640 ≈ half a row sum).
  The pipeliner splits the loop into 3 stages and **desyncs the serial carry recurrence**
  — a silent correctness break. The carry is a genuine loop-carried scalar that
  T.Pipelined cannot preserve. **REVERT.** (And even if it compiled correctly, the SM
  already runs 6 independent row-scans concurrently, so intra-block prefetch has little
  left to hide — see Iter 6.)

## Iter 6 — DIRECT roofline measurement: pure copy vs cumsum (definitive floor)
- **Hypothesis:** if the kernel is truly HBM-bound, a bare streaming copy (Y=X, no scan)
  on the same (32768,32768) tensor should run no faster than the cumsum kernel.
- **Measurement (same GPU, 50-trial timing loop):**
  - pure copy  (T.copy X→shared→Y, no scan): **10.65 ms / 806 GB/s**
  - cumsum kernel (baseline):                **10.56 ms / 813 GB/s**
  - torch.cumsum reference:                   12.95 ms / 664 GB/s
- The cumsum kernel is **as fast as a bare memory copy** (a hair faster, within noise) —
  the scan compute is 100% hidden behind HBM traffic. ~810 GB/s ≈ **84–85% of the
  960 GB/s spec peak**, the realistic GDDR6+ECC streaming ceiling.

## Conclusion: AT FLOOR (confirmed by 4 distinct directions)
(1) vectorization already float4 load+store; (2) already max occupancy 6 blocks/SM,
no reg spills; (3) T.Pipelined is correctness-incompatible with the carry recurrence and
has ~nothing to hide given 6 concurrent per-SM scans; (4) **cumsum == pure-copy speed**,
so it cannot go faster than moving the 8.59 GB of read-once/write-once traffic. Traffic is
already optimal. No iteration beat baseline. **Baseline kept verbatim (git-diff-clean)**,
best = 10.8 ms / 1.2130x on GPU 2.
