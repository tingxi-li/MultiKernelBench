# Iteration Log — cumsum / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/cumsum/triton/solution/cumsum.py`,
Triton speedup 1.2264x); benched against the same `reference/math/cumsum.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of cumsum | 1.1927x | 10.9000 ms | 13.0000 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** row cumsum dim=1; chunked scan with carry. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=10.9000 ms, REF=13.0000 ms, **SPEEDUP=1.1927x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.2264x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 1b (float4) — vectorized global load+store
- Change: cast global xr/yr to float4*, stride-TT vectorized copy into/out of shared buf. Scan order unchanged.
- Bench: COMPILED=True, CORRECT=True, RUNTIME=10.8 ms, SPEEDUP=1.2130x.
- Verdict: <3% over baseline -> issue rate is not the bottleneck; occupancy/latency-bound. KEEP (tiny win, harmless), pivot to occupancy.

## Iter 2 — TT=512 (full 48 warps/SM occupancy)
- Change: TT 256->512, EPT 16->8. Lifts occupancy from ~40 to 48 warps/SM.
- Bench: CORRECT=True, RUNTIME=10.8 ms, SPEEDUP=1.2130x. No change vs iter-1 -> not warp-occupancy-bound. REVERT (no gain over iter-1).

## Iter 3 — CHK=8192 (fewer chunks / sync barriers, larger in-flight)
- Change: CHK 4096->8192, EPT 16, TT 512. 4 chunks/row instead of 8.
- Bench: CORRECT=True, RUNTIME=10.9 ms, SPEEDUP=1.2018x. Slightly worse. REVERT.

## Conclusion: AT FLOOR
- cumsum = read 4GB + write 4GB = 8GB irreducible global traffic. 8GB/10.8ms = 740 GB/s achieved,
  vs torch ref 610 GB/s (we beat ref by 21%). Mixed read+write DRAM bandwidth on AD102 tops out
  well below the 960 GB/s read-only peak (read/write turnaround). float4 (issue), TT=512 (occupancy),
  CHK=8192 (fewer passes) ALL within noise (std 0.27ms) -> bandwidth-saturated, not compute/issue/latency-bound.
- BEST = iter-1 (float4 IO, TT=256), 1.2130x / 10.8 ms.

## FINAL — restored iter-1 (float4 IO)
- COMPILED=True, CORRECT=True, RUNTIME=10.8 ms, REF=13.1 ms, SPEEDUP=1.2130x.
- Status: at_floor. Genuine attempts on all non-memory levers (issue rate via float4, occupancy
  via TT=512, fewer passes via CHK=8192) stayed within run-to-run noise. Bandwidth-bound at 740 GB/s
  (8GB irreducible read+write), already 21% faster than torch.cumsum ref.

---

# SESSION 2 (2026-07-02) — re-investigation vs depth-policy "headroom" claim

Baseline re-benched on GPU 0 (RTX 6000 Ada): RUNTIME=10.8 ms, SPEEDUP=1.2037x, std 0.27ms.

## Decisive probe — measured read+write BW ceiling on THIS GPU (bw_probe.py, GPU 0)
- torch `copy_` (4GB read + 4GB write = 8GB): **10.6 ms → 752 GB/s**
- torch `add` scalar (read+write 8GB): **10.5 ms → 761 GB/s**
- read-only rowsum (4GB read): 4.82 ms → 830 GB/s
- torch.cumsum ref: 12.86 ms → 622 GB/s
- **Conclusion:** mixed read+write on this GDDR6 part tops out ~752-761 GB/s (well below the
  830 GB/s read-only ceiling and 960 GB/s spec peak — read/write turnaround). Our cumsum at
  10.6-10.8 ms is **98% of the pure-copy floor**. cumsum traffic (out size == in size, fp32
  fixed by ref) is irreducibly 8GB → no traffic lever exists. RFO physics check: 10.8ms already
  implies ≤10GB traffic, so writes are NOT incurring full read-for-ownership.

## Iter 1 — __stcs streaming store (isolated RFO test)
- Hypothesis: write-once output; evict-first store avoids write-allocate/RFO traffic.
- Change: store loop `yr4[k]=buf4[k]` → `__stcs(&yr4[k], buf4[k])` (intrinsic, no PTX). Rest = baseline.
- Fast-signal: CORRECT=True, min 10.6, mean 11.0 (baseline min 10.6, mean 11.1). **Neutral.**
- Verdict: writes already skip RFO (physics + sibling agree). REVERT.

## Iter 2 — register-staged DOUBLE-BUFFER software pipeline (num_stages analog)
- Hypothesis: overlap chunk c+1 global-load latency with chunk c scan (what Triton num_stages=3 does).
- Change: two shared bufs (CHK=2048 ea, same 16KB total); prefetch c+1 into registers before scanning c,
  land into nxt buffer after store. __stcs store.
- Fast-signal: CORRECT=True, min 10.6, mean 11.0. **Neutral (0%).**
- Verdict: latency already hidden by block-level parallelism (32768 blocks / 142 SMs). REVERT.

## Iter 3 — __ldcs streaming load (read-path evict-first)
- Hypothesis: symmetric evict-first read reduces L2 pollution.
- Change: load `buf4[k]=xr4[k]` → `__ldcs(&xr4[k])` + __stcs store.
- Fast-signal: CORRECT=True, min 10.7, mean 11.1. **Slightly WORSE** (matches sibling iter-4). REVERT.

## Iter 4 — register-resident segment + warp-shuffle block scan (barrier reduction)
- Hypothesis: replace 16 Hillis-Steele syncs/chunk with warp __shfl_up_sync (2 syncs/chunk); keep
  segment scan in registers (halves shared traffic). Canonical efficient scan.
- Change: EPT-segment scan → registers r[EPT]; __shfl_up_sync intra-warp scan + 8-way broadcast combine;
  __stcs store; CHK=2048.
- Fast-signal: CORRECT=True, min 10.5, mean 10.9 (best fast-signal seen).
- **Controlled same-session VERDICT A/B (--num-warmup 200):**
  - warp-shuffle: RUNTIME=10.7 ms, mean 10.7, min 10.5, **SPEEDUP=1.2150x**, std 0.266
  - baseline:     RUNTIME=10.8 ms, mean 10.8, min 10.6, **SPEEDUP=1.2037x**, std 0.261
  - Delta = 0.1 ms (~0.9%), well inside std 0.26ms → **within noise, NOT a real >3% win.** Independently
    equals sibling cuda_unlimited's kept best (1.2150x), confirming it is the same noise-level plateau.
- Verdict: REVERT (does not clear the 3% real-win bar; keeping it risks a phantom regression on re-bench).

## Conclusion — AT FLOOR (confirmed from 5 distinct directions)
1. Direct copy-BW microbenchmark: pure copy floor = 10.6 ms / 752 GB/s; we are at 10.6-10.8 ms (98%).
2. Streaming store (__stcs): neutral → no RFO to remove.
3. Software pipeline / double-buffer: neutral → latency already hidden by 32768-block parallelism.
4. Streaming load (__ldcs): slightly worse.
5. Warp-shuffle register scan: 1.2150x = within noise of baseline, equals sibling's independent best.
- cumsum is DRAM read+write bandwidth-bound on 8GB irreducible traffic; the mixed BW ceiling on this
  AD102 part is ~752-761 GB/s (not 830-960). No lever beats baseline by a real margin.
- **BEST kept = committed baseline (float4 IO, Hillis-Steele), restored VERBATIM (git-diff-clean).**
