# Iteration Log — cumsum / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/cumsum/solution/cumsum.py`,
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
- **vs Triton baseline 1.2264x:** see ako_dsl_runs/RESULTS.md for the cross-DSL table.

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
