# Iteration Log — relu / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/relu/triton/solution/relu.py`,
Triton speedup 1.0000x); benched against the same `reference/activation/relu.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of relu | 0.9877x | 16.2000 ms | 16.0000 ms | correct |

## Iter 1 — tilelang port

- **Hypothesis:** Unary elementwise; HBM-bandwidth roofline. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.2000 ms, REF=16.0000 ms, **SPEEDUP=0.9877x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0000x:** see ako_runs/RESULTS.md for the cross-DSL table.

## iter-1 (vectorized T.copy)
- Change: VEC=4 inner T.vectorized loop over BLK//4 (float4-style coalesced access)
- SPEEDUP: 1.0323x  RUNTIME: 15.5ms  CORRECT: True
- vs baseline 0.9938x/16.1ms -> ~3.8% faster. KEEP.

## Iter 2 — FLOOR PROOF (re-run; committed baseline = iter-1 VEC=4)

Baseline re-benched on GPU 3 (RTX 6000 Ada): **SPEEDUP=1.0323x, RUNTIME=15.5ms,
REF=16.0ms, CORRECT=True** (bench.sh, --num-warmup 200). This is the committed
baseline; goal was to confirm the memory floor via distinct directions, not pad.

**Roofline math.** Input batch=4096, dim=393216 -> N=1,610,612,736 elts.
Traffic = read+write = 2 * N * 4B = 12.88 GB. At 15.5 ms -> **831 GB/s effective**,
= **~86.5% of the ~960 GB/s HBM peak**. torch.relu (ref) hits only ~805 GB/s.
N % BLK(8192) == 0 exactly (N/8192 = 196608), so the bounds guard never fires.

**Direction A — vectorization / branchless (codegen inspection, noise-free).**
Dumped the generated CUDA via `k.get_kernel_source()`. The baseline ALREADY emits
optimal streaming code: `*(float4*)(X+...)` loads and `*(float4*)(Y+...)` stores
(128-bit `ld/st.global.v4.f32`), the `if idx<N` guard fully eliminated by the
compiler (no bounds check in codegen), and `#pragma unroll` over 8 iters, under
`__launch_bounds__(256,1)`. Transaction width is already maximal -> no gain
available. This is a definitive, timer-independent floor proof.

**Direction B — occupancy sweep (block-size x threads).** Fast-signal bench
(--no-ref, 30 trials, 50 warmup), ranked by min-of-trials:
  - baseline  8192/256 : min 15.4  mean 16.1  std 0.607
  - variant   8192/512 : min 15.2  mean 16.2  std 0.749
  - variant   4096/256 : min 15.2  mean 16.1  std 0.729
  - variant  16384/256 : min 15.3  mean 16.1  std 0.713
Means flat at 16.1-16.2 ms across 256/512 threads and 4096/8192/16384 blocks; the
0.2 ms min spread is << 1 std (~0.7 ms). Kernel is occupancy-insensitive ->
classic memory-bound floor. No variant clears the noise margin. REVERT all.

**Verdict: AT FLOOR.** Both distinct directions confirm the memory roofline.
Baseline kept verbatim (git-diff-clean). improved=false, at_floor=true.

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 2 | floor proof (codegen + occupancy sweep) | 1.0323x | 15.5 ms | 16.0 ms | at floor |
