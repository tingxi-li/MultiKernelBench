# Iteration Log — sigmoid / tilelang

DSL: **TileLang DSL (JIT tile kernels)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/sigmoid/triton/solution/sigmoid.py`,
Triton speedup 1.0190x); benched against the same `reference/activation/sigmoid.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | tilelang port of sigmoid | 1.0000x | 16.1000 ms | 16.1000 ms | correct |
| 2 | vectorization (drop bounds guard) | — | 16.6 ms (fast min) | 16.6 baseline | REVERT (flat) |
| 3 | occupancy / block-count sweep | — | 16.5–16.6 ms (fast min) | 16.6 baseline | REVERT (flat) |

## Iter 1 — tilelang port

- **Hypothesis:** Unary elementwise; HBM roofline. Porting the verified Triton algorithm to tilelang should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.1000 ms, REF=16.1000 ms, **SPEEDUP=1.0000x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0190x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Baseline re-bench (this run)

- COMPILED=True, CORRECT=True, RUNTIME=16.1000 ms, REF=16.1000 ms, **SPEEDUP=1.0000x**.
- Unary elementwise sigmoid: reads N fp32, writes N fp32 → pure HBM-bandwidth bound.
  Runtime equals ref exactly (16.1 ms). No compute lever (single exp) can move a
  memory-bound op; vectorization already saturates bus (Triton port topped at 1.019x ~ noise).
- **Verdict: at floor.** No iteration attempted — no lever exists above HBM roofline.

## Floor-proof run (this session, GPU 0)

Re-benched baseline on GPU 0: COMPILED=True, CORRECT=True, RUNTIME=16.1 ms, REF=16.1 ms,
**SPEEDUP=1.0000x**. Fast-signal (20 trials, 200 warmup) noise floor: min 16.6 ms
(clock never ramps in the short run; the 100-trial verdict ramps to ~15.8–15.9 → mean 16.1).
Rank candidates against the fast-signal baseline (min 16.6), NOT the verdict 16.1.

**Roofline:** N = 4096 × 393216 = 1.61e9 fp32. Read+write ≈ 12.9 GB / 16.1 ms ≈ 800 GB/s,
i.e. ~83% of the RTX 6000 Ada ~960 GB/s theoretical peak. For read-once/write-once streaming
fp32, 80–85% of peak IS the wall (DRAM refresh/ECC/bus turnaround). Strongest evidence: we
equal `torch.sigmoid` EXACTLY (1.0000x), and torch's elementwise is an already-tuned vectorized
kernel — so ~800 GB/s is the practical ceiling, not something a kernel rewrite can beat.

Two genuinely distinct directions tested this session (both flat, both REVERTED):

### Iter 2 — vectorization (drop the per-element bounds guard)
- **Hypothesis:** the `if idx < N` predicate may block 128-bit (float4) vectorized loads/stores;
  N is an exact multiple of BLK=8192, so a guard-free kernel body (emitted only when
  `N % BLK == 0`, guarded path kept for the general case — no correctness landmine) could widen
  memory transactions.
- **Result:** fast min **16.6 ms** = baseline. Flat. Confirms tilelang was already vectorizing
  (consistent with matching torch). **REVERT.**

### Iter 3 — occupancy / block-count sweep
- **Hypothesis:** more resident blocks or a different thread count could raise memory saturation.
- **Change/Result (fast min):** BLK=8192/TH=256 (baseline) 16.6 ms; BLK=4096/TH=256 16.5 ms;
  BLK=16384/TH=512 16.6 ms. All within the ±0.1 ms / <1% noise band (std 0.25–0.45), far under
  the >3% keep-bar. Grid is already ~196k blocks (>>142 SMs), so occupancy was never the limiter.
  **REVERT** to baseline verbatim (git-diff clean).

**Final: at floor, improved=false.** Baseline (BLK=8192/TH=256, guarded) kept verbatim. Detector
passes (regression_type None). Full verdict re-run below confirms COMPILED/CORRECT and 1.0000x.
