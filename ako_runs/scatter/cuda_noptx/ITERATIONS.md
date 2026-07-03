# Iteration Log — scatter / cuda_noptx

DSL: **plain CUDA C++ via cpp_extension.load_inline (no inline PTX)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/scatter/triton/solution/scatter.py`,
Triton speedup 5.3079x); benched against the same `reference/index/scatter.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_noptx port of scatter | 6.5580x | 0.0276 ms | 0.1810 ms | correct |

## Iter 1 — cuda_noptx port

- **Hypothesis:** deterministic last-wins (atomicMax); scored --deterministic. Porting the verified Triton algorithm to cuda_noptx should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=0.0276 ms, REF=0.1810 ms, **SPEEDUP=6.5580x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 5.3079x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Re-bench 2026-06-29
- baseline: 6.4643x CORRECT, 0.0280ms
- final (unchanged): 6.3604x CORRECT, 0.0283ms
- At deterministic-scatter floor (~27us); beats Triton 5.31x. No edit attempted; run-to-run noise <3%. at_floor.

## Deep floor-confirmation pass 2026-07-02 (GPU 0)
- Baseline (my GPU): SPEEDUP 6.9466x, RUNTIME 0.0262 ms (ref 0.182 ms). Final verdict after
  restore: 0.0265 ms, 6.4151x (ref drifted to 0.170 ms — speedup delta is pure reference-side
  clock noise; my-GPU RUNTIME is flat 0.0262->0.0265 ms).
- **Profiling (nsys):** warm the wall looked dispatch-bound (kernels ~10us in a 26us wall), but
  that was a measurement artifact: the RTX 6000 Ada has **96 MB L2** and the bench's
  `clear_l2_cache` writes a fresh **256 MB** buffer each trial, so the kernels run fully cold.
  True cold GPU times (256 MB clear): **argk 13.4us + gather 8.34us + fill(init) 3.2us = 24.9us**,
  which matches a captured CUDA-graph replay floor of **24.58us**. The wall is genuine cold-DRAM
  memory + scattered-atomic execution, not CPU dispatch.
- **7 distinct hypotheses, none beat baseline:**
  1. Cooperative-groups single kernel (1 launch, grid.sync init/atomic/resolve): **30us WORSE** —
     `cudaLaunchCooperativeKernel` is heavy and the co-resident grid + 2 grid.syncs slowed the
     kernel (12.84us vs 10.2us warm); launch-count reduction only wins if the replacement launch
     is cheap.
  2. `cudaMemsetAsync(0xFF)` init instead of `torch::full`: neutral (26us) — full's dispatch was
     not in the critical path.
  3. Persistent self-cleaning winner (gather resets -1, drops per-call memset): **36us WORSE** —
     the memset was priming the winner buffer in L2; removing it made argk's atomics hit cold
     DRAM (argk 4.9->8.85us).
  4. Persistent (static) out+win buffers, no per-call ATen alloc: neutral (26us) — allocation is
     not the bottleneck; the wall is cold-kernel execution.
  5. CUDA graph capture+replay: pure-replay floor 24.58us (only ~1.7us under baseline); a real
     C++-hidden-graph solution would add pybind/forward overhead back to ~27us. Not worth the
     statefulness/correctness risk.
  6. Packed-64 atomicMax (position<<32 | value_bits) to read updates COALESCED instead of the
     scattered winner-gather: **38us WORSE** — 64-bit atomics + doubled 4 MB accumulator cost
     more than the coalesced updates read saved.
  7. 32-bit indexing + `k=i-r*K` (one div, avoid 64-bit div/mod): neutral (26us) — the divisions
     were hidden behind memory latency.
- **Verdict: at_floor.** Restored the committed baseline VERBATIM (git-diff-clean). Final:
  COMPILED=True, CORRECT=True (5/5), 0.0265 ms, 6.4151x. Still beats Triton 5.31x / TileLang.
  The dominant cost is argk's scattered atomicMax (13.4us cold) which is inherent to deterministic
  last-wins scatter; no atomic-free reformulation is cheaper at K=4096/row.
