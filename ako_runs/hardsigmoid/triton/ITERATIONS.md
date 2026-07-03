# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Autotuned Triton clamp(x/6+1/2,0,1) | 1.00x | 16.0 ms | roofline |
| 2 | Copy-floor proof + 3 distinct probes | 1.00x | 16.0 ms | AT FLOOR (confirmed) |

## Iterations

### Iter 1 — Autotuned Triton clamp(x/6+1/2,0,1)

- **Hypothesis:** Bandwidth-bound unary op; tuned Triton matches torch.
- **Changes:** Replaced the identity baseline with the optimized solution in `solution/hardsigmoid.py`.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 16.0 ms (mean); Reference: 16.0 ms
  - Speedup: 1.00x (mean)
- **Analysis:** 1.00x == roofline. Closed-form clamp, no branches; memory-bound. Floor reached.
- **Next:** At roofline — stop.

### Iter 2 — Copy-floor proof + 3 distinct probes (AT FLOOR)

Re-benched baseline on GPU 0 (this run): SPEEDUP 1.00x, 16.0 ms mean (min 15.6),
CORRECT 5/5. Rather than pad with trivial block tweaks, confirmed the floor
positively via a copy-kernel proof plus 3 genuinely distinct directions. All
timings are standalone cuda-event benches (60 iters / 30 warmup) at fixed BLOCK;
traffic = 2·n·4 B = 12.88 GB (n = 4096·393216 = 1.61e9 fp32).

- **Direction A — Positive copy-floor proof (arithmetic hiding):** a bare Triton
  copy kernel (load x, store y, zero math) runs in 15.71–15.80 ms (~818 GB/s);
  the hardsigmoid clamp kernel runs in 15.72–15.77 ms (~819 GB/s). **arith
  overhead = −0.2%** (within noise) → the clamp math is 100% hidden behind HBM.
  Reference points: `torch.hardsigmoid` = 15.92 ms (809 GB/s); pure D2D
  `torch y.copy_(x)` (same 2n traffic) = 16.00 ms (806 GB/s). Our kernel is
  already at/below the cost of a bare memory copy — nothing that moves
  read-n+write-n bytes can be faster.
- **Direction B — BLOCK_SIZE / vectorization:** copy & hs both flat across
  BLOCK ∈ {4096, 8192, 16384} (15.71→15.80 ms). Triton already emits 128-bit
  vectorized loads on contiguous aligned fp32; autotuner already sweeps this.
- **Direction C — Traffic/fusion:** out-of-place elementwise is fundamentally
  read-n + write-n = 2n; in-place would not reduce traffic. Single kernel launch
  already; no fusion opportunity for a standalone unary op.
- **Direction D — Cache/streaming eviction hint:** `eviction_policy='evict_first'`
  on load+store gave +0.00% / −0.29% vs plain (within noise). Single-pass stream
  gains nothing from cache hints, as expected.

- **Achievable ceiling:** ~818 GB/s ≈ 85% of the RTX 6000 Ada 960 GB/s spec —
  the normal GDDR6 streaming ceiling. The ~19% gap to spec is hardware-achievable
  bandwidth, not a kernel deficiency (torch and a bare memcpy hit the same wall).
- **Decision:** No probe beat baseline by a repeatable >3% margin. KEEP committed
  baseline verbatim (solution git-diff-clean). Scratch benches removed.
- **Next:** AT FLOOR — stop. improved=false, at_floor=true, changed_solution=false.
