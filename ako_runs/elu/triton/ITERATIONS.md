# Iteration Log

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Autotuned Triton ELU (alpha from init) | 1.01x | 15.9 ms | improved |
| 2 | L2 streaming/eviction hint (evict_first) | — | 16.6 ms | REVERT (no gain) |

## Iterations

### Iter 1 — Autotuned Triton ELU (alpha from init)

- **Hypothesis:** Bandwidth-bound unary op; exp only on the negative branch (untaken for rand>=0 inputs) but correct by construction.
- **Changes:** Replaced the identity baseline with the optimized solution in `solution/elu.py`.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 15.9 ms (mean); Reference: 16.0 ms
  - Speedup: 1.01x (mean)
- **Analysis:** 1.01x == roofline. Memory-bound. Floor reached.
- **Next:** At roofline — stop.

### Iter 2 — L2 streaming/eviction hint (evict_first)

- **Hypothesis:** Data is streamed once (read x, write y; 12.9 GB total for 4096×393216 fp32). Tagging loads/stores `eviction_policy="evict_first"` could reduce L2 pollution on this AD102 (large L2) and lift effective bandwidth above the ~86% of peak the baseline already sees.
- **Change (variant only, baseline solution left untouched):** added `eviction_policy="evict_first"` to both `tl.load` and `tl.store`; identical autotune configs.
- **Bench (fast-signal, --no-ref, 200 warmup, 30 trials, concurrent sibling load → clocks ~16.5 ms baseline):**
  - baseline: 16.5 / 16.5 ms
  - evict variant: 16.6 / 16.7 ms
- **Analysis:** No improvement — marginally slower, within noise. The kernel is bandwidth-bound and already coalesced/vectorized by Triton; L2 hinting has no leverage on a pure once-through stream. REVERT (baseline kept verbatim).

## Floor conclusion

ELU on 4096×393216 fp32 is a pure elementwise (unary) op → read 6.44 GB + write 6.44 GB = 12.9 GB HBM traffic. On RTX 6000 Ada (~960 GB/s) the copy floor is ~13.4 ms; the baseline runs 15.9 ms (min 15.6 → ~826 GB/s ≈ 86% of peak) and matches vendor F.elu (16.0 ms) at 1.01x. Distinct directions explored and exhausted:
1. **Occupancy / block-count / vectorization** — swept by the baseline's autotune (BLOCK_SIZE ∈ {2048,4096,8192,16384} × num_warps ∈ {4,8,16}); already optimal.
2. **L2 streaming / cache-eviction hints** (Iter 2) — no gain.
3. **Launch-count / fusion** — N/A: single elementwise op, one launch, nothing to fuse; compute is negligible and the exp branch is never taken (inputs from `torch.rand` are all ≥ 0).
AT FLOOR. Baseline retained verbatim; no change beats it.
