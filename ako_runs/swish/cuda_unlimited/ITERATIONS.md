# Iteration Log — swish / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/swish/triton/solution/swish.py`,
Triton speedup 2.5253x); benched against the same `reference/activation/swish.py` golden.

Workload: 4096×393216 fp32 = 1.61e9 elems = 12.88 GB read+write. RTX6000-Ada peak ~960 GB/s.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| base | G=1 grid-stride float4 (prior committed best) | 2.4534x | 16.1 ms | 39.5 ms | correct |
| win  | **G=6 consecutive-float4 per thread (MLP)** | **2.5321x** | **15.6 ms** | 39.5 ms | **correct, KEPT** |

## Re-run context

Prior pass declared "AT FLOOR" after only 2 compute-side probes (streaming-load,
fast-expf). This re-run establishes the EMPIRICAL ceiling first, then finds the
lever the prior pass missed: this op is **MLP/latency-limited, not a flat
bandwidth roofline**.

## Baseline (my GPU 2, verdict --num-warmup 200)
- G=1 grid-stride float4 (committed): RUNTIME=16.1 ms mean (min 15.8), SPEEDUP=2.4534x, CORRECT.

## Iter 1 — copy-ceiling probe (diagnostic, not in solution)
- **Hypothesis:** discriminate "at floor" vs "MLP-limited" — time a bare float4 copy (same 12.88 GB).
- **Result (perf_counter):** torch.clone 16.04 ms; float4 copy (ldg+cs store) 15.79 ms;
  a **2× float4-per-thread copy hit 15.47 ms (835 GB/s)** — meaningfully below the 1× copy.
  => the 1× kernel is NOT bandwidth-saturated; there is MLP headroom. Ceiling ≈ 15.1–15.5 ms.

## Iter 2 — load/store cache policy (controlled cuda-event A/B)
- **Hypothesis:** __ldcs streaming load (compiler-scheduled, distinct from prior inline-asm ld.cs that regressed) or store policy could help.
- **Result:** __ldg == __ldcs == plain-load (all 15.79 ms); store .cs == plain == identical.
  Compute + cache policy are fully HBM-hidden. **No change** (kept __ldg + st.global.cs).

## Iter 3 — interleaved MLP unroll U∈{1,2,3,4,8} × block{256,512} × cap (controlled)
- **Hypothesis:** classic interleaved grid-stride unroll (thread reads i, i+T, i+2T…) raises MLP.
- **Result:** FLAT — all 15.65–15.92 ms, no clear winner. Interleaved layout does NOT help;
  the win must come from the ACCESS LAYOUT, not just issuing more loads. **Reverted.**

## Iter 4 — consecutive-group vs interleaved (controlled round-robin, cancels clock drift)
- **Hypothesis:** grouping *consecutive* float4 per thread (contiguous per-thread span) beats interleaved.
- **Result:** A plain-float4 15.82 ms; **B consec-pair-2× 15.47 ms**; C interleaved-2× 15.88 ms.
  Consecutive-pair is a real, reproducible ~2.2% win (all 3 stats agree); interleaved is *slower*.
  Larger contiguous per-thread access (32 B) → better coalescing granularity + back-to-back MLP loads. **Promising.**

## Iter 5 — consecutive-group size sweep G∈{1,2,3,4,6,8} (copy, controlled)
- **Result:** G1 15.81 / G2 15.44 / G3 15.44 / **G4 15.27** / G6 15.27 / G8 19.83 (spill).
  512-thread rows bogus (launch_bounds(256,6) forbids 512 → silent launch fail).

## Iter 6 — real-swish A/B: G, store policy, thread count (controlled)
- **Result:** G1 15.81; G4-cs-256 15.22; G4-plain-256 15.21 (store policy tie); G4-cs-512 15.32;
  **G6-cs-256 15.16**; G4-cs-384 15.27. Compute is fully hidden (swish≈copy). 256 threads best.

## Iter 7 — final group selection at shipping launch_bounds(256,6): G∈{4,5,6,7} (controlled)
- **Result:** G4 15.29 (846) / G5 16.00 (regress, odd-group) / **G6 15.14 min, 15.21 mean (851)** / G7 18.40 (spill).
  G6 is the last non-spilling group; all 3 stats agree it edges G4 by ~0.085 ms reproducibly. **CHOSEN G=6.**
- **VERDICT (--num-warmup 200):** COMPILED=True, CORRECT=True, RUNTIME=15.6 ms mean (min 15.2),
  REF=39.5 ms, **SPEEDUP=2.5321x**. Clear win over 16.1 ms / 2.4534x baseline. **KEPT.**

## Iter 8 — grid-cap tuning for G=6 (controlled cuda-event)
- **Hypothesis:** raising the 131072 block cap (fewer grid-stride passes) could reduce loop overhead.
- **Result:** cap131072 mean 15.250 (min 15.146) / cap262144 15.355 / cap=full 15.384.
  Current cap is best-or-tied on every stat; more blocks add scheduling overhead. **No change.**

## Iter 9 — branchless full-group fast path (controlled)
- **Hypothesis:** replacing the per-element `if(i<n4)` in full groups with an unguarded fast path removes branches.
- **Result:** fast cap131072 15.326 / fast cap262144 15.234 vs guarded-current 15.250. No gain —
  the bound branches are perfectly predictable and fully HBM-hidden. **Reverted (kept guarded).**

## Conclusion — AT FLOOR (real MLP win banked)

Best = **G=6 consecutive-float4-per-thread, 256 threads, launch_bounds(256,6), __ldg load +
st.global.cs streaming store, cap 131072.** VERDICT 15.6 ms / **2.5321x** (min 15.2 ≈ 851 GB/s
≈ 89% of the 960 GB/s peak — the practical GDDR6 read+write ceiling incl. bus-turnaround).

The prior pass's "at floor" was premature: it probed only compute-side levers (which ARE hidden)
and missed the **memory-access-layout** lever. Grouping G=6 *consecutive* float4 per thread
(vs 1/thread grid-stride or interleaved) lifted 16.1→15.6 ms (2.4534→2.5321x), a real +3.1%
that also edges the Triton baseline (2.5253x). Nine distinct directions (copy-ceiling, load/store
cache policy, interleaved unroll, consecutive-group size, thread count, launch_bounds/register
cliff, grid cap, branchless) now converge on this config; further gains would require sub-1%
that the ~0.4 ms verdict noise cannot resolve.
