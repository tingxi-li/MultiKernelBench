# Iteration Log — sigmoid / cuda_unlimited

DSL: **CUDA + inline PTX (float4 vec, st.global.cs streaming store, red.global.max)**.
Cross-DSL port of the optimized Triton kernel (`ako_runs/sigmoid/triton/solution/sigmoid.py`,
Triton speedup 1.0190x); benched against the same `reference/activation/sigmoid.py` golden.

## Summary

| Iter | Title | Speedup | Runtime | Ref | Status |
|------|-------|---------|---------|-----|--------|
| 1 | cuda_unlimited port of sigmoid | 1.0063x | 16.0000 ms | 16.1000 ms | correct (KEEP) |
| 2 | MLP unroll x4 (independent loads) | — | min 15.8 ms | — | noise, REVERT |
| 3 | plain float4 store (vs st.global.cs) | — | min 15.8 ms | — | noise, REVERT |
| 4 | plain load (vs __ldg non-coherent) | — | min 15.8 ms | — | noise, REVERT |

**Verdict: AT FLOOR.** Bandwidth-bound elementwise (12.9 GB traffic / ~960 GB/s = 13.4 ms
theoretical; min 15.8 ms = 85% of peak, the normal band for a streaming kernel). Three
genuinely-distinct memory levers all land at the identical 15.8 ms min as the Iter-1 baseline
AND as torch.sigmoid itself — the practical wall. Baseline restored verbatim (git-clean).

## Iter 1 — cuda_unlimited port

- **Hypothesis:** Unary elementwise; HBM roofline. Porting the verified Triton algorithm to cuda_unlimited should
  reproduce its correctness and approach its speedup, with DSL-specific levers
  (vectorized/streaming memory for CUDA-unlimited; tile primitives for TileLang).
- **Bench (verdict, --num-warmup 200):** COMPILED=True, CORRECT=True,
  RUNTIME=16.0000 ms, REF=16.1000 ms, **SPEEDUP=1.0063x**.
- **forward() is glue-only** (passes utils/cheating_detection.py); all compute is
  in the kernel (CUDA kernel body / TileLang prim_func reached only via a
  subscript-dispatch, mirroring Triton's `kernel[grid](...)` exemption).
- **vs Triton baseline 1.0190x:** see ako_runs/RESULTS.md for the cross-DSL table.

## Iter 2 — MLP unroll x4 (memory-level parallelism)

- **Hypothesis:** The grid-stride loop's per-iteration load→compute→store may serialize loads;
  issuing 4 independent `__ldg` float4 loads into registers before any compute would keep 4
  memory requests/thread in flight and close the last ~15% gap on a streaming kernel.
- **Change:** Unrolled the main loop x4 (4 loads, then 4 `actf4`, then 4 streaming stores) with
  a scalar remainder tail. COMPILED=True, CORRECT=True (5/5).
- **Bench (clock-ramped, --num-warmup 200 --num-perf-trials 40):** min **15.8 ms**, mean 16.4 ms
  vs baseline min 15.8 / mean 16.0. **Identical min — REVERT.** The compiler already pipelines the
  grid-stride loop; explicit MLP adds nothing → traffic, not latency/MLP, is the limiter.
- **Next:** test the store cache hint.

## Iter 3 — plain float4 store vs st.global.cs streaming store

- **Hypothesis:** The `st.global.cs` (evict-first) streaming store is the one exotic lever in the
  kernel; for a write-once output a plain 128-bit store could be neutral or slightly better
  (evict-first can interfere with write-combining on large sequential writes).
- **Change:** replaced `stcs_v4(...)` with `((float4*)y)[i] = v`. COMPILED=True, CORRECT=True.
- **Bench (clock-ramped):** min **15.8 ms**, mean 16.4 ms. **Identical min — REVERT.** The streaming
  store hint is perf-neutral here; both variants sit on the bandwidth wall.
- **Next:** test the load cache hint.

## Iter 4 — plain float4 load vs __ldg non-coherent load

- **Hypothesis:** `__ldg` routes the read through the non-coherent (texture) cache; for a
  read-once streaming input a plain `float4` load could differ.
- **Change:** replaced `__ldg(x4 + i)` with `x4[i]`. COMPILED=True, CORRECT=True.
- **Bench (clock-ramped):** min **15.8 ms**, mean 16.4 ms. **Identical min — REVERT.** Read-path
  cache hint is perf-neutral.

## Conclusion — AT FLOOR (improved=false, baseline kept)

Sigmoid is a pure unary elementwise op: 6.44 GB read + 6.44 GB write = 12.9 GB HBM traffic.
Theoretical floor at ~960 GB/s = 13.4 ms; observed min 15.8 ms = 85% of peak, the normal
efficiency band for a coalesced streaming kernel. **The decisive evidence: torch.sigmoid — a
heavily-tuned first-party kernel — hits the same 15.8 ms min.** Three genuinely-distinct memory
levers (MLP unroll, store cache hint, load cache hint) all reproduce that identical 15.8 ms min,
so there is no measurable headroom. Iter-1 float4 + streaming-store baseline (1.0063x) restored
verbatim (git-clean); final verdict re-confirmed COMPILED=True, CORRECT=True, SPEEDUP=1.0063x.
