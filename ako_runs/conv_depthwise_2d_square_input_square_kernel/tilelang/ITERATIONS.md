# Iteration Log

<!--
Per-iteration template (copy when adding a new iter entry under "## Iterations"):

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False
  - Runtime: ___ ms (mean), ___ ~ ___ ms (min ~ max)
  - Speedup: ___x (mean), ___ ~ ___x (min ~ max)
- **Analysis:** Why it worked or failed
- **Next:** What to try next

Append one row per iter to the Summary table below.
Status values: improved / no-change / regression / failed.
-->

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | identity baseline (cuDNN depthwise conv2d) | 1.02x | 3.34 ms | baseline |
| 2 | smem-halo depthwise conv bh6 bw170 (fp32, divisor-tiled) | 2.44x* | 2.58 ms | improved / KEPT |

## Iterations

### Op summary (MEMORY-BOUND depthwise conv, win tier)
- 16x64x512x512, k3, stride1, pad0 -> out 510x510. Only 9 MACs/pixel -> plain fp32
  accumulation matches cuDNN exactly (maxabs ~1.2e-7). ~2 GB traffic -> memory-bound
  (roofline ~1.97 ms / ~1015 GB/s).
- **Two TileLang eager-builder pitfalls hit & solved:** (a) raw `for x in list` and
  `for k in range()` get intercepted as *device* loops (and a device `if`/`&` boundary
  guard throws "is_bool() is false"); (b) 510 doesn't tile by powers of 2. Fix: choose
  tiles that EVENLY divide 510 (=2*3*5*17) so NO boundary guard is needed, and sum the
  3x3 taps with a **Python generator expression** over a module-level TAPS list (pure
  compile-time unroll, never touches the builder's range/if override).
- Shared-memory halo tiling (load (bh+2)x(bw+2) once, reuse) -> 833 GB/s (2.57 ms);
  direct L2-reuse version -> 810 GB/s. 9 configs all plateaued 2.57-2.66 ms.
- **Result: kernel 2.58 ms, ~833 GB/s = ~82% of roofline (halo scatter + 510-not-%4
  blocks float4 vectorization -> can't reach the pure-stream 92%). Beats cuDNN.**
  Logged speedup 2.44x is inflated by a cuDNN reference OUTLIER (ref max 286 ms this
  run); the honest, robust win vs cuDNN's typical 2.8-3.4 ms is ~1.1-1.3x.
- STOP: at achievable memory roofline (plateau confirmed across 9 tile configs); beats
  the cuDNN reference. Detector-clean.
