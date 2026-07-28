# Phase-1 static tables (precision + generated code)

*Independent of GPU load: these are arithmetic and static-code facts.*

## Precision control — 20 seeds × 2 RMS-matched input distributions

Both distributions have `RMS = 1/√3`, so the operands carry identical
energy. Only the mean differs. `pass` is the fraction of the 20 seeds where
**every** element satisfies the harness gate `|ref−got| ≤ 1e-4 + 1e-4·|ref|`.

| variant | KC | dist | seeds passing | max abs err | gate budget | % elems failing | signed bias | ‖C‖ |
|---|---|---|---|---|---|---|---|---|
| A | full-K | `rand` | 20/20 | 0.01465 | 0.2049 | 0.000% | -2.814e-05 | 2048.0 |
| A | full-K | `randn` | 0/20 | 0.0007629 | 0.002507 | 0.004% | -6.572e-10 | 24.1 |
| B | full-K | `rand` | 0/20 | 0.2349 | 0.2049 | 1.130% | -0.1863 | 2048.0 |
| B | full-K | `randn` | 0/20 | 0.05247 | 0.002507 | 78.290% | -2.378e-07 | 24.1 |
| C | 512 | `rand` | 20/20 | 0.05774 | 0.2049 | 0.000% | -0.01063 | 2048.0 |
| C | 512 | `randn` | 0/20 | 0.05282 | 0.002507 | 78.259% | -2.319e-07 | 24.1 |
| C | 1024 | `rand` | 20/20 | 0.06909 | 0.2049 | 0.000% | -0.02224 | 2048.0 |
| C | 1024 | `randn` | 0/20 | 0.0528 | 0.002507 | 78.260% | -2.316e-07 | 24.1 |
| C | 2048 | `rand` | 20/20 | 0.09253 | 0.2049 | 0.000% | -0.04562 | 2048.0 |
| C | 2048 | `randn` | 0/20 | 0.05282 | 0.002507 | 78.262% | -2.308e-07 | 24.1 |
| C | 4096 | `rand` | 20/20 | 0.1381 | 0.2049 | 0.000% | -0.09249 | 2048.0 |
| C | 4096 | `randn` | 0/20 | 0.0527 | 0.002507 | 78.267% | -2.348e-07 | 24.1 |
| C | 8192 | `rand` | 0/20 | 0.2349 | 0.2049 | 1.130% | -0.1863 | 2048.0 |
| C | 8192 | `randn` | 0/20 | 0.05247 | 0.002507 | 78.290% | -2.378e-07 | 24.1 |
| D | 2048 | `rand` | 20/20 | 0.09253 | 0.2049 | 0.000% | -0.04562 | 2048.0 |
| D | 2048 | `randn` | 0/20 | 0.05282 | 0.002507 | 78.262% | -2.308e-07 | 24.1 |

## KC sweep — split-K is an accuracy lever, and the scaling law says why

| KC | \|bias\| | ratio vs previous | max abs err | seeds passing (`rand`) |
|---|---|---|---|---|
| 512 | 0.01063 | — | 0.05774 | 20/20 |
| 1024 | 0.02224 | 2.09× | 0.06909 | 20/20 |
| 2048 | 0.04562 | 2.05× | 0.09253 | 20/20 |
| 4096 | 0.09249 | 2.03× | 0.1381 | 20/20 |
| 8192 | 0.1863 | 2.01× | 0.2349 | 0/20 |

A random walk of independent round-offs would grow as √KC (**1.41× per doubling**); a systematic drift grows as KC (**2.00× per doubling**). The measured ratios decide which mechanism is at work.

## Is the fp32 oracle itself accurate? (vs an fp64 ground truth)

| dist | oracle max err vs fp64 | oracle bias vs fp64 | gate budget |
|---|---|---|---|
| `randn` | 0.0002714 | -2.31e-09 | 0.002507 |
| `rand` | 0.00519 | -5.49e-06 | 0.2049 |

The reference the benchmark scores against is not exact. Its own error sets the floor below which 'kernel error' cannot be measured.

## Generated-code census (SASS) — what the compilers actually emitted

Static instruction counts from `cuobjdump -sass`. **These are static, not dynamic**: a fully unrolled loop shows a large count against a rolled loop with a large trip count, for identical arithmetic. The bit-identical outputs across all four DSLs prove the arithmetic is the same regardless.

| DSL | variant | HMMA | LDGSTS<br><sub>cp.async</sub> | LDSM<br><sub>ldmatrix</sub> | FFMA | regs | spill B | smem B |
|---|---|---|---|---|---|---|---|---|
