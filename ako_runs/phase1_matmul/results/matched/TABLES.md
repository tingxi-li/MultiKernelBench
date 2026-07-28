# Phase-1 tables — tag `matched`

*169 process records, 169 successful.*

### Matched configuration — geom=primary, inputs=rand

Absolute median runtime in ms (median of per-process medians). ✓/✗ = passes / fails the harness gate `|ref−got| ≤ 1e-4 + 1e-4·|ref|`.

| DSL | A<br><sub>fp32 / full-K / native pipe</sub> | B<br><sub>fp16 TC / full-K / no pipe</sub> | C<br><sub>fp16 TC / KC=2048 / no pipe</sub> | D<br><sub>fp16 TC / KC=2048 / 3-stage pipe</sub> |
|---|---|---|---|---|
| torch | 4.717 ✓ | 1.168 ✗ | — | — |
| tilelang | 6.462 ✓ | 1.042 ✗ | 1.102 ✓ | 1.048 ✓ |
| triton | 5.152 ✓ | 1.140 ✗ | 1.577 ✓ | 1.171 ✓ |
| cuda_noptx | 5.407 ✓ | 1.166 ✗ | 1.642 ✓ | 1.376 ✓ |
| cuda_unlimited | 5.508 ✓ | 1.128 ✗ | 1.210 ✓ | 1.253 ✓ |

### Achieved TFLOP/s (2·M·N·K ÷ median runtime)

| DSL | A | B | C | D |
|---|---|---|---|---|
| torch | 29.1 | 117.6 | — | — |
| tilelang | 21.3 | 131.8 | 124.7 | 131.2 |
| triton | 26.7 | 120.6 | 87.2 | 117.3 |
| cuda_noptx | 25.4 | 117.9 | 83.7 | 99.9 |
| cuda_unlimited | 25.0 | 121.8 | 113.6 | 109.7 |

### Decomposition — each step isolates one factor

Speedup of the later variant over the earlier one, derived from the medians above. >1 means the step made it faster.

| DSL | B/A<br><sub>fp16 tensor cores</sub> | C/B<br><sub>split-K chunk flush</sub> | D/C<br><sub>software pipeline</sub> | D/A<br><sub>total</sub> |
|---|---|---|---|---|
| torch | 4.04× | — | — | — |
| tilelang | 6.20× | 0.95× | 1.05× | 6.17× |
| triton | 4.52× | 0.72× | 1.35× | 4.40× |
| cuda_noptx | 4.64× | 0.71× | 1.19× | 3.93× |
| cuda_unlimited | 4.88× | 0.93× | 0.97× | 4.39× |

### Error against the fp32 oracle — inputs=rand

| DSL | variant | max abs err | gate budget | % elems failing | signed bias | gate |
|---|---|---|---|---|---|---|
| torch | A | 0 | ~0.205 | 0.0000% | +0 | PASS |
| torch | B | 1.821 | ~0.205 | 72.7576% | -0.071 | FAIL |
| tilelang | A | 0.01416 | ~0.205 | 0.0000% | -2.79e-05 | PASS |
| tilelang | B | 0.2334 | ~0.205 | 1.0964% | -0.1862 | FAIL |
| tilelang | C | 0.08618 | ~0.205 | 0.0000% | -0.04552 | PASS |
| tilelang | D | 0.08618 | ~0.205 | 0.0000% | -0.04552 | PASS |
| triton | A | 0.01416 | ~0.205 | 0.0000% | -2.79e-05 | PASS |
| triton | B | 0.2334 | ~0.205 | 1.0964% | -0.1862 | FAIL |
| triton | C | 0.08618 | ~0.205 | 0.0000% | -0.04552 | PASS |
| triton | D | 0.08618 | ~0.205 | 0.0000% | -0.04552 | PASS |
| cuda_noptx | A | 0.01416 | ~0.205 | 0.0000% | -2.79e-05 | PASS |
| cuda_noptx | B | 0.2334 | ~0.205 | 1.0964% | -0.1862 | FAIL |
| cuda_noptx | C | 0.08618 | ~0.205 | 0.0000% | -0.04552 | PASS |
| cuda_noptx | D | 0.08618 | ~0.205 | 0.0000% | -0.04552 | PASS |
| cuda_unlimited | A | 0.01416 | ~0.205 | 0.0000% | -2.79e-05 | PASS |
| cuda_unlimited | B | 0.2334 | ~0.205 | 1.0964% | -0.1862 | FAIL |
| cuda_unlimited | C | 0.08618 | ~0.205 | 0.0000% | -0.04552 | PASS |
| cuda_unlimited | D | 0.08618 | ~0.205 | 0.0000% | -0.04552 | PASS |

### Measurement quality — per-process spread

| key | n proc | median ms | min–max ms | 95% CI ms | spread % | compile s |
|---|---|---|---|---|---|---|
| torch/A | 5 | 4.7171 | 4.6597–4.7176 | 4.674–4.738 | 1.2 | 0.1 |
| torch/B | 5 | 1.1684 | 1.1510–1.1694 | 1.155–1.175 | 1.6 | 0.1 |
| tilelang/A | 5 | 6.4625 | 6.3580–6.4645 | 6.384–6.501 | 1.6 | 10.4 |
| tilelang/B | 5 | 1.0424 | 1.0035–1.0629 | 1.011–1.066 | 5.7 | 4.1 |
| tilelang/C | 5 | 1.1018 | 1.0864–1.1121 | 1.088–1.111 | 2.3 | 4.0 |
| tilelang/D | 5 | 1.0476 | 1.0220–1.0598 | 1.028–1.066 | 3.6 | 4.8 |
| triton/A | 5 | 5.1517 | 5.0600–5.2014 | 5.071–5.217 | 2.7 | 0.4 |
| triton/B | 5 | 1.1397 | 1.1351–1.1505 | 1.134–1.149 | 1.3 | 0.4 |
| triton/C | 5 | 1.5769 | 1.5673–1.5872 | 1.566–1.585 | 1.3 | 0.4 |
| triton/D | 5 | 1.1715 | 1.1602–1.1837 | 1.162–1.183 | 2.0 | 0.4 |
| cuda_noptx/A | 5 | 5.4066 | 5.2500–5.4282 | 5.283–5.463 | 3.3 | 0.2 |
| cuda_noptx/B | 5 | 1.1658 | 1.1367–1.1798 | 1.138–1.182 | 3.7 | 0.2 |
| cuda_noptx/C | 5 | 1.6424 | 1.6208–1.6476 | 1.625–1.652 | 1.6 | 0.2 |
| cuda_noptx/D | 5 | 1.3763 | 1.3619–1.4060 | 1.355–1.410 | 3.2 | 0.2 |
| cuda_unlimited/A | 5 | 5.5076 | 5.4810–5.5660 | 5.475–5.553 | 1.5 | 0.2 |
| cuda_unlimited/B | 5 | 1.1279 | 1.1131–1.1412 | 1.116–1.143 | 2.5 | 0.2 |
| cuda_unlimited/C | 5 | 1.2103 | 1.2042–1.2406 | 1.199–1.238 | 3.0 | 0.2 |
| cuda_unlimited/D | 5 | 1.2534 | 1.2012–1.2800 | 1.206–1.285 | 6.3 | 0.2 |
