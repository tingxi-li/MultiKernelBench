# Fused closure v2 results

Status: **COMPLETE**. Raw measurements: 135/135.

## Candidate process medians

| candidate | eligible | median ms | exact median interval ms | n |
|---|---:|---:|---:|---:|
| torch_historical_exact | False | 1.442816 | [1.441792, 1.446912] | 15 |
| torch_historical_precast | False | 1.425408 | [1.400832, 1.426432] | 15 |
| torch_contract_fp32 | True | 1.504256 | [1.476608, 1.519104] | 15 |
| tilelang_full_g08 | True | 1.301504 | [1.294336, 1.310720] | 15 |
| triton_full_g05 | True | 1.295360 | [1.285120, 1.308160] | 15 |
| tilelang_common_g03 | True | 1.506816 | [1.503744, 1.509888] | 15 |
| triton_common_g00 | True | 1.456640 | [1.455648, 1.465856] | 15 |
| cuda_noptx_common_g04 | True | 1.866672 | [1.864704, 1.874944] | 15 |
| cuda_unlimited_common_g02 | True | 1.811456 | [1.790976, 1.834496] | 15 |

## Preregistered paired contrasts

Ratios are numerator time divided by denominator time; values below one favor the numerator.

| family | numerator / denominator | median ratio | exact interval | Holm p | claim |
|---|---|---:|---:|---:|---:|
| historical_diagnostic | tilelang_full_g08 / torch_historical_exact | 0.900639 | [0.885653, 0.916962] | 0.00012207 | False |
| historical_diagnostic | triton_full_g05 / torch_historical_exact | 0.897381 | [0.884078, 0.907315] | 0.00012207 | False |
| same_contract_vendor | tilelang_full_g08 / torch_contract_fp32 | 0.863853 | [0.858407, 0.888349] | 0.00738525 | True |
| same_contract_vendor | triton_full_g05 / torch_contract_fp32 | 0.863172 | [0.852713, 0.870409] | 0.00195312 | True |
| torch_decomposition | torch_historical_exact / torch_historical_precast | 1.015086 | [1.010768, 1.029971] | 0.0351562 | False |
| torch_decomposition | torch_historical_precast / torch_contract_fp32 | 0.947583 | [0.931246, 0.951678] | 0.00195312 | False |

## Strict-common frozen-recipe spread

Median within-block slowest/fastest spread: 1.280591x (exact interval [1.279078, 1.283656]x).

Historical-torch contrasts are diagnostic because the historical arms do not satisfy the fused-v2 structural arithmetic contract. The common spread concerns four frozen recipes, not universal DSL optima.
