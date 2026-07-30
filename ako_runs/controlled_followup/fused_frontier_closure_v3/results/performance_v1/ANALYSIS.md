# Fused frontier closure v3 results

Status: **COMPLETE**. Raw measurements: 120/120.

## Candidate medians

| candidate | median ms | exact interval ms | n |
|---|---:|---:|---:|
| torch_contract_fp32 | 1.449984 | [1.449984, 1.476608] | 15 |
| tilelang_full_g08 | 1.283456 | [1.274368, 1.286656] | 15 |
| triton_full_g05 | 1.273856 | [1.271808, 1.284096] | 15 |
| cuda_noptx_old_g04 | 1.845248 | [1.840992, 1.852352] | 15 |
| cuda_unlimited_old_g02 | 1.767936 | [1.764352, 1.789440] | 15 |
| cuda_noptx_streamed_g05 | 1.596416 | [1.585664, 1.600000] | 15 |
| cuda_noptx_streamed_g09 | 1.592320 | [1.586176, 1.597440] | 15 |
| cuda_unlimited_streamed_g07 | 1.405952 | [1.395312, 1.422336] | 15 |

## Preregistered paired ratios

Ratios are numerator time divided by denominator time.

| family | numerator / denominator | median | exact interval | Holm p | result |
|---|---|---:|---:|---:|---|
| old_vs_new_within_cuda | cuda_noptx_streamed_g05 / cuda_noptx_old_g04 | 0.862385 | [0.861279, 0.867097] | 0.000183105 | numerator_faster |
| old_vs_new_within_cuda | cuda_noptx_streamed_g09 / cuda_noptx_old_g04 | 0.862385 | [0.856836, 0.867150] | 0.000183105 | numerator_faster |
| old_vs_new_within_cuda | cuda_unlimited_streamed_g07 / cuda_unlimited_old_g02 | 0.789461 | [0.784999, 0.796866] | 0.000183105 | numerator_faster |
| compiler_vs_new_cuda_fixed | tilelang_full_g08 / cuda_noptx_streamed_g05 | 0.803681 | [0.802417, 0.809169] | 0.000366211 | numerator_faster |
| compiler_vs_new_cuda_fixed | triton_full_g05 / cuda_noptx_streamed_g05 | 0.799680 | [0.796487, 0.810776] | 0.000366211 | numerator_faster |
| compiler_vs_new_cuda_fixed | tilelang_full_g08 / cuda_noptx_streamed_g09 | 0.806029 | [0.800384, 0.810637] | 0.000366211 | numerator_faster |
| compiler_vs_new_cuda_fixed | triton_full_g05 / cuda_noptx_streamed_g09 | 0.802695 | [0.797180, 0.808210] | 0.000366211 | numerator_faster |
| compiler_vs_new_cuda_fixed | tilelang_full_g08 / cuda_unlimited_streamed_g07 | 0.911743 | [0.903528, 0.918759] | 0.000366211 | numerator_faster |
| compiler_vs_new_cuda_fixed | triton_full_g05 / cuda_unlimited_streamed_g07 | 0.907962 | [0.899881, 0.914181] | 0.000366211 | numerator_faster |
| new_cuda_vs_contract_torch | cuda_noptx_streamed_g05 / torch_contract_fp32 | 1.093574 | [1.082842, 1.101695] | 0.000183105 | numerator_slower |
| new_cuda_vs_contract_torch | cuda_noptx_streamed_g09 / torch_contract_fp32 | 1.097494 | [1.082178, 1.101695] | 0.000183105 | numerator_slower |
| new_cuda_vs_contract_torch | cuda_unlimited_streamed_g07 / torch_contract_fp32 | 0.967406 | [0.959459, 0.980932] | 0.000976562 | numerator_faster |

## Fixed-frontier spread

Median within-block spread: 1.250500x (exact interval [1.243647, 1.256544]x).

Eligibility is imported from the original frozen 4-case × 64-seed mixed validation gates. It is not fresh-stress eligibility.

The no-PTX g05/g09 winner remains unresolved by design. All conclusions concern these fixed recipes on one GPU, not universal DSL optima.
