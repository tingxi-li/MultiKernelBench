# Archived versus current fused rebenchmark — results

Campaign: `archived-current-fused-v1`. All 60 preregistered records passed the bound per-process correctness diagnostic.

Each number below is the median of 15 independent process medians; brackets are the exact [x4,x12] order-statistic interval (96.484% achieved coverage).

| Subject | Median ms | Exact interval ms |
|---|---:|---:|
| triton_current | 1.055744 | [1.043456, 1.068032] |
| tilelang_archived_opus48 | 1.350656 | [1.334272, 1.364992] |
| tilelang_current | 1.474048 | [1.469440, 1.488384] |
| triton_archived_opus48 | 1.887232 | [1.788416, 1.945600] |

The preregistered paired estimand is `current / archived` within each randomized block; values below 1 favor the current file.

| Contrast | Median ratio | Exact interval | Holm p | Decision |
|---|---:|---:|---:|---|
| tilelang_current_over_archived | 1.090398 | [1.079490, 1.097570] | 0.00012207 | archived_faster |
| triton_current_over_archived | 0.553445 | [0.543730, 0.578872] | 0.00012207 | current_faster |

Interpretation is deliberately artifact-level: archived and current files differ in more than code generation, so these data resolve contemporaneous file performance, not an intrinsic DSL effect. The performance-input diagnostic is not a full fused-v2 4×64 gate acceptance.
