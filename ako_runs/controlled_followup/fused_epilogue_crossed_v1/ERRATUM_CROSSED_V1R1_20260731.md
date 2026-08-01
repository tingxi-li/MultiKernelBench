# Erratum for crossed epilogue v1r1

> Date: 2026-07-31
>
> Scope: interpretation of result tag `crossed_v1r1`. The sealed campaign,
> locks, result records, summaries, and evidence archives remain unchanged.

This erratum controls later descriptions of the distribution, feasibility,
and four-lane timing results.

## Distribution correction

For `global_intermediate.tilelang.g01`, the published
`per_cell_signed_over_positive` value `0.8318727700287634` (`0.8319`) is an
unpaired ratio of two independently sampled process medians. It crosses the
observed fast/slow process-mode boundary and must not be reported as a 17%
input-distribution effect.

The preregistered block pairing already present in the same summary gives the
applicable estimate: **`0.9873116536809582` with exact 96.484375% interval
`[0.7986992376178009, 1.2250099052701044]`**, reported compactly as **0.9873
[0.7987, 1.2250]**. The interval includes 1.0; this cell does not establish a
distribution effect. The settled-tail diagnostic further limits the largest
deviation among all 30 timed cells to 2.4%, with one interval excluding 1.0,
and does not change the frozen result decisions.

## Feasibility taxonomy quarantine

The frozen outcome census remains valid as a record of what this implementation
did: 150 `GATE_PASSED`, 38 declared `UNSUPPORTED`, 24 `BUILD_FAILED`, 12
`LAUNCH_FAILED`, and 4 `GATE_FAILED`. The published
`feasibility_strategy_x_lane_interactions` table must not be cited as a
language-capability taxonomy, because the 40 measured non-passes reduce to two
allocation mechanisms while their labels also depend on CUDA error handling:

| Mechanism | Affected measured cells | Count | Interpretation |
|---|---|---:|---|
| fp32 epilogue tile exceeds the 101,376-byte sm_89 dynamic-shared-memory limit | `g05`--`g12` for `smem_staged` in TileLang and CUDA-no-PTX, plus `register_fused` and `smem_staged` in CUDA-unlimited | 32 | CUDA-unlimited discarded the attribute error, so the same wall surfaced as launch/gate failures; 8 `register_fused` CUDA-unlimited cells also requested a tile that strategy never uses. |
| Triton stages=4 operand pipeline requires 106,496 bytes | `g07`, `g11`, `g15`, and `g18` for `register_fused` and `global_intermediate` | 8 | A measured recipe/operand-footprint limit, not evidence about the separately declared unsupported strategy. |

The 38 `UNSUPPORTED` cells (`register_fused` × CUDA-no-PTX and `smem_staged` ×
Triton, 19 each) were preregistered declarations, not measured falsification
results. They remain outcomes of this frozen campaign, but capability claims
require the successor's source-retaining probes. No crossed-v1r1 threshold,
outcome, or label is retroactively changed.

## Four-lane timing scope

`global_intermediate`, the only strategy timed in all four lanes, consists of a
lane-native GEMM writing fp32 global scratch followed by one shared CUDA kernel
for bias, exact-erf GELU, and row-softmax. Every confirmation record reports the
same postprocess CUDA-source SHA-256:

```text
129e4a188a6bf0ecc51cbe4d6af7b8249f2f74057187d3a62c35255e12e31a1e
```

Four-lane ratios therefore compare **lane-native GEMM plus a common additive
CUDA epilogue**. They are not four-language fused-epilogue implementation
contrasts and must not be described that way.

## Evidence bindings

| Artifact | SHA-256 |
|---|---|
| `results/crossed_v1r1/final_summary.json` | `98552e48aa67411ba738e160d454e5c387e1c33cb1c543aa53a20198e4c0a70d` |
| `results/crossed_v1r1/audit_summary.json` | `229b8a34c7a0f6be24668a14a60cfe4606e523f6adc3138c1b60ea7b095ad74f` |
| `results/crossed_v1r1/confirmation_selection.json` | `59b57ed184e44f60d2e7a310b05496a8927df2da8e897c5ae1ac19d0e4573c22` |
| `results/crossed_v1r1/reanalysis_tail_v1.json` | `e7296ce10585169e7dcf1f0d66c6bc59bf079d367d30491c46f42a8b3d03d1e5` |
