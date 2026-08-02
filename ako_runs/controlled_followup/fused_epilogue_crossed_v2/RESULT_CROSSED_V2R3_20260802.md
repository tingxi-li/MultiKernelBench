# Crossed epilogue v2 result memo — `crossed_v2r3`

Date: 2026-08-02 UTC
Status: **COMPLETE**; `crossed_v2r3` is the controlling Ada result.

This is a non-frozen interpretation memo. It does not alter the sealed campaign
source, launch lock, results, or evidence archive. The earlier `crossed_v2`,
`crossed_v2r1`, and `crossed_v2r2` attempts remain preserved and
non-controlling; no timing or selection was reused from them.

## Protocol and provenance

| Binding | Value |
|---|---|
| Campaign | `fused-epilogue-crossed-v2` |
| Prelaunch commit | `e5ab8617f1d95b866a55c5eb8e2648e831cb834b` |
| Launch-lock SHA-256 | `206b75387acf4d60b688a8e19b3aff094a8ef69d5a0a1b72269d8b24f48be1b6` |
| Source-bundle SHA-256 | `abe6873e89686ca94a2dc096b633feff615c7dbfd8fb4e91546850ed26cae20b` |
| GPU | NVIDIA RTX 6000 Ada Generation, compute capability 8.9 |
| Driver | `610.43.02` |
| Timing device | Physical GPU 0, UUID `GPU-45af34ad-0c74-74d0-ef3a-652090d837ae` |

The configured upstream contained the prelaunch commit before the first GPU
process. Every phase used fresh driver/GPU identity and occupancy checks.

## Feasibility and reachability

The audit is complete at `304/304` cells:

| Outcome | Cells |
|---|---:|
| `GATE_PASSED` | 249 |
| `BUILD_FAILED` | 36 |
| `UNSUPPORTED` | 19 |
| `LAUNCH_FAILED` | 0 |
| `GATE_FAILED` | 0 |

The 249 gate-passed cells bind `127,488` gate rows (`249 × 512`). Here,
**supported means source-expressible**, not successful at every grid point.
Resource-bound build/setup outcomes remain `BUILD_FAILED`; they are not relabeled
unsupported. This includes
`register_common_postprocess.triton.{g07,g11,g15,g18}`, which require 106,496 B
of shared memory with 101,376 B available. The 19 `UNSUPPORTED` cells are the
audit projection of the measured 19-grid Triton explicit-smem public-API
limitation.

`register_fused`, `global_intermediate`, and
`register_common_postprocess` are jointly reachable in all four lanes at 15
grids: `g00,g01,g02,g03,g04,g05,g06,g08,g09,g10,g12,g13,g14,g16,g17`.
The excluded grids are `g07,g11,g15,g18`. `smem_staged` has no grid common to
all four lanes. The audit also confirms the eight recovered
`register_fused.cuda_unlimited.g05` through `g12` cells.

## Selection and confirmation

- Screen: `498/498` records, exactly two processes for each of 249 gate-legal
  cells.
- Selection: `N=44`, derived mechanically as the top two gate-legal screen
  cells plus legal `g01` within each strategy/lane, with duplicate cells
  collapsed and no manual substitutions.
- Confirmation: `1,380/1,380` records, exactly `30N + 60`; the extra 60 are
  duplicate-label sham records.
- Each confirmation record contains 100 trials. Trials 60–99 are controlling;
  full-window, first/last-decile, and drift results are diagnostics only.

## Sham floor and reportable Ada contrasts

`sham_a` and `sham_b` bind the same byte-identical implementation,
`fdc96ede2b4be3cf308b47427eb543627c454d5517035a728b26c3ee6032bc76`.
The sham ratio is `sham_a / sham_b`:

| Distribution | Median | Exact median interval |
|---|---:|---:|
| Positive | 1.001340 | [0.988614, 1.012375] |
| Withheld-signed | 1.002677 | [0.994867, 1.015070] |

The derived resolution floor is `0.014957935591828302` in absolute log-ratio.
The table below contains every preregistered positive-distribution contrast
whose entire exact median interval clears that floor. These are 96.484375%
order-statistic intervals, the first exact intervals meeting at least 95%
coverage. Ratios are settled-tail milliseconds in numerator/denominator, so a
ratio above one means the numerator was slower.

| Contrast | Grid | Median ratio | Exact median interval |
|---|---|---:|---:|
| Register-common / global, Triton | `g01` | 1.064220 | [1.017226, 1.096966] |
| Register-common / global, Triton | `g05` | 0.978482 | [0.970593, 0.982281] |
| Triton / TileLang, register-fused | `g01` | 1.107191 | [1.094122, 1.108998] |
| CUDA-no-PTX / TileLang, register-fused | `g01` | 1.257732 | [1.194240, 1.272727] |
| CUDA-unlimited / TileLang, register-fused | `g01` | 1.163770 | [1.153925, 1.181479] |
| Triton / TileLang, register-fused | `g05` | 0.972200 | [0.956693, 0.979142] |
| CUDA-unlimited / TileLang, register-fused | `g09` | 1.137561 | [1.117825, 1.155478] |
| CUDA-no-PTX / TileLang, smem-staged | `g00` | 1.248569 | [1.157032, 1.250342] |
| CUDA-no-PTX / TileLang, smem-staged | `g01` | 1.249496 | [1.205932, 1.266734] |
| CUDA-unlimited / TileLang, smem-staged | `g01` | 1.165939 | [1.159391, 1.182339] |
| CUDA-no-PTX / TileLang, smem-staged | `g04` | 1.237579 | [1.220008, 1.253444] |
| CUDA-unlimited / TileLang, smem-staged | `g04` | 1.200408 | [1.190687, 1.214876] |
| CUDA-no-PTX / TileLang, global-intermediate | `g01` | 1.250494 | [1.203647, 1.276792] |
| CUDA-unlimited / TileLang, global-intermediate | `g01` | 1.158657 | [1.143140, 1.175397] |
| Triton / TileLang, register-common | `g01` | 1.127927 | [1.113105, 1.142415] |
| CUDA-no-PTX / TileLang, register-common | `g01` | 1.269804 | [1.228232, 1.273478] |
| CUDA-unlimited / TileLang, register-common | `g01` | 1.167125 | [1.153212, 1.180943] |
| CUDA-no-PTX / TileLang, register-common | `g08` | 1.165068 | [1.153572, 1.178363] |

These are cell-specific, factor-isolated contrasts, not generalized lane
rankings. No distribution-stability claim is promoted. The `0.8319`
distribution claim, RQ(e), convergence, reciprocal-transfer, and
effort-frontier claims are excluded.

## Evidence bindings

| Artifact | SHA-256 |
|---|---|
| `results/crossed_v2r3/audit_summary.json` | `096a9833d47f858752fd4a31bbee3db7df525d7d36e91593bd6a011b05ff86aa` |
| `results/crossed_v2r3/confirmation_selection.json` | `a008ba2a5e6a81432eee7e94e7ebb34eeb90e3b88031f901cf10129a4be432ca` |
| `results/crossed_v2r3/final_summary.json` | `b38e01849024c285e14ea1eba7477c85443c68ca54b08a885c32f2413fa5ca03` |
| `results/crossed_v2r3/screen/launch_receipt.json` | `bc54fe68db4eeefba98dbef4b35ba30f2af8ca1e38d55f38ea11a4b60f95c454` |
| `results/crossed_v2r3/confirmation/launch_receipt.json` | `c901ed10278e16cbe60b0881e49ee16a79212635fd316eb0f13ff966394598ec` |
| `evidence/crossed_v2r3_complete_v1.tar.gz` | `4e6dda75b0516fd1cac928ec1a30ec84e03ec0774be10c178ef3b4d9e650bded` |
| `evidence/crossed_v2r3_complete_v1.index.json` | `3340a86f4f64404a96461fa5bb5262ff3e1cebacd645d4bd63aaa1bedda212c5` |

The archive contains 2,638 indexed entries and is 31,801,353 bytes. Independent
verification passed, and a separately rederived `final_summary.json` was
byte-identical to the sealed summary.

## Scope and remaining blocker

Performance evidence is Ada-only and covers one frozen fused
GEMM+bias+exact-GELU+row-softmax shape. No non-sm_89 GPU is currently available,
so the hardware-bound second-architecture feasibility matrix does not yet
exist. Per policy, no second-architecture timing will be run and paper drafting
remains blocked until that 304-cell feasibility audit is sealed and verified.
