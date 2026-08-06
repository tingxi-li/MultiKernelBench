# TileLang abstraction v3 admission result — `ada_v3r1`

Date: 2026-08-06 UTC
Status: **ADMISSION COMPLETE; NO PERFORMANCE TIMING AUTHORIZED**

This is a non-frozen interpretation memo. It does not alter the sealed source,
campaign lock, or retained GPU evidence.

## Question and reasoning chain

The research question is whether TileLang's lower abstraction levels improve
efficiency. A defensible answer requires the following chain:

1. compare high- and low-level implementations of the same operation;
2. hold workload, shape, dtype, algorithmic work, tile, pipeline depth, and
   input distribution fixed;
3. prove both arms satisfy the same current correctness gate on withheld inputs;
4. confirm the generated CUDA/PTX/SASS and launch geometry for each arm;
5. only then compare randomized, blocked timings against duplicate-label sham
   controls.

This run tested step 3. It did not reach steps 4 or 5 because no pair passed
admission. That boundary is the result: `ada_v3r1` supplies no evidence that
either abstraction level is faster.

## Controlled setup

The fixed denominator contains one preregistered high/low pair in each of three
families:

| Family | High-level arm | Low-level arm | Current gate |
|---|---|---|---|
| Matmul | H1 | M2 | Matmul v4 |
| Fused softmax | F1 | F4c | Fused v2 |
| SDPA | S3H | S3M | Unavailable |

Each executable arm was built in a fresh process on physical GPU 0, then run
over the complete frozen validation split. Generated CUDA, PTX, and SASS were
retained and hashed. Matched arms used identical coordinate sets and seeds.
The SDPA row was retained in the denominator but stopped before build because
there is no accepted current robust SDPA gate.

The run used an NVIDIA RTX 6000 Ada Generation (sm_89), UUID
`GPU-45af34ad-0c74-74d0-ef3a-652090d837ae`, driver `610.43.02`, PyTorch
`2.10.0+cu128`, TileLang `0.1.11`, and Triton `3.6.0`.

## Results

| Pair/arm | Gate records | Failed | Terminal outcome | Generated identity |
|---|---:|---:|---|---|
| Matmul H1 | 6,144 | 2,048 | `GATE_FAILED` | `4228483d070b...` |
| Matmul M2 | 6,144 | 2,048 | `GATE_FAILED` | `ce0038cb4d16...` |
| Fused softmax F1 | 512 | 506 | `GATE_FAILED` | `80aab1308b95...` |
| Fused softmax F4c | 512 | 506 | `GATE_FAILED` | `52daa5be6075...` |
| SDPA S3H/S3M | 0 | — | `CURRENT_GATE_UNAVAILABLE` | — |

The high and low arms have different generated identities, but their binary
pass/fail vectors are identical within each family:

- matmul coordinate/outcome-vector SHA-256:
  `68f4942a54e9ac58c6113b6196e3239d48d5104a2ea0d7b90f198bc9d69bddf2`;
- fused-softmax coordinate/outcome-vector SHA-256:
  `4add8edbd0bcd3a64b6166511772c5f3de8c41006ec7573cb1f289a205a7b10c`.

For both matmul arms, all `legacy_u01` and `opposing_means` records failed in
both semantic and conformance gates, while all four other cases passed. For
both fused-softmax arms, 61/64 `legacy_u01_gain1` seeds and all seeds in the
other three cases failed under both gates. The matched signatures localize the
problem to behavior shared by each pair or its held-fixed contract, rather
than supplying an abstraction-level ranking.

The independently re-derived admission census is:

| Outcome | Pairs |
|---|---:|
| Requested | 3 |
| Timing eligible | 0 |
| Excluded fail-closed | 2 |
| Current gate unavailable | 1 |

The mechanically generated timing manifest therefore contains zero rows. No
timing process was launched.

## What can and cannot be concluded

`ada_v3r1` supports a methodological conclusion useful beyond TileLang:
abstraction-efficiency comparisons must treat semantic equivalence as an
empirical admission condition. Syntactically matched kernels, matching launch
geometry, or successful compilation are insufficient. When both abstraction
arms share an input-dependent failure signature, timing them would rank two
incorrect implementations and confound abstraction level with contract error.

This run cannot answer whether lower TileLang abstractions improve efficiency,
nor whether higher abstractions are faster. It also cannot generalize across
families or architectures.

## Minimal next experiment

Diagnose one family at a time using the retained failing/pass coordinates.
First reproduce matmul's two failing cases and four passing cases against the
frozen references, varying only the shared arithmetic/contract choice. Admit a
new successor pair only after both unchanged abstraction arms pass all 12,288
matmul records. Apply the analogous procedure to fused softmax. Add a current
SDPA gate under a new lock before building either SDPA arm. Performance timing
remains blocked until at least one complete matched pair passes admission.

## Evidence bindings

| Artifact | SHA-256 |
|---|---|
| Campaign lock | `4514c83bc9e61efa09bf2f6964b9bb2888d821e61843028cc4f2387d5b03e16e` |
| Admission summary | `a4c7162b105583bd58da913eb5e4f58fe19f185b18ebefdf28b92a0eff14c7a8` |
| Admission evidence bundle | `2b9782fb3a114bc78431e6e051d2436d050e0326fded7427c320ad2b54c16ec3` |
| Timing manifest (zero rows) | `10b2d18d4c4edd71c8e633a8717ae80f98a16d75ed84104373dbf2c4586f7ca5` |

The retained result tree contains 25 files (38 MiB). CPU protocol validation
passed, all six campaign tests passed, and a separately re-derived summary was
byte-identical to the retained summary.
