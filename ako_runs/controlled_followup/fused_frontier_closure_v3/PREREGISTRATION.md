# Fused frontier/reachability closure v3: preregistration

Status: prospective and append-only. No receipt-bound v1 or v2 file is edited.

## Question and fixed candidates

This experiment closes the reachability mechanism on one GPU and in one
randomized session. It asks whether replacing the old CUDA shared-memory
epilogue recipes with the streamed-global recipes changes their paired runtime
relative to fixed compiler-frontier and contract-matched torch recipes.

Eight candidates are frozen in `campaign.json`: contract-matched torch;
TileLang full g08; Triton full g05; old CUDA-no-PTX common g04; old
CUDA-unlimited common g02; streamed CUDA-no-PTX g05 and g09; and streamed
CUDA-unlimited g07. Both no-PTX recipes are retained because reachability v2
did not resolve a strict winner between g05 and g09.

## Imported correctness eligibility

Timing requires a deterministic eligibility receipt that verifies the complete
fused_closure_v2 and fused_reachability_v2 evidence and establishes that every
selected candidate passed both frozen fused-v2 mixed gates, all four registered
cases, and validation seed indices 0--63 with zero failures. This is imported
original-gate eligibility only. It is explicitly not fresh-stress eligibility
and says nothing about any later stress namespace.

## Same-GPU performance protocol

- physical GPU 3, UUID `GPU-eafdd6ce-8857-40fd-f494-47a7240bf6b5`;
- CUDA 13.1 nvcc required and captured;
- 15 randomized complete blocks from order seed 2026073005;
- one fresh serialized process per candidate in each block;
- seed-0 `torch.rand` benchmark inputs;
- precast fp16 activation and cached fp16 `(K,N)` weight where required;
- 2.0 seconds fixed-time warm-up;
- 100 CUDA-event trials with a 128 MB L2 flush before every trial;
- each process median is the independent observation.

The checked launcher refuses a busy or mismatched GPU and captures UUID,
clocks, persistence mode, driver, temperature, memory, and the CUDA 13.1 nvcc
fingerprint before writing its immutable launch receipt.

## Frozen inference

Cells and paired ratios use medians of 15 block observations and the exact
distribution-free `[x4,x12]` interval with 0.96484375 coverage. Ratios are
numerator time divided by denominator time. Each ratio receives an exact
two-sided sign test around one; p-values are Holm-adjusted only within each of
the three families in `campaign.json`.

A directional paired result requires all 15 pairs, imported original-gate
eligibility for both arms, an interval strictly on one side of one, and a
within-family Holm-adjusted p-value below 0.05. The fixed four-arm spread is
computed inside each block as slowest/fastest for TileLang g08, Triton g05,
CUDA-no-PTX streamed g05, and CUDA-unlimited streamed g07, then summarized by
the same median and exact interval.

No result is a universal DSL optimum or language-ceiling claim. In particular,
the fixed spread does not resolve the no-PTX g05/g09 selection uncertainty.

## Failure and evidence policy

All build, runtime, correctness, source, GPU, missing-record, and foreign-record
failures are retained and fail closed. Raw records are atomic and never
overwritten. Any retry after a recorded failure or any source/protocol change
requires a new result tag. The analyzer is source-bound before launch. After
analysis, a deterministic tar/index evidence bundle binds sources, imported
evidence, launch receipts, logs, raw trials, and analysis outputs.
