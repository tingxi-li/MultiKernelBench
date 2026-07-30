# Fused GBGS closure v2: preregistration

Status: prospective. This document and the machine-readable `campaign.json`
must be content-addressed by `source_receipt.json` before either correctness or
performance execution. No v1 file is modified or normalized by this campaign.

## Questions

1. Do the two frozen compiler-frontier recipes beat a contemporaneous vendor
   implementation under the same fused-v2 mixed-precision contract?
2. How much of the historical torch comparison is activation-cast placement or
   fp16 epilogue arithmetic?
3. What is the performance spread at recipes frozen from the strict all-four
   feasible intersection?

The historical torch path remains scientifically useful, but it is a diagnostic
denominator: it casts the fp32 activation inside the timed region and performs
bias, GELU, and softmax in fp16 before returning fp32. It is not relabeled as a
fused-v2-conforming implementation. The precast historical-arithmetic arm only
removes the activation cast from timing. The contract arm uses fp16 operands,
`torch.mm(..., out_dtype=torch.float32)`, fp32 bias, exact-erf GELU, fp32
softmax, and fp32 output.

## Frozen candidates and selection

There are nine candidates, in the exact order stored in `campaign.json`:

- three torch definitions above;
- confirmed full-frontier TileLang g08 and Triton g05;
- strict-common TileLang g03, Triton g00, CUDA-no-PTX g04, and
  CUDA-unlimited g02.

The strict intersection is exactly
`{g00,g01,g02,g03,g04,g13,g14,g16,g17}`. It is derived from timing eligibility
in the existing 19-point screen. The four common recipes are the screen minima
within that set. They are fixed recipes, not asserted latent lane optima.

## Correctness adjudication

Before performance launch, all nine candidates receive complete locked
validation under the already accepted fused-v2 gate:

- both `semantic_mixed` and `conformance_mixed`;
- all four registered cases;
- validation seed indices 0--63;
- unchanged thresholds from `gate_spec_fused_v2.json`.

Outputs are evaluated once per candidate/case/seed and reused across the two
references. Collection failures and every threshold failure are retained.
Structural contract status is adjudicated separately from empirical threshold
status. The first two torch candidates remain `diagnostic_nonconforming` even
if their metrics happen to pass. A candidate is same-contract eligible only if
it is declared `fused_v2_required`, has no structural mismatch, has complete
coverage, and has zero collection or threshold failures under both gates.

Performance collection requires a complete adjudication artifact, not universal
gate success. This avoids selectively suppressing an informative failed vendor
arm. Same-contract conclusions exclude ineligible candidates.

## Performance protocol

- one physical RTX 6000 Ada GPU, requested as physical GPU 0;
- `CUDA_VISIBLE_DEVICES=0` fixed before importing torch in child processes;
- 15 randomized complete blocks;
- each block contains one fresh serialized process for every candidate;
- order is deterministic from seed 2026073002 and randomized within block;
- `torch.rand`, seed 0, benchmark shape `(1024,8192) @ (8192,8192)`;
- cached fp16 `(K,N)` weight;
- 2.0 seconds fixed-time warmup;
- 100 CUDA-event trials with an L2 flush before every trial;
- the process median is the independent observation.

Build time, warmup iterations, all 100 trials, environment fingerprint, launch
position, source binding, output dtype/shape, and the legacy diagnostic gate are
retained. GPU occupancy, clocks, temperature, persistence mode, driver, CUDA,
nvcc, torch, host, and git state are captured in the launch receipt when the
platform exposes them.

## Frozen inference

Cell point estimates are medians of 15 process medians. Their estimator-aligned
interval is the exact central distribution-free interval `[x4,x12]`, whose
coverage for a continuous population median is 0.96484375. Paired contrasts use
the median of 15 within-block time ratios and the same interval. The null ratio
of 1 is tested by the exact two-sided sign test. P-values are Holm-adjusted
within, but never across, the three preregistered families in `campaign.json`.

The common-four spread is computed inside each block as slowest/fastest among
the four frozen common recipes; its point and interval use the same median and
exact order-statistic rule. This is a spread of frozen recipes, not an inferred
universal DSL order.

Historical-diagnostic ratios cannot support a same-contract vendor claim. A
compiler recipe may be described as faster than the contract-matched torch arm
only when both are same-contract eligible, all 15 paired observations exist,
the preregistered paired interval lies strictly below 1, and the exact
sign-test p-value is below 0.05 after Holm adjustment within its frozen family.
No claim is based on CI overlap.

## Failure and amendment policy

Raw records are append-only and atomically written. A build, runtime,
correctness, missing-record, foreign-record, or source-hash failure remains in
the campaign. Retrying a failed measurement requires a new result tag; `--force`
is not an accepted analysis input. Any source or protocol amendment requires a
new deterministic source receipt and is disclosed as a new launch, never folded
into an existing receipt.
