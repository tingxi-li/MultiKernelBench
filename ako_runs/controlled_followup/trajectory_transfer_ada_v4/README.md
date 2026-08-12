# Trajectory transfer Ada v4

Status: **implemented source contract; not yet frozen or executed**.

This fresh successor tests one bounded question: does replacing a common CUDA
softmax with a destination-native softmax improve a paired two-kernel
`matmul+bias+exact-GELU+softmax` implementation in TileLang, Triton, or CUDA
on the frozen Ada GPU? CUDA-no-PTX and CUDA-unlimited remain separate compiler
policy lanes, but they are not counted as independent programming models.

The donor step is Triton's
`register_common_postprocess.g01 -> register_fused.g01`. Its previous native
comparison selected the mechanism because its interval cleared the same-campaign
sham floor on both distributions. This campaign re-admits and re-times every
artifact; the prior result is selection provenance, not new performance data.

## Orthogonal transfer dimensions

The structural route is assigned before performance and has one value per
destination. Triton uses `tl.max`/`tl.sum`, and TileLang uses the F1
`T.reduce_max`/`T.reduce_sum` implementation whose softmax correctness was
previously gated on real GEMM scratch; these are
`direct_primitive_mapping`. That predecessor receipt is not evidence for a full
two-kernel artifact: this campaign must freshly build, gate, and dynamically
audit the full pair. The prior crossed-v2 TileLang path is not reused as
the direct arm because it manually spells out shuffle and shared-memory trees.

CUDA-no-PTX and CUDA-unlimited use `manual_reconstruction`. The frozen CUDA C++
core surface has shuffle, shared-memory, and synchronization mechanisms but no
corresponding destination-level row-max/row-sum operator, so both lanes reuse
one pre-reviewed hand-written softmax recipe. The primitive-absence receipt is
strictly scoped to that frozen surface: it does not claim that external CUDA
libraries such as CUB or Thrust lack reductions. The two CUDA lanes share this
recipe and constitute compiler-policy contrasts, not independent model
replications.

The configuration adaptation is independently either:

- `donor_fixed`: use `g01` with no search; or
- `bounded_retune`: test the frozen, origin-bound 19-grid plan with the same
  budget for mechanism-off and mechanism-on. Build failures consume attempts.

The mechanism-off implementation is lane-native register GBG followed by one
byte-identical common CUDA softmax. Mechanism-on replaces only that second
kernel through the destination's preregistered route. Both remain two-kernel
implementations. This is not a generic kernel-fusion treatment.

Every admitted artifact must also pass a fresh, performance-blind dynamic-work
audit. One already-compiled call is observed through CUDA profiler activity and
must contain exactly two CUDA device events (including memcpy or memset), produce a finite fp32
`1024x8192` output, and retain no duration or other performance observation.
Declared `n_kernels` metadata alone is insufficient.

Each artifact also records `held_gemm_source_sha256`, a canonical projection of
the generated first-kernel GEMM body and schedule with the softmax and arm label
excluded. Off/on artifacts are paired only when that digest is identical.
For TileLang this successor uses the stronger control required by the sealed v2
and v3 incidents: each fresh v4 off arm is compiled and verified first, then its
complete five-file GEMM cache entry is copied into only its paired fresh v4
on-arm cache. The on arm bypasses TileLang's unstable frontend-alias lookup and
loads that entry through the frozen `tvm_ffi` cache backend by its exact kernel
key and artifact path. It must not create a held-GEMM frontend alias, lower, or
compile that GEMM, and it freshly compiles only the F1 softmax. Device source, host
source, executable, cache key, and the complete held-entry file census must all
remain byte-identical. A missing or mismatched exact-key load excludes the pair;
there is no fallback compilation. No v2 or v3 artifact byte is reused.

The predecessor incident at
`trajectory_transfer_ada_v3/INCIDENT_SEEDED_CACHE_LOOKUP_20260812.json` is itself
a frozen material. Validation independently rehashes its 502-file combined and
501-file result closures, rederives the exact 20 TileLang-off passes, 20
TileLang-on `BUILD_FAILED` frontend misses, 10,240 passing gate rows, 20 intact
six-file v3 seeds, 120 absent entries, and absence of every timing phase. It
binds source commit `a8a7c5270a3e780371c60c1c6d1f8b965803cf30`, result commit
`c9a34ded2a8e6b1b88fb1046b4e9cf525a7e2139`, and corrected incident receipt
commit `c404784d6ac9bd839fe2a986530f8fc5dad61f28`.

Before the remaining 158 admissions run, the same fresh v4 admission executes
the donor-fixed TileLang off/on pair first. A retained, hash-bound canary requires
both full gates, both dynamic audits, both load-only verifications, the exact
five-file held-GEMM key/path/source/object binding, one fresh F1 compile during
admission, zero compiles during verification, and two distinct admitted
load-only objects. A canary failure burns the tag before any other cell runs;
the probe records no durations and reuses no predecessor artifact.

## Frozen census

- Admission: `4 destinations x 2 states x (1 fixed + 19 retuned) = 160` artifacts.
- Screening ceiling: `4 x 2 x 19 x 2 = 304` fresh positive-input processes.
- Selection: one retuned artifact per destination and state, selected only from
  the positive screen; withheld inputs never influence selection.
- Confirmation: 15 blocks containing fixed and selected-retuned off/on pairs on
  two distributions, plus a byte-identical two-label sham per destination:
  `15 x (32 candidate + 16 sham) = 720` fresh processes.
- Fixed and retuned transfer are both required estimands in that single frozen
  confirmation plan. Fixed results never trigger, suppress, or otherwise branch
  retuned timing.
- Each timing record has 100 trials; trials 60--99 control.

The frozen effect orientation is `log(off_tail/on_tail)`, and sham ratios use
label A over label B. Exact distribution-free median intervals reuse the bound
core's maximum admissible order statistic `k`. A direction clears the global
sham floor only with a strict CI endpoint inequality; equality is unresolved,
and a block effect equal to the floor counts nonpositive in the sign test.

All failure states are retained exactly as `PRIMITIVE_ABSENT`,
`TRANSLATION_FAILED`, `AUDIT_FAILED`, `UNSUPPORTED`, `BUILD_FAILED`,
`LAUNCH_FAILED`, `GATE_FAILED`, or `GATE_PASSED`. Only `GATE_PASSED` artifacts
may be timed.

## CPU lifecycle

```bash
python -m unittest \
  ako_runs.controlled_followup.trajectory_transfer_ada_v4.test_protocol_contract
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.protocol check
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.protocol prepare
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.protocol freeze
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.protocol check-frozen
```

`freeze` only writes deterministic source-side manifests and a lock. GPU
readiness, provenance, admission, screening, selection checkpointing, and
confirmation remain separate lifecycle boundaries in `runner.py`.

After committing and pushing that exact frozen source closure, run the GPU
lifecycle on physical GPU 0 with one fresh tag:

```bash
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.runner provenance
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.runner ready
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.runner admit --tag ada_v4r1
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.runner screen --tag ada_v4r1
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.runner select --tag ada_v4r1
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.runner confirm --tag ada_v4r1
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.analyze final --tag ada_v4r1
python -m ako_runs.controlled_followup.trajectory_transfer_ada_v4.analyze verify --tag ada_v4r1
```

## Claim boundary

The result can classify fixed or tuned transfer of this one mechanism through a
direct-primitive or manual-reconstruction route on this workload and Ada
device. It cannot establish general trajectory/order portability, generic
fusion benefits, translator independence, cross-architecture transfer, or two
independent CUDA-model replications.
