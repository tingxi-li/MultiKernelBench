# Ada same-campaign native-strategy recurrence

Status: **implementation complete; no GPU launch is authorized until the
source/execution freezes and artifact-admission evidence are committed and
verified on the live upstream**.

This is a fresh successor to the non-controlling v2 admission incident. V2
built and gated its first artifact, then deferred provenance validation resolved
a foreign bare `runner` module after benchmark-path shadowing. V3 binds the
complete retained v2 closure, resolves only this package's runner, and forbids
reuse of every v2 artifact byte.

This successor replaces the unexecutable translation direction with a narrower
question that the existing implementations can answer: at the frozen fused
workload and `g01`, do the directions of two native strategy contrasts recur
across TileLang, Triton, CUDA-no-PTX, and CUDA-unlimited? TileLang is fixed
before timing as the same-campaign reference; no prior TileLang performance
estimate is claimed or bound.

The native prefix in every lane is:

```text
global_intermediate
  -> register_common_postprocess  (bias + exact GELU on accumulator values)
  -> register_fused               (lane-native softmax)
```

All 12 prefix implementations are re-admitted only after the complete
`crossed_v2r3` 304-cell audit is re-derived, its source/dependency lock is
re-hashed, and each selected audit record and 512-record gate file agrees with
the independently verified evidence index.

## Why this design identifies the bounded claim

The study changes one frozen native strategy contract at a time while holding the
operator, `g01` configuration, lane, hardware, input distribution, correctness
gate, timing method, and block fixed. Every complete block contains each of the
12 native cells on both positive and withheld-signed inputs. The 28 records in
each of 15 blocks are ordered by one frozen SHA-256 randomization, and every
record runs in a fresh process on physical GPU 0. This makes step ratios paired
within block while distributing thermal and temporal drift across conditions.
The first contrast is the whole `global_intermediate` to
`register_common_postprocess` implementation transition. It is associated with
accumulator bias/exact-GELU fusion, but it also crosses native builder paths and
does not isolate that mechanism causally. The second contrast is likewise
reported as a whole native-strategy transition associated with native softmax.

The exact raw census is 420 records: `15 x (12 cells x 2 distributions + 2
sham labels x 2 distributions)`. The two shams execute the same admitted
`register_common_postprocess.tilelang.g01` source-and-code-object artifact. The
crossed-run implementation hash remains a diagnostic only; cache-only resource
metadata cannot reject an otherwise byte-identical artifact.

Trials 60--99 are controlling. For every lane and distribution, analysis forms
paired blockwise speedup ratios for step 1, step 2, and the cumulative prefix,
then uses the exact median interval. The global resolution floor is the largest
absolute log-ratio endpoint from both sham-label intervals. A destination step
is classified as `same_campaign_speedup_recurrence` only when its interval and
the TileLang reference interval clear that floor in the speedup direction on both
distributions. A same-direction slowdown is reported separately as
`same_campaign_slowdown_recurrence` and is not evidence of a recurring gain.

This can support only same-campaign recurrence of these whole native-strategy
contrasts at this workload/configuration on Ada. It cannot support causal
mechanism isolation, independent replication of a prior donor estimate, literal source
translation, optimization-order portability, translator independence, or a
general optimization-trajectory transfer claim.
The per-step, per-destination intervals are not multiplicity-adjusted, so the
study also makes no global all-destination or familywise cross-lane conclusion.

## Launch gates

Stage one freezes all source, material, and plan bytes, after which that lock
must be committed and pushed:

```bash
python -m ako_runs.controlled_followup.native_trajectory_replication_ada_v3.runner prepare
python -m ako_runs.controlled_followup.native_trajectory_replication_ada_v3.runner freeze-material
python -m unittest ako_runs.controlled_followup.native_trajectory_replication_ada_v3.test_protocol
# commit and push stage one
python -m ako_runs.controlled_followup.native_trajectory_replication_ada_v3.runner provenance
python -m ako_runs.controlled_followup.native_trajectory_replication_ada_v3.runner freeze-execution
```

`provenance` verifies the stage-one commit with live `git ls-remote`, validates
the crossed instrument closure, and captures an idle static GPU0/toolchain
identity while holding the
host-global physical-GPU0 lock keyed by UUID. The provenance and execution lock
must then be committed and pushed as stage two:

```bash
# commit and push stage two, then run performance-blind admission
python -m ako_runs.controlled_followup.native_trajectory_replication_ada_v3.runner admit-artifacts
# commit and push the complete 12-artifact evidence closure
python -m ako_runs.controlled_followup.native_trajectory_replication_ada_v3.runner ready
python -m ako_runs.controlled_followup.native_trajectory_replication_ada_v3.runner execute
python -m ako_runs.controlled_followup.native_trajectory_replication_ada_v3.analyze \
  --out ako_runs/controlled_followup/native_trajectory_replication_ada_v3/results/analysis.json
```

The toolchain identity binds the resolved Python executable and version,
PyTorch/CUDA, Triton, TileLang, `/usr/local/cuda-13.1` nvcc, imported module
files, and the crossed builder/runtime source closure. `ready` and every timing
child rederive it; every raw record retains it for analysis.

`ready` rechecks clean tracked bytes, ancestry of stage one, the final live
upstream head, and idle GPU/toolchain identity. Execution holds that host-global
lock for the whole campaign and verifies GPU0 idle before every child, idle
except for that child immediately after timing, and idle again after child exit.
Raw records,
per-position timestamp/hash receipts, the launch receipt, and run status are all
write-once; analysis rejects missing, duplicate, extra, reordered, overlapping,
or hash-changed evidence.
