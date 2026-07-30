# Reciprocal recipe-transfer scaffold

This campaign tests the home-recipe alternative left open by Phase 1: does a
destination look best only when every DSL is forced to realize a schedule
family discovered in that destination's competitor?

It remains a prospective performance scaffold. The accepted v4 robust gate is
now bound, including its validation summary, acceptance receipt, and raw
holdout hashes. No destination implementation, recipe-resolution lock, retune
plan, audit receipt, reciprocal result, or reciprocal GPU launch is present.
Historical Phase-1 and native-search files are evidence inputs and are never
modified.

## Factorial design

Both `manifests/primary.json` and `manifests/audit.json` contain the same
deterministic 2 × 4 × 2 Cartesian product:

- recipe origin: `tilelang_phase1_confirmed`, `triton_grouped_autotuned`;
- destination: TileLang, Triton, CUDA without inline PTX, CUDA with inline PTX;
- treatment: literal donor transfer, recipient-native retuning.

That is 16 primary cells and 16 audit cells. Primary cells use five independent
processes after the candidate is frozen. Audit cells verify source/config
identity, work mapping, dynamic Tensor Core work, prohibited operations,
correctness, and generated-code resources before a primary cell is eligible.

The manifests deliberately contain no random timestamps or machine fields.
`make_manifests.py` derives them in a stable axis order and content-addresses
every recipe card.

The robust gate generates fp32 semantic inputs. Every performance cell receives
the corresponding operands rounded to fp16 before the timed kernel region and
returns fp32. This keeps casting outside the recipe-origin treatment, matching
Phase 1's kernel-only controlled comparison. End-to-end `.half()` cost remains
a separate casting estimand and must not be folded into an origin effect.

## Frozen recipe origins

The TileLang card is the confirmed Phase-1 winner:

```text
BM=128, BN=256, BK=64, stages=2, threads=256, KC=2048
plain two-dimensional grid, fp16 Tensor Cores, chunk accumulator -> fp32 outer accumulator
```

Its card is derived from `phase1_matmul/jobs/confirm.json` and binds the exact
Phase-1 implementation, specification, and five-process confirmation artifacts.
Literal transfer may translate syntax only. If the new robust gate rejects
`KC=2048`, the literal cell fails; it is not silently repaired.

The Triton card is a genuinely Triton-native schedule family rather than the
shared Phase-1 grid. `make_manifests.py` parses the 13 ordered configurations
and `GROUP_M=8` mapping from the current fp16 fused-GEMM artifact without
importing Triton. The recipe projects only the GEMM through its fp32
accumulator onto standard matmul; bias, GELU, softmax, and fp16 intermediate
storage are outside the recipe.

The donor's full-K accumulator may not satisfy the new signed-input contract.
The only permitted amendment is a common in-block KC flush. After the robust
gate is frozen, select the largest passing value in this preregistered order:

```text
8192, 4096, 2048, 1024, 512
```

The value is selected once and bound across all four destinations before any
performance split is opened. `dependencies/recipe_resolution_lock.json` must
bind it to the gate and both exact recipe-card hashes.

## Literal versus retuned

Literal treatment preserves the donor's work mapping, tile family, pipeline
depths, thread/warp counts, order, and accumulator structure. The Triton family
is executed offline in destinations without a native autotuner so the same 13
points are still charged and compared.

Recipient-retuned treatment preserves the arithmetic/dataflow and the donor's
plain-versus-grouped mapping, but permits destination-native tile, stage, and
thread/warp choices. Every destination gets 19 attempted candidates; failed
builds consume the budget. Its exact plan must be written and hashed before any
screening result exists and may never inspect the held-out performance split.
Two-process screening selects two candidates, which are independently
confirmed in five processes. Performance inputs remain
hidden until the winner is frozen.

## Dependency gate

The accepted v4 robust gate lives at
`../robust_gate/calibration/gate_spec_matmul_v4.json`. The failed v3 pilot at
`../robust_gate/gate_spec.json` is retained as historical evidence and is not
used by this campaign. Mere gate-file presence is insufficient. After the
robust-gate validator emits the exact `gate_spec_sha256` in its validation
summary, bind it explicitly:

```bash
python ako_runs/controlled_followup/reciprocal/bind_gate.py \
  --gate-spec ako_runs/controlled_followup/robust_gate/calibration/gate_spec_matmul_v4.json \
  --validation-summary ako_runs/controlled_followup/robust_gate/validation/matmul_holdout_summary_v4.json \
  --acceptance-receipt ako_runs/controlled_followup/robust_gate/validation/v4_acceptance_receipt.json \
  --out ako_runs/controlled_followup/reciprocal/dependencies/gate_lock.json \
  --validated-sha256 <sha256-printed-by-robust-gate-validator>
```

The resulting `dependencies/gate_lock.json` must bind the gate campaign ID,
the manifest and calibration-record hashes, the robust validator's canonical
gate-spec hash, the exact gate-spec file bytes, the exact successful
holdout-summary bytes, and the acceptance receipt that hashes all three raw
holdout files. Smoke or incomplete specs
(anything other than 640 calibration seeds per anchor/case or 512 locked
validation seeds per case) are rejected. These v4 counts implement the
preregistered nonparametric tolerance experiment documented in
`../robust_gate/V4_PREREGISTRATION.md`. The checked-in
`gate_lock.example.json` is intentionally non-launchable.

The gate is now bound, but launch remains blocked until the recipe
resolution, implementation registry/files, retune plans, and (for primary
jobs) per-cell audit receipts exist. These are explicit manifest dependencies,
not conventions hidden in prose.

## CPU-only inspection and validation

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  ako_runs/controlled_followup/reciprocal/make_manifests.py --check

PYTHONDONTWRITEBYTECODE=1 python \
  ako_runs/controlled_followup/reciprocal/validate.py

PYTHONDONTWRITEBYTECODE=1 python \
  ako_runs/controlled_followup/reciprocal/launch.py --manifest audit --list

PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  ako_runs/controlled_followup/reciprocal/test_scaffold.py
```

`validate.py --launch-ready` and `launch.py --validate-only` intentionally
return nonzero while the robust gate or any later binding is missing.
`launch.py --execute` additionally requires an explicit external runner; this
scaffold cannot accidentally discover or invoke a GPU harness by import.

## Interpretation

The primary estimands are origin effects within destination, destination
effects within origin, and the origin × destination interaction, reported
separately for literal and retuned treatment. A home advantage that reverses
with recipe origin is evidence of recipe anchoring. A destination effect that
survives both origins and recipient retuning is stronger finite-budget compiler
evidence. Neither result is a theoretical DSL ceiling.
