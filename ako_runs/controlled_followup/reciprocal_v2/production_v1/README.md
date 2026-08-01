# Reciprocal-v2 production supplement v1

This append-only supplement turns the frozen reciprocal-v2 design into
fail-closed production plumbing.  It does **not** claim that translator
isolation, translation, GPU validation, KC resolution, or a benchmark launch
has happened.

The original 61-file freeze and `evidence/prereg_v1` remain the immutable study
design.  This directory adds operational contracts and is frozen separately;
`evidence/prereg_v2` is a superseding production-protocol preregistration that
nests and hash-binds `prereg_v1` rather than replacing it.

## Prepared state

- Two non-claiming boundary requests and 48 non-claiming translation requests
  are generated deterministically from the frozen audit manifest.
- Isolation becomes valid only after an external, content-addressed boundary
  wrapper records five raw exit-code probes and 24 produced sources for each
  translator.  A git worktree name alone is not accepted as isolation.  The
  accepted boundary kinds are a container, mount namespace, or distinct OS
  user; peer source, performance results, the repository root, and the network
  must be inaccessible.
- The implementation registry is derived from exactly 48 regular, non-empty,
  non-symlink sources plus their bound receipts.  Missing or extra files,
  reused paths/inodes, stale hashes, and stale isolation bindings fail closed.
  Equal byte hashes are disclosed as groups; they are not silently treated as
  independent implementations.
- The common-KC ladder is fixed at `8192, 4096, 2048, 1024, 512`.  Exactly one
  outstanding request is allowed, every attempt contains the 24 literal cells,
  raw robust-v4 records are independently validated, and advancement is only
  allowed after a complete failed predecessor.  The first all-cell pass stops
  the ladder.  Exhaustion writes only a terminal `no_common_kc` receipt and
  leaves launch blocked.
- KC execution additionally requires a content-addressed runner, a pushed
  commit/external timestamp authorization, and four distinct authorized GPU
  UUIDs.  A live GPU UUID/name/compute-capability check is repeated before the
  runner starts.  These prerequisites are deliberately absent at
  preregistration time.

JSON Schemas in `schemas/` describe interchange records.  Runtime checks are
dependency-free and stricter where repository-derived census, path, hash,
ordering, or raw-record recomputation is required.

## Preregistration lifecycle

From the repository root:

```bash
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.treatment_plan
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.treatment_plan --check
python -m unittest ako_runs.controlled_followup.reciprocal_v2.production_v1.test_production_v1
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.freeze --freeze
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.capture_evidence build --name prereg_v2
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.capture_evidence verify \
  --index ako_runs/controlled_followup/reciprocal_v2/evidence/prereg_v2.index.json
```

Both freeze and evidence build refuse to run if treatment outputs, a KC
authorization, implementations, or any of the three success locks already
exist.  All freeze/evidence outputs are exclusive-create and immutable.

## Later real execution

The external boundary wrapper must consume each file under
`requests/isolation/` and `requests/translations/`, create the requested source
and receipt paths, and write raw transcripts under `outputs/isolation/`.
Nothing in this supplement fabricates those transcripts or sources.

After an operator supplies real artifacts, the fail-closed sequence is:

```bash
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.isolation status
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.isolation freeze
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.registry status
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.registry build
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.kc_ladder status
```

For each authorized KC, use `kc_ladder request` or `kc_ladder execute`, then
`kc_ladder capture`.  Only after an independently recomputed all-cell pass may
`kc_ladder freeze` create the frozen campaign's
`dependencies/recipe_resolution_lock.json`.  The original
`reciprocal_v2/validate.py --stage ...` remains the final launch blocker and
continues to require its other preregistered provenance and result contracts.

## Current claim

As sealed in `prereg_v2`: request/capture/validation tooling is prepared and
tested; there are zero treatment implementations, zero boundary transcripts,
zero KC raw results, and zero isolation/registry/resolution success locks.
