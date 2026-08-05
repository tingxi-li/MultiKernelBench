# Finite-frontier F0/F1 protocol scaffold

This directory is a CPU-only, fail-closed implementation of the Q1 experiment
contracts in `../GPU_DSL_CLAIM_EXPERIMENT_DESIGN_20260805_ZH.md`. It neither
launches GPU work nor changes any sealed campaign.

## F0: second-architecture feasibility

F0 changes one intended factor: GPU architecture. It preserves the four
strategies, four implementation lanes, and 19 grids, producing exactly 304
feasibility rows. The only allowed stages are support probe, build, setup,
launch, and correctness gate. Timing is forbidden.

Manifest generation requires a JSON binding with this minimum shape:

```json
{
  "schema_version": 1,
  "gpu_identity": {
    "uuid": "GPU-...",
    "name": "...",
    "driver_version": "...",
    "compute_capability": "9.0"
  },
  "toolchain": {
    "python": "...", "torch": "...", "triton": "...",
    "tilelang": "...", "nvcc": "..."
  },
  "source_sha256": {
    "identity.json": "64 lowercase hex characters",
    "toolchain.json": "...",
    "support_probe.py": "...",
    "gate_lock.json": "...",
    "runner.py": "...",
    "dependency_inventory.json": "..."
  },
  "closure_roles": {
    "hardware_identity_capture": "identity.json",
    "toolchain_capture": "toolchain.json",
    "support_probe": "support_probe.py",
    "gate_lock": "gate_lock.json",
    "runner": "runner.py",
    "dependency_inventory": "dependency_inventory.json"
  }
}
```

Missing or sm_89 identity fails before the manifest exists:

```bash
python ako_runs/controlled_followup/finite_frontier_v1/protocol.py \
  f0-manifest HARDWARE_BINDING.json --source-root REPOSITORY_ROOT
```

Every closure file is read and re-hashed. The identity/toolchain captures must
agree with their binding, the dependency inventory must enumerate every other
closure file, and malformed capabilities are rejected. The generated JSON is a
requested-cell contract, not permission to run it. `validate_f0_results`
re-derives that manifest and admits exactly one unique, re-hashed terminal receipt
for every one of the 304 cells. Receipt schemas are exact and timing fields are
rejected.

## F1: multi-task finite frontier

`f1-contract` prints only the required material inputs and mechanical record
count formulas:

```bash
python ako_runs/controlled_followup/finite_frontier_v1/protocol.py f1-contract
```

F1 remains `design_only_blocked`. There is deliberately no F1 manifest or
launcher until a new explicit post-paper authorization and all frozen corpus,
candidate-count, hardware, gate, source, stage-count, and sham inputs exist.

## CPU check

```bash
python -m unittest discover \
  -s ako_runs/controlled_followup/finite_frontier_v1 -p 'test_*.py'
```
