# Ada four-device feasibility replication protocol

Status: **prepared and noncontrolling; GPU launch remains fail-closed until the
separate execution lock is frozen, committed, pushed, and verified**.

This isolated successor describes the smallest valid use of the four currently
available RTX 6000 Ada GPUs. It repeats the frozen `crossed_v2r3` feasibility
factorial on four named device instances; it does not pretend that four copies
of `sm_89` are four architectures.

## Estimand and boundary

Each bound GPU receives the same `4 strategies × 4 lanes × 19 grids = 304`
requested cells. The full census is therefore exactly `4 × 304 = 1,216`
terminal records. The only estimand is cellwise terminal-status concordance
within these four devices on this host, source, gate, and software stack.

The claim scope is fixed to
`noncontrolling_same_host_same_sku_device_feasibility_reproducibility`.
It cannot support cross-architecture reachability, a DSL performance ceiling or
ranking, trajectory transfer, abstraction efficiency, convergence, or an RTX
6000 Ada population claim.

Timing, screening, confirmation, sham timing, and search are forbidden. The
only allowed stages are source support, build, setup, launch, and the frozen
correctness gate. Every request retains exactly one of:

- `UNSUPPORTED`
- `BUILD_FAILED`
- `LAUNCH_FAILED`
- `GATE_FAILED`
- `GATE_PASSED`

Discordance is an outcome, not an analyzer failure: the validator lists every
cell whose four status labels differ. Source-level `UNSUPPORTED` rows must be
interpreted as one common classification repeated through four device-bound
receipts, not as four independent architecture observations.

## Contract and validation

`protocol.py` hard-binds the four UUIDs captured at kickoff, the exact RTX 6000
Ada SKU, compute capability `8.9`, the instrument campaign ID
`fused-epilogue-crossed-v2`, and the distinct reference result tag
`crossed_v2r3`.
A contract must remain `design_only_not_authorized`, set
`timing_allowed=false`, and bind five distinct regular files by SHA-256 under
the roles `source_lock`, `gate_lock`, `analyzer`, `runner`, and
`instrument_launch_lock`. The last role is the sealed crossed-v2 campaign lock;
the source role is its independently verified complete evidence index.

`make_manifest()` re-hashes those dependencies and mechanically emits all 1,216
requests. `validate_results()` re-derives the manifest and requires one unique,
content-addressed, exact-schema terminal receipt for every request. Each outer
result and receipt must also bind the underlying crossed audit record, shard
receipt, per-device audit summary, and campaign launch lock by path and SHA-256.
The validator re-derives cell identity and terminal status from the instrument
record and gate JSONL, cross-checks the shard receipt's UUID/assignment, and
reconstructs each device summary's counts and eligible-cell set. Internally
consistent outer labels without those materials fail closed. Extra fields,
including any latency field, are rejected.

`refuse_launch()` always raises. GPU execution requires a separately frozen
successor lock, an explicit policy amendment, CPU validation, a commit verified
on the configured upstream, and a fresh UUID/occupancy preflight. This design
directory itself grants none of those permissions.

## Execution scaffold

`launch.py` invokes the sealed crossed-v2 audit runner without changing its
internal campaign ID. Each GPU has one unique result tag. Four cyclic waves use
shard `(gpu + wave) % 4`, so every wave runs four different 76-cell shards and
every device eventually retains all 304 cells. Per-GPU caches are isolated;
failure in one wave prevents every later wave.

Prepare or verify the deterministic contract and 1,216-row manifest:

```bash
python ako_runs/controlled_followup/ada_device_replication_v1/launch.py \
  prepare --write
python ako_runs/controlled_followup/ada_device_replication_v1/launch.py prepare
python ako_runs/controlled_followup/ada_device_replication_v1/launch.py list
```

Freeze the separate execution lock only after the wrapper is final. The first
command writes it; the second re-verifies every current source/instrument hash:

```bash
python ako_runs/controlled_followup/ada_device_replication_v1/launch.py \
  freeze --write --authorization-basis "explicit user authorization for Ada-only noncontrolling feasibility replication"
python ako_runs/controlled_followup/ada_device_replication_v1/launch.py \
  freeze --authorization-basis "ignored during verification"
```

Commit and push the contract, manifest, wrapper, tests, and execution lock.
Then check the configured remote directly and execute; `execute` performs no
screening, confirmation, search, or timing. Based on the frozen audit's prior
throughput, the four cyclic waves are expected to take roughly 9–10 hours:

```bash
python ako_runs/controlled_followup/ada_device_replication_v1/launch.py ready
python ako_runs/controlled_followup/ada_device_replication_v1/launch.py execute
```

After all 16 shard processes finish, re-run the sealed analyzer on each
per-device tag and build the independently revalidated outer census:

```bash
python ako_runs/controlled_followup/ada_device_replication_v1/analyze.py --derive
```

## CPU check

```bash
python -m unittest \
  ako_runs.controlled_followup.ada_device_replication_v1.test_protocol \
  ako_runs.controlled_followup.ada_device_replication_v1.test_execution
```
