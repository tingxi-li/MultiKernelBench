# Replicated convergence v2

Status: **retired; never launch**. The official `launch.py` only refuses execution.
The frozen protocol, manifests, receipts, and preregistration evidence remain as
historical records; `validate_launch.py` is now a historical diagnostic and cannot
authorize work. The 128-trajectory prompt extension is preserved but permanently
excluded from this campaign and any successor launch. See
`RETIREMENT_20260731.md` and `retirement.json`.

This directory records the retired round-2 convergence protocol. It does **not**
claim provider calls, GPU trajectories, completed sum/SDPA gates, or performance
results.

## Frozen design

- Core: `3 operations × 4 DSLs × 2 model families × 8 replicates = 192`.
- Prompt extension: `matmul × 4 DSLs × 2 models × 2 extra prompt arms × 8 = 128`.
- Total: 320 trajectories and 56.5333 completed-evaluation GPU-hours.
- Within every eight-replicate cell, each of four frozen GPU slots occurs exactly
  twice. Slots 0--3 are now identity-bound to the four host-verified RTX 6000 Ada
  UUIDs in `receipts/gpu_assignment_binding_20260731.json`; live driver/UUID equality
  remains mandatory at launch and is currently unavailable.
- Requested models are GPT-5.6-sol through OpenAI and Claude Opus 4.8 through
  Anthropic. A provider-attested immutable revision is mandatory; aliases cannot
  launch.
- Both providers are treated as lacking deterministic sampling-seed control. Every
  replicate is an independently identified stochastic request, and the limitation is
  recorded with its usage event.

Generate or verify the deterministic factorial and protocol receipts:

```bash
python make_manifest.py
python make_manifest.py --check
python freeze_protocol.py
python freeze_protocol.py --check
```

`freeze_protocol.py` freezes protocol bytes, not execution. The sum and SDPA files
under `gates/` explicitly remain `preregistered_not_executed`; their thresholds are
unset. Sum uses six cases, 128 calibration and 512 locked validation seeds per case,
plus structural/semantic negative controls. SDPA reuses the six registered case
families with 32 calibration and 64 locked validation seeds per case, plus structural
and wrong-attention controls.

## Historical launch barrier

Run:

```bash
python validate_launch.py
```

The historical command exits 2 and says `launch_forbidden` until all of the following are real
and hash-bound: immutable model resolutions, four live GPU UUIDs, completed/frozen
sum and SDPA gates, distinct hidden-tuning and terminal-holdout services/principals,
gate-legal frozen reference latencies, provider credentials/SDKs, and a remote
preregistration receipt. `--skip-gpu-runtime` is diagnostic and can never authorize a
launch. Retirement now supersedes that conditional readiness: satisfying every old
check still does not authorize launch. Secrets are read from environment variables
only and are rejected from event payloads.

The reusable model-lock API is:

```python
from convergence_v2.model_resolution import load_model_resolution_lock
from convergence_v2.provider_adapters import OpenAIResponsesAdapter

resolutions = load_model_resolution_lock(lock_path, expected_models=(model_spec,))
adapter = OpenAIResponsesAdapter.from_env(resolutions["openai:gpt-5.6-sol"])
```

Provider SDK imports are lazy. An injected SDK-compatible client can be passed to the
adapter constructor for tests. `audit_metadata()` and normalized provider-usage
events contain no credentials.

## Controller and analysis invariants

`controller.py` writes a hash-chained, fsync'd JSONL stream. Event IDs are
semantically idempotent; a conflicting replay fails. Monotonic time around every
completed build/evaluation attempt is charged regardless of success, build failure,
gate failure, runtime failure, or timeout. NCU time is separate. If a process resumes
with an unmatched attempt start, the trajectory is right-censored rather than
silently undercharging the attempt or replacing the replicate.

Only the `SearcherGateFacade` enters a search process; it reveals pass/fail and failed
metric names. `TerminalEvaluator` requires a different principal and dataset hash.
The dependency-free analyzer supplies Kaplan–Meier curves, pairwise log-rank tests,
restricted mean survival time, and Holm correction. Empty outcome streams are an
error, so the scaffold cannot fabricate an analysis.

Run CPU checks with:

```bash
python -m pytest -q tests
```

## Deterministic evidence capture

`capture_evidence.py` is an explicitly post-freeze packaging utility. It is not
listed in `locks/protocol_freeze_receipt.json`, is never a launch input, and does
not change or regenerate any historical receipt. A preregistration bundle can be
captured now:

```bash
python capture_evidence.py prereg \
  --output-prefix evidence/prereg_v2
```

The embedded manifest states `preregistration_only` and explicitly claims no
provider calls, GPU trajectories, completed campaign, or performance results. It
also retains the identity-bound GPU assignment and the currently unresolved live
GPU/runtime, model, robust-gate, reference-latency, and remote-preregistration
checks.

Verify the canonical index, normalized gzip/tar metadata, exact member census,
embedded manifest/state hashes, and every retained member hash independently:

```bash
python capture_evidence.py verify --index evidence/prereg_v2.index.json
```

`prereg_v1` remains an immutable, internally verifiable earlier snapshot. It is
superseded by `prereg_v2`, which additionally contains this verifier and its
tests; neither v1 file is overwritten or deleted.

The retired protocol's historical complete-capture contract was:

```bash
python capture_evidence.py complete \
  --launch-receipt results/launch_authorization.json \
  --trajectory-root results/trajectories \
  --outcomes results/outcomes.jsonl \
  --analysis results/analysis.json \
  --output-prefix evidence/complete_v2
```

This command is retained for artifact interpretation, not as a current workflow;
retirement forbids producing the prerequisite launch. Complete mode requires a
`launch_permitted` receipt whose checks all passed and
whose `artifact_sha256` map binds every frozen and mutable lock named by the tool.
It then verifies exactly 320 terminal hash-chained journals, their immutable model
revision and frozen GPU UUID bindings, at least one retained provider-usage event
per trajectory, the exact 320-outcome census, and byte-for-byte reproduction of the
survival analysis. Until all those conditions hold it cannot make execution or
results claims.

Both archive and sidecar index are reproducible from identical inputs: archive
member order and metadata and the gzip timestamp are normalized. Caches, build
products, active locks, temporary/partial files, and prior evidence/results are
excluded from preregistration capture. Results enter complete evidence only through
the explicitly validated paths above.
