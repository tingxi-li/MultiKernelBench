# Effort frontier v1 (RQ5)

Status: **preregistered and implemented; not launched**.

The missing executor registry is a substantive treatment artifact, not a
mechanically fillable command list. The repository audit and blocker classes
are recorded in [EXECUTOR_TREATMENT_ARTIFACT.md](EXECUTOR_TREATMENT_ARTIFACT.md).

This campaign measures an implementation-specific effort frontier for one
`M=1024, K=8192, N=8192` matmul+bias+GELU+softmax contract on RTX 6000 Ada.
It does not estimate a universal DSL ceiling or vendor-expert performance.  The
four programmable lanes are cuBLASLt, Triton, TileLang, and hand CUDA with an
unlimited source budget.  A frozen PyTorch contract is a timing control, not a
fifth search trajectory.

## Frozen design

There are 20 independent search trajectories: four lanes by five contexts.
Each freezes a snapshot after the first completed action crossing 0.5, 2, and 8
hours of cumulative active effort.  Provider wait, controller compute, GPU
evaluation, human intervention, and token categories are recorded separately.
Measured controller time covers prompt/feedback construction, response parsing,
candidate persistence, evaluator-response validation, and checkpoint selection;
event serialization/lifecycle bookkeeping is protocol overhead. Queue/outage
time is excluded.  Overshoot is retained.  A build or frozen
fused-v2 gate failure consumes effort.

At a checkpoint, the fastest tuning-legal candidate is evaluated once on the
hidden terminal holdout.  Failure remains a failure; the controller does not
search the hidden holdout or silently substitute another candidate.  Terminal
feedback is never put into a later prompt.  In inference, a failed/missing
terminal selection ranks worse than every finite legal latency, with failures
tied.  Every cell also reports its success count.

Search-stage lane contrasts enumerate all `10 choose 5 = 252` label assignments
for a tie-aware, two-sided Mann–Whitney randomization test.  Holm correction is
applied to the six lane pairs separately at each checkpoint.

Every eligible checkpoint selection is subsequently timed on physical GPU 0 in
15 randomized complete blocks under both `positive_rand_seed0` and the withheld
`signed_mixed_withheld` distribution.  Each distribution/block contains the
control once and all eligible selections once, in a frozen SHA-256 permutation;
each record has 25 warmups and 100 timed trials.  Candidate/control ratios are
paired within block and reduced to one median per independent search
trajectory.  The analysis reports exact lane contrasts, exact control sign
tests, and exact four-lane positive-versus-signed rank stability.  The 15 timing
blocks remain technical replicates and are not treated as 15 independent
searches.

## Fail-closed external contracts

No executor is embedded or inferred.  Before launch,
`locks/executor_registry.json` must be supplied with this shape:

```json
{
  "schema_version": 1,
  "campaign_id": "fused-effort-frontier-v1-20260731",
  "lanes": {
    "cublaslt": {
      "tuning_command": ["..."],
      "terminal_holdout_command": ["..."],
      "confirmation_command": ["..."],
      "timeout_s": 7200,
      "source_hashes": {"repository/relative/file": "64 lowercase hex"}
    }
  },
  "control": {
    "confirmation_command": ["..."],
    "timeout_s": 7200,
    "source_hashes": {"repository/relative/file": "64 lowercase hex"}
  }
}
```

All four lane keys are mandatory.  Commands are argv arrays (never shell
strings), and every executor source is content-addressed.  Search evaluators
read one JSON request on stdin and emit one JSON object binding campaign,
trajectory, lane, split, candidate, gate, GPU allocation and observed GPU UUID.
They report boolean `lane_policy_pass`, `build_ok`, and `gate_pass`, and a finite
positive `median_ms` only when all three pass. Lane policy must reject calls or
generated code outside the assigned implementation. Confirmation executors additionally echo the
record/block/distribution identity and return exactly 100 positive finite trial
times.  Candidate records must affirm the fused-v2 gate; the control must affirm
its exact contract check.

`locks/model_resolution_lock.json` is intentionally unresolved.  It must be
resolved through the public `convergence_v2.model_resolution` contract to a
provider-attested immutable revision distinct from the `gpt-5.6-sol` alias.
The controller uses the direct OpenAI Responses adapter and requires
`OPENAI_API_KEY`; secrets are never written to events.

`locks/prelaunch_provenance.json` must bind a full immutable commit, verified
remote push, external UTC timestamp, the manifest hash, the exact campaign-file
hash map, model-lock hash, and executor-registry hash.  These files are not
created with placeholder success claims.  GPU identity is checked against the
four frozen UUIDs and Ada compute capability 8.9.  Any missing lock, credential,
executor, provenance field, driver, or GPU identity prevents all provider and
GPU work.

## Commands

CPU-only structural validation:

```bash
python ako_runs/controlled_followup/effort_frontier_v1/validate.py
python -m unittest discover -s ako_runs/controlled_followup/effort_frontier_v1/tests -v
```

Readiness (expected to be blocked until real locks, credentials, and GPUs are
available):

```bash
python ako_runs/controlled_followup/effort_frontier_v1/validate.py --launch-ready
python ako_runs/controlled_followup/effort_frontier_v1/launch.py
```

Launch requires the explicit `--execute` switch.  The launcher holds a file
lock, runs five waves with one trajectory per GPU, and starts no later wave
after a failure.  Hash-chained events use durable external-action start records;
an interrupted billed request or GPU action refuses automatic retry until it is
reconciled.

After all trajectories complete, freeze and run confirmation:

```bash
python ako_runs/controlled_followup/effort_frontier_v1/confirmation.py \
  --result-root ako_runs/controlled_followup/effort_frontier_v1/results/search_v1 \
  --plan ako_runs/controlled_followup/effort_frontier_v1/results/confirmation_v1/plan.json \
  --make-plan
python ako_runs/controlled_followup/effort_frontier_v1/confirmation.py \
  --result-root ako_runs/controlled_followup/effort_frontier_v1/results/search_v1 \
  --plan ako_runs/controlled_followup/effort_frontier_v1/results/confirmation_v1/plan.json \
  --records ako_runs/controlled_followup/effort_frontier_v1/results/confirmation_v1/records.jsonl \
  --execute
```

Analyze search alone, or pass both confirmation files for the complete result:

```bash
python ako_runs/controlled_followup/effort_frontier_v1/analyze.py \
  --result-root ako_runs/controlled_followup/effort_frontier_v1/results/search_v1 \
  --confirmation-plan ako_runs/controlled_followup/effort_frontier_v1/results/confirmation_v1/plan.json \
  --confirmation-records ako_runs/controlled_followup/effort_frontier_v1/results/confirmation_v1/records.jsonl \
  --out ako_runs/controlled_followup/effort_frontier_v1/results/analysis_v1.json
```

No lock, candidate, event, confirmation record, or analysis output currently in
this directory asserts that a provider/GPU experiment ran.

## Evidence lifecycle

`capture_evidence.py` implements the common `prereg` / `complete` lifecycle.
Both stages create a deterministic gzip/tar (zero timestamps, owners, and stable
ordering) plus a canonical hash index, refuse overwrite, and support a full
member-by-member `verify`. Prereg evidence contains the frozen sources, prompts,
manifest, tests, unresolved/resolved model-resolution record, and fused-v2 gate
dependencies. It contains no result claim.

Complete evidence is refused unless all 20 trajectories, all three checkpoints,
the frozen confirmation plan, every expected confirmation record, and the
canonical combined analysis validate as complete. It adds immutable JSON
prelaunch locks/receipts and content-addressed executor sources. Runtime mutex
files, caches, build trees, compiled objects/binaries, PTX/cubin, and partial
artifacts are excluded. The result root must live in the repository so every
archive member has a stable repository-relative name.

```bash
python ako_runs/controlled_followup/effort_frontier_v1/capture_evidence.py build \
  --stage prereg --name effort_frontier_prereg_v1 \
  --output-dir ako_runs/controlled_followup/effort_frontier_v1/evidence

python ako_runs/controlled_followup/effort_frontier_v1/capture_evidence.py build \
  --stage complete --name effort_frontier_complete_v1 \
  --output-dir ako_runs/controlled_followup/effort_frontier_v1/evidence \
  --result-root ako_runs/controlled_followup/effort_frontier_v1/results/search_v1 \
  --confirmation-plan ako_runs/controlled_followup/effort_frontier_v1/results/confirmation_v1/plan.json \
  --confirmation-records ako_runs/controlled_followup/effort_frontier_v1/results/confirmation_v1/records.jsonl \
  --analysis ako_runs/controlled_followup/effort_frontier_v1/results/analysis_v1.json

python ako_runs/controlled_followup/effort_frontier_v1/capture_evidence.py verify \
  --index ako_runs/controlled_followup/effort_frontier_v1/evidence/effort_frontier_complete_v1.index.json
```
