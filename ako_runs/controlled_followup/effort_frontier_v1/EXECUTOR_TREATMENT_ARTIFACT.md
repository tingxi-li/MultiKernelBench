# RQ5 executor treatment artifact and blocker classification

This note distinguishes a missing experimental treatment artifact from an
unfinished campaign controller.

## Why `executor_registry.json` is not generated from existing runners

The repository does not currently contain four interchangeable executors for
arbitrary model-proposed implementations of this exact fused contract.
Automatically naming existing scripts in a registry would create false
readiness and asymmetric treatments:

- `fused_epilogue_crossed_v1` and the Phase-2 `variants2` builders construct a
  finite, checked configuration grid. They do not accept arbitrary candidate
  source emitted during an optimization trajectory.
- The generic MultiKernelBench evaluator can execute generated CUDA and Triton
  Python modules, but it has no CUDA TileLang backend, no cuBLASLt lane, no
  frozen fused-v2 4-case × 64-seed gate, no tuning/terminal split, and no RQ5
  lane-policy or GPU-identity evidence contract.
- The repository has no cuBLASLt implementation of this workload. In
  particular, no audited `GELU_BIAS`/exact-softmax wrapper exists to hash. A
  Phase-1 CUDA GEMM or Phase-2 hand-CUDA builder is not a cuBLASLt substitute.
- Existing timing runners consume frozen candidate objects/configurations and
  campaign-specific receipts. Adapting them to generated source would require
  choosing a candidate ABI, import surface, compiler flags, artifact cache,
  lane purity rules, error semantics, and cuBLASLt postprocess. Those choices
  affect search difficulty and therefore are part of the RQ5 treatment.

Consequently, deriving a registry from those runners would favor lanes with an
existing generated-code harness, mislabel fixed builders as open-ended search,
and leave the new cuBLASLt treatment undefined. The validator deliberately
refuses that substitution.

## What must be frozen as the treatment artifact

Before preregistration is made launch-controlling, an independently reviewed
artifact must provide:

1. One common candidate-source ABI and identical base task information for all
   programmable lanes, with only the assigned lane capability differing.
2. Four source-to-callable executors (`cublaslt`, `triton`, `tilelang`, and
   `cuda_unlimited`) plus the nonprogrammable PyTorch control executor.
3. Auditable lane-policy checks. Passing source must use only its named
   implementation surface; fallback to PyTorch, another DSL, a fixed incumbent,
   or an out-of-lane vendor GEMM is ineligible.
4. The frozen fused-v2 tuning gate and a genuinely hidden terminal holdout,
   with complete gate evidence hashes and no terminal metrics returned to the
   optimizer.
5. Identical shape/dtype/input/timing semantics, GPU UUID observation, 25/100
   confirmation timing, and exact positive/signed input generation.
6. For cuBLASLt, a checked CUDA-13.1 wrapper that records whether
   `GELU_BIAS` is expressible and fused-v2 legal, and otherwise permits only the
   preregistered exact nonvendor postprocess. Its claim remains
   implementation-specific, never vendor-expert.

The resulting `locks/executor_registry.json` must contain frozen argv arrays,
timeouts, and SHA-256 maps for every source above. The existing validator then
content-checks it, the prelaunch provenance lock binds it, each trajectory
binds it at event zero, and complete evidence archives its sources.

This artifact is intentionally not fabricated with placeholder commands or
hashes. Creating it is substantive treatment construction and independent
review work, not safe mechanical reuse.

## Blocker classes as of 2026-07-31

Treatment artifact (must be constructed and frozen before launch):

- The executor registry described above.

External environment/provider blockers:

- A provider-attested immutable revision for `gpt-5.6-sol` is unavailable; the
  checked-in model lock is explicitly `unresolved`.
- `OPENAI_API_KEY` is absent.
- The NVIDIA driver is unavailable to `nvidia-smi`, so all four frozen Ada UUIDs
  cannot be revalidated.

Prelaunch coordination blockers:

- The completed campaign and treatment sources must be committed, pushed, and
  externally timestamped from a clean worktree. Only then can
  `prelaunch_provenance.json` truthfully be created.

Locally resolvable controller/protocol gaps:

- None known after static validation and the CPU-only protocol suite. The
  controller, event clock, analyzers, confirmation lifecycle, validation, and
  deterministic evidence lifecycle are implemented. This statement does not
  reclassify the executor treatment artifact as complete.
