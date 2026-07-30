# Matmul v4 fixed-threshold instrument audit

This is a new append-only campaign. It does not amend, copy, or refit the
matmul-v4 gate. `manifest.json` binds the original gate byte-for-byte and
canonically, and every production row repeats those bindings.

The campaign has three independent endpoints:

1. Four mandatory 512-seed-per-case native replications (`r0` through `r3`).
   Each block is judged separately; a pooled result cannot rescue a block.
2. Registered wrong-answer controls under every applicable frozen gate. The
   paired-cancellation zero/row-roll/column-roll cases are exact-zero exclusions:
   they must pass, not be mislabeled as successful rejections. NaN and wrong-shape
   controls separately verify fail-closed preflight.
3. First fixed-gate contact with the Phase-1 Triton A/B/C/D kernels: A is checked
   under `semantic_q32`; B/C/D are each checked under both mixed gates.

All pseudorandom namespaces are new and pairwise distinct. A seed identifies one
input tensor pair; tensor elements and output rows are not treated as independent
samples.

The original gate does not use ordinary NRMSE. For a nonzero wrong answer
against the bitwise-zero paired-cancellation reference, that diagnostic alone
would overflow under the canonical tiny-denominator convention. The runner
saturates only this ungated diagnostic at finite float64 maximum so the five
unchanged registered gate metrics can make and record the intended rejection.

## Workflow

Run CPU contract tests and optional smoke collection before freezing:

```bash
python -m unittest discover -s ako_runs/controlled_followup/robust_gate/audits/matmul_v4_instrument_v1/tests -t . -v
python -m ako_runs.controlled_followup.robust_gate.audits.matmul_v4_instrument_v1.runner \
  --arm synthetic --device cpu --cpu-smoke --max-seeds 1 --out /tmp/matmul_v4_audit_smoke.jsonl
```

After all source and manifest review is complete, create one-time receipts:

```bash
python -m ako_runs.controlled_followup.robust_gate.audits.matmul_v4_instrument_v1.freeze --freeze
python -m ako_runs.controlled_followup.robust_gate.audits.matmul_v4_instrument_v1.freeze --prepare-launch
```

The launch receipt contains the exact six production commands. Every command is
pinned to physical GPU 1 with `CUDA_VISIBLE_DEVICES=1`. The runner refuses a
different GPU binding, an unregistered output, a changed source bundle, a
changed original gate, or an existing result/partial file.

When a process is interrupted, completed rows remain in `OUTPUT.partial`.
Ordinary kernel, control, and preflight failures are rows rather than reasons to
discard a run. Once all six complete raw files exist, analyze them with:

```bash
python -m ako_runs.controlled_followup.robust_gate.audits.matmul_v4_instrument_v1.analyze
```

The summary reports evidence completeness, every block/candidate/gate endpoint,
and every case separately. A failed real candidate is a retained result; it does
not reopen or authorize mutation of the v4 thresholds.
