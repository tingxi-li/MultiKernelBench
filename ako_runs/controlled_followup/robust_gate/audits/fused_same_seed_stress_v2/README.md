# Corrected fused same-seed robustness stress v2

This correctness-only campaign evaluates exactly ten frozen candidates on the
same 512 signed-normal, gain-16 inputs under both registered fused-v2 mixed
gates. The first 256 seed tuples exactly replay the old-winner stress namespace;
the remaining 256 tuples use a newly frozen namespace. The effective sample
size is 512 shared inputs, never 1,024 gate views or 10,240 records.

The preserved v1 launcher is non-controlling: it used the wrong namespace,
never produced measurements, and has not been overwritten or deleted. Its
byte-level disposition is recorded in `legacy_v1_status.json`.

Lifecycle:

```bash
python -m ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v2.freeze --freeze
python -m ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v2.freeze --prepare-launch
python -m ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v2.launch --preflight
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 \
  python -m ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v2.runner
python -m ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v2.analyze
python -m ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v2.capture_evidence build --name complete_v1
```

The runner is fail-closed and append-only. An interrupted `.partial` stream is
validated and resumed without duplicating completed candidate/gate/seed keys.
Build, setup, reference, execution, and metric failures remain records and
consume their preregistered cells. No timing or performance selection is
authorized.

