# Crossed epilogue v1r1 reporting recovery

This append-only recovery preserves the pushed v1 protocol and its failed
four-shard launch. The v1 audit completed the first cell's 512 frozen gate
evaluations in each process, then all four processes stopped in the diagnostic
`gate_summary`: exact-zero `negative_count` and `nonfinite_count` constraints
were incorrectly treated as denominators. No gate JSONL, cell outcome, shard
status, timing record, or performance result was sealed.

`incident_receipt.json` content-addresses the five retained v1 artifacts and
records the four identical `ZeroDivisionError` exits. The parent source, cell
manifest, gate thresholds, gate decisions, candidate builders, execution
order, and analysis estimands remain byte-identical.

The only computational change is reporting:

- strictly positive thresholds retain `value / threshold` utilization;
- exact-zero count constraints are reported as observed maxima and violation
  counts, never as ratios;
- eligibility continues to use the frozen per-row `gate_pass` decisions.

All v1r1 JSON and JSONL receives both the original parent identities and the
recovery source/lock/commit binding. The old `crossed_v1` result tree is never
resumed or modified; recovery writes only to `results/crossed_v1r1`.

## Freeze and launch

From the repository root:

```bash
python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.freeze --write
python -m pytest -q ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/tests
python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.validate
```

Commit the recovery sources and `recovery_lock.json`, push that exact commit,
then require host-level readiness before launching:

```bash
python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.validate --launch-ready --gpu 0

python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.audit --tag crossed_v1r1 --gpu 0 --shard-index 0 --shard-count 4
python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.audit --tag crossed_v1r1 --gpu 1 --shard-index 1 --shard-count 4
python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.audit --tag crossed_v1r1 --gpu 2 --shard-index 2 --shard-count 4
python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.audit --tag crossed_v1r1 --gpu 3 --shard-index 3 --shard-count 4
```

Analyze and time through the recovery wrappers so every derived artifact keeps
the v1r1 binding:

```bash
python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.analyze audit \
  --result-root ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1r1 \
  --out ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1r1/audit_summary.json

python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.launch screen \
  --tag crossed_v1r1 --gpu 0 \
  --eligibility ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1r1/audit_summary.json
```

Use the same recovery `analyze` and `launch` modules for confirmation. Complete
evidence is accepted only from a bound complete final summary.

No v1r1 launch is authorized merely by this directory. Its exact commit must
be independently present on the configured upstream and the live GPU check
must pass.
