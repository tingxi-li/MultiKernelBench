# Crossed-v1r1 provenance supplement

This append-only supplement closes the self-containment gap in the completed
`crossed_v1r1` primary evidence bundle. It does not alter or reinterpret any
experimental result.

The archive embeds the verified primary evidence archive/index, the complete
49-file union of all parent and recovery source/dependency lock maps, the
complete final summary, the recovery lock, and the frozen supplement builder.
The primary archive omitted 23 parent-locked files; those bytes are included
and checked against the original launch lock here.

The standalone verifier needs only the supplement archive and its external
index. It recursively validates the embedded primary archive, reconstructs all
four source/dependency maps from the embedded locks, checks normalized archive
metadata and every member hash, and proves that the direct final-summary bytes
match the copy in the primary archive.

From the repository root:

```bash
python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.provenance_supplement_v1.freeze --write
python -m pytest -q ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/provenance_supplement_v1/tests
python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.provenance_supplement_v1.build build
python -m ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.provenance_supplement_v1.build verify
```

Publication is exclusive: neither the primary pair nor an existing supplement
output is ever replaced.
