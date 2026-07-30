# Post-hoc fused-run evidence bundle v1

This directory preserves the essential ignored result bytes from the completed
fused screen, robust validation, and confirmation. It is deliberately labeled
**post hoc**: the index can show that the preserved bytes match the completed
receipts and summaries, but it cannot establish preregistration, a launch-time
source freeze, or an external timestamp.

`build_evidence.py` is read-only with respect to
`controlled_followup/fused_grid/results/`. It selects and hashes:

- all 152 screen process records, the launch/status files, both screen
  summaries, and the frozen confirmation selection;
- all 80 confirmation process records and its launch/status/analysis summary;
- the aggregate `records.jsonl` stream and run/summary controls for the robust
  smoke, tuning, and locked-validation runs, including the validation
  normalization receipt.

Build caches, binaries, scratch runner copies, and redundant per-record robust
files are excluded. The selected aggregate JSONL streams contain 608, 3,520,
and 28,672 records. A normalized GNU tar archive is compressed with gzip level
9, an empty filename header, and `mtime=0`, so identical inputs produce
identical bundle bytes.

Build and verify from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  ako_runs/controlled_followup/provenance/evidence_v1/build_evidence.py build

PYTHONDONTWRITEBYTECODE=1 python \
  ako_runs/controlled_followup/provenance/evidence_v1/build_evidence.py verify
```

The build writes `evidence_bundle.tar.gz` and `evidence_index.json` only in this
directory. Verification rehashes every live source, checks the expected record
counts, verifies the bundle hash, and rehashes every archive member. Completed
result files and historical launch receipts are never rewritten.
