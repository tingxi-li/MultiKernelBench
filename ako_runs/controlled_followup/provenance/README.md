# Provenance capture

Create the snapshot immediately before a campaign launch, after its protocol,
gate specification, and job manifest have been frozen:

```bash
python capture.py snapshot \
  --campaign 20260729_controlled_followup_v1 \
  --out snapshots/20260729_controlled_followup_v1.json
python capture.py verify snapshots/20260729_controlled_followup_v1.json
```

`verify` is expected to fail after source changes.  That is an integrity signal,
not a reason to update a completed campaign's snapshot.  Make a new campaign ID
and snapshot for a changed protocol or implementation.

Limited corrections to historical Markdown are separately recorded in
[`historical_document_corrections_20260731.json`](historical_document_corrections_20260731.json),
which preserves each document's pre-edit and post-edit SHA-256. This receipt is
not a substitute for a prelaunch campaign snapshot.

## Round-2 implementation handoff

[`evidence_v3/`](evidence_v3/) packages the corrected documents, completed
matmul-v4 and same-seed evidence, plain closure/reachability files, and the
fixed preregistration bundles for the four prospective programs. It excludes
active locks, partials, caches/build products, prospective result trees, and
the non-controlling evidence-v2 v3--v8 archives.

```bash
python evidence_v3/build_evidence.py dry-run
python evidence_v3/build_evidence.py build --name round2_complete_v1
python evidence_v3/build_evidence.py verify \
  --index evidence_v3/round2_complete_v1.index.json
```

“Complete” names the implementation/evidence handoff. The embedded state stays
`empirical_round2_program_complete=false` until a future, separately named
bundle can truthfully bind all completed experiments.
