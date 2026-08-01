# Evidence builder fix v1

This is an append-only correction to the already-frozen `production_v1`
evidence builder.  The frozen builder has SHA-256
`5e74703976d1844143fff232c82102da000af4b5eb437fff2498195491c440ff`.
Its `_write_bundle` loop reused the destination variable name for each source
member; after writing a temporary archive, the exclusive hard-link therefore
targeted the last source member (`production_v1/receipts/source_freeze.json`)
instead of `evidence/prereg_v2.tar.gz`.  The existing receipt correctly caused
`FileExistsError`.  No evidence bundle/index, treatment output, source, GPU
record, or success lock was created.

The faulty source and production-v1 freeze are preserved byte-for-byte.  This
directory changes only the producer: it uses distinct `bundle_path` and
`source_path` variables, binds both the defective and effective builders by
hash, has its own immutable source freeze, and adds itself to `prereg_v2`.
Verification remains compatible with the frozen verifier, whose archive read
path was not affected.

Lifecycle from the repository root:

```bash
python -m unittest ako_runs.controlled_followup.reciprocal_v2.production_v1.evidence_fix_v1.test_evidence_fix_v1
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.evidence_fix_v1.freeze --freeze
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.evidence_fix_v1.capture_evidence build --name prereg_v2
python -m ako_runs.controlled_followup.reciprocal_v2.production_v1.evidence_fix_v1.capture_evidence verify \
  --index ako_runs/controlled_followup/reciprocal_v2/evidence/prereg_v2.index.json
```

The repair does not broaden the scientific protocol or claim treatment
completion.  It only makes the intended deterministic preregistration archive
creatable and records the supersession explicitly.
