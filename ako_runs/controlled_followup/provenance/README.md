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
