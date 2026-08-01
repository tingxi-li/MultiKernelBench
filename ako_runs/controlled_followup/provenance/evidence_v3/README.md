# Round-2 umbrella evidence v3

`build_evidence.py` creates a deterministic, post-hoc preservation bundle. It
does not turn a local file into an external preregistration and it does not
claim that every round-2 campaign ran.

The selector includes:

- the review, response, run-ledger, errata, historical-report pointers, and
  hash-bound historical-correction receipt;
- receipt-bound completed matmul-v4 results, including all six raw streams and
  the controlling margin-v2 report;
- receipt- and archive-bound completed same-seed-v2 evidence (`10,240/10,240`
  records over 512 effective shared seeds);
- every plain eligible file in the completed closure, reachability, and
  frontier-closure trees, including the formerly gitignored reachability
  results;
- all four new campaign protocols: convergence v2, crossed epilogue v1,
  reciprocal v2, and effort frontier v1;
- ten fixed, hash-checked index/archive pairs, including same-seed-v2
  completion and reciprocal-v2 base/production, convergence-v2, and effort-frontier
  preregistration evidence.

Loose selection excludes Python/test/extension caches, compiled objects and
CUDA/Triton build products, runtime `*.lock` files, and unvalidated result trees
for prospective campaigns. A partial or temporary file in a selected tree is a
hard error. Blocked preflight JSON receipts are included only when they
explicitly say launch is not ready; they are not empirical results. The
non-controlling `provenance/evidence_v2/fused_postreview_v3` through `v8`
archives are never selected.

Nested archives are preserved byte-for-byte and verified against fixed hashes,
their indexes, embedded manifests, membership, sizes, and payload hashes. This
means an older sealed nested archive may itself preserve historical runtime
sentinels; those bytes are not promoted to loose v3 entries or rewritten.

The conventional output name is `round2_complete_v1`. “Complete” refers only
to the implementation/evidence handoff. The embedded manifest and external
index record the current campaign states and always keep
`empirical_round2_program_complete` false while prospective programs have no
validated results.

Run the dry-run and CPU tests while campaign sources are still changing:

```bash
python ako_runs/controlled_followup/provenance/evidence_v3/build_evidence.py dry-run
python -m unittest -v \
  ako_runs.controlled_followup.provenance.evidence_v3.test_evidence_v3
```

After all campaign agents have stopped editing, rerun those checks and create
the immutable archive exactly once:

```bash
python ako_runs/controlled_followup/provenance/evidence_v3/build_evidence.py build \
  --name round2_complete_v1
python ako_runs/controlled_followup/provenance/evidence_v3/build_evidence.py verify \
  --index ako_runs/controlled_followup/provenance/evidence_v3/round2_complete_v1.index.json
```

Publication uses temporary files, full archive self-verification, and
exclusive hard-link publication. Existing outputs are never overwritten.
