# Controlled follow-up umbrella evidence v2

This directory contains an append-only, post-hoc evidence-preservation
builder. It does not change a campaign source, result, or receipt and does not
claim an external timestamp or preregistration.

The fixed core selection contains:

- `REVIEW_RESPONSE_20260730.md` and `ERRATA_20260730.md`;
- the original `evidence_v1` index and bundle;
- receipt-bound source plus complete gate/performance raw evidence for
  `fused_closure_v2`;
- receipt-bound source, receipts, raw streams, and summary for the prior-winner
  `fused_row_sum_stress_v1` audit; and
- the exact, independently verifiable index/bundle pairs for reachability v2,
  streamed reachability row-sum stress, archived/current reconstruction, and
  frontier closure v3.

Compiled objects, CUDA/Triton build products, caches, active locks, partial
streams, temporary files, and nested evidence directories discovered by broad
directory traversal are refused. The five specifically named nested bundles
are evidence payloads, not executable build products.

The matmul-v4 audit is deliberately deferred. Inspect the complete core
selection without writing an archive:

```bash
python ako_runs/controlled_followup/provenance/evidence_v2/build_evidence.py dry-run
```

After that audit has all six final raw streams, a bound completion receipt, and
a bound summary, test the deferred include:

```bash
python ako_runs/controlled_followup/provenance/evidence_v2/build_evidence.py dry-run \
  --include-matmul-v4
```

Seal the completed fused campaigns while explicitly recording that active
matmul-v4 is deferred:

```bash
python ako_runs/controlled_followup/provenance/evidence_v2/build_evidence.py build \
  --name fused_postreview_v1 --defer-active-matmul-v4
```

A later superseding build is fail-closed unless `--include-matmul-v4` is
supplied and that audit validates as complete:

```bash
python ako_runs/controlled_followup/provenance/evidence_v2/build_evidence.py build \
  --name complete_with_matmul_v1 --include-matmul-v4
python ako_runs/controlled_followup/provenance/evidence_v2/build_evidence.py verify \
  --index ako_runs/controlled_followup/provenance/evidence_v2/complete_with_matmul_v1.index.json
```

Output names are immutable. The archive is a deterministic GNU tar stream in
gzip with timestamp zero and an empty gzip filename. Every regular member uses
mode `0644`, timestamp `0`, uid/gid `0`, and empty owner/group names. The index
and embedded manifest use canonical JSON.
