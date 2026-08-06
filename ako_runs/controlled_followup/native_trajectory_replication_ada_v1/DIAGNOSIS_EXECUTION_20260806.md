# Native trajectory v1 execution diagnosis

Date: 2026-08-06 UTC
Status: **ROOT CAUSE CONFIRMED; V1 REMAINS NON-CONTROLLING**

This non-frozen memo interprets the immutable incident receipt
`INCIDENT_EXECUTION_20260806.json` (SHA-256
`fa23bbcfc7063c145c911585d809e82f51d14afe0b0a987608c1fe75a77a68b7`).
No v1 source or retained result was edited, and no v1 record may be retried or
reused.

## Failure

The first planned record, `register_fused.cuda_unlimited.g01` on positive
inputs, stopped before correctness and timing because the fresh build did not
match the implementation identity imported from `crossed_v2r3`. Thus v1 has
zero performance observations.

## Root cause

The frozen shared fingerprint walker in
`fused_epilogue_crossed_v2/candidates.py` classifies every string-valued field
whose key *contains* `source` as source material:

```python
if isinstance(item, str) and "source" in str(key):
```

`"source" in "kernel_resources_error"` is true. A cache-hit CUDA build records
`kernel_resources_error = "ptxas log contains 0 entries for mma_gemm"`, while
a fresh compile records parsed kernel resources and no error string. The
diagnostic therefore entered the alleged implementation identity even though
the generated CUDA source was unchanged.

For this cell, the CUDA source SHA-256 is stable at
`6e8b43c6a621cc4df9968dbd0c334db33ebbf7a83ef385618ee2477a230c5bf5`.
Recomputing the frozen formula from actual source fields only gives
`6c7f064216ef5c4a83f5956d78d87c7cf9b9c7207bcb3bf164584b789081841a`.
Adding the cache-only `kernel_resources_error` field reproduces the imported
identity exactly:
`993ea554e4f3a2578425e40aca3d287b9eddbc2fc1d1289704760b6c03b86641`.
A subsequent cache-hit diagnostic rebuild also reproduced `993ea554...`, two
kernels, and the same CUDA source hash.

## Consequence and successor rule

This is a measurement-instrument identity bug, not evidence of implementation
drift and not a performance result. The successor must admit artifacts before
timing, bind generated source and loadable code objects explicitly, ignore
diagnostic/resource fields when defining implementation identity, and require
timing children to load only those admitted artifacts. V1 remains preserved as
a fail-closed incident under its original lock.
