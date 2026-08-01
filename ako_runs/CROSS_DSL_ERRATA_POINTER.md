# Controlled cross-DSL report correction pointer

The historical controlled cross-DSL reports now carry a one-line pointer to the
current interpretation. The top-level report also corrects one archived-Triton
artifact description. Their pre-edit and post-edit identities are retained by
SHA-256 in the correction receipt. Current documents are:

- [`controlled_followup/REVIEW_ROUND2_RESPONSE_20260731.md`](controlled_followup/REVIEW_ROUND2_RESPONSE_20260731.md)
- [`controlled_followup/RUN_20260731.md`](controlled_followup/RUN_20260731.md)
- [`controlled_followup/ERRATA_20260730.md`](controlled_followup/ERRATA_20260730.md)
- [`controlled_followup/REVIEW_RESPONSE_20260730.md`](controlled_followup/REVIEW_RESPONSE_20260730.md)
- [`controlled_followup/REVIEW_ROUND2_20260730.md`](controlled_followup/REVIEW_ROUND2_20260730.md)
- [`controlled_followup/robust_gate/audits/matmul_v4_instrument_v1/results/summary.json`](controlled_followup/robust_gate/audits/matmul_v4_instrument_v1/results/summary.json)
- [`controlled_followup/robust_gate/audits/matmul_v4_instrument_v1/results/margin_report_v2.json`](controlled_followup/robust_gate/audits/matmul_v4_instrument_v1/results/margin_report_v2.json)
- [`controlled_followup/provenance/historical_document_corrections_20260731.json`](controlled_followup/provenance/historical_document_corrections_20260731.json)

These documents correct scope, provenance, artifact-version, robustness, and
matmul-v4 legality interpretations without changing any frozen gate or result.
