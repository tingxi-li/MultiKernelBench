#!/usr/bin/env python3
"""Build or verify the deterministic pre-launch v3 source/evidence receipt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.fused_frontier_closure_v3 import (
        core,
        eligibility,
    )
else:  # pragma: no cover
    from . import core, eligibility


LOCAL_SOURCE_NAMES = (
    "README.md",
    "PREREGISTRATION.md",
    "__init__.py",
    "analyze.py",
    "campaign.json",
    "candidates.py",
    "capture_evidence.py",
    "core.py",
    "eligibility.py",
    "eligibility_receipt.json",
    "launch.py",
    "measure.py",
    "provenance.py",
    "test_frontier.py",
)

EVIDENCE_PATHS = (
    "ako_runs/controlled_followup/fused_closure_v2/source_receipt.json",
    "ako_runs/controlled_followup/fused_closure_v2/results/gate_v1/gate_summary.json",
    "ako_runs/controlled_followup/fused_closure_v2/results/performance_v1/launch_receipt.json",
    "ako_runs/controlled_followup/fused_closure_v2/results/performance_v1/analysis_summary.json",
    "ako_runs/controlled_followup/fused_reachability_v2/launch_lock.json",
    "ako_runs/controlled_followup/fused_reachability_v2/evidence/complete_v1.index.json",
    "ako_runs/controlled_followup/fused_reachability_v2/evidence/complete_v1.tar.gz",
    "ako_runs/controlled_followup/fused_reachability_v2/results/screen_analysis_v1/selection.json",
    "ako_runs/controlled_followup/fused_reachability_v2/results/screen_analysis_v1/confirmation_selection.json",
    "ako_runs/controlled_followup/fused_reachability_v2/results/confirmation_analysis_v1/confirmation_summary.json",
    "ako_runs/controlled_followup/fused_reachability_v2/results/robust_noptx_v1/summary.json",
    "ako_runs/controlled_followup/fused_reachability_v2/results/robust_noptx_v1/records.jsonl",
    "ako_runs/controlled_followup/fused_reachability_v2/results/robust_unlimited_v1/summary.json",
    "ako_runs/controlled_followup/fused_reachability_v2/results/robust_unlimited_v1/records.jsonl",
)


def _hash_paths(paths) -> dict[str, str]:
    result = {}
    for relative in paths:
        path = core.REPO_ROOT / relative
        if not path.is_file():
            raise core.ClosureError(f"missing receipt input: {relative}")
        result[relative] = core.sha256_file(path)
    return result


def _local_hashes() -> dict[str, str]:
    return _hash_paths(
        tuple(
            str((core.HERE / name).relative_to(core.REPO_ROOT))
            for name in LOCAL_SOURCE_NAMES
        )
    )


def _transitive_sources() -> dict[str, str]:
    from ako_runs.controlled_followup.fused_closure_v2 import (
        provenance as closure_provenance,
    )
    from ako_runs.controlled_followup.fused_reachability_v2 import protocol

    closure = closure_provenance.verify_receipt()
    reach = protocol.verify_lock()
    declared = {
        **closure["local_source_sha256"],
        **closure["external_source_sha256"],
        **reach["source_sha256"],
    }
    current = _hash_paths(tuple(sorted(declared)))
    if current != declared:
        raise core.ClosureError("transitive candidate source map differs")
    return current


def expected_receipt() -> dict[str, Any]:
    campaign = core.load_campaign()
    eligibility_receipt = eligibility.verify_receipt()
    return {
        "schema_version": 1,
        "record_type": "fused_frontier_closure_v3_source_receipt",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": core.canonical_sha256(campaign),
        "candidate_sha256": {
            row["candidate_id"]: core.candidate_sha256(row)
            for row in campaign["candidates"]
        },
        "performance_protocol_sha256": core.protocol_sha256(campaign),
        "eligibility_receipt_sha256": core.sha256_file(
            core.ELIGIBILITY_RECEIPT_PATH
        ),
        "eligibility_receipt_canonical_sha256": core.canonical_sha256(
            eligibility_receipt
        ),
        "local_source_sha256": _local_hashes(),
        "transitive_candidate_source_sha256": _transitive_sources(),
        "imported_evidence_sha256": _hash_paths(EVIDENCE_PATHS),
        "freeze_policy": (
            "deterministic pre-launch receipt; any byte change requires a new "
            "receipt and result tag"
        ),
    }


def verify_receipt(path: Path = core.SOURCE_RECEIPT_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise core.ClosureError(f"missing source receipt: {path}")
    observed = core.read_json(path)
    expected = expected_receipt()
    if observed != expected or path.read_bytes() != core.stable_json_bytes(observed):
        raise core.ClosureError("source receipt differs from frozen inputs")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.emit:
        print(core.stable_json_bytes(expected_receipt()).decode("utf-8"), end="")
    else:
        receipt = verify_receipt()
        print(
            f"OK {receipt['campaign_id']} source receipt "
            f"sha256={core.sha256_file(core.SOURCE_RECEIPT_PATH)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
