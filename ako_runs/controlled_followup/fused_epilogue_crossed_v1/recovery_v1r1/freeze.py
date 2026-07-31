#!/usr/bin/env python3
"""Create or verify the append-only crossed-v1r1 recovery lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from . import common
except ImportError:  # pragma: no cover - direct-script CLI
    import common  # type: ignore


SOURCE_RELATIVES = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/__init__.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/README.md",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/analyze.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/audit.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/capture_evidence.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/common.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/freeze.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/incident_receipt.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/launch.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/run_one.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/validate.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/tests/__init__.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/tests/test_recovery.py",
)

DEPENDENCY_RELATIVES = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/analyze.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/audit.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/campaign.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/capture_evidence.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/cells.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/core.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/launch.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/launch_lock.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/run_one.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/validate.py",
    "ako_runs/controlled_followup/fused_grid/robust_adapter.py",
    "ako_runs/controlled_followup/fused_grid/robust_adapter_manifest.json",
    "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json",
)


def _hash_map(relatives: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for relative in relatives:
        path = common.REPO_ROOT / relative
        common.require(
            path.is_file() and not path.is_symlink(),
            f"freeze input missing/symlinked: {relative}",
        )
        values[relative] = common.file_sha256(path)
    return values


def expected_lock() -> dict[str, Any]:
    common.verify_incident_receipt()
    sources = _hash_map(SOURCE_RELATIVES)
    dependencies = _hash_map(DEPENDENCY_RELATIVES)
    return {
        "campaign_id": common.CAMPAIGN_ID,
        "dependency_bundle_sha256": common.canonical_sha256(dependencies),
        "dependency_sha256": dependencies,
        "incident_receipt_sha256": common.file_sha256(common.INCIDENT_PATH),
        "parent_git_commit": common.PARENT_GIT_COMMIT,
        "parent_launch_lock_sha256": common.PARENT_LAUNCH_LOCK_SHA256,
        "parent_source_bundle_sha256": common.PARENT_SOURCE_BUNDLE_SHA256,
        "record_type": "fused_crossed_v1r1_recovery_lock",
        "recovery_id": common.RECOVERY_ID,
        "reporting_change": {
            "exact_zero_thresholds": "report maxima and violation counts; do not divide",
            "frozen_gate_decisions_changed": False,
            "frozen_thresholds_changed": False,
            "positive_thresholds": "retain value/threshold utilization",
        },
        "result_tag": common.RESULT_TAG,
        "schema_version": 1,
        "source_bundle_sha256": common.canonical_sha256(sources),
        "source_sha256": sources,
    }


def verify() -> dict[str, Any]:
    common.require(common.LOCK_PATH.is_file(), "recovery lock is missing")
    observed = common.read_json(common.LOCK_PATH)
    expected = expected_lock()
    common.require(observed == expected, "recovery lock differs from current bytes")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        common.require(not common.LOCK_PATH.exists(), "recovery lock already exists")
        common.exclusive_json(common.LOCK_PATH, expected_lock())
    value = verify()
    print(
        json.dumps(
            {
                "dependency_bundle_sha256": value["dependency_bundle_sha256"],
                "files": len(value["source_sha256"]),
                "source_bundle_sha256": value["source_bundle_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
