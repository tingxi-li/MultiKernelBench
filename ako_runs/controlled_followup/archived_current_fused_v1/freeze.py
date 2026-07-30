#!/usr/bin/env python3
"""Create the deterministic pre-launch source, job, and launch-lock receipts."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.archived_current_fused_v1 import protocol
else:  # pragma: no cover
    from . import protocol


def _preflight_targets(campaign: dict) -> None:
    for subject in campaign["subjects"]:
        path = protocol.REPO_ROOT / subject["source_path"]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        if "Model" not in classes:
            raise protocol.CampaignError(
                f"faithful loader requires a top-level Model class: {path}"
            )


def main() -> int:
    campaign = protocol.load_campaign()
    _preflight_targets(campaign)
    targets = (
        (protocol.SOURCE_RECEIPT_PATH, protocol.expected_receipt(campaign)),
        (protocol.JOBS_PATH, protocol.expected_jobs(campaign)),
        (protocol.LOCK_PATH, protocol.expected_lock(campaign)),
    )
    existing = [str(path) for path, _ in targets if path.exists()]
    if existing:
        raise protocol.CampaignError(
            "refusing to overwrite frozen artifacts; verify instead or create v2: "
            + ", ".join(existing)
        )
    for path, value in targets:
        protocol.atomic_json(path, value)
    _campaign, receipt, jobs, lock = protocol.verify_lock()
    print(
        f"FROZEN {campaign['campaign_id']} sources={len(receipt['source_sha256'])} "
        f"jobs={jobs['expected_records']} lock_sha256={protocol.sha256_file(protocol.LOCK_PATH)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

