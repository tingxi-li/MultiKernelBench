#!/usr/bin/env python3
"""Bind complete frozen-gate summaries into an immutable confirmation set."""
from __future__ import annotations

import argparse
from pathlib import Path

from protocol import LOCK, file_sha256, read_json, stable_write, verify_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-selection", required=True)
    parser.add_argument("--robust-noptx", required=True)
    parser.add_argument("--robust-unlimited", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def validate_summary(
    path: Path,
    lane: str,
    selection_sha256: str,
    selected_jobs: set[str],
    lock: dict,
) -> dict:
    summary = read_json(path)
    if summary.get("prospective_campaign_id") != lock["campaign_id"]:
        raise RuntimeError(f"foreign robust campaign: {path}")
    if summary.get("lane") != lane:
        raise RuntimeError(f"robust lane mismatch: {path}")
    if summary.get("selection_sha256") != selection_sha256:
        raise RuntimeError(f"robust summary selection mismatch: {path}")
    if summary.get("source_bundle_sha256") != lock["source_bundle_sha256"]:
        raise RuntimeError(f"robust summary source mismatch: {path}")
    if summary.get("complete_frozen_validation_split") is not True:
        raise RuntimeError(f"robust summary is not the full validation split: {path}")
    coverage = summary.get("launch_coverage", {})
    if coverage.get("complete") is not True:
        raise RuntimeError(f"robust summary has incomplete coverage: {path}")
    if set(summary.get("selected_jobs", [])) != selected_jobs:
        raise RuntimeError(f"robust summary selected-job mismatch: {path}")
    if summary.get("gate_spec_sha256") != lock["frozen_gate"]["gate_spec_sha256"]:
        raise RuntimeError(f"robust summary gate mismatch: {path}")
    receipt_path = path.parent / "launch_receipt.json"
    receipt = read_json(receipt_path)
    if (
        receipt.get("campaign_id") != lock["campaign_id"]
        or receipt.get("lane") != lane
        or receipt.get("selection_sha256") != selection_sha256
        or set(receipt.get("selected_jobs", [])) != selected_jobs
        or receipt.get("source_bundle_sha256") != lock["source_bundle_sha256"]
        or receipt.get("gate_spec_sha256") != lock["frozen_gate"]["gate_spec_sha256"]
        or receipt.get("seed_indices") != list(range(64))
    ):
        raise RuntimeError(f"robust launch receipt mismatch: {receipt_path}")
    return summary


def main() -> int:
    args = parse_args()
    lock = verify_lock()
    screen_path = Path(args.screen_selection)
    screen = read_json(screen_path)
    screen_hash = file_sha256(screen_path)
    if screen.get("campaign_id") != lock["campaign_id"]:
        raise RuntimeError("screen selection campaign mismatch")
    selected_by_lane = {
        lane: {
            row["job_id"] for row in screen.get("selected", []) if row.get("lane") == lane
        }
        for lane in ("cuda_noptx", "cuda_unlimited")
    }
    if any(len(values) != 3 for values in selected_by_lane.values()):
        raise RuntimeError("screen selection must contain three jobs per lane")
    summaries = {
        "cuda_noptx": (
            Path(args.robust_noptx),
            validate_summary(
                Path(args.robust_noptx), "cuda_noptx", screen_hash,
                selected_by_lane["cuda_noptx"], lock,
            ),
        ),
        "cuda_unlimited": (
            Path(args.robust_unlimited),
            validate_summary(
                Path(args.robust_unlimited), "cuda_unlimited", screen_hash,
                selected_by_lane["cuda_unlimited"], lock,
            ),
        ),
    }
    selected = []
    for row in screen["selected"]:
        lane = row["lane"]
        summary = summaries[lane][1]
        groups = [group for group in summary.get("groups", []) if group.get("grid_job_id") == row["job_id"]]
        gate_ids = {group.get("gate_id") for group in groups}
        passed = bool(
            len(groups) == 2
            and gate_ids == {"semantic_mixed", "conformance_mixed"}
            and all(
                group.get("success") is True
                and group.get("coverage_complete") is True
                and group.get("n_failed_records") == 0
                for group in groups
            )
        )
        selected.append({**row, "robust_eligible": passed})
    value = {
        "schema_version": 1,
        "record_type": "fused_reachability_v2_confirmation_selection",
        "campaign_id": lock["campaign_id"],
        "launch_lock_sha256": file_sha256(LOCK),
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "screen_selection_path": str(screen_path),
        "screen_selection_sha256": screen_hash,
        "robust_summaries": {
            lane: {"path": str(path), "sha256": file_sha256(path)}
            for lane, (path, _summary) in summaries.items()
        },
        "selected": selected,
        "status": (
            "COMPLETE_ALL_PASS" if all(row["robust_eligible"] for row in selected)
            else "COMPLETE_WITH_FAILURES"
        ),
        "all_selected_pass": all(row["robust_eligible"] for row in selected),
    }
    stable_write(Path(args.out), value)
    print(f"robust-eligible={sum(row['robust_eligible'] for row in selected)}/{len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
