#!/usr/bin/env python3
"""Freeze the prospective campaign before any performance launch."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from protocol import (
    ADAPTER,
    BASE_JOBS,
    CAMPAIGN,
    JOBS,
    LOCK,
    ORIGINAL_ADAPTER,
    REPO_ROOT,
    V1_CONFIRMATION_SUMMARY,
    build_jobs,
    canonical_sha256,
    file_sha256,
    source_hashes,
    stable_write,
)


def main() -> None:
    if JOBS.exists() or LOCK.exists() or ADAPTER.exists():
        raise RuntimeError(
            "jobs.json/launch_lock.json already exist; frozen v2 is immutable. "
            "Use a new campaign version for source or protocol changes."
        )
    results = LOCK.parent / "results"
    if results.exists() and any(results.iterdir()):
        raise RuntimeError("refusing to freeze after result artifacts exist")
    jobs = build_jobs()
    hashes = source_hashes()
    stable_write(JOBS, jobs)
    robust_manifest = REPO_ROOT / "ako_runs/controlled_followup/robust_gate/manifest.json"
    gate_spec = REPO_ROOT / "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json"
    original_adapter = json.loads(ORIGINAL_ADAPTER.read_text(encoding="utf-8"))
    prospective_adapter = {
        "schema_version": 1,
        "campaign_id": "fused-reachability-streamed-epilogue-v2",
        "operation": "fused_softmax",
        "source_sha256": hashes,
        "source_bundle_sha256": canonical_sha256(hashes),
        "grid": {
            "manifest_path": str(CAMPAIGN.relative_to(REPO_ROOT)),
            "manifest_sha256": file_sha256(CAMPAIGN),
            "manifest_canonical_sha256": canonical_sha256(
                json.loads(CAMPAIGN.read_text(encoding="utf-8"))
            ),
            "jobs_path": str(JOBS.relative_to(REPO_ROOT)),
            "jobs_sha256": file_sha256(JOBS),
            "jobs_canonical_sha256": canonical_sha256(jobs),
            "job_count": len(jobs),
            "job_sha256": {row["job_id"]: canonical_sha256(row) for row in jobs},
        },
        "robust_gate": original_adapter["robust_gate"],
    }
    stable_write(ADAPTER, prospective_adapter)
    lock = {
        "schema_version": 1,
        "record_type": "fused_reachability_v2_launch_lock",
        "campaign_id": "fused-reachability-streamed-epilogue-v2",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_sha256": file_sha256(CAMPAIGN),
        "base_jobs_path": str(BASE_JOBS.relative_to(REPO_ROOT)),
        "base_jobs_sha256": file_sha256(BASE_JOBS),
        "jobs_path": str(JOBS.relative_to(REPO_ROOT)),
        "jobs_sha256": file_sha256(JOBS),
        "jobs_canonical_sha256": canonical_sha256(jobs),
        "job_sha256": {row["job_id"]: canonical_sha256(row) for row in jobs},
        "source_sha256": hashes,
        "source_bundle_sha256": canonical_sha256(hashes),
        "adapter_manifest_path": str(ADAPTER.relative_to(REPO_ROOT)),
        "adapter_manifest_sha256": file_sha256(ADAPTER),
        "adapter_manifest_canonical_sha256": canonical_sha256(prospective_adapter),
        "frozen_gate": {
            "manifest_path": str(robust_manifest.relative_to(REPO_ROOT)),
            "manifest_sha256": file_sha256(robust_manifest),
            "gate_spec_path": str(gate_spec.relative_to(REPO_ROOT)),
            "gate_spec_sha256": file_sha256(gate_spec),
            "original_adapter_path": str(ORIGINAL_ADAPTER.relative_to(REPO_ROOT)),
            "original_adapter_sha256": file_sha256(ORIGINAL_ADAPTER),
        },
        "v1_baseline": {
            "summary_path": str(V1_CONFIRMATION_SUMMARY.relative_to(REPO_ROOT)),
            "summary_sha256": file_sha256(V1_CONFIRMATION_SUMMARY),
            "point_minima_ms": {
                "cuda_noptx": 1.8677760362625122,
                "cuda_unlimited": 1.8053120374679565
            }
        },
        "launch_policy": {
            "screen": {"reps": 2, "randomization_seed": 2026073001},
            "confirmation": {"reps": 15, "randomization_seed": 2026073002},
            "all_failures_retained": True,
            "source_drift_fails_closed": True,
            "overwrite_forbidden": True,
        },
    }
    stable_write(LOCK, lock)
    print(f"froze {len(jobs)} jobs at {LOCK}")
    print(f"source_bundle_sha256={lock['source_bundle_sha256']}")


if __name__ == "__main__":
    main()
