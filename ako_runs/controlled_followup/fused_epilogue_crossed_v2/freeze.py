#!/usr/bin/env python3
"""Create or verify probe-stage and final-campaign source locks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

try:
    from .core import (
        CAMPAIGN_ID, CAMPAIGN_PATH, DEPENDENCY_PATHS, LOCK_PATH, PROBE_LOCK_PATH,
        REPO_ROOT, ROBUST_ADAPTER_PATH, SOURCE_PATHS, SUPPORT_RESOLUTION_PATH,
        canonical_sha256, file_sha256, load_cells, read_json, stable_write,
        support_evidence_paths, validate_campaign, validate_support_resolution,
    )
except ImportError:  # direct script execution
    from core import (
    CAMPAIGN_ID,
    CAMPAIGN_PATH,
    DEPENDENCY_PATHS,
    LOCK_PATH,
    PROBE_LOCK_PATH,
    REPO_ROOT,
    ROBUST_ADAPTER_PATH,
    SOURCE_PATHS,
    SUPPORT_RESOLUTION_PATH,
    canonical_sha256,
    file_sha256,
    load_cells,
    read_json,
    stable_write,
    support_evidence_paths,
    validate_campaign,
    validate_support_resolution,
    )


def make_lock(stage: str) -> dict:
    campaign = read_json(CAMPAIGN_PATH)
    resolution = read_json(SUPPORT_RESOLUTION_PATH)
    validate_campaign(campaign)
    validate_support_resolution(resolution, require_resolved=stage == "campaign")
    cells = load_cells(require_resolved=stage == "campaign")
    missing = [relative for relative in (*SOURCE_PATHS, *DEPENDENCY_PATHS) if not (REPO_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze missing files: {missing}")
    sources = {relative: file_sha256(REPO_ROOT / relative) for relative in SOURCE_PATHS}
    dependencies = {relative: file_sha256(REPO_ROOT / relative) for relative in DEPENDENCY_PATHS}
    evidence_paths = support_evidence_paths(resolution) if stage == "campaign" else ()
    support_evidence = {
        relative: file_sha256(REPO_ROOT / relative) for relative in evidence_paths
    }
    adapter = read_json(ROBUST_ADAPTER_PATH)
    return {
        "campaign_id": CAMPAIGN_ID,
        "campaign_sha256": file_sha256(CAMPAIGN_PATH),
        "cells_sha256": canonical_sha256(cells),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dependency_bundle_sha256": canonical_sha256(dependencies),
        "dependency_sha256": dependencies,
        "frozen_gate": {
            "adapter_manifest_sha256": file_sha256(ROBUST_ADAPTER_PATH),
            "gate_spec_sha256": adapter["robust_gate"]["gate_spec_sha256"],
            "manifest_sha256": adapter["robust_gate"]["manifest_sha256"],
        },
        "lock_stage": stage,
        "schema_version": 2,
        "source_bundle_sha256": canonical_sha256(sources),
        "source_sha256": sources,
        "support_evidence_bundle_sha256": canonical_sha256(support_evidence),
        "support_evidence_sha256": support_evidence,
        "support_resolution_sha256": file_sha256(SUPPORT_RESOLUTION_PATH),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probes", "campaign"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = PROBE_LOCK_PATH if args.stage == "probes" else LOCK_PATH
    expected = make_lock(args.stage)
    if args.write:
        stable_write(path, expected)
    observed = read_json(path)
    for key, value in expected.items():
        if key != "created_utc" and observed.get(key) != value:
            raise RuntimeError(f"{args.stage} lock differs at {key}")
    print(f"stage={args.stage} source_bundle_sha256={observed['source_bundle_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
