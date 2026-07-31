#!/usr/bin/env python3
"""Create or verify the immutable source/dependency launch lock."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from core import (
    CAMPAIGN_PATH,
    CELLS_PATH,
    DEPENDENCY_PATHS,
    LOCK_PATH,
    REPO_ROOT,
    SOURCE_PATHS,
    canonical_sha256,
    file_sha256,
    read_json,
    stable_write,
    validate_campaign,
    validate_cells,
)


def make_lock() -> dict:
    campaign = read_json(CAMPAIGN_PATH)
    cells = read_json(CELLS_PATH)
    validate_campaign(campaign)
    validate_cells(cells)
    missing = [relative for relative in (*SOURCE_PATHS, *DEPENDENCY_PATHS) if not (REPO_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze missing files: {missing}")
    sources = {relative: file_sha256(REPO_ROOT / relative) for relative in SOURCE_PATHS}
    dependencies = {relative: file_sha256(REPO_ROOT / relative) for relative in DEPENDENCY_PATHS}
    adapter = read_json(REPO_ROOT / "ako_runs/controlled_followup/fused_grid/robust_adapter_manifest.json")
    return {
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": file_sha256(CAMPAIGN_PATH),
        "cells_sha256": file_sha256(CELLS_PATH),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dependency_bundle_sha256": canonical_sha256(dependencies),
        "dependency_sha256": dependencies,
        "frozen_gate": {
            "adapter_manifest_sha256": file_sha256(REPO_ROOT / "ako_runs/controlled_followup/fused_grid/robust_adapter_manifest.json"),
            "gate_spec_sha256": adapter["robust_gate"]["gate_spec_sha256"],
            "manifest_sha256": adapter["robust_gate"]["manifest_sha256"],
        },
        "schema_version": 1,
        "source_bundle_sha256": canonical_sha256(sources),
        "source_sha256": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    expected = make_lock()
    if args.write:
        stable_write(LOCK_PATH, expected)
    observed = read_json(LOCK_PATH)
    # Timestamp is documentary; every substantive field must reproduce.
    for key, value in expected.items():
        if key != "created_utc" and observed.get(key) != value:
            raise RuntimeError(f"launch lock differs at {key}")
    print(f"source_bundle_sha256={observed['source_bundle_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

