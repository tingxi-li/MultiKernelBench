#!/usr/bin/env python3
"""Capture a complete, hash-indexed campaign evidence archive."""
from __future__ import annotations

import argparse
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from core import (
    DEPENDENCY_PATHS,
    LOCK_PATH,
    REPO_ROOT,
    SOURCE_PATHS,
    file_sha256,
    load_contract,
    read_json,
    result_root,
    stable_write,
)


EXCLUDED_NAMES = {"active.lock"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".torch_ext"}
EXCLUDED_SUFFIXES = {".o", ".so", ".pyc", ".ninja"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    campaign, _cells, lock = load_contract()
    summary_path = Path(args.summary).resolve()
    summary = read_json(summary_path)
    if summary.get("record_type") != "fused_crossed_final_summary" or summary.get("complete") is not True:
        raise RuntimeError("only a complete final summary can become controlling evidence")
    if summary.get("launch_lock_sha256") != file_sha256(LOCK_PATH) or summary.get("source_bundle_sha256") != lock["source_bundle_sha256"]:
        raise RuntimeError("final summary is not bound to the current lock")
    results = result_root(args.tag)
    summary_path.relative_to(results)
    files = {REPO_ROOT / relative for relative in (*SOURCE_PATHS, *DEPENDENCY_PATHS)}
    files.add(LOCK_PATH)
    for path in results.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if path.name in EXCLUDED_NAMES or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or ".partial." in path.name:
            continue
        files.add(path)
    ordered = sorted(files, key=lambda path: str(path.relative_to(REPO_ROOT)))
    entries = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in ordered
    ]
    prefix = Path(args.out_prefix).resolve()
    archive = prefix.with_suffix(".tar.gz")
    index = prefix.with_suffix(".index.json")
    if archive.exists() or index.exists():
        raise FileExistsError("refusing to overwrite an evidence archive/index")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        for path in ordered:
            handle.add(path, arcname=str(path.relative_to(REPO_ROOT)), recursive=False)
    stable_write(
        index,
        {
            "archive_path": str(archive),
            "archive_sha256": file_sha256(archive),
            "campaign_id": campaign["campaign_id"],
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "entries": entries,
            "entry_count": len(entries),
            "launch_lock_sha256": file_sha256(LOCK_PATH),
            "record_type": "fused_crossed_complete_evidence_index",
            "schema_version": 1,
            "source_bundle_sha256": lock["source_bundle_sha256"],
            "summary_path": str(summary_path.relative_to(REPO_ROOT)),
            "summary_sha256": file_sha256(summary_path),
        },
    )
    print(f"archive={archive} entries={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

