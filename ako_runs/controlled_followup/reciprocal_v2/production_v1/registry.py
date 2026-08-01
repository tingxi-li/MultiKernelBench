#!/usr/bin/env python3
"""Derive the 48-cell implementation registry from real isolated outputs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:  # Support both ``python file.py`` and ``python -m package.module``.
    from . import common, isolation, treatment_plan
except ImportError:  # pragma: no cover - exercised by CLI smoke tests
    import common
    import isolation
    import treatment_plan


class RegistryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def _isolation_rows() -> dict[str, dict[str, Any]]:
    require(common.TRANSLATOR_ISOLATION_LOCK.is_file(), "translator isolation lock is missing")
    observed = common.load_json(common.TRANSLATOR_ISOLATION_LOCK)
    require(observed == isolation.lock_document(), "translator isolation lock is stale")
    return {row["translator"]: row for row in observed["translators"]}


def _validate_receipt(job: dict[str, Any], boundary: dict[str, Any]) -> tuple[Path, Path]:
    cid, translator = job["cell_id"], job["translator"]
    request_path = treatment_plan.translation_request_path(cid, translator)
    source_path = treatment_plan.implementation_source_path(cid, translator)
    receipt_path = treatment_plan.translation_receipt_path(cid, translator)
    require(source_path.is_file() and not source_path.is_symlink(), f"{cid}: source missing/symlinked")
    require(source_path.stat().st_size > 0, f"{cid}: source empty")
    require(receipt_path.is_file() and not receipt_path.is_symlink(), f"{cid}: receipt missing/symlinked")
    value = common.load_json(receipt_path)
    expected = {
        "schema_version": 1,
        "record_type": "reciprocal_v2_translation_receipt",
        "supplement_id": common.SUPPLEMENT_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "state": "completed_treatment_output",
        "cell_id": cid,
        "recipe_origin": job["recipe_origin"],
        "destination_dsl": job["destination_dsl"],
        "transfer_mode": job["transfer_mode"],
        "translator": translator,
        "legacy_source_freeze_sha256": common.EXPECTED_LEGACY["source_freeze_sha256"],
        "supplement_source_freeze_sha256": common.file_sha256(common.SOURCE_FREEZE),
        "request_path": common.repo_path(request_path),
        "request_sha256": common.file_sha256(request_path),
        "source_path": common.repo_path(source_path),
        "source_sha256": common.file_sha256(source_path),
        "workspace_id": boundary["worktree_id"],
        "execution_id": boundary["execution_id"],
        "isolation_transcript_path": boundary["isolation_transcript_path"],
        "isolation_transcript_sha256": boundary["isolation_transcript_sha256"],
        "other_translator_source_accessible": False,
        "performance_results_accessible": False,
    }
    for key, expected_value in expected.items():
        require(value.get(key) == expected_value, f"{cid}: receipt {key} mismatch")
    require(isinstance(value.get("completed_utc"), str) and value["completed_utc"], f"{cid}: completion time missing")
    return source_path.resolve(), receipt_path.resolve()


def document() -> dict[str, Any]:
    boundaries = _isolation_rows()
    manifest = common.load_json(common.BASE / "manifests/audit.json")
    implementations = []
    seen_paths, seen_inodes = set(), set()
    expected_sources, expected_receipts = set(), set()
    by_digest: dict[str, list[str]] = defaultdict(list)
    for job in manifest["jobs"]:
        cid, translator = job["cell_id"], job["translator"]
        source, receipt = _validate_receipt(job, boundaries[translator])
        expected_sources.add(source)
        expected_receipts.add(receipt)
        inode = (source.stat().st_dev, source.stat().st_ino)
        require(source not in seen_paths and inode not in seen_inodes, f"{cid}: source path/inode reused")
        seen_paths.add(source)
        seen_inodes.add(inode)
        digest = common.file_sha256(source)
        by_digest[digest].append(cid)
        request = treatment_plan.translation_request_path(cid, translator)
        implementations.append(
            {
                "cell_id": cid,
                "recipe_origin": job["recipe_origin"],
                "destination_dsl": job["destination_dsl"],
                "transfer_mode": job["transfer_mode"],
                "translator": translator,
                "source": common.repo_path(source),
                "sha256": digest,
                "translation_request_path": common.repo_path(request),
                "translation_request_sha256": common.file_sha256(request),
                "translation_receipt_path": common.repo_path(receipt),
                "translation_receipt_sha256": common.file_sha256(receipt),
            }
        )
    observed_sources = set()
    for translator in common.base.TRANSLATORS:
        root = common.BASE / "translators" / translator / "implementations"
        if root.exists():
            for path in root.rglob("*"):
                require(not path.is_symlink(), f"implementation tree contains symlink: {path}")
                if path.is_file():
                    observed_sources.add(path.resolve())
    require(observed_sources == expected_sources, "implementation tree has missing or extra files")
    observed_receipts = set()
    if common.TRANSLATION_RECEIPT_ROOT.exists():
        for path in common.TRANSLATION_RECEIPT_ROOT.rglob("*"):
            require(not path.is_symlink(), f"translation receipt tree contains symlink: {path}")
            if path.is_file():
                observed_receipts.add(path.resolve())
    require(
        observed_receipts == expected_receipts,
        "translation receipt tree has missing or extra files",
    )
    identical = [
        {"sha256": digest, "cell_ids": cell_ids}
        for digest, cell_ids in sorted(by_digest.items())
        if len(cell_ids) > 1
    ]
    request_map = {
        row["translation_request_path"]: row["translation_request_sha256"]
        for row in implementations
    }
    return {
        "schema_version": 1,
        "record_type": "reciprocal_v2_implementation_registry",
        "campaign_id": common.base.CAMPAIGN_ID,
        "state": "frozen",
        "source_freeze_sha256": common.EXPECTED_LEGACY["source_freeze_sha256"],
        "supplement_source_freeze_sha256": common.file_sha256(common.SOURCE_FREEZE),
        "translator_isolation_sha256": common.file_sha256(common.TRANSLATOR_ISOLATION_LOCK),
        "translation_request_bundle_sha256": common.canonical_sha256(request_map),
        "implementation_count": len(implementations),
        "identical_source_groups_disclosed": identical,
        "implementations": implementations,
    }


def build() -> Path:
    value = document()
    require(value["implementation_count"] == 48, "implementation census is not 48")
    isolation.exclusive_json(common.IMPLEMENTATION_REGISTRY, value)
    return common.IMPLEMENTATION_REGISTRY


def status() -> dict[str, Any]:
    manifest = common.load_json(common.BASE / "manifests/audit.json")
    missing_sources, missing_receipts = [], []
    for job in manifest["jobs"]:
        cid, translator = job["cell_id"], job["translator"]
        source = treatment_plan.implementation_source_path(cid, translator)
        receipt = treatment_plan.translation_receipt_path(cid, translator)
        if not source.is_file():
            missing_sources.append(common.repo_path(source))
        if not receipt.is_file():
            missing_receipts.append(common.repo_path(receipt))
    return {
        "ready": not missing_sources and not missing_receipts and common.TRANSLATOR_ISOLATION_LOCK.is_file(),
        "missing_sources": missing_sources,
        "missing_receipts": missing_receipts,
        "isolation_lock_present": common.TRANSLATOR_ISOLATION_LOCK.is_file(),
        "registry_present": common.IMPLEMENTATION_REGISTRY.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("build")
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "status":
        result = status()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ready"] else 2
    if args.command == "build":
        output = build()
    else:
        require(common.load_json(common.IMPLEMENTATION_REGISTRY) == document(), "registry is stale")
        output = common.IMPLEMENTATION_REGISTRY
    print(json.dumps({"path": common.repo_path(output), "sha256": common.file_sha256(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
