#!/usr/bin/env python3
"""Freeze or verify the reciprocal-v2 preregistration source bundle."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import make_manifests
import make_retune_plans
import protocol
import validate


LOCAL_NAMES = (
    ".gitignore", "README.md", "__init__.py", "protocol.py", "make_manifests.py",
    "make_retune_plans.py", "validate.py", "launch.py", "analyze.py", "freeze.py",
    "capture_evidence.py", "test_reciprocal_v2.py",
)


def _exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite of {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def selected_source_files() -> list[Path]:
    validate.validate_static()
    files = {protocol.HERE / name for name in LOCAL_NAMES}
    files.update((protocol.HERE / "translators" / translator / "README.md") for translator in protocol.TRANSLATORS)
    files.update((protocol.HERE / "recipe_cards").glob("*.json"))
    files.update((protocol.HERE / "manifests").glob("*.json"))
    files.update((protocol.HERE / "retune_plans").glob("*/*.json"))
    files.add(protocol.GATE_LOCK)
    files.update(make_manifests.ORIGIN_FILES.values())
    files.update((protocol.GATE_SPEC, protocol.GATE_SUMMARY, protocol.GATE_RECEIPT))
    files.add(make_retune_plans.SOURCE_GRID)
    for card_path in make_manifests.ORIGIN_FILES.values():
        for evidence in protocol.load_json(card_path).get("evidence", []):
            files.add(protocol.REPO_ROOT / evidence["path"])
    acceptance = protocol.load_json(protocol.GATE_RECEIPT)
    for evidence in acceptance.get("validation", {}).get("raw_evidence", []):
        files.add(protocol.REPO_ROOT / evidence["path"])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return sorted((path.resolve() for path in files), key=lambda path: protocol.repo_path(path))


def expected_source_map() -> dict[str, str]:
    return {protocol.repo_path(path): protocol.file_sha256(path) for path in selected_source_files()}


def verify() -> dict[str, Any]:
    if not protocol.SOURCE_FREEZE.is_file():
        raise FileNotFoundError(protocol.SOURCE_FREEZE)
    receipt = protocol.load_json(protocol.SOURCE_FREEZE)
    expected = expected_source_map()
    if (
        receipt.get("campaign_id") != protocol.CAMPAIGN_ID
        or receipt.get("source_sha256") != expected
        or receipt.get("source_bundle_sha256") != protocol.canonical_sha256(expected)
        or receipt.get("block_order_sha256") != protocol.block_order_sha256()
        or receipt.get("audit_manifest_sha256") != protocol.file_sha256(make_manifests.AUDIT_MANIFEST)
        or receipt.get("primary_manifest_sha256") != protocol.file_sha256(make_manifests.PRIMARY_MANIFEST)
    ):
        raise ValueError("source freeze receipt is stale or inconsistent")
    return receipt


def freeze() -> dict[str, Any]:
    if protocol.SOURCE_FREEZE.exists():
        raise FileExistsError(f"refusing overwrite of {protocol.SOURCE_FREEZE}")
    if any((protocol.HERE / name).exists() for name in ("results", "evidence")):
        raise FileExistsError("source freeze must precede results and evidence")
    sources = expected_source_map()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=protocol.REPO_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    receipt = {
        "schema_version": 2,
        "record_type": "reciprocal_v2_source_freeze_receipt",
        "campaign_id": protocol.CAMPAIGN_ID,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_local_freeze": git_commit,
        "remote_push_claimed": False,
        "source_sha256": sources,
        "source_bundle_sha256": protocol.canonical_sha256(sources),
        "audit_manifest_sha256": protocol.file_sha256(make_manifests.AUDIT_MANIFEST),
        "primary_manifest_sha256": protocol.file_sha256(make_manifests.PRIMARY_MANIFEST),
        "gate_lock_sha256": protocol.file_sha256(protocol.GATE_LOCK),
        "factorial": {"origins": 3, "destinations": 4, "modes": 2, "translators": 2, "audit_cells": 48, "primary_cells": 48},
        "retuned_attempts_per_cell": protocol.RETUNE_ATTEMPTS,
        "kc_ladder": list(protocol.KC_LADDER),
        "confirmation_blocks": protocol.CONFIRM_REPS,
        "block_order_seed": protocol.BLOCK_ORDER_SEED,
        "block_order_sha256": protocol.block_order_sha256(),
        "implementation_results_opened": False,
        "performance_results_opened": False,
    }
    _exclusive(protocol.SOURCE_FREEZE, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    receipt = freeze() if args.freeze else verify()
    print(json.dumps({"source_bundle_sha256": receipt["source_bundle_sha256"], "files": len(receipt["source_sha256"]), "block_order_sha256": receipt["block_order_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
