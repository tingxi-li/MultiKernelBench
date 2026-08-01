#!/usr/bin/env python3
"""Freeze or verify the append-only evidence-builder correction."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .. import common, freeze as production_freeze
else:  # pragma: no cover - exercised by CLI smoke tests
    _REPO_ROOT = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(_REPO_ROOT))
    from ako_runs.controlled_followup.reciprocal_v2.production_v1 import common
    from ako_runs.controlled_followup.reciprocal_v2.production_v1 import freeze as production_freeze


HERE = Path(__file__).resolve().parent
SOURCE_FREEZE = HERE / "receipts/source_freeze.json"
FIX_ID = "reciprocal-v2-production-v1-evidence-fix-v1-20260731"
PARENT_FREEZE_SHA256 = "abf550ce0b71083b36af076c319b995ccb94aefa681536fa4e76760c275bcaa4"
DEFECTIVE_BUILDER_SHA256 = "5e74703976d1844143fff232c82102da000af4b5eb437fff2498195491c440ff"
DEFECTIVE_BUILDER = common.HERE / "capture_evidence.py"


class FixFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixFreezeError(message)


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"unreconciled temporary file: {temporary}")
    temporary.write_bytes(common.stable_json_bytes(value))
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_parent() -> dict[str, Any]:
    receipt = production_freeze.verify()
    require(
        common.file_sha256(common.SOURCE_FREEZE) == PARENT_FREEZE_SHA256,
        "production-v1 source-freeze identity changed",
    )
    require(
        common.file_sha256(DEFECTIVE_BUILDER) == DEFECTIVE_BUILDER_SHA256,
        "frozen defective builder identity changed",
    )
    require(
        receipt["source_sha256"][common.repo_path(DEFECTIVE_BUILDER)]
        == DEFECTIVE_BUILDER_SHA256,
        "production-v1 freeze does not bind the defective builder",
    )
    return receipt


def selected_source_files() -> list[Path]:
    verify_parent()
    files = set(HERE.glob("*.py"))
    files.add(HERE / "README.md")
    missing = [common.repo_path(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    require(not any(path.is_symlink() for path in files), "fix sources may not be symlinks")
    return sorted((path.resolve() for path in files), key=common.repo_path)


def expected_source_map() -> dict[str, str]:
    return {
        common.repo_path(path): common.file_sha256(path)
        for path in selected_source_files()
    }


def verify() -> dict[str, Any]:
    verify_parent()
    require(SOURCE_FREEZE.is_file() and not SOURCE_FREEZE.is_symlink(), "fix source freeze missing/symlinked")
    receipt = common.load_json(SOURCE_FREEZE)
    sources = expected_source_map()
    require(
        receipt.get("schema_version") == 1
        and receipt.get("record_type") == "reciprocal_v2_evidence_builder_fix_freeze"
        and receipt.get("fix_id") == FIX_ID
        and receipt.get("campaign_id") == common.base.CAMPAIGN_ID
        and receipt.get("state") == "frozen_append_only_erratum"
        and receipt.get("parent_source_freeze_sha256") == PARENT_FREEZE_SHA256
        and receipt.get("defective_builder_sha256") == DEFECTIVE_BUILDER_SHA256
        and receipt.get("source_sha256") == sources
        and receipt.get("source_bundle_sha256") == common.canonical_sha256(sources)
        and receipt.get("treatment_artifacts_opened") is False,
        "evidence-builder fix freeze is stale or inconsistent",
    )
    return receipt


def freeze() -> dict[str, Any]:
    verify_parent()
    if SOURCE_FREEZE.exists():
        raise FileExistsError(f"refusing overwrite: {SOURCE_FREEZE}")
    opened = production_freeze.treatment_artifacts()
    require(not opened, "fix freeze must precede all treatment artifacts")
    sources = expected_source_map()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=common.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    effective_builder = HERE / "capture_evidence.py"
    receipt = {
        "schema_version": 1,
        "record_type": "reciprocal_v2_evidence_builder_fix_freeze",
        "fix_id": FIX_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "state": "frozen_append_only_erratum",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_local_freeze": commit,
        "remote_push_claimed": False,
        "parent_source_freeze_sha256": PARENT_FREEZE_SHA256,
        "defective_builder_path": common.repo_path(DEFECTIVE_BUILDER),
        "defective_builder_sha256": DEFECTIVE_BUILDER_SHA256,
        "defect": "bundle_destination_variable_shadowed_by_source_loop_variable",
        "effective_builder_path": common.repo_path(effective_builder),
        "effective_builder_sha256": common.file_sha256(effective_builder),
        "source_sha256": sources,
        "source_bundle_sha256": common.canonical_sha256(sources),
        "treatment_artifacts_opened": False,
        "performance_results_opened": False,
        "claim_limit": "evidence-production repair only; no treatment or outcome claim",
    }
    _exclusive_json(SOURCE_FREEZE, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    receipt = freeze() if args.freeze else verify()
    print(json.dumps({"files": len(receipt["source_sha256"]), "source_bundle_sha256": receipt["source_bundle_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
