#!/usr/bin/env python3
"""Freeze or verify the append-only reciprocal-v2 production supplement."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Support both direct-script and package-module execution.
    from . import common, treatment_plan
except ImportError:  # pragma: no cover - exercised by CLI smoke tests
    import common
    import treatment_plan


class FreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


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


def _files_below(root: Path) -> list[Path]:
    if not root.exists():
        return []
    require(root.is_dir() and not root.is_symlink(), f"unsafe artifact root: {root}")
    return [path for path in root.rglob("*") if path.is_file() or path.is_symlink()]


def treatment_artifacts() -> list[Path]:
    """Return artifacts that would make a preregistration freeze post-treatment."""
    paths = _files_below(common.OUTPUT_ROOT)
    for translator in common.base.TRANSLATORS:
        paths.extend(
            _files_below(
                common.BASE / "translators" / translator / "implementations"
            )
        )
    paths.extend(
        path
        for path in (
            common.KC_EXECUTION_AUTHORIZATION,
            common.TRANSLATOR_ISOLATION_LOCK,
            common.IMPLEMENTATION_REGISTRY,
            common.RESOLUTION_LOCK,
        )
        if path.exists()
    )
    return sorted(set(path.resolve() for path in paths), key=common.repo_path)


def selected_source_files() -> list[Path]:
    common.verify_legacy_identities()
    treatment_plan.check()
    files = set(common.HERE.glob("*.py"))
    files.add(common.HERE / "README.md")
    files.update(common.SCHEMA_ROOT.glob("*.json"))
    files.update(common.REQUEST_ROOT.rglob("*.json"))
    files.add(common.KC_PLAN)
    missing = [common.repo_path(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    unsafe = [common.repo_path(path) for path in files if path.is_symlink()]
    require(not unsafe, f"supplement sources may not be symlinks: {unsafe}")
    return sorted((path.resolve() for path in files), key=common.repo_path)


def expected_source_map() -> dict[str, str]:
    return {
        common.repo_path(path): common.file_sha256(path)
        for path in selected_source_files()
    }


def verify() -> dict[str, Any]:
    common.verify_legacy_identities()
    require(common.SOURCE_FREEZE.is_file(), "supplement source freeze is missing")
    require(not common.SOURCE_FREEZE.is_symlink(), "supplement source freeze is symlinked")
    receipt = common.load_json(common.SOURCE_FREEZE)
    sources = expected_source_map()
    checks = (
        receipt.get("schema_version") == 1,
        receipt.get("record_type")
        == "reciprocal_v2_production_supplement_source_freeze",
        receipt.get("supplement_id") == common.SUPPLEMENT_ID,
        receipt.get("campaign_id") == common.base.CAMPAIGN_ID,
        receipt.get("state") == "preregistered_no_treatment",
        receipt.get("legacy_identities") == common.EXPECTED_LEGACY,
        receipt.get("source_sha256") == sources,
        receipt.get("source_bundle_sha256") == common.canonical_sha256(sources),
        receipt.get("treatment_artifacts_opened") is False,
        receipt.get("performance_results_opened") is False,
        receipt.get("remote_push_claimed") is False,
    )
    require(all(checks), "supplement source freeze is stale or inconsistent")
    return receipt


def freeze() -> dict[str, Any]:
    common.verify_legacy_identities()
    if common.SOURCE_FREEZE.exists():
        raise FileExistsError(f"refusing overwrite: {common.SOURCE_FREEZE}")
    opened = treatment_artifacts()
    require(
        not opened,
        "supplement freeze must precede treatment artifacts: "
        + ", ".join(common.repo_path(path) for path in opened[:10]),
    )
    sources = expected_source_map()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=common.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = {
        "schema_version": 1,
        "record_type": "reciprocal_v2_production_supplement_source_freeze",
        "supplement_id": common.SUPPLEMENT_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "state": "preregistered_no_treatment",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_local_freeze": commit,
        "remote_push_claimed": False,
        "legacy_identities": dict(common.EXPECTED_LEGACY),
        "legacy_source_file_count": len(
            common.load_json(common.LEGACY_SOURCE_FREEZE)["source_sha256"]
        ),
        "source_sha256": sources,
        "source_bundle_sha256": common.canonical_sha256(sources),
        "treatment_artifacts_opened": False,
        "performance_results_opened": False,
        "success_locks_opened": False,
        "claim_limit": (
            "protocol_and_fail_closed_tooling_only; no isolation, translation, "
            "GPU execution, KC result, or launch readiness is claimed"
        ),
    }
    _exclusive_json(common.SOURCE_FREEZE, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    receipt = freeze() if args.freeze else verify()
    print(
        json.dumps(
            {
                "files": len(receipt["source_sha256"]),
                "legacy_source_bundle_sha256": receipt["legacy_identities"][
                    "source_bundle_sha256"
                ],
                "source_bundle_sha256": receipt["source_bundle_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
