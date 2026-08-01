#!/usr/bin/env python3
"""Correctly build and verify the append-only reciprocal-v2 ``prereg_v2``."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

if __package__:
    from .. import common, freeze as production_freeze
    from . import freeze as fix_freeze
else:  # pragma: no cover - exercised by CLI smoke tests
    _REPO_ROOT = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(_REPO_ROOT))
    from ako_runs.controlled_followup.reciprocal_v2.production_v1 import common
    from ako_runs.controlled_followup.reciprocal_v2.production_v1 import freeze as production_freeze
    from ako_runs.controlled_followup.reciprocal_v2.production_v1.evidence_fix_v1 import freeze as fix_freeze


EVIDENCE_ROOT = common.BASE / "evidence"
PREREG_NAME = "prereg_v2"


class EvidenceFixError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceFixError(message)


def _canonical_line(value: Any) -> bytes:
    return common.canonical_bytes(value) + b"\n"


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"unreconciled temporary file: {temporary}")
    temporary.write_bytes(common.stable_json_bytes(value))
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tar_add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def selected_files() -> list[Path]:
    parent = production_freeze.verify()
    correction = fix_freeze.verify()
    opened = production_freeze.treatment_artifacts()
    require(not opened, "prereg_v2 must precede treatment artifacts")
    files = {common.REPO_ROOT / relative for relative in parent["source_sha256"]}
    files.update(common.REPO_ROOT / relative for relative in correction["source_sha256"])
    files.update(
        {
            common.SOURCE_FREEZE,
            fix_freeze.SOURCE_FREEZE,
            common.LEGACY_SOURCE_FREEZE,
            common.LEGACY_PREREG_INDEX,
            common.LEGACY_PREREG_BUNDLE,
        }
    )
    missing = [common.repo_path(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    require(not any(path.is_symlink() for path in files), "evidence input is symlinked")
    return sorted((path.resolve() for path in files), key=common.repo_path)


def manifest(files: list[Path]) -> dict[str, Any]:
    parent = production_freeze.verify()
    correction = fix_freeze.verify()
    effective_builder = fix_freeze.HERE / "capture_evidence.py"
    return {
        "schema_version": 1,
        "record_type": "reciprocal_v2_production_supplement_evidence_manifest",
        "supplement_id": common.SUPPLEMENT_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "stage": "prereg",
        "supersedes_for_production_protocol": "prereg_v1",
        "preserves_legacy_preregistration": True,
        "legacy_identities": dict(common.EXPECTED_LEGACY),
        "supplement_source_freeze_sha256": common.file_sha256(common.SOURCE_FREEZE),
        "supplement_source_bundle_sha256": parent["source_bundle_sha256"],
        "evidence_fix_id": fix_freeze.FIX_ID,
        "evidence_fix_source_freeze_sha256": common.file_sha256(fix_freeze.SOURCE_FREEZE),
        "evidence_fix_source_bundle_sha256": correction["source_bundle_sha256"],
        "superseded_builder_path": common.repo_path(fix_freeze.DEFECTIVE_BUILDER),
        "superseded_builder_sha256": fix_freeze.DEFECTIVE_BUILDER_SHA256,
        "superseded_builder_defect": correction["defect"],
        "effective_builder_path": common.repo_path(effective_builder),
        "effective_builder_sha256": common.file_sha256(effective_builder),
        "treatment_artifacts_included": False,
        "performance_results_included": False,
        "success_locks_included": False,
        "claim_limit": (
            "prepared fail-closed production protocol plus append-only evidence-builder "
            "repair; no boundary, treatment, GPU, KC, or launch-success claim"
        ),
        "entries": [
            {
                "path": common.repo_path(source_path),
                "size": source_path.stat().st_size,
                "sha256": common.file_sha256(source_path),
            }
            for source_path in files
        ],
    }


def _write_bundle(
    bundle_path: Path, files: list[Path], value: dict[str, Any]
) -> None:
    """Write exclusively; destination cannot be shadowed by source iteration."""
    temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"unreconciled temporary file: {temporary}")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as archive:
                    _tar_add(archive, "EVIDENCE_MANIFEST.json", _canonical_line(value))
                    for source_path, entry in zip(files, value["entries"], strict=True):
                        _tar_add(archive, entry["path"], source_path.read_bytes())
        os.link(temporary, bundle_path)
    finally:
        temporary.unlink(missing_ok=True)


def build(name: str = PREREG_NAME) -> Path:
    require(name == PREREG_NAME, "only the immutable name prereg_v2 is permitted")
    files = selected_files()
    value = manifest(files)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_path = EVIDENCE_ROOT / f"{name}.tar.gz"
    index_path = EVIDENCE_ROOT / f"{name}.index.json"
    if bundle_path.exists() or index_path.exists():
        raise FileExistsError("prereg_v2 evidence outputs are immutable")
    _write_bundle(bundle_path, files, value)
    index = {
        "schema_version": 1,
        "record_type": "reciprocal_v2_production_supplement_evidence_index",
        "supplement_id": common.SUPPLEMENT_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "bundle_path": common.repo_path(bundle_path),
        "bundle_sha256": common.file_sha256(bundle_path),
        "bundle_size": bundle_path.stat().st_size,
        "manifest_sha256": common.canonical_sha256(value),
        "entry_count": len(value["entries"]),
        "manifest": value,
    }
    _exclusive_json(index_path, index)
    return index_path


def verify(index_path: Path) -> dict[str, Any]:
    common.verify_legacy_identities()
    production_freeze.verify()
    correction = fix_freeze.verify()
    require(index_path.is_file() and not index_path.is_symlink(), "evidence index missing/symlinked")
    index = common.load_json(index_path)
    require(
        index.get("record_type") == "reciprocal_v2_production_supplement_evidence_index"
        and index.get("supplement_id") == common.SUPPLEMENT_ID
        and index.get("campaign_id") == common.base.CAMPAIGN_ID,
        "evidence index header mismatch",
    )
    bundle_path = common.REPO_ROOT / str(index.get("bundle_path", ""))
    require(bundle_path.is_file() and not bundle_path.is_symlink(), "evidence bundle missing/symlinked")
    require(common.file_sha256(bundle_path) == index.get("bundle_sha256"), "bundle SHA mismatch")
    require(bundle_path.stat().st_size == index.get("bundle_size"), "bundle size mismatch")
    value = index.get("manifest")
    require(isinstance(value, dict), "index lacks embedded manifest")
    require(common.canonical_sha256(value) == index.get("manifest_sha256"), "manifest SHA mismatch")
    require(
        value.get("legacy_identities") == common.EXPECTED_LEGACY
        and value.get("supplement_source_freeze_sha256") == fix_freeze.PARENT_FREEZE_SHA256
        and value.get("evidence_fix_source_freeze_sha256") == common.file_sha256(fix_freeze.SOURCE_FREEZE)
        and value.get("evidence_fix_source_bundle_sha256") == correction["source_bundle_sha256"]
        and value.get("superseded_builder_sha256") == fix_freeze.DEFECTIVE_BUILDER_SHA256
        and value.get("effective_builder_sha256") == common.file_sha256(Path(__file__).resolve())
        and value.get("treatment_artifacts_included") is False
        and value.get("performance_results_included") is False
        and value.get("success_locks_included") is False,
        "manifest lifecycle/erratum binding mismatch",
    )
    entries = value.get("entries")
    require(isinstance(entries, list), "manifest entries missing")
    expected = {row["path"]: row for row in entries}
    require(len(expected) == len(entries) == index.get("entry_count"), "duplicate/count mismatch")
    with tarfile.open(bundle_path, "r:gz") as archive:
        member_list = archive.getmembers()
        require(all(member.isfile() for member in member_list), "archive contains non-file member")
        require(len({member.name for member in member_list}) == len(member_list), "archive has duplicate names")
        members = {member.name: member for member in member_list}
        embedded_member = members.pop("EVIDENCE_MANIFEST.json", None)
        require(embedded_member is not None, "archive manifest missing")
        stream = archive.extractfile(embedded_member)
        require(stream is not None and stream.read() == _canonical_line(value), "embedded manifest mismatch")
        require(set(members) == set(expected), "archive membership mismatch")
        for name, entry in expected.items():
            stream = archive.extractfile(members[name])
            require(stream is not None, f"archive member unreadable: {name}")
            data = stream.read()
            require(
                len(data) == entry["size"] and hashlib.sha256(data).hexdigest() == entry["sha256"],
                f"archive member mismatch: {name}",
            )
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--name", default=PREREG_NAME)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        index_path = build(args.name)
        result = common.load_json(index_path)
    else:
        index_path = args.index.resolve()
        result = verify(index_path)
    print(json.dumps({"bundle_sha256": result["bundle_sha256"], "entries": result["entry_count"], "index": common.repo_path(index_path), "ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
