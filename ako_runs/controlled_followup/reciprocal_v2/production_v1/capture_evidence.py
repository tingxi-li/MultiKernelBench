#!/usr/bin/env python3
"""Build or verify the deterministic reciprocal-v2 ``prereg_v2`` supplement."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

try:  # Support both direct-script and package-module execution.
    from . import common, freeze
except ImportError:  # pragma: no cover - exercised by CLI smoke tests
    import common
    import freeze


EVIDENCE_ROOT = common.BASE / "evidence"
PREREG_NAME = "prereg_v2"


class EvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


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
    receipt = freeze.verify()
    opened = freeze.treatment_artifacts()
    require(
        not opened,
        "prereg_v2 must precede treatment artifacts: "
        + ", ".join(common.repo_path(path) for path in opened[:10]),
    )
    files = {common.REPO_ROOT / relative for relative in receipt["source_sha256"]}
    files.update(
        {
            common.SOURCE_FREEZE,
            common.LEGACY_SOURCE_FREEZE,
            common.LEGACY_PREREG_INDEX,
            common.LEGACY_PREREG_BUNDLE,
        }
    )
    missing = [common.repo_path(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    unsafe = [common.repo_path(path) for path in files if path.is_symlink()]
    require(not unsafe, f"evidence inputs may not be symlinks: {unsafe}")
    return sorted((path.resolve() for path in files), key=common.repo_path)


def manifest(files: list[Path]) -> dict[str, Any]:
    supplement = freeze.verify()
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
        "supplement_source_bundle_sha256": supplement["source_bundle_sha256"],
        "treatment_artifacts_included": False,
        "performance_results_included": False,
        "success_locks_included": False,
        "claim_limit": (
            "prepared fail-closed production protocol only; no boundary, treatment, "
            "GPU, KC, or launch-success claim"
        ),
        "entries": [
            {
                "path": common.repo_path(path),
                "size": path.stat().st_size,
                "sha256": common.file_sha256(path),
            }
            for path in files
        ],
    }


def _write_bundle(path: Path, files: list[Path], value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"unreconciled temporary file: {temporary}")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as archive:
                    _tar_add(archive, "EVIDENCE_MANIFEST.json", _canonical_line(value))
                    for path, entry in zip(files, value["entries"], strict=True):
                        _tar_add(archive, entry["path"], path.read_bytes())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(name: str = PREREG_NAME) -> Path:
    require(name == PREREG_NAME, "this supplement only permits the immutable name prereg_v2")
    files = selected_files()
    value = manifest(files)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    bundle = EVIDENCE_ROOT / f"{name}.tar.gz"
    index_path = EVIDENCE_ROOT / f"{name}.index.json"
    if bundle.exists() or index_path.exists():
        raise FileExistsError("prereg_v2 evidence outputs are immutable")
    _write_bundle(bundle, files, value)
    index = {
        "schema_version": 1,
        "record_type": "reciprocal_v2_production_supplement_evidence_index",
        "supplement_id": common.SUPPLEMENT_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "bundle_path": common.repo_path(bundle),
        "bundle_sha256": common.file_sha256(bundle),
        "bundle_size": bundle.stat().st_size,
        "manifest_sha256": common.canonical_sha256(value),
        "entry_count": len(value["entries"]),
        "manifest": value,
    }
    _exclusive_json(index_path, index)
    return index_path


def verify(index_path: Path) -> dict[str, Any]:
    common.verify_legacy_identities()
    freeze.verify()
    require(index_path.is_file() and not index_path.is_symlink(), "evidence index missing/symlinked")
    index = common.load_json(index_path)
    require(
        index.get("record_type")
        == "reciprocal_v2_production_supplement_evidence_index"
        and index.get("supplement_id") == common.SUPPLEMENT_ID
        and index.get("campaign_id") == common.base.CAMPAIGN_ID,
        "evidence index header mismatch",
    )
    bundle = common.REPO_ROOT / str(index.get("bundle_path", ""))
    require(bundle.is_file() and not bundle.is_symlink(), "evidence bundle missing/symlinked")
    require(common.file_sha256(bundle) == index.get("bundle_sha256"), "bundle SHA mismatch")
    require(bundle.stat().st_size == index.get("bundle_size"), "bundle size mismatch")
    value = index.get("manifest")
    require(isinstance(value, dict), "index lacks embedded manifest")
    require(common.canonical_sha256(value) == index.get("manifest_sha256"), "manifest SHA mismatch")
    require(
        value.get("legacy_identities") == common.EXPECTED_LEGACY
        and value.get("supplement_source_freeze_sha256")
        == common.file_sha256(common.SOURCE_FREEZE)
        and value.get("treatment_artifacts_included") is False
        and value.get("performance_results_included") is False
        and value.get("success_locks_included") is False,
        "manifest lifecycle binding mismatch",
    )
    entries = value.get("entries")
    require(isinstance(entries, list), "manifest entries missing")
    expected = {row["path"]: row for row in entries}
    require(len(expected) == len(entries) == index.get("entry_count"), "duplicate/count mismatch")
    with tarfile.open(bundle, "r:gz") as archive:
        members_list = archive.getmembers()
        require(all(member.isfile() for member in members_list), "archive contains non-file member")
        require(len({member.name for member in members_list}) == len(members_list), "archive has duplicate names")
        members = {member.name: member for member in members_list}
        embedded_member = members.pop("EVIDENCE_MANIFEST.json", None)
        require(embedded_member is not None, "archive manifest missing")
        embedded_stream = archive.extractfile(embedded_member)
        require(embedded_stream is not None and embedded_stream.read() == _canonical_line(value), "embedded manifest mismatch")
        require(set(members) == set(expected), "archive membership mismatch")
        for name, entry in expected.items():
            stream = archive.extractfile(members[name])
            require(stream is not None, f"archive member unreadable: {name}")
            data = stream.read()
            require(
                len(data) == entry["size"]
                and hashlib.sha256(data).hexdigest() == entry["sha256"],
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
    print(
        json.dumps(
            {
                "bundle_sha256": result["bundle_sha256"],
                "entries": result["entry_count"],
                "index": common.repo_path(index_path),
                "ok": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
