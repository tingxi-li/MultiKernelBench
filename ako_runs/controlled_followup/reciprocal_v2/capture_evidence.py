#!/usr/bin/env python3
"""Create or verify deterministic preregistration/complete evidence bundles."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Any

import freeze
import make_manifests
import protocol
import validate


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _canonical_bytes(value: Any) -> bytes:
    return protocol.canonical_bytes(value) + b"\n"


def _exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite of {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def selected_files(stage: str) -> list[Path]:
    receipt = freeze.verify()
    files = {protocol.REPO_ROOT / relative for relative in receipt["source_sha256"]}
    files.add(protocol.SOURCE_FREEZE)
    if stage == "complete":
        blockers = validate.dependency_blockers("primary")
        if blockers:
            raise ValueError("complete evidence blocked: " + "; ".join(blockers[:10]))
        for path in (
            protocol.IMPLEMENTATION_REGISTRY, protocol.TRANSLATOR_ISOLATION_LOCK,
            protocol.RESOLUTION_LOCK,
            protocol.PROVENANCE_LOCK, protocol.PRIMARY_RAW,
            protocol.PRIMARY_ANALYSIS, protocol.PRIMARY_COMPLETION,
            protocol.HERE / "results/audit/launch_receipt.json",
            protocol.HERE / "results/screen/launch_receipt.json",
            protocol.HERE / "results/primary/launch_receipt.json",
        ):
            files.add(path)
        manifest = protocol.load_json(make_manifests.PRIMARY_MANIFEST)
        for job in manifest["jobs"]:
            files.add(protocol.HERE / job["audit_receipt"])
            files.add(protocol.HERE / "results" / "selection" / "receipts" / f"{job['cell_id']}.json")
        registry = protocol.load_json(protocol.IMPLEMENTATION_REGISTRY)
        files.update(protocol.REPO_ROOT / row["source"] for row in registry["implementations"])
        resolution = protocol.load_json(protocol.RESOLUTION_LOCK)
        for attempt in resolution["attempts"]:
            files.update(protocol.REPO_ROOT / row["v4_summary_path"] for row in attempt["cells"])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return sorted((path.resolve() for path in files), key=lambda path: protocol.repo_path(path))


def _tar_add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def build(stage: str, name: str) -> Path:
    if stage not in ("prereg", "complete") or name in (".", "..") or not NAME_RE.fullmatch(name):
        raise ValueError("unsafe evidence stage/name")
    files = selected_files(stage)
    entries = [{"path": protocol.repo_path(path), "size": path.stat().st_size, "sha256": protocol.file_sha256(path)} for path in files]
    manifest = {
        "schema_version": 2,
        "record_type": "reciprocal_v2_evidence_manifest",
        "campaign_id": protocol.CAMPAIGN_ID,
        "stage": stage,
        "source_freeze_sha256": protocol.file_sha256(protocol.SOURCE_FREEZE),
        "source_bundle_sha256": freeze.verify()["source_bundle_sha256"],
        "entries": entries,
    }
    output = protocol.HERE / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / f"{name}.tar.gz"
    index_path = output / f"{name}.index.json"
    if bundle.exists() or index_path.exists():
        raise FileExistsError("evidence outputs are immutable")
    temporary = bundle.with_suffix(bundle.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                _tar_add(archive, "EVIDENCE_MANIFEST.json", _canonical_bytes(manifest))
                for path, entry in zip(files, entries, strict=True):
                    _tar_add(archive, entry["path"], path.read_bytes())
    temporary.replace(bundle)
    index = {
        "schema_version": 2,
        "record_type": "reciprocal_v2_evidence_index",
        "bundle_path": protocol.repo_path(bundle),
        "bundle_sha256": protocol.file_sha256(bundle),
        "bundle_size": bundle.stat().st_size,
        "manifest_sha256": protocol.canonical_sha256(manifest),
        "entry_count": len(entries),
        "manifest": manifest,
    }
    _exclusive(index_path, index)
    print(json.dumps({"index": str(index_path), "bundle_sha256": index["bundle_sha256"], "entries": len(entries)}, sort_keys=True))
    return index_path


def verify(index_path: Path) -> None:
    index = protocol.load_json(index_path)
    bundle = protocol.REPO_ROOT / index["bundle_path"]
    if protocol.file_sha256(bundle) != index["bundle_sha256"]:
        raise ValueError("bundle SHA mismatch")
    manifest = index["manifest"]
    if protocol.canonical_sha256(manifest) != index["manifest_sha256"]:
        raise ValueError("manifest SHA mismatch")
    expected = {row["path"]: row for row in manifest["entries"]}
    with tarfile.open(bundle, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        embedded = archive.extractfile(members.pop("EVIDENCE_MANIFEST.json")).read()
        if embedded != _canonical_bytes(manifest) or set(members) != set(expected):
            raise ValueError("archive manifest/membership mismatch")
        for name, entry in expected.items():
            data = archive.extractfile(members[name]).read()
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"archive member mismatch: {name}")
    print(json.dumps({"ok": True, "entries": len(expected), "bundle_sha256": index["bundle_sha256"]}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--stage", choices=("prereg", "complete"), required=True)
    build_parser.add_argument("--name", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.stage, args.name)
    else:
        verify(args.index.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
