#!/usr/bin/env python3
"""Build or verify a deterministic archive of source receipts and result evidence."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import tarfile
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.archived_current_fused_v1 import protocol
else:  # pragma: no cover
    from . import protocol


def resolve_include(value: str) -> Path:
    path = (protocol.REPO_ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(protocol.REPO_ROOT)
    except ValueError as exc:
        raise protocol.CampaignError(f"evidence include is outside repository: {path}") from exc
    if not path.exists():
        raise protocol.CampaignError(f"missing evidence include: {path}")
    return path


def collect(includes: list[str]) -> list[Path]:
    _campaign, receipt, _jobs, _lock = protocol.verify_lock()
    paths = {protocol.REPO_ROOT / relative for relative in receipt["source_sha256"]}
    paths.update({protocol.SOURCE_RECEIPT_PATH, protocol.JOBS_PATH, protocol.LOCK_PATH})
    for value in includes:
        path = resolve_include(value)
        if path.is_file():
            paths.add(path)
        else:
            paths.update(item for item in path.rglob("*") if item.is_file())
    normalized = []
    for path in paths:
        resolved = path.resolve()
        if resolved.is_symlink():
            raise protocol.CampaignError(f"evidence refuses symlink: {resolved}")
        normalized.append(resolved)
    return sorted(set(normalized), key=lambda path: str(path.relative_to(protocol.REPO_ROOT)))


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def build(name: str, includes: list[str]) -> int:
    protocol.validate_tag(name)
    evidence_dir = protocol.HERE / "evidence"
    bundle = evidence_dir / f"{name}.tar.gz"
    index_path = evidence_dir / f"{name}.index.json"
    if bundle.exists() or index_path.exists():
        raise protocol.CampaignError("refusing to overwrite evidence bundle/index")
    files = collect(includes)
    entries = [
        {
            "path": str(path.relative_to(protocol.REPO_ROOT)),
            "size": path.stat().st_size,
            "sha256": protocol.sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "record_type": "archived_current_fused_v1_evidence_manifest",
        "name": name,
        "campaign_id": protocol.load_campaign()["campaign_id"],
        "launch_lock_file_sha256": protocol.sha256_file(protocol.LOCK_PATH),
        "entries": entries,
    }
    manifest_bytes = protocol.stable_json_bytes(manifest)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    with bundle.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.GNU_FORMAT) as archive:
                add_bytes(archive, "EVIDENCE_MANIFEST.json", manifest_bytes)
                for path in files:
                    add_bytes(
                        archive,
                        str(path.relative_to(protocol.REPO_ROOT)),
                        path.read_bytes(),
                    )
    index = {
        "schema_version": 1,
        "record_type": "archived_current_fused_v1_evidence_index",
        "name": name,
        "bundle": str(bundle.relative_to(protocol.REPO_ROOT)),
        "bundle_sha256": protocol.sha256_file(bundle),
        "bundle_size": bundle.stat().st_size,
        "manifest_canonical_sha256": protocol.canonical_sha256(manifest),
        "entry_count": len(entries),
        "entries": entries,
    }
    protocol.atomic_json(index_path, index)
    print(
        f"BUILT {index_path.relative_to(protocol.REPO_ROOT)} entries={len(entries)} "
        f"bundle_sha256={index['bundle_sha256']}"
    )
    return 0


def verify(index_path: Path) -> int:
    index_path = index_path.resolve()
    index = protocol.read_json(index_path)
    if index_path.read_bytes() != protocol.stable_json_bytes(index):
        raise protocol.CampaignError("evidence index is not stable JSON")
    bundle = protocol.REPO_ROOT / index["bundle"]
    if protocol.sha256_file(bundle) != index["bundle_sha256"]:
        raise protocol.CampaignError("evidence bundle hash mismatch")
    with tarfile.open(bundle, mode="r:gz") as archive:
        manifest_stream = archive.extractfile("EVIDENCE_MANIFEST.json")
        if manifest_stream is None:
            raise protocol.CampaignError("cannot extract evidence manifest")
        import json

        manifest = json.load(manifest_stream)
        expected_names = {row["path"] for row in index["entries"]}
        observed_names = {
            member.name
            for member in archive.getmembers()
            if member.isfile() and member.name != "EVIDENCE_MANIFEST.json"
        }
        if observed_names != expected_names:
            raise protocol.CampaignError("evidence archive membership mismatch")
        if protocol.canonical_sha256(manifest) != index["manifest_canonical_sha256"]:
            raise protocol.CampaignError("evidence manifest hash mismatch")
        if manifest.get("entries") != index.get("entries"):
            raise protocol.CampaignError("evidence manifest/index entries differ")
        for entry in index["entries"]:
            stream = archive.extractfile(entry["path"])
            if stream is None:
                raise protocol.CampaignError(f"cannot extract {entry['path']}")
            data = stream.read()
            if len(data) != entry["size"]:
                raise protocol.CampaignError(f"evidence size mismatch: {entry['path']}")
            import hashlib

            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise protocol.CampaignError(f"evidence hash mismatch: {entry['path']}")
    print(f"VERIFIED {index_path} entries={index['entry_count']} bundle_sha256={index['bundle_sha256']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder = subparsers.add_parser("build")
    builder.add_argument("--name", required=True)
    builder.add_argument("--include", action="append", default=[])
    checker = subparsers.add_parser("verify")
    checker.add_argument("--index", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build":
        return build(args.name, args.include)
    return verify(args.index)


if __name__ == "__main__":
    raise SystemExit(main())
