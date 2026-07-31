#!/usr/bin/env python3
"""Capture or verify complete v1r1 evidence without rewriting prior artifacts."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from . import common, validate
except ImportError:  # pragma: no cover - direct-script CLI
    import common  # type: ignore
    import validate  # type: ignore


MANIFEST_NAME = "EVIDENCE_MANIFEST.json"


def _safe_name(name: str) -> None:
    value = PurePosixPath(name)
    common.require(
        name
        and not value.is_absolute()
        and value.as_posix() == name
        and all(part not in {"", ".", ".."} for part in value.parts),
        f"unsafe archive name: {name!r}",
    )


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=common.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def selection(summary: Path) -> tuple[list[Path], dict[str, str]]:
    state = validate.validate_structure()
    binding = common.binding_for_commit(_source_commit())
    common.validate_retained_tree(binding)
    summary = summary.resolve()
    common.require(
        summary.is_file()
        and summary.is_relative_to(common.RESULT_ROOT.resolve()),
        "final summary missing or outside v1r1 results",
    )
    final = common.read_json(summary)
    common.validate_binding(final, binding, "final summary")
    common.require(
        final.get("record_type") == "fused_crossed_final_summary"
        and final.get("complete") is True,
        "complete final summary required",
    )
    files: set[Path] = {common.LOCK_PATH, common.INCIDENT_PATH}
    lock = state["lock"]
    files.update(common.REPO_ROOT / name for name in lock["source_sha256"])
    files.update(common.REPO_ROOT / name for name in lock["dependency_sha256"])
    for root in (common.PARENT_RESULT_ROOT, common.RESULT_ROOT):
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name.endswith(".lock") or ".partial." in path.name:
                continue
            common.require(
                path.suffix in {".json", ".jsonl"},
                f"unexpected evidence type: {path}",
            )
            files.add(path.resolve())
    ordered = sorted(files, key=common.repo_path)
    common.require(
        all(path.is_file() and not path.is_symlink() for path in ordered),
        "evidence selection contains missing/symlinked input",
    )
    return ordered, binding


def _tar_member(name: str, data: bytes) -> tuple[tarfile.TarInfo, io.BytesIO]:
    _safe_name(name)
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info, io.BytesIO(data)


def build(summary: Path, out_prefix: Path) -> dict[str, Any]:
    files, binding = selection(summary)
    bundle = out_prefix.with_suffix(".tar.gz")
    index_path = out_prefix.with_suffix(".index.json")
    common.require(not bundle.exists() and not index_path.exists(), "evidence output exists")
    entries = [
        {
            "path": common.repo_path(path),
            "sha256": common.file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in files
    ]
    manifest = common.add_binding(
        {
            "campaign_id": common.CAMPAIGN_ID,
            "complete": True,
            "entries": entries,
            "failed_parent_launch_preserved": True,
            "record_type": "fused_crossed_v1r1_evidence_manifest",
            "schema_version": 1,
            "summary_path": common.repo_path(summary.resolve()),
            "summary_sha256": common.file_sha256(summary.resolve()),
        },
        binding,
    )
    payload = common.stable_json_bytes(manifest)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_name(f".{bundle.name}.partial.{os.getpid()}")
    common.require(not temporary.exists(), "unreconciled evidence partial")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as archive:
                    info, stream = _tar_member(MANIFEST_NAME, payload)
                    archive.addfile(info, stream)
                    for path, entry in zip(files, entries, strict=True):
                        info, stream = _tar_member(entry["path"], path.read_bytes())
                        archive.addfile(info, stream)
        os.link(temporary, bundle)
    finally:
        temporary.unlink(missing_ok=True)
    index = common.add_binding(
        {
            "bundle_path": common.repo_path(bundle),
            "bundle_sha256": common.file_sha256(bundle),
            "bundle_size": bundle.stat().st_size,
            "campaign_id": common.CAMPAIGN_ID,
            "entry_count": len(entries),
            "manifest_sha256": common.canonical_sha256(manifest),
            "record_type": "fused_crossed_v1r1_evidence_index",
            "schema_version": 1,
        },
        binding,
    )
    common.exclusive_json(index_path, index)
    return verify(index_path)


def verify(index_path: Path) -> dict[str, Any]:
    index = common.read_json(index_path)
    binding = common.binding_for_commit(index.get("recovery_git_commit", ""))
    common.validate_binding(index, binding, "evidence index")
    bundle = common.REPO_ROOT / str(index.get("bundle_path", ""))
    common.require(
        bundle.is_file()
        and common.file_sha256(bundle) == index.get("bundle_sha256")
        and bundle.stat().st_size == index.get("bundle_size"),
        "evidence bundle size/hash mismatch",
    )
    with tarfile.open(bundle, "r:gz") as archive:
        members_list = archive.getmembers()
        common.require(
            all(member.isfile() for member in members_list)
            and len({member.name for member in members_list}) == len(members_list),
            "evidence members are not unique regular files",
        )
        for member in members_list:
            _safe_name(member.name)
            common.require(
                member.mode == 0o644
                and member.mtime == 0
                and member.uid == member.gid == 0
                and member.uname == member.gname == "",
                f"evidence metadata differs: {member.name}",
            )
        members = {member.name: member for member in members_list}
        manifest_member = members.pop(MANIFEST_NAME, None)
        common.require(manifest_member is not None, "evidence manifest absent")
        stream = archive.extractfile(manifest_member)
        common.require(stream is not None, "evidence manifest unreadable")
        manifest = json.loads(stream.read())
        common.validate_binding(manifest, binding, "evidence manifest")
        common.require(
            common.canonical_sha256(manifest) == index.get("manifest_sha256")
            and manifest.get("complete") is True,
            "evidence manifest identity/state mismatch",
        )
        entries = manifest.get("entries")
        common.require(
            isinstance(entries, list)
            and len(entries) == index.get("entry_count"),
            "evidence entry count mismatch",
        )
        expected = {entry["path"]: entry for entry in entries}
        common.require(len(expected) == len(entries) and set(members) == set(expected), "evidence membership differs")
        for name, entry in expected.items():
            member = members[name]
            stream = archive.extractfile(member)
            common.require(stream is not None, f"evidence member unreadable: {name}")
            data = stream.read()
            common.require(
                len(data) == entry["size"]
                and hashlib.sha256(data).hexdigest() == entry["sha256"],
                f"evidence member differs: {name}",
            )
    return {
        "bundle_sha256": index["bundle_sha256"],
        "entries": index["entry_count"],
        "ok": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--summary", type=Path, required=True)
    build_parser.add_argument("--out-prefix", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    result = (
        build(args.summary, args.out_prefix)
        if args.command == "build"
        else verify(args.index)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
