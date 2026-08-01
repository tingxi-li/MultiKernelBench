#!/usr/bin/env python3
"""Build or verify deterministic complete v2 evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import tarfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from .core import (
        DEPENDENCY_PATHS, LOCK_PATH, REPO_ROOT, SOURCE_PATHS,
        SUPPORT_RESOLUTION_PATH, file_sha256, load_contract, read_json,
        result_root, stable_write,
    )
except ImportError:  # direct script execution
    from core import (
    DEPENDENCY_PATHS,
    LOCK_PATH,
    REPO_ROOT,
    SOURCE_PATHS,
    SUPPORT_RESOLUTION_PATH,
    file_sha256,
    load_contract,
    read_json,
    result_root,
    stable_write,
    )


EXCLUDED_NAMES = {"active.lock"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".torch_ext"}
EXCLUDED_SUFFIXES = {".o", ".so", ".pyc", ".ninja"}


def _normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def _files(tag: str) -> list[Path]:
    _campaign, _cells, lock = load_contract()
    files = {REPO_ROOT / relative for relative in (*SOURCE_PATHS, *DEPENDENCY_PATHS)}
    files.update({LOCK_PATH, SUPPORT_RESOLUTION_PATH})
    files.update(REPO_ROOT / relative for relative in lock["support_evidence_sha256"])
    for path in result_root(tag).rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if path.name in EXCLUDED_NAMES or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or ".partial." in path.name:
            continue
        files.add(path)
    return sorted(files, key=lambda path: str(path.relative_to(REPO_ROOT)))


def build(args) -> int:
    campaign, _cells, lock = load_contract()
    summary_path = Path(args.summary).resolve()
    try:
        summary_path.relative_to(result_root(args.tag).resolve())
    except ValueError as exc:
        raise RuntimeError("final summary is outside the requested result tag") from exc
    summary = read_json(summary_path)
    if summary.get("record_type") != "fused_crossed_v2_final_summary" or summary.get("complete") is not True:
        raise RuntimeError("only a complete v2 final summary can become evidence")
    if summary.get("launch_lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("summary is not bound to the final lock")
    if (
        summary.get("campaign_id") != campaign["campaign_id"]
        or summary.get("source_bundle_sha256") != lock["source_bundle_sha256"]
    ):
        raise RuntimeError("summary is not bound to the frozen campaign sources")
    files = _files(args.tag)
    files.append(summary_path)
    files = sorted(set(files), key=lambda path: str(path.relative_to(REPO_ROOT)))
    entries = [
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path), "size": path.stat().st_size}
        for path in files
    ]
    prefix = Path(args.out_prefix).resolve()
    archive, index = prefix.with_suffix(".tar.gz"), prefix.with_suffix(".index.json")
    if archive.exists() or index.exists():
        raise FileExistsError("refusing to overwrite v2 evidence")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as handle:
                for path in files:
                    handle.add(path, arcname=str(path.relative_to(REPO_ROOT)), recursive=False, filter=_normalized)
    stable_write(
        index,
        {
            "archive_path": str(archive.relative_to(REPO_ROOT)),
            "archive_sha256": file_sha256(archive),
            "campaign_id": campaign["campaign_id"],
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "entries": entries,
            "entry_count": len(entries),
            "launch_lock_sha256": file_sha256(LOCK_PATH),
            "record_type": "fused_crossed_v2_complete_evidence_index",
            "schema_version": 2,
            "source_bundle_sha256": lock["source_bundle_sha256"],
            "summary_path": str(summary_path.relative_to(REPO_ROOT)),
            "summary_sha256": file_sha256(summary_path),
        },
    )
    print(f"archive={archive} entries={len(entries)}")
    return 0


def _validated_entries(index: dict, campaign: dict, lock: dict, lock_sha256: str) -> dict:
    entries = index.get("entries")
    if (
        index.get("record_type") != "fused_crossed_v2_complete_evidence_index"
        or index.get("schema_version") != 2
        or index.get("campaign_id") != campaign["campaign_id"]
        or index.get("launch_lock_sha256") != lock_sha256
        or index.get("source_bundle_sha256") != lock["source_bundle_sha256"]
        or not isinstance(entries, list)
        or index.get("entry_count") != len(entries)
    ):
        raise RuntimeError("invalid v2 evidence index header or count")
    expected = {row.get("path"): row for row in entries if isinstance(row, dict)}
    if len(expected) != len(entries) or None in expected:
        raise RuntimeError("evidence index contains duplicate or malformed entries")
    controlling = {
        index.get("summary_path"): index.get("summary_sha256"),
        str(LOCK_PATH.relative_to(REPO_ROOT)): lock_sha256,
    }
    if any(expected.get(path, {}).get("sha256") != digest for path, digest in controlling.items()):
        raise RuntimeError("evidence index lost its summary or launch-lock binding")
    return expected


def verify(args) -> int:
    campaign, _cells, lock = load_contract()
    index_path = Path(args.index).resolve()
    index = read_json(index_path)
    expected = _validated_entries(index, campaign, lock, file_sha256(LOCK_PATH))
    archive = (REPO_ROOT / index["archive_path"]).resolve()
    summary = (REPO_ROOT / index["summary_path"]).resolve()
    for path in (archive, summary):
        try:
            path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"evidence path escapes repository: {path}") from exc
    if not summary.is_file() or file_sha256(summary) != index["summary_sha256"]:
        raise RuntimeError("current summary hash mismatch")
    if file_sha256(archive) != index["archive_sha256"]:
        raise RuntimeError("archive hash mismatch")
    observed = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile() or member.name in observed:
                raise RuntimeError("archive contains a non-file or duplicate member")
            extracted = handle.extractfile(member)
            assert extracted is not None
            payload = extracted.read()
            observed[member.name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    if set(observed) != set(expected):
        raise RuntimeError("archive member census mismatch")
    for path, row in expected.items():
        if observed[path] != {"sha256": row["sha256"], "size": row["size"]}:
            raise RuntimeError(f"archive member mismatch: {path}")
    print(f"verified={archive} entries={len(observed)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--tag", required=True)
    build_parser.add_argument("--summary", required=True)
    build_parser.add_argument("--out-prefix", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--index", required=True)
    args = parser.parse_args()
    return build(args) if args.command == "build" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
