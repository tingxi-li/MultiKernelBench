"""Build or verify a deterministic evidence archive for this audit."""

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

from ako_runs.controlled_followup.robust_gate.schema import canonical_sha256, file_sha256, load_json

from .analyze import COMPLETION_PATH, SUMMARY_PATH
from .runner import (
    BUILD_PATH,
    COLLECTION_PATH,
    EXECUTION_PATH,
    FREEZE_PATH,
    HERE,
    LAUNCH_PATH,
    MANIFEST_PATH,
    REPO_ROOT,
    verify_campaign,
)


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite of {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_complete() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, _gate, _lock, _seeds = verify_campaign(require_freeze=True)
    required = (
        FREEZE_PATH,
        LAUNCH_PATH,
        EXECUTION_PATH,
        BUILD_PATH,
        COLLECTION_PATH,
        COMPLETION_PATH,
        SUMMARY_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    completion = load_json(COMPLETION_PATH)
    summary = load_json(SUMMARY_PATH)
    if (
        completion.get("summary_sha256") != file_sha256(SUMMARY_PATH)
        or completion.get("collection_receipt_sha256") != file_sha256(COLLECTION_PATH)
        or completion.get("evidence_complete") != summary.get("evidence_complete")
        or completion.get("all_candidate_gate_groups_success")
        != summary.get("all_candidate_gate_groups_success")
    ):
        raise ValueError("completion/summary binding mismatch")
    raw = HERE / manifest["workload"]["output"]
    collection = load_json(COLLECTION_PATH)
    if collection.get("raw_sha256") != file_sha256(raw):
        raise ValueError("collection/raw binding mismatch")
    return manifest, summary


def selected_files(manifest: dict[str, Any]) -> list[Path]:
    freeze = load_json(FREEZE_PATH)
    files = {REPO_ROOT / relative for relative in freeze["source_sha256"]}
    reach = manifest["reachability_binding"]
    gate = manifest["registered_gate_binding"]
    prior = manifest["prior_stress_binding"]
    for relative in (
        reach["launch_lock_path"],
        reach["screen_selection_path"],
        reach["screen_summary_path"],
        gate["gate_spec_path"],
        gate["manifest_path"],
        prior["manifest_path"],
    ):
        files.add(REPO_ROOT / relative)
    files.update(
        {
            FREEZE_PATH,
            LAUNCH_PATH,
            EXECUTION_PATH,
            BUILD_PATH,
            COLLECTION_PATH,
            COMPLETION_PATH,
            SUMMARY_PATH,
            HERE / manifest["workload"]["output"],
        }
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return sorted((path.resolve() for path in files), key=lambda path: str(path.relative_to(REPO_ROOT)))


def _tar_add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def build(name: str) -> Path:
    if name in (".", "..") or not NAME_RE.fullmatch(name):
        raise ValueError(f"unsafe evidence name: {name!r}")
    manifest, summary = _validate_complete()
    files = selected_files(manifest)
    entries = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    evidence_manifest = {
        "schema_version": "1.0",
        "record_type": "fused_reachability_row_sum_stress_evidence_manifest",
        "campaign_id": manifest["campaign_id"],
        "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
        "launch_receipt_sha256": file_sha256(LAUNCH_PATH),
        "completion_receipt_sha256": file_sha256(COMPLETION_PATH),
        "summary_sha256": file_sha256(SUMMARY_PATH),
        "evidence_complete": summary["evidence_complete"],
        "all_candidate_gate_groups_success": summary["all_candidate_gate_groups_success"],
        "performance_selection_feedback_authorized": False,
        "entries": entries,
    }
    output = HERE / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / f"{name}.tar.gz"
    index_path = output / f"{name}.index.json"
    if bundle.exists() or index_path.exists():
        raise FileExistsError("evidence outputs are immutable; choose another name")
    embedded = canonical_bytes(evidence_manifest) + b"\n"
    temporary = bundle.with_suffix(bundle.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                _tar_add(archive, "EVIDENCE_MANIFEST.json", embedded)
                for path, entry in zip(files, entries, strict=True):
                    _tar_add(archive, entry["path"], path.read_bytes())
    temporary.replace(bundle)
    index = {
        "schema_version": "1.0",
        "record_type": "fused_reachability_row_sum_stress_evidence_index",
        "bundle_path": str(bundle.relative_to(REPO_ROOT)),
        "bundle_sha256": file_sha256(bundle),
        "bundle_size": bundle.stat().st_size,
        "manifest_sha256": canonical_sha256(evidence_manifest),
        "entry_count": len(entries),
        "manifest": evidence_manifest,
    }
    _exclusive(index_path, index)
    print(json.dumps({"index": str(index_path), "bundle_sha256": index["bundle_sha256"], "entries": len(entries)}, sort_keys=True))
    return index_path


def verify(index_path: Path) -> None:
    index = load_json(index_path)
    bundle = REPO_ROOT / index["bundle_path"]
    if file_sha256(bundle) != index["bundle_sha256"]:
        raise ValueError("bundle SHA mismatch")
    manifest = index["manifest"]
    if canonical_sha256(manifest) != index["manifest_sha256"]:
        raise ValueError("embedded manifest SHA mismatch")
    expected = {entry["path"]: entry for entry in manifest["entries"]}
    with tarfile.open(bundle, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        embedded = archive.extractfile(members.pop("EVIDENCE_MANIFEST.json")).read()
        if embedded != canonical_bytes(manifest) + b"\n":
            raise ValueError("embedded manifest bytes mismatch")
        if set(members) != set(expected):
            raise ValueError("archive membership mismatch")
        for name, entry in expected.items():
            data = archive.extractfile(members[name]).read()
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"archive member mismatch: {name}")
    print(json.dumps({"ok": True, "entries": len(expected), "bundle_sha256": index["bundle_sha256"]}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--name", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--index", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.name)
    else:
        verify(Path(args.index).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
