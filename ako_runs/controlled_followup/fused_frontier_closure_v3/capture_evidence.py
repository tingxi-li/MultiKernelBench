#!/usr/bin/env python3
"""Build or verify a deterministic complete v3 evidence bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import sys
import tarfile
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.fused_frontier_closure_v3 import (
        core,
        eligibility,
        provenance,
    )
else:  # pragma: no cover
    from . import core, eligibility, provenance


def _tar_add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def _validate_results(tag: str) -> tuple[Path, dict]:
    root = core.RESULTS_ROOT / tag
    summary_path = root / "analysis_summary.json"
    summary = core.read_json(summary_path)
    if (
        summary.get("record_type") != "fused_frontier_closure_v3_analysis"
        or summary.get("status") != "COMPLETE"
        or summary.get("source_receipt_sha256")
        != core.sha256_file(core.SOURCE_RECEIPT_PATH)
        or summary.get("eligibility_receipt_sha256")
        != core.sha256_file(core.ELIGIBILITY_RECEIPT_PATH)
        or len(summary.get("raw_record_sha256", {})) != 120
    ):
        raise core.ClosureError("v3 analysis is not complete/bound")
    for relative, expected in summary["raw_record_sha256"].items():
        path = core.REPO_ROOT / relative
        if not path.is_file() or core.sha256_file(path) != expected:
            raise core.ClosureError(f"analysis raw hash differs: {relative}")
    return root, summary


def _selected_files(root: Path, source: dict) -> list[Path]:
    relatives = set(source["local_source_sha256"])
    relatives.update(source["transitive_candidate_source_sha256"])
    relatives.update(source["imported_evidence_sha256"])
    relatives.add(str(core.SOURCE_RECEIPT_PATH.relative_to(core.REPO_ROOT)))
    relatives.add(str(core.ELIGIBILITY_RECEIPT_PATH.relative_to(core.REPO_ROOT)))
    files = {core.REPO_ROOT / relative for relative in relatives}
    for path in root.rglob("*"):
        if path.is_file() and not path.name.endswith(".tmp"):
            files.add(path)
    return sorted(files, key=lambda path: str(path.relative_to(core.REPO_ROOT)))


def build(name: str, tag: str) -> Path:
    if not name or any(value in name for value in ("/", "..")):
        raise core.ClosureError("unsafe evidence name")
    source = provenance.verify_receipt()
    eligibility.verify_receipt()
    root, analysis = _validate_results(tag)
    files = _selected_files(root, source)
    entries = [
        {
            "path": str(path.relative_to(core.REPO_ROOT)),
            "size": path.stat().st_size,
            "sha256": core.sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "record_type": "fused_frontier_closure_v3_evidence_manifest",
        "campaign_id": source["campaign_id"],
        "source_receipt_sha256": core.sha256_file(core.SOURCE_RECEIPT_PATH),
        "eligibility_receipt_sha256": core.sha256_file(
            core.ELIGIBILITY_RECEIPT_PATH
        ),
        "analysis_sha256": core.sha256_file(root / "analysis_summary.json"),
        "launch_receipt_sha256": analysis["launch_receipt_sha256"],
        "tag": tag,
        "entries": entries,
    }
    output = core.HERE / "evidence"
    bundle, index_path = output / f"{name}.tar.gz", output / f"{name}.index.json"
    if bundle.exists() or index_path.exists():
        raise FileExistsError("evidence output is immutable; choose a new name")
    output.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_suffix(bundle.suffix + ".tmp")
    manifest_bytes = core.canonical_json_bytes(manifest) + b"\n"
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                _tar_add(archive, "EVIDENCE_MANIFEST.json", manifest_bytes)
                for path, entry in zip(files, entries):
                    _tar_add(archive, entry["path"], path.read_bytes())
    temporary.replace(bundle)
    index = {
        "schema_version": 1,
        "record_type": "fused_frontier_closure_v3_evidence_index",
        "bundle_path": str(bundle.relative_to(core.REPO_ROOT)),
        "bundle_sha256": core.sha256_file(bundle),
        "bundle_size": bundle.stat().st_size,
        "manifest_sha256": core.canonical_sha256(manifest),
        "entry_count": len(entries),
        "manifest": manifest,
    }
    core.atomic_json(index_path, index)
    print(
        f"built {index_path} entries={len(entries)} "
        f"bundle_sha256={index['bundle_sha256']}"
    )
    return index_path


def verify(index_path: Path) -> None:
    index = core.read_json(index_path)
    bundle = core.REPO_ROOT / index["bundle_path"]
    if core.sha256_file(bundle) != index["bundle_sha256"]:
        raise core.ClosureError("evidence bundle hash differs")
    manifest = index["manifest"]
    if core.canonical_sha256(manifest) != index["manifest_sha256"]:
        raise core.ClosureError("evidence manifest hash differs")
    expected = {row["path"]: row for row in manifest["entries"]}
    with tarfile.open(bundle, mode="r:gz") as archive:
        members = {item.name: item for item in archive.getmembers() if item.isfile()}
        embedded = members.pop("EVIDENCE_MANIFEST.json", None)
        if embedded is None or archive.extractfile(embedded).read() != (
            core.canonical_json_bytes(manifest) + b"\n"
        ):
            raise core.ClosureError("embedded evidence manifest differs")
        if set(members) != set(expected):
            raise core.ClosureError("evidence membership differs")
        for name, row in expected.items():
            data = archive.extractfile(members[name]).read()
            if len(data) != row["size"] or hashlib.sha256(data).hexdigest() != row[
                "sha256"
            ]:
                raise core.ClosureError(f"evidence payload differs: {name}")
    print(f"OK evidence entries={len(expected)} bundle_sha256={index['bundle_sha256']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--name", required=True)
    build_parser.add_argument("--tag", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--index", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.name, args.tag)
    else:
        verify(args.index.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
