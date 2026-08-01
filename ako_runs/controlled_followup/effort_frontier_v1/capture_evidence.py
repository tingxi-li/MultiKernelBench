#!/usr/bin/env python3
"""Build or verify deterministic preregistration/complete evidence bundles."""
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

try:
    from . import analyze, campaign, validate
except ImportError:  # direct script execution
    import analyze  # type: ignore
    import campaign  # type: ignore
    import validate  # type: ignore


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EXCLUDED_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".torch_ext", "build", "dist", "evidence"}
)
EXCLUDED_NAMES = frozenset(
    {".launcher.lock", ".confirmation.lock", "active.lock"}
)
EXCLUDED_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".o",
        ".so",
        ".a",
        ".cubin",
        ".ptx",
        ".ninja",
        ".lock",
        ".gz",
        ".tar",
        ".zip",
    }
)


def _excluded(path: Path) -> bool:
    relative = path.relative_to(campaign.REPO_ROOT)
    return (
        path.name in EXCLUDED_NAMES
        or any(part in EXCLUDED_PARTS for part in relative.parts)
        or path.suffix in EXCLUDED_SUFFIXES
        or ".partial." in path.name
        or path.name.endswith(".partial")
    )


def _source_files() -> set[Path]:
    files = {campaign.HERE / relative for relative in campaign.PROVENANCE_FILES}
    files.update(
        {
            campaign.HERE / ".gitignore",
            campaign.HERE / "tests/__init__.py",
            campaign.HERE / "tests/test_protocol.py",
            campaign.MODEL_LOCK,
            campaign.GATE_SPEC,
            campaign.GATE_RECEIPT,
        }
    )
    return files


def _assert_files(files: set[Path]) -> list[Path]:
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("evidence input is missing: " + "; ".join(sorted(missing)))
    escaped = []
    for path in files:
        try:
            path.resolve().relative_to(campaign.REPO_ROOT.resolve())
        except ValueError:
            escaped.append(str(path))
    if escaped:
        raise ValueError("evidence input escapes repository: " + "; ".join(sorted(escaped)))
    excluded = [campaign.repo_path(path) for path in files if _excluded(path)]
    if excluded:
        raise ValueError("selected evidence contains an excluded artifact: " + "; ".join(excluded))
    return sorted((path.resolve() for path in files), key=campaign.repo_path)


def selected_prereg_files() -> list[Path]:
    validate.validate_static()
    return _assert_files(_source_files())


def _canonical_analysis(
    result_root: Path,
    plan_path: Path,
    record_path: Path,
    analysis_path: Path,
) -> dict[str, Any]:
    search, _ = analyze.validate_search(result_root)
    expected_plan = analyze.build_confirmation_plan(result_root)
    supplied_plan = campaign.load_json(plan_path)
    if (
        supplied_plan != expected_plan
        or plan_path.read_bytes() != campaign.stable_json_bytes(supplied_plan)
    ):
        raise ValueError("complete evidence confirmation plan is stale/noncanonical")
    records = analyze._validate_confirmation_records(expected_plan, record_path)
    confirmation = analyze.analyze_confirmation(search, expected_plan, records)
    if confirmation.get("complete") is not True:
        raise ValueError("complete evidence is blocked by failed confirmation records")
    expected = {"search": search, "confirmation": confirmation}
    supplied = campaign.load_json(analysis_path)
    if supplied != expected or analysis_path.read_bytes() != campaign.stable_json_bytes(supplied):
        raise ValueError("complete evidence analysis is stale/noncanonical")
    return expected


def selected_complete_files(
    result_root: Path,
    plan_path: Path,
    record_path: Path,
    analysis_path: Path,
) -> tuple[list[Path], dict[str, Any]]:
    analysis_value = _canonical_analysis(result_root, plan_path, record_path, analysis_path)
    files = _source_files()
    # Immutable JSON locks/receipts are evidence, unlike transient runtime
    # *.lock mutex files, which are excluded by _excluded.
    files.update(
        {
            campaign.EXECUTOR_REGISTRY,
            campaign.PROVENANCE_LOCK,
            plan_path.resolve(),
            record_path.resolve(),
            analysis_path.resolve(),
        }
    )
    result_root = result_root.resolve()
    try:
        result_root.relative_to(campaign.REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("complete result root must be inside the repository") from exc
    for path in result_root.rglob("*"):
        if path.is_file() and not _excluded(path):
            files.add(path)
    registry = campaign.load_json(campaign.EXECUTOR_REGISTRY)
    for entry in [*registry["lanes"].values(), registry["control"]]:
        files.update(campaign.REPO_ROOT / relative for relative in entry["source_hashes"])
    return _assert_files(files), analysis_value


def _tar_add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    data = campaign.stable_json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The exclusive file is intentionally left visible if writing fails;
        # callers never mistake a partial index for a valid canonical index.
        raise


def build(
    stage: str,
    name: str,
    output_dir: Path,
    *,
    result_root: Path | None = None,
    plan_path: Path | None = None,
    record_path: Path | None = None,
    analysis_path: Path | None = None,
) -> Path:
    if stage not in {"prereg", "complete"} or not NAME_RE.fullmatch(name):
        raise ValueError("unsafe evidence stage/name")
    analysis_sha = None
    if stage == "prereg":
        if any(value is not None for value in (result_root, plan_path, record_path, analysis_path)):
            raise ValueError("prereg evidence does not accept result inputs")
        files = selected_prereg_files()
    else:
        if any(value is None for value in (result_root, plan_path, record_path, analysis_path)):
            raise ValueError("complete evidence requires result root, plan, records, and analysis")
        files, analysis_value = selected_complete_files(
            result_root, plan_path, record_path, analysis_path  # type: ignore[arg-type]
        )
        analysis_sha = campaign.canonical_sha256(analysis_value)
    entries = [
        {
            "path": campaign.repo_path(path),
            "size": path.stat().st_size,
            "sha256": campaign.file_sha256(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "record_type": "effort_frontier_evidence_manifest",
        "campaign_id": campaign.CAMPAIGN_ID,
        "stage": stage,
        "analysis_canonical_sha256": analysis_sha,
        "exclusion_policy": {
            "parts": sorted(EXCLUDED_PARTS),
            "names": sorted(EXCLUDED_NAMES),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
            "partial_names": True,
        },
        "entries": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = output_dir / f"{name}.tar.gz"
    index = output_dir / f"{name}.index.json"
    if bundle.exists() or index.exists():
        raise FileExistsError("evidence bundle/index is immutable and already exists")
    partial = output_dir / f".{name}.tar.gz.partial.{os.getpid()}"
    try:
        with partial.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    _tar_add(
                        archive,
                        "EVIDENCE_MANIFEST.json",
                        campaign.stable_json_bytes(manifest),
                    )
                    for path, entry in zip(files, entries, strict=True):
                        _tar_add(archive, entry["path"], path.read_bytes())
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(partial, bundle)
    except BaseException:
        # A crash artifact retains `.partial.` in its name and is never selected
        # into a later evidence bundle. It is not silently promoted or removed.
        raise
    index_value = {
        "schema_version": 1,
        "record_type": "effort_frontier_evidence_index",
        "campaign_id": campaign.CAMPAIGN_ID,
        "stage": stage,
        "bundle_filename": bundle.name,
        "bundle_sha256": campaign.file_sha256(bundle),
        "bundle_size": bundle.stat().st_size,
        "manifest_canonical_sha256": campaign.canonical_sha256(manifest),
        "entry_count": len(entries),
        "manifest": manifest,
    }
    _exclusive_json(index, index_value)
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "bundle_sha256": index_value["bundle_sha256"],
                "entries": len(entries),
                "index": str(index),
                "stage": stage,
            },
            sort_keys=True,
        )
    )
    return index


def verify(index_path: Path) -> dict[str, Any]:
    index_path = index_path.resolve()
    index = campaign.load_json(index_path)
    if index_path.read_bytes() != campaign.stable_json_bytes(index):
        raise ValueError("evidence index is not canonical stable JSON")
    if (
        index.get("schema_version") != 1
        or index.get("record_type") != "effort_frontier_evidence_index"
        or index.get("campaign_id") != campaign.CAMPAIGN_ID
        or index.get("stage") not in {"prereg", "complete"}
    ):
        raise ValueError("evidence index identity differs")
    bundle_name = index.get("bundle_filename")
    if not isinstance(bundle_name, str) or Path(bundle_name).name != bundle_name:
        raise ValueError("evidence bundle filename is unsafe")
    bundle = index_path.parent / bundle_name
    if (
        not bundle.is_file()
        or campaign.file_sha256(bundle) != index.get("bundle_sha256")
        or bundle.stat().st_size != index.get("bundle_size")
    ):
        raise ValueError("evidence bundle bytes differ from index")
    manifest = index.get("manifest")
    if (
        not isinstance(manifest, dict)
        or campaign.canonical_sha256(manifest) != index.get("manifest_canonical_sha256")
    ):
        raise ValueError("evidence manifest hash differs")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("record_type") != "effort_frontier_evidence_manifest"
        or manifest.get("campaign_id") != campaign.CAMPAIGN_ID
        or manifest.get("stage") != index["stage"]
    ):
        raise ValueError("embedded evidence manifest identity differs")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != index.get("entry_count"):
        raise ValueError("evidence entry count differs")
    expected = {row["path"]: row for row in entries}
    if len(expected) != len(entries):
        raise ValueError("evidence manifest has duplicate paths")
    for row in entries:
        name = row.get("path")
        digest = row.get("sha256")
        size = row.get("size")
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError("evidence manifest has an unsafe/malformed entry")
    with tarfile.open(bundle, "r:gz") as archive:
        all_members = archive.getmembers()
        if any(not member.isfile() for member in all_members):
            raise ValueError("evidence archive contains a non-file member")
        if any(
            member.mtime != 0
            or member.uid != 0
            or member.gid != 0
            or member.mode != 0o644
            for member in all_members
        ):
            raise ValueError("evidence archive metadata is not deterministic")
        file_members = all_members
        members = {member.name: member for member in file_members}
        if len(members) != len(file_members):
            raise ValueError("evidence archive has duplicate file members")
        embedded_member = members.pop("EVIDENCE_MANIFEST.json", None)
        if embedded_member is None:
            raise ValueError("embedded evidence manifest is missing")
        embedded_handle = archive.extractfile(embedded_member)
        if embedded_handle is None or embedded_handle.read() != campaign.stable_json_bytes(manifest):
            raise ValueError("embedded evidence manifest differs")
        if set(members) != set(expected):
            raise ValueError("evidence archive membership differs")
        for name, entry in expected.items():
            handle = archive.extractfile(members[name])
            if handle is None:
                raise ValueError(f"cannot read evidence member {name}")
            data = handle.read()
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"evidence member differs: {name}")
    result = {
        "ok": True,
        "stage": index["stage"],
        "entries": len(entries),
        "bundle_sha256": index["bundle_sha256"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--stage", choices=("prereg", "complete"), required=True)
    build_parser.add_argument("--name", required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--result-root", type=Path)
    build_parser.add_argument("--confirmation-plan", type=Path)
    build_parser.add_argument("--confirmation-records", type=Path)
    build_parser.add_argument("--analysis", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.index)
    else:
        build(
            args.stage,
            args.name,
            args.output_dir,
            result_root=args.result_root,
            plan_path=args.confirmation_plan,
            record_path=args.confirmation_records,
            analysis_path=args.analysis,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
