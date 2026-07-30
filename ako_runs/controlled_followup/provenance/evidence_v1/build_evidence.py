#!/usr/bin/env python3
"""Build and verify a compact, post-hoc evidence-preservation bundle.

This script never writes below ``fused_grid/results``.  It reads the completed
screen, confirmation, and robust-validation artifacts, records their hashes,
and stores deterministic copies in a gzip-compressed tar archive next to this
script.  The resulting bundle preserves evidence; it is not a launch-time
freeze, preregistration record, or external timestamp.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
GENERATOR = Path(__file__).resolve()
RESULTS = REPO / "ako_runs/controlled_followup/fused_grid/results"
INDEX = HERE / "evidence_index.json"
BUNDLE = HERE / "evidence_bundle.tar.gz"

SCREEN = RESULTS / "fused_gbgs_grid_rank"
CONFIRMATION = RESULTS / "fused_gbgs_confirm_robust_v1"
ROBUST = RESULTS / "robust"

SCREEN_TOP = (
    "confirmation_jobs.robust_validation.json",
    "screen_summary.json",
    "screen_summary.robust_validation.json",
    "status.json",
)
CONFIRMATION_TOP = ("status.json", "summary.json")
ROBUST_RUNS = {
    "tuning_smoke_all76": {
        "records": 608,
        "files": ("records.jsonl", "run.json", "summary.json"),
    },
    "tuning_full_screened55": {
        "records": 3520,
        "files": ("records.jsonl", "run.json", "summary.json"),
    },
    "validation_screened56": {
        "records": 28672,
        "files": (
            "records.jsonl",
            "run.json",
            "summary.json",
            "summary.validation.json",
            "summary.validation.receipt.json",
        ),
    },
}


class EvidenceError(RuntimeError):
    """The completed evidence tree does not satisfy the preservation contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as source:
        for count, _line in enumerate(source, start=1):
            pass
    return count


def exactly_one_launch(directory: Path) -> Path:
    launches = sorted(directory.glob("launch_*.json"))
    if len(launches) != 1:
        raise EvidenceError(
            f"expected exactly one launch receipt below {directory}, found {len(launches)}"
        )
    return launches[0]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise EvidenceError(f"required evidence file is absent: {path}")
    return path


def selected_files() -> tuple[list[Path], dict[str, Any]]:
    screen_raw = sorted((SCREEN / "raw").glob("*.json"))
    confirmation_raw = sorted((CONFIRMATION / "raw").glob("*.json"))
    if len(screen_raw) != 152:
        raise EvidenceError(f"expected 152 screen records, found {len(screen_raw)}")
    if len(confirmation_raw) != 80:
        raise EvidenceError(
            f"expected 80 confirmation records, found {len(confirmation_raw)}"
        )

    files: list[Path] = []
    files.extend(screen_raw)
    files.append(exactly_one_launch(SCREEN))
    files.extend(require_file(SCREEN / name) for name in SCREEN_TOP)
    files.extend(confirmation_raw)
    files.append(exactly_one_launch(CONFIRMATION))
    files.extend(require_file(CONFIRMATION / name) for name in CONFIRMATION_TOP)

    robust_counts: dict[str, int] = {}
    for run_name, contract in ROBUST_RUNS.items():
        run_dir = ROBUST / run_name
        for name in contract["files"]:
            files.append(require_file(run_dir / name))
        observed = count_lines(run_dir / "records.jsonl")
        expected = int(contract["records"])
        if observed != expected:
            raise EvidenceError(
                f"{run_name}/records.jsonl: expected {expected} records, found {observed}"
            )
        robust_counts[run_name] = observed

    unique = sorted(set(files), key=lambda path: path.relative_to(REPO).as_posix())
    if len(unique) != len(files):
        raise EvidenceError("the evidence selection contains duplicate paths")
    counts = {
        "screen_process_records": len(screen_raw),
        "confirmation_process_records": len(confirmation_raw),
        "robust_aggregate_records": robust_counts,
        "selected_files": len(unique),
    }
    return unique, counts


def category(path: Path) -> str:
    relative = path.relative_to(RESULTS)
    if relative.parts[0] == SCREEN.name:
        return "screen_raw" if relative.parts[1] == "raw" else "screen_control"
    if relative.parts[0] == CONFIRMATION.name:
        return (
            "confirmation_raw"
            if relative.parts[1] == "raw"
            else "confirmation_control"
        )
    if path.name == "records.jsonl":
        return "robust_aggregate_records"
    return "robust_control"


def entry(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "category": category(path),
        "path": path.relative_to(REPO).as_posix(),
        "sha256": sha256_file(path),
    }
    if path.name == "records.jsonl":
        value["jsonl_records"] = count_lines(path)
    return value


def normalized_tarinfo(path: Path, arcname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=arcname)
    info.size = path.stat().st_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def write_bundle(paths: Iterable[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as raw_tmp:
        raw_path = Path(raw_tmp.name)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as gzip_tmp:
        gzip_path = Path(gzip_tmp.name)
    try:
        with tarfile.open(raw_path, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for path in paths:
                arcname = path.relative_to(REPO).as_posix()
                with path.open("rb") as source:
                    archive.addfile(normalized_tarinfo(path, arcname), source)
        with raw_path.open("rb") as source, gzip_path.open("wb") as sink:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=sink, compresslevel=9, mtime=0
            ) as compressor:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    compressor.write(chunk)
        os.replace(gzip_path, destination)
        destination.chmod(0o644)
    finally:
        raw_path.unlink(missing_ok=True)
        gzip_path.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(encoded)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
    path.chmod(0o644)


def build() -> dict[str, Any]:
    paths, counts = selected_files()
    entries = [entry(path) for path in paths]
    write_bundle(paths, BUNDLE)
    payload = {
        "schema_version": 1,
        "evidence_id": "controlled-followup-fused-posthoc-evidence-v1",
        "attestation": {
            "kind": "post_hoc_evidence_preservation",
            "source_files_modified": False,
            "statement": (
                "This index and bundle were created after campaign completion. "
                "They preserve the observed result bytes but do not establish "
                "preregistration, a launch-time source freeze, or an external timestamp."
            ),
        },
        "bundle": {
            "bytes": BUNDLE.stat().st_size,
            "format": "deterministic GNU tar + gzip(level=9, mtime=0, empty filename)",
            "path": BUNDLE.relative_to(REPO).as_posix(),
            "sha256": sha256_file(BUNDLE),
        },
        "generator": {
            "path": GENERATOR.relative_to(REPO).as_posix(),
            "sha256": sha256_file(GENERATOR),
        },
        "counts": counts,
        "entries": entries,
        "selection_policy": {
            "included": [
                "all 152 screen process records",
                "screen launch/status, both screen summaries, and frozen confirmation selection",
                "all 80 confirmation process records",
                "confirmation launch/status and analysis summary",
                "aggregate records plus run/summary/normalization controls for all three robust runs",
            ],
            "excluded": [
                "build caches and compiled binaries",
                "scratch runner copies",
                "per-record robust duplicates already represented by aggregate records.jsonl streams",
            ],
        },
    }
    atomic_json(INDEX, payload)
    return payload


def verify() -> dict[str, Any]:
    if not INDEX.is_file() or not BUNDLE.is_file():
        raise EvidenceError("run the build command before verification")
    payload = json.loads(INDEX.read_text(encoding="utf-8"))
    expected_paths, observed_counts = selected_files()
    expected_names = [path.relative_to(REPO).as_posix() for path in expected_paths]
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise EvidenceError("index entries must be a list")
    indexed_names = [value.get("path") for value in entries]
    if indexed_names != expected_names:
        raise EvidenceError("indexed evidence selection differs from the current contract")
    if payload.get("counts") != observed_counts:
        raise EvidenceError("indexed record/file counts do not reproduce")

    for value, path in zip(entries, expected_paths, strict=True):
        if value.get("bytes") != path.stat().st_size:
            raise EvidenceError(f"size mismatch: {path}")
        if value.get("sha256") != sha256_file(path):
            raise EvidenceError(f"source hash mismatch: {path}")
        if path.name == "records.jsonl" and value.get("jsonl_records") != count_lines(path):
            raise EvidenceError(f"record-count mismatch: {path}")

    bundle = payload.get("bundle", {})
    if bundle.get("bytes") != BUNDLE.stat().st_size:
        raise EvidenceError("bundle size mismatch")
    if bundle.get("sha256") != sha256_file(BUNDLE):
        raise EvidenceError("bundle hash mismatch")
    generator = payload.get("generator", {})
    if generator.get("path") != GENERATOR.relative_to(REPO).as_posix():
        raise EvidenceError("generator path mismatch")
    if generator.get("sha256") != sha256_file(GENERATOR):
        raise EvidenceError("generator hash mismatch")

    with tarfile.open(BUNDLE, mode="r:gz") as archive:
        members = archive.getmembers()
        member_names = [member.name for member in members]
        if member_names != expected_names:
            raise EvidenceError("bundle member selection/order differs from the index")
        for member, indexed in zip(members, entries, strict=True):
            if not member.isfile() or member.size != indexed["bytes"]:
                raise EvidenceError(f"invalid bundle member metadata: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise EvidenceError(f"cannot read bundle member: {member.name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != indexed["sha256"]:
                raise EvidenceError(f"bundle member hash mismatch: {member.name}")

    return {
        "bundle_sha256": bundle["sha256"],
        "checked_files": len(entries),
        "counts": observed_counts,
        "index": INDEX.relative_to(REPO).as_posix(),
        "ok": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    try:
        result = build() if args.command == "build" else verify()
    except (EvidenceError, OSError, tarfile.TarError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error), "ok": False}, indent=2, sort_keys=True))
        return 1
    if args.command == "build":
        result = {
            "bundle": result["bundle"],
            "counts": result["counts"],
            "index": INDEX.relative_to(REPO).as_posix(),
            "ok": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
