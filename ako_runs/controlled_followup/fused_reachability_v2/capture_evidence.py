#!/usr/bin/env python3
"""Build or verify a deterministic evidence bundle for reachability v2."""
from __future__ import annotations

import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

from protocol import (
    ADAPTER,
    CAMPAIGN,
    HERE,
    JOBS,
    LOCK,
    REPO_ROOT,
    SOURCE_PATHS,
    TAG_RE,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    read_json,
    record_filename,
    stable_write,
    validate_launch_receipt,
    validate_process_record,
    verify_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--name", required=True)
    build.add_argument("--include", action="append", default=[])
    verify = sub.add_parser("verify")
    verify.add_argument("--index", required=True)
    return parser.parse_args()


def resolve_include(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    path.relative_to(REPO_ROOT.resolve())
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def validate_result_root(root: Path, lock: dict) -> None:
    receipt_path = root / "launch_receipt.json"
    if not receipt_path.is_file():
        return
    receipt = read_json(receipt_path)
    if receipt.get("record_type") == "fused_reachability_v2_launch_receipt":
        contract = receipt["contract"]
        receipt, jobs = validate_launch_receipt(
            root, phase=contract["phase"], lane=contract["lane"], lock=lock
        )
        for job in jobs:
            for rep in range(contract["reps"]):
                validate_process_record(
                    root / "raw" / record_filename(job["job_id"], rep),
                    job=job,
                    phase=contract["phase"],
                    rep=rep,
                    physical_gpu=contract["physical_gpu"],
                    lock=lock,
                )
        return
    if receipt.get("record_type") == "fused_reachability_v2_robust_receipt":
        if (
            receipt.get("campaign_id") != lock["campaign_id"]
            or receipt.get("source_bundle_sha256") != lock["source_bundle_sha256"]
            or receipt.get("gate_spec_sha256") != lock["frozen_gate"]["gate_spec_sha256"]
            or receipt.get("seed_indices") != list(range(64))
        ):
            raise RuntimeError(f"invalid robust receipt: {receipt_path}")
        summary = read_json(root / "summary.json")
        if (
            summary.get("complete_frozen_validation_split") is not True
            or summary.get("launch_coverage", {}).get("complete") is not True
            or summary.get("source_bundle_sha256") != lock["source_bundle_sha256"]
        ):
            raise RuntimeError(f"invalid robust summary: {root / 'summary.json'}")
        expected = len(receipt["selected_jobs"]) * len(receipt["case_ids"]) * 64
        observed = len(list((root / "raw").glob("*/*/seed*.json")))
        if observed != expected:
            raise RuntimeError(f"robust raw coverage mismatch at {root}: {observed}/{expected}")
        return
    raise RuntimeError(f"unknown launch receipt type: {receipt_path}")


def selected_files(includes: list[Path]) -> list[Path]:
    base = [REPO_ROOT / relative for relative in SOURCE_PATHS]
    base.extend((CAMPAIGN, JOBS, ADAPTER, LOCK))
    result = set(path.resolve() for path in base)
    for include in includes:
        if include.is_file():
            result.add(include)
            continue
        for path in include.rglob("*"):
            if not path.is_file():
                continue
            if any(part in (".torch_ext", "__pycache__", ".pytest_cache", "evidence") for part in path.parts):
                continue
            if path.name.endswith(".tmp") or path.name == "active.lock":
                continue
            result.add(path.resolve())
    return sorted(result, key=lambda path: str(path.relative_to(REPO_ROOT)))


def tar_add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def build(name: str, include_values: list[str]) -> Path:
    if name in (".", "..") or not TAG_RE.fullmatch(name):
        raise ValueError(f"unsafe evidence name: {name!r}")
    lock = verify_lock()
    includes = [resolve_include(value) for value in include_values]
    for include in includes:
        validate_result_root(include, lock)
    files = selected_files(includes)
    entries = []
    for path in files:
        relative = str(path.relative_to(REPO_ROOT))
        entries.append(
            {"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)}
        )
    manifest = {
        "schema_version": 1,
        "record_type": "fused_reachability_v2_evidence_manifest",
        "campaign_id": lock["campaign_id"],
        "frozen_utc": lock["frozen_utc"],
        "launch_lock_sha256": file_sha256(LOCK),
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "includes": [str(path.relative_to(REPO_ROOT)) for path in includes],
        "entries": entries,
    }
    manifest_bytes = canonical_bytes(manifest) + b"\n"
    output = HERE / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / f"{name}.tar.gz"
    index_path = output / f"{name}.index.json"
    if bundle.exists() or index_path.exists():
        raise FileExistsError("evidence output is immutable; choose a new name")
    temporary = bundle.with_suffix(bundle.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                tar_add_bytes(archive, "EVIDENCE_MANIFEST.json", manifest_bytes)
                for path, entry in zip(files, entries):
                    tar_add_bytes(archive, entry["path"], path.read_bytes())
    temporary.replace(bundle)
    index = {
        "schema_version": 1,
        "record_type": "fused_reachability_v2_evidence_index",
        "bundle_path": str(bundle.relative_to(REPO_ROOT)),
        "bundle_sha256": file_sha256(bundle),
        "bundle_size": bundle.stat().st_size,
        "manifest_sha256": canonical_sha256(manifest),
        "entry_count": len(entries),
        "manifest": manifest,
    }
    stable_write(index_path, index)
    print(json.dumps({"index": str(index_path), "bundle_sha256": index["bundle_sha256"], "entries": len(entries)}, sort_keys=True))
    return index_path


def verify(index_path: Path) -> None:
    index = read_json(index_path)
    bundle = REPO_ROOT / index["bundle_path"]
    if file_sha256(bundle) != index["bundle_sha256"]:
        raise RuntimeError("evidence bundle SHA mismatch")
    manifest = index["manifest"]
    if canonical_sha256(manifest) != index["manifest_sha256"]:
        raise RuntimeError("evidence manifest SHA mismatch")
    expected = {entry["path"]: entry for entry in manifest["entries"]}
    with tarfile.open(bundle, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        manifest_data = archive.extractfile(members.pop("EVIDENCE_MANIFEST.json")).read()
        if manifest_data != canonical_bytes(manifest) + b"\n":
            raise RuntimeError("embedded evidence manifest mismatch")
        if set(members) != set(expected):
            raise RuntimeError("evidence archive membership mismatch")
        for name, entry in expected.items():
            data = archive.extractfile(members[name]).read()
            import hashlib
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"evidence member mismatch: {name}")
    print(json.dumps({"ok": True, "entries": len(expected), "bundle_sha256": index["bundle_sha256"]}, sort_keys=True))


def main() -> int:
    args = parse_args()
    if args.command == "build":
        build(args.name, args.include)
    else:
        verify(Path(args.index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

