#!/usr/bin/env python3
"""Select, build, and verify the deterministic post-review umbrella evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable


REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FOLLOWUP = REPO / "ako_runs/controlled_followup"
GENERATOR = Path(__file__).resolve()
README = HERE / "README.md"
INIT = HERE / "__init__.py"
TEST = HERE / "test_build_evidence.py"
AUDIT_GITIGNORE = FOLLOWUP / "robust_gate/audits/.gitignore"

CLOSURE = FOLLOWUP / "fused_closure_v2"
ROW_STRESS = FOLLOWUP / "robust_gate/audits/fused_row_sum_stress_v1"
MATMUL_V4 = FOLLOWUP / "robust_gate/audits/matmul_v4_instrument_v1"

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

FORBIDDEN_PARTS = {
    ".torch_extensions",
    ".torch_ext",
    "__pycache__",
    ".pytest_cache",
}
FORBIDDEN_NAMES = {
    "active.lock",
    "build.ninja",
    ".ninja_deps",
    ".ninja_log",
}
FORBIDDEN_SUFFIXES = {
    ".a",
    ".cubin",
    ".fatbin",
    ".o",
    ".obj",
    ".ptx",
    ".pyc",
    ".pyo",
    ".so",
}
TEMPORARY_SUFFIXES = (".partial", ".part", ".tmp", "~")

CLOSURE_EXPECTED = {
    "campaign_id": "fused-gbgs-closure-v2",
    "source_receipt_sha256": "d38ba70aa667ee873e3c4c75708881ddf87ab3f48beb6ddc10cfcf94126156c7",
    "source_counts": {"local_source_sha256": 12, "external_source_sha256": 20, "frozen_evidence_sha256": 8},
    "candidate_count": 9,
    "gate_launch_sha256": "4cf8c97344842f724af574f6f9fd29cd93ea1c7c573ca266fe86f13e110f1be4",
    "gate_summary_sha256": "0d7955543c262a5dff7fcfd8a3466678f8c55b13b5290c48dd40c90328a068e4",
    "gate_raw_files": 256,
    "gate_records": 4608,
    "performance_launch_sha256": "b56214e92a941c273b081f266e5430b91c2dcc6b9196993b9fa5150a432e2e92",
    "performance_status_sha256": "1e40be7e0bd6e5916499ec41106d458da4efc4523d151662abeaf8a18333545e",
    "performance_summary_sha256": "072ce7ea841dad51fe63c50d33c7f02f1bc2cf8c12cc97f3d1d75d387d9bbcec",
    "performance_records": 135,
}

ROW_EXPECTED = {
    "campaign_id": "controlled-followup-fused-row-sum-stress-v1",
    "manifest_sha256": "3b4171c07a7d7bc943bb8228f7ba45ed80dbd4e16c64b3f859b3b8026d20f330",
    "freeze_sha256": "71dc0087a5d550dd698847a70d1f89191d2cade1ff453bf290117f32230bc498",
    "launch_sha256": "91f8c5d130b9c6a9a1ec93bca7c2110dd17a4fc5c92e343ca469859826cd7642",
    "completion_sha256": "fae663bca0ee2802cd6afcb24cb200a909da59099b5ce48896d54db53b39e39e",
    "summary_sha256": "df51077cf16f5a67abc6ab821bda99ecd87ee93b77a8e8cd3246cdcf38859891",
    "source_files": 10,
    "winner_records": 2048,
    "boundary_records": 6,
}

MATMUL_EXPECTED = {
    "campaign_id": "controlled-followup-matmul-v4-instrument-audit-v1",
    "freeze_sha256": "56c297207e6b3a6104967d36d059d1a23fdab83a4da6e937c59d6ec94e495c72",
    "launch_sha256": "8568636ca4d7d77578051d3d76aa837b81635d60b88e0713da1c6d9e50edc3f0",
    "source_files": 19,
    "workloads": 6,
}


class EvidenceError(RuntimeError):
    """A source tree or archive violates the umbrella evidence contract."""


@dataclass(frozen=True)
class NestedEvidence:
    label: str
    index: str
    bundle: str
    index_sha256: str
    bundle_sha256: str
    entries: int
    kind: str


NESTED = (
    NestedEvidence(
        "original_evidence_v1",
        "ako_runs/controlled_followup/provenance/evidence_v1/evidence_index.json",
        "ako_runs/controlled_followup/provenance/evidence_v1/evidence_bundle.tar.gz",
        "70e939efdad1b41aafc99f94588ee5795311482c7b455fb50261cf5a174aef96",
        "6305573cbb13e47902d0b0cd58d629feea812678ca6e4b1e13e4884f96432ea6",
        251,
        "legacy_v1",
    ),
    NestedEvidence(
        "fused_reachability_v2",
        "ako_runs/controlled_followup/fused_reachability_v2/evidence/complete_v1.index.json",
        "ako_runs/controlled_followup/fused_reachability_v2/evidence/complete_v1.tar.gz",
        "d228ba509e02f2ed7e9a75346e2e960e3ec9fa6f8cfea943193da6a91b7f06cb",
        "efcd54b68885b2bdd2c60202d68b858fbe3f40a8288d5815ef5fa7eeb6536572",
        1712,
        "manifest_index",
    ),
    NestedEvidence(
        "fused_reachability_row_sum_stress_v1",
        "ako_runs/controlled_followup/robust_gate/audits/fused_reachability_row_sum_stress_v1/evidence/complete_v1.index.json",
        "ako_runs/controlled_followup/robust_gate/audits/fused_reachability_row_sum_stress_v1/evidence/complete_v1.tar.gz",
        "00f985e92bbfdc5697d88b5e5334093d1e5c0124a82715c070c077bc6cb87cbe",
        "8acc2e8927f95761781688647cddcb5ef3e6dba9df2e9a8d650aced96aa41028",
        25,
        "manifest_index",
    ),
    NestedEvidence(
        "archived_current_fused_v1",
        "ako_runs/controlled_followup/archived_current_fused_v1/evidence/main_v1.index.json",
        "ako_runs/controlled_followup/archived_current_fused_v1/evidence/main_v1.tar.gz",
        "c4b84f8feecfadf452620e6494e2b13445d9c06ff173c812407735d133362b95",
        "4b4557b468492c201f796ff34b8375f066c0234fb22d7e5f7aebbf181a75f617",
        143,
        "archived_index",
    ),
    NestedEvidence(
        "fused_frontier_closure_v3",
        "ako_runs/controlled_followup/fused_frontier_closure_v3/evidence/complete_v1.index.json",
        "ako_runs/controlled_followup/fused_frontier_closure_v3/evidence/complete_v1.tar.gz",
        "5e12dc8676d4694a05e6970bbd3b54ab9cace3b09976e3e558563ed1b5cf0771",
        "8b0a05a24013e4944be27accdbcbb7b9fe24bd027bade08fddfbe5c63dc44760",
        319,
        "manifest_index",
    ),
)
ALLOWED_BINARY_EVIDENCE = {str((REPO / item.bundle).resolve()) for item in NESTED}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected a JSON object: {path}")
    return value


def line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"path escapes repository: {path}") from exc


def _is_forbidden(path: Path) -> str | None:
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        return "cache/build directory"
    if path.name in FORBIDDEN_NAMES:
        return "active lock or build-control file"
    if path.name.endswith(TEMPORARY_SUFFIXES) or ".tmp." in path.name:
        return "temporary or partial file"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return "compiled/build product"
    return None


def require_file(path: Path, expected_sha256: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"required regular file is missing or a symlink: {path}")
    resolved = path.resolve()
    relative(resolved)
    reason = _is_forbidden(resolved)
    if reason and str(resolved) not in ALLOWED_BINARY_EVIDENCE:
        raise EvidenceError(f"refusing {reason}: {relative(resolved)}")
    if expected_sha256 is not None:
        if not HASH_RE.fullmatch(expected_sha256):
            raise EvidenceError(f"invalid expected SHA-256 for {path}")
        observed = sha256_file(resolved)
        if observed != expected_sha256:
            raise EvidenceError(
                f"SHA-256 mismatch for {relative(resolved)}: {observed} != {expected_sha256}"
            )
    return resolved


class Selection:
    def __init__(self) -> None:
        self.categories: dict[Path, set[str]] = defaultdict(set)

    def add(
        self, path: Path, category: str, expected_sha256: str | None = None
    ) -> None:
        resolved = require_file(path, expected_sha256)
        self.categories[resolved].add(category)

    def add_hash_map(
        self,
        values: Any,
        category: str,
        *,
        expected_count: int,
        base: Path = REPO,
    ) -> None:
        if not isinstance(values, dict) or len(values) != expected_count:
            observed = len(values) if isinstance(values, dict) else "not-an-object"
            raise EvidenceError(
                f"{category}: expected {expected_count} hashed paths, found {observed}"
            )
        for name, digest in values.items():
            if not isinstance(name, str) or not isinstance(digest, str):
                raise EvidenceError(f"{category}: malformed hashed-path entry")
            self.add(base / name, category, digest)

    def paths(self) -> list[Path]:
        return sorted(self.categories, key=relative)


def _tar_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members = archive.getmembers()
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise EvidenceError("archive contains duplicate member names")
    if any(not member.isfile() for member in members):
        raise EvidenceError("archive contains a non-regular member")
    return {member.name: member for member in members}


def _verify_payload_members(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    entries: list[dict[str, Any]],
) -> None:
    expected: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise EvidenceError("nested evidence entry is not an object")
        name = entry.get("path")
        size = entry.get("size", entry.get("bytes"))
        digest = entry.get("sha256")
        if (
            not isinstance(name, str)
            or name in expected
            or not isinstance(size, int)
            or not isinstance(digest, str)
            or not HASH_RE.fullmatch(digest)
        ):
            raise EvidenceError(f"malformed nested evidence entry: {entry!r}")
        expected[name] = {"size": size, "sha256": digest}
    if set(members) != set(expected):
        raise EvidenceError("nested evidence archive membership differs from index")
    for name, entry in expected.items():
        member = members[name]
        stream = archive.extractfile(member)
        if stream is None:
            raise EvidenceError(f"cannot extract nested member {name}")
        observed = sha256_stream(stream)
        if member.size != entry["size"] or observed != entry["sha256"]:
            raise EvidenceError(f"nested evidence payload mismatch: {name}")


def verify_nested(spec: NestedEvidence) -> dict[str, Any]:
    index_path = require_file(REPO / spec.index, spec.index_sha256)
    bundle_path = require_file(REPO / spec.bundle, spec.bundle_sha256)
    index = load_json(index_path)
    if spec.kind == "legacy_v1":
        bundle = index.get("bundle", {})
        entries = index.get("entries")
        declared_path = bundle.get("path") if isinstance(bundle, dict) else None
        declared_sha = bundle.get("sha256") if isinstance(bundle, dict) else None
        declared_count = index.get("counts", {}).get("selected_files")
        has_manifest = False
    elif spec.kind == "archived_index":
        entries = index.get("entries")
        declared_path = index.get("bundle")
        declared_sha = index.get("bundle_sha256")
        declared_count = index.get("entry_count")
        has_manifest = True
    else:
        manifest = index.get("manifest")
        entries = manifest.get("entries") if isinstance(manifest, dict) else None
        declared_path = index.get("bundle_path")
        declared_sha = index.get("bundle_sha256")
        declared_count = index.get("entry_count")
        if not isinstance(manifest, dict) or canonical_sha256(manifest) != index.get(
            "manifest_sha256"
        ):
            raise EvidenceError(f"nested manifest binding mismatch: {spec.label}")
        has_manifest = True
    if (
        declared_path != spec.bundle
        or declared_sha != spec.bundle_sha256
        or declared_count != spec.entries
        or not isinstance(entries, list)
        or len(entries) != spec.entries
    ):
        raise EvidenceError(f"nested index contract mismatch: {spec.label}")
    with tarfile.open(bundle_path, mode="r:gz") as archive:
        members = _tar_members(archive)
        if has_manifest:
            manifest_member = members.pop("EVIDENCE_MANIFEST.json", None)
            if manifest_member is None:
                raise EvidenceError(f"nested manifest is absent: {spec.label}")
            stream = archive.extractfile(manifest_member)
            if stream is None:
                raise EvidenceError(f"cannot extract nested manifest: {spec.label}")
            embedded = stream.read()
            embedded_value = json.loads(embedded)
            if spec.kind == "archived_index":
                if (
                    embedded_value.get("entries") != entries
                    or hashlib.sha256(
                        canonical_bytes(embedded_value) + b"\n"
                    ).hexdigest()
                    != index.get("manifest_canonical_sha256")
                ):
                    raise EvidenceError(
                        f"archived nested manifest binding mismatch: {spec.label}"
                    )
            elif embedded != canonical_bytes(index["manifest"]) + b"\n":
                raise EvidenceError(f"nested embedded manifest mismatch: {spec.label}")
        _verify_payload_members(archive, members, entries)
    return {
        "label": spec.label,
        "index_path": spec.index,
        "index_sha256": spec.index_sha256,
        "bundle_path": spec.bundle,
        "bundle_sha256": spec.bundle_sha256,
        "nested_entry_count": spec.entries,
        "verified": True,
    }


def select_nested(selection: Selection) -> list[dict[str, Any]]:
    validations = []
    for spec in NESTED:
        validations.append(verify_nested(spec))
        selection.add(REPO / spec.index, f"nested:{spec.label}", spec.index_sha256)
        selection.add(REPO / spec.bundle, f"nested:{spec.label}", spec.bundle_sha256)
    return validations


def select_closure(selection: Selection) -> dict[str, Any]:
    expected = CLOSURE_EXPECTED
    receipt_path = CLOSURE / "source_receipt.json"
    receipt = load_json(require_file(receipt_path, expected["source_receipt_sha256"]))
    if (
        receipt.get("campaign_id") != expected["campaign_id"]
        or not isinstance(receipt.get("candidate_sha256"), dict)
        or len(receipt["candidate_sha256"]) != expected["candidate_count"]
    ):
        raise EvidenceError("fused closure source receipt contract mismatch")
    selection.add(receipt_path, "fused_closure_v2:source_receipt", expected["source_receipt_sha256"])
    for field, count in expected["source_counts"].items():
        selection.add_hash_map(
            receipt.get(field), f"fused_closure_v2:{field}", expected_count=count
        )

    gate_root = CLOSURE / "results/gate_v1"
    gate_launch = gate_root / "gate_launch_receipt.json"
    gate_summary_path = gate_root / "gate_summary.json"
    gate_summary = load_json(
        require_file(gate_summary_path, expected["gate_summary_sha256"])
    )
    if (
        gate_summary.get("record_type") != "fused_closure_v2_gate_summary"
        or gate_summary.get("coverage_complete") is not True
        or gate_summary.get("performance_launch_allowed") is not True
        or gate_summary.get("expected_records") != expected["gate_records"]
        or gate_summary.get("observed_records") != expected["gate_records"]
        or len(gate_summary.get("adjudications", [])) != expected["candidate_count"]
        or gate_summary.get("gate_launch_receipt_sha256")
        != expected["gate_launch_sha256"]
    ):
        raise EvidenceError("fused closure gate summary is incomplete or unbound")
    selection.add(gate_launch, "fused_closure_v2:gate_receipt", expected["gate_launch_sha256"])
    selection.add(gate_summary_path, "fused_closure_v2:gate_summary", expected["gate_summary_sha256"])
    selection.add_hash_map(
        gate_summary.get("raw_bundle_sha256"),
        "fused_closure_v2:gate_raw",
        expected_count=expected["gate_raw_files"],
        base=gate_root,
    )

    performance_root = CLOSURE / "results/performance_v1"
    performance_summary_path = performance_root / "analysis_summary.json"
    performance_summary = load_json(
        require_file(performance_summary_path, expected["performance_summary_sha256"])
    )
    if (
        performance_summary.get("record_type")
        != "fused_closure_v2_performance_analysis"
        or performance_summary.get("campaign_id") != expected["campaign_id"]
        or performance_summary.get("status") != "COMPLETE"
        or performance_summary.get("expected_records")
        != expected["performance_records"]
        or performance_summary.get("observed_records")
        != expected["performance_records"]
        or performance_summary.get("source_receipt_sha256")
        != expected["source_receipt_sha256"]
        or performance_summary.get("gate_summary_sha256")
        != expected["gate_summary_sha256"]
        or performance_summary.get("launch_receipt_sha256")
        != expected["performance_launch_sha256"]
    ):
        raise EvidenceError("fused closure performance analysis is incomplete or unbound")
    performance_controls = (
        ("launch_receipt.json", "performance_receipt", expected["performance_launch_sha256"]),
        ("launch_status.json", "performance_status", expected["performance_status_sha256"]),
        ("analysis_summary.json", "performance_summary", expected["performance_summary_sha256"]),
        ("ANALYSIS.md", "performance_report", None),
    )
    for name, category, digest in performance_controls:
        selection.add(performance_root / name, f"fused_closure_v2:{category}", digest)
    selection.add_hash_map(
        performance_summary.get("raw_record_sha256"),
        "fused_closure_v2:performance_raw",
        expected_count=expected["performance_records"],
    )
    return {
        "campaign_id": expected["campaign_id"],
        "source_receipt_sha256": expected["source_receipt_sha256"],
        "receipt_bound_source_files": sum(expected["source_counts"].values()),
        "generated_candidates": expected["candidate_count"],
        "gate_raw_files": expected["gate_raw_files"],
        "gate_logical_records": expected["gate_records"],
        "gate_summary_sha256": expected["gate_summary_sha256"],
        "performance_raw_files": expected["performance_records"],
        "performance_summary_sha256": expected["performance_summary_sha256"],
    }


def select_row_stress(selection: Selection) -> dict[str, Any]:
    expected = ROW_EXPECTED
    manifest_path = ROW_STRESS / "manifest.json"
    freeze_path = ROW_STRESS / "receipts/freeze_receipt.json"
    launch_path = ROW_STRESS / "receipts/launch_receipt.json"
    completion_path = ROW_STRESS / "receipts/completion_receipt.json"
    summary_path = ROW_STRESS / "results/summary.json"
    manifest = load_json(require_file(manifest_path, expected["manifest_sha256"]))
    freeze = load_json(require_file(freeze_path, expected["freeze_sha256"]))
    completion = load_json(require_file(completion_path, expected["completion_sha256"]))
    summary = load_json(require_file(summary_path, expected["summary_sha256"]))
    if (
        manifest.get("campaign_id") != expected["campaign_id"]
        or freeze.get("campaign_id") != expected["campaign_id"]
        or completion.get("campaign_id") != expected["campaign_id"]
        or summary.get("campaign_id") != expected["campaign_id"]
        or freeze.get("manifest_sha256") != expected["manifest_sha256"]
        or completion.get("freeze_receipt_sha256") != expected["freeze_sha256"]
        or completion.get("summary_sha256") != expected["summary_sha256"]
        or completion.get("evidence_complete") is not True
        or summary.get("evidence_complete") is not True
        or summary.get("coverage", {}).get("winner_expected")
        != expected["winner_records"]
        or summary.get("coverage", {}).get("winner_observed_unique")
        != expected["winner_records"]
    ):
        raise EvidenceError("prior-winner row-sum stress evidence is incomplete or unbound")
    selection.add_hash_map(
        freeze.get("source_sha256"),
        "fused_row_sum_stress_v1:source",
        expected_count=expected["source_files"],
    )
    for path, category, digest in (
        (freeze_path, "freeze_receipt", expected["freeze_sha256"]),
        (launch_path, "launch_receipt", expected["launch_sha256"]),
        (completion_path, "completion_receipt", expected["completion_sha256"]),
        (summary_path, "summary", expected["summary_sha256"]),
    ):
        selection.add(path, f"fused_row_sum_stress_v1:{category}", digest)
    raw_by_name = {
        row.get("path"): row.get("sha256")
        for row in summary.get("raw_files", [])
        if isinstance(row, dict)
    }
    expected_raw = {
        "results/raw/winners_gain16.jsonl": expected["winner_records"],
        "results/raw/boundary.jsonl": expected["boundary_records"],
    }
    if set(raw_by_name) != set(expected_raw):
        raise EvidenceError("prior-winner row-sum raw-file set differs")
    for name, records in expected_raw.items():
        path = ROW_STRESS / name
        selection.add(path, "fused_row_sum_stress_v1:raw", raw_by_name[name])
        observed = line_count(path)
        if observed != records:
            raise EvidenceError(f"{relative(path)}: expected {records} rows, found {observed}")
    return {
        "campaign_id": expected["campaign_id"],
        "freeze_receipt_sha256": expected["freeze_sha256"],
        "receipt_bound_source_files": expected["source_files"],
        "winner_records": expected["winner_records"],
        "boundary_records": expected["boundary_records"],
        "summary_sha256": expected["summary_sha256"],
        "evidence_complete": True,
        "all_winner_groups_success": summary.get("all_winner_groups_success"),
    }


def matmul_status() -> dict[str, Any]:
    partials = sorted(relative(path) for path in MATMUL_V4.rglob("*.partial") if path.is_file())
    completion = MATMUL_V4 / "receipts/completion_receipt.json"
    summary = MATMUL_V4 / "results/summary.json"
    launch_path = MATMUL_V4 / "receipts/launch_receipt.json"
    finals: list[str] = []
    missing: list[str] = []
    if launch_path.is_file():
        launch = load_json(launch_path)
        for workload in launch.get("workloads", []):
            path = REPO / workload.get("output", "")
            (finals if path.is_file() else missing).append(relative(path))
    state = "complete_candidate" if not partials and not missing and completion.is_file() and summary.is_file() else "active_or_incomplete"
    return {
        "state": state,
        "partial_files": partials,
        "present_final_raw_files": sorted(finals),
        "missing_final_raw_files": sorted(missing),
        "completion_receipt_present": completion.is_file(),
        "summary_present": summary.is_file(),
    }


def select_matmul(selection: Selection) -> dict[str, Any]:
    status = matmul_status()
    if status["state"] != "complete_candidate":
        raise EvidenceError(
            "matmul-v4 deferred include is active or incomplete: "
            + json.dumps(status, sort_keys=True)
        )
    expected = MATMUL_EXPECTED
    freeze_path = MATMUL_V4 / "receipts/freeze_receipt.json"
    launch_path = MATMUL_V4 / "receipts/launch_receipt.json"
    completion_path = MATMUL_V4 / "receipts/completion_receipt.json"
    summary_path = MATMUL_V4 / "results/summary.json"
    freeze = load_json(require_file(freeze_path, expected["freeze_sha256"]))
    launch = load_json(require_file(launch_path, expected["launch_sha256"]))
    completion = load_json(require_file(completion_path))
    summary = load_json(require_file(summary_path))
    if (
        freeze.get("campaign_id") != expected["campaign_id"]
        or launch.get("campaign_id") != expected["campaign_id"]
        or completion.get("campaign_id") != expected["campaign_id"]
        or summary.get("campaign_id") != expected["campaign_id"]
        or len(launch.get("workloads", [])) != expected["workloads"]
        or completion.get("summary_sha256") != sha256_file(summary_path)
        or completion.get("freeze_receipt_sha256") != expected["freeze_sha256"]
        or completion.get("evidence_complete") is not True
        or summary.get("evidence_complete") is not True
        or summary.get("expected_records") != summary.get("observed_unique_records")
    ):
        raise EvidenceError("matmul-v4 completion/summary binding mismatch")
    selection.add_hash_map(
        freeze.get("source_files"),
        "matmul_v4_instrument_v1:source",
        expected_count=expected["source_files"],
    )
    for path, category, digest in (
        (freeze_path, "freeze_receipt", expected["freeze_sha256"]),
        (launch_path, "launch_receipt", expected["launch_sha256"]),
        (completion_path, "completion_receipt", None),
        (summary_path, "summary", None),
    ):
        selection.add(path, f"matmul_v4_instrument_v1:{category}", digest)
    summary_raw = {
        row.get("path"): row.get("sha256")
        for row in summary.get("raw_files", [])
        if isinstance(row, dict)
    }
    expected_outputs = {item["output"]: item["expected_records"] for item in launch["workloads"]}
    if set(summary_raw) != set(expected_outputs):
        raise EvidenceError("matmul-v4 raw file set differs from frozen launch")
    for name, records in expected_outputs.items():
        path = REPO / name
        selection.add(path, "matmul_v4_instrument_v1:raw", summary_raw[name])
        if line_count(path) != records:
            raise EvidenceError(f"matmul-v4 row count differs: {name}")
    if any(path.is_file() for path in MATMUL_V4.rglob("*.partial")):
        raise EvidenceError("matmul-v4 partial stream appeared during selection")
    return {
        "campaign_id": expected["campaign_id"],
        "freeze_receipt_sha256": expected["freeze_sha256"],
        "receipt_bound_source_files": expected["source_files"],
        "raw_files": expected["workloads"],
        "expected_records": summary["expected_records"],
        "summary_sha256": sha256_file(summary_path),
        "evidence_complete": True,
    }


def select(include_matmul_v4: bool) -> tuple[Selection, dict[str, Any]]:
    selection = Selection()
    selection.add(GENERATOR, "umbrella:generator")
    selection.add(README, "umbrella:documentation")
    selection.add(INIT, "umbrella:source")
    selection.add(TEST, "umbrella:test")
    selection.add(AUDIT_GITIGNORE, "umbrella:evidence_hygiene")
    selection.add(FOLLOWUP / "REVIEW_RESPONSE_20260730.md", "review_response")
    selection.add(FOLLOWUP / "ERRATA_20260730.md", "review_response")
    validations: dict[str, Any] = {
        "nested_evidence": select_nested(selection),
        "fused_closure_v2": select_closure(selection),
        "fused_row_sum_stress_v1": select_row_stress(selection),
        "matmul_v4_status": matmul_status(),
    }
    if include_matmul_v4:
        validations["matmul_v4_instrument_v1"] = select_matmul(selection)
    return selection, validations


def selection_report(include_matmul_v4: bool) -> dict[str, Any]:
    selection, validations = select(include_matmul_v4)
    paths = selection.paths()
    counts: Counter[str] = Counter()
    for path in paths:
        counts.update(selection.categories[path])
    return {
        "ok": True,
        "record_type": "controlled_followup_umbrella_evidence_v2_dry_run",
        "matmul_v4_included": include_matmul_v4,
        "selected_entries": len(paths),
        "selected_uncompressed_bytes": sum(path.stat().st_size for path in paths),
        "category_memberships": dict(sorted(counts.items())),
        "validations": validations,
        "excluded_classes": sorted(
            [
                "active locks",
                "build caches",
                "compiled objects and CUDA/Triton binaries",
                "partial and temporary files",
                "Python and pytest caches",
            ]
        ),
    }


def entries_for(selection: Selection) -> list[dict[str, Any]]:
    return [
        {
            "categories": sorted(selection.categories[path]),
            "path": relative(path),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in selection.paths()
    ]


def normalized_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    archive.addfile(normalized_info(name, len(data)), io.BytesIO(data))


def write_bundle(path: Path, manifest: dict[str, Any], files: Iterable[Path]) -> None:
    manifest_bytes = canonical_bytes(manifest) + b"\n"
    with path.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as zipped:
            with tarfile.open(
                fileobj=zipped, mode="w", format=tarfile.GNU_FORMAT
            ) as archive:
                add_bytes(archive, "UMBRELLA_MANIFEST.json", manifest_bytes)
                for source in files:
                    with source.open("rb") as stream:
                        archive.addfile(
                            normalized_info(relative(source), source.stat().st_size),
                            stream,
                        )


def _verify_umbrella_bundle(path: Path, manifest: dict[str, Any]) -> None:
    expected = {entry["path"]: entry for entry in manifest["entries"]}
    if len(expected) != len(manifest["entries"]):
        raise EvidenceError("umbrella manifest contains duplicate paths")
    with tarfile.open(path, mode="r:gz") as archive:
        members = _tar_members(archive)
        embedded = members.pop("UMBRELLA_MANIFEST.json", None)
        if embedded is None:
            raise EvidenceError("umbrella manifest member is absent")
        embedded_stream = archive.extractfile(embedded)
        if embedded_stream is None or embedded_stream.read() != canonical_bytes(manifest) + b"\n":
            raise EvidenceError("embedded umbrella manifest differs")
        if set(members) != set(expected):
            raise EvidenceError("umbrella archive membership differs")
        for name, entry in expected.items():
            member = members[name]
            if (
                member.mode != 0o644
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
            ):
                raise EvidenceError(f"non-normalized tar metadata: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise EvidenceError(f"cannot extract umbrella member: {name}")
            if member.size != entry["size"] or sha256_stream(stream) != entry["sha256"]:
                raise EvidenceError(f"umbrella payload differs: {name}")


def _exclusive_publish(temporary: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"immutable evidence output already exists: {destination}")
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"immutable evidence output appeared concurrently: {destination}"
        ) from exc
    destination.chmod(0o644)


def build(
    name: str,
    include_matmul_v4: bool,
    defer_active_matmul_v4: bool = False,
) -> Path:
    if not NAME_RE.fullmatch(name) or name in {".", ".."}:
        raise EvidenceError(f"unsafe evidence name: {name!r}")
    if include_matmul_v4 == defer_active_matmul_v4:
        raise EvidenceError(
            "choose exactly one of --include-matmul-v4 or "
            "--defer-active-matmul-v4"
        )
    bundle = HERE / f"{name}.tar.gz"
    index_path = HERE / f"{name}.index.json"
    if bundle.exists() or index_path.exists():
        raise FileExistsError("evidence outputs are immutable; choose a new name")
    selection, validations = select(include_matmul_v4=include_matmul_v4)
    if defer_active_matmul_v4 and validations["matmul_v4_status"]["state"] != "active_or_incomplete":
        raise EvidenceError(
            "--defer-active-matmul-v4 is only valid while the audit is active "
            "or incomplete"
        )
    entries = entries_for(selection)
    manifest = {
        "schema_version": 2,
        "record_type": "controlled_followup_umbrella_evidence_v2_manifest",
        "evidence_id": name,
        "attestation": {
            "kind": "post_hoc_evidence_preservation",
            "external_timestamp_claimed": False,
            "preregistration_claimed": False,
            "source_files_modified": False,
        },
        "generator": {
            "path": relative(GENERATOR),
            "sha256": sha256_file(GENERATOR),
        },
        "matmul_v4_included": include_matmul_v4,
        "matmul_v4_disposition": (
            "included_complete"
            if include_matmul_v4
            else "explicitly_deferred_active_or_incomplete"
        ),
        "validations": validations,
        "selection_policy": {
            "exact_completed_artifacts_only": True,
            "compiled_binaries_excluded": True,
            "caches_excluded": True,
            "active_locks_excluded": True,
            "temporary_and_partial_files_excluded": True,
            "nested_evidence_pairs": [item.label for item in NESTED],
        },
        "entries": entries,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    bundle_tmp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{name}.", suffix=".tar.gz.tmp", dir=HERE, delete=False
    )
    bundle_tmp = Path(bundle_tmp_handle.name)
    bundle_tmp_handle.close()
    bundle_tmp.unlink()
    index_tmp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{name}.", suffix=".index.json.tmp", dir=HERE, delete=False
    )
    index_tmp = Path(index_tmp_handle.name)
    index_tmp_handle.close()
    try:
        write_bundle(bundle_tmp, manifest, selection.paths())
        _verify_umbrella_bundle(bundle_tmp, manifest)
        index = {
            "schema_version": 2,
            "record_type": "controlled_followup_umbrella_evidence_v2_index",
            "evidence_id": name,
            "bundle_path": relative(bundle),
            "bundle_sha256": sha256_file(bundle_tmp),
            "bundle_size": bundle_tmp.stat().st_size,
            "manifest_sha256": canonical_sha256(manifest),
            "entry_count": len(entries),
            "manifest": manifest,
        }
        index_tmp.write_bytes(canonical_bytes(index) + b"\n")
        _exclusive_publish(bundle_tmp, bundle)
        try:
            _exclusive_publish(index_tmp, index_path)
        except Exception:
            bundle.unlink(missing_ok=True)
            raise
    finally:
        bundle_tmp.unlink(missing_ok=True)
        index_tmp.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "bundle_sha256": index["bundle_sha256"],
                "entries": index["entry_count"],
                "index": relative(index_path),
            },
            sort_keys=True,
        )
    )
    return index_path


def resolve_index(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    return require_file(path)


def verify(index_path: Path) -> dict[str, Any]:
    index = load_json(index_path)
    if index_path.read_bytes() != canonical_bytes(index) + b"\n":
        raise EvidenceError("umbrella index is not canonical JSON")
    manifest = index.get("manifest")
    if (
        index.get("record_type")
        != "controlled_followup_umbrella_evidence_v2_index"
        or not isinstance(manifest, dict)
        or canonical_sha256(manifest) != index.get("manifest_sha256")
        or index.get("entry_count") != len(manifest.get("entries", []))
        or not isinstance(manifest.get("matmul_v4_included"), bool)
        or manifest.get("matmul_v4_disposition")
        not in {
            "included_complete",
            "explicitly_deferred_active_or_incomplete",
        }
        or (
            manifest["matmul_v4_included"]
            != (manifest["matmul_v4_disposition"] == "included_complete")
        )
    ):
        raise EvidenceError("umbrella index/manifest contract mismatch")
    bundle_path = (REPO / index.get("bundle_path", "")).resolve()
    relative(bundle_path)
    if (
        not bundle_path.is_file()
        or bundle_path.stat().st_size != index.get("bundle_size")
        or sha256_file(bundle_path) != index.get("bundle_sha256")
    ):
        raise EvidenceError("umbrella bundle hash or size differs")
    _verify_umbrella_bundle(bundle_path, manifest)
    result = {
        "ok": True,
        "bundle_sha256": index["bundle_sha256"],
        "entries": index["entry_count"],
        "index": relative(index_path),
    }
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run", help="validate and report selection without writing")
    dry.add_argument("--include-matmul-v4", action="store_true")
    builder = sub.add_parser("build", help="build an immutable final umbrella archive")
    builder.add_argument("--name", required=True)
    disposition = builder.add_mutually_exclusive_group(required=True)
    disposition.add_argument("--include-matmul-v4", action="store_true")
    disposition.add_argument("--defer-active-matmul-v4", action="store_true")
    checker = sub.add_parser("verify", help="verify an immutable umbrella archive")
    checker.add_argument("--index", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "dry-run":
        print(json.dumps(selection_report(args.include_matmul_v4), indent=2, sort_keys=True))
    elif args.command == "build":
        build(
            args.name,
            args.include_matmul_v4,
            args.defer_active_matmul_v4,
        )
    else:
        verify(resolve_index(args.index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
