#!/usr/bin/env python3
"""Build and verify the round-2 implementation/evidence preservation bundle."""
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
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FOLLOWUP = REPO / "ako_runs/controlled_followup"
DEFAULT_NAME = "round2_complete_v1"
MANIFEST_MEMBER = "EVIDENCE_MANIFEST.json"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

DOCUMENTS = (
    REPO / "ako_runs/CONTROLLED_CROSS_DSL_REPORT.md",
    REPO / "ako_runs/CONTROLLED_CROSS_DSL_REVIEW.md",
    REPO / "ako_runs/CROSS_DSL_ERRATA_POINTER.md",
    REPO / "ako_runs/phase1_matmul/PHASE1_REPORT.md",
    REPO / "ako_runs/phase2_fused_sdpa/PHASE2_REPORT.md",
    REPO / "ako_runs/phase2_fused_sdpa/PHASE2_REPORT.built.md",
    FOLLOWUP / "README.md",
    FOLLOWUP / "REVIEW_20260730.md",
    FOLLOWUP / "REVIEW_RESPONSE_20260730.md",
    FOLLOWUP / "REVIEW_ROUND2_20260730.md",
    FOLLOWUP / "REVIEW_ROUND2_RESPONSE_20260731.md",
    FOLLOWUP / "ERRATA_20260730.md",
    FOLLOWUP / "RUN_20260730.md",
    FOLLOWUP / "RUN_20260731.md",
    FOLLOWUP / "provenance/README.md",
    FOLLOWUP / "provenance/historical_document_corrections_20260731.json",
)

HISTORICAL_CORRECTION_DOCUMENTS = (
    REPO / "ako_runs/CONTROLLED_CROSS_DSL_REPORT.md",
    REPO / "ako_runs/CONTROLLED_CROSS_DSL_REVIEW.md",
    REPO / "ako_runs/phase1_matmul/PHASE1_REPORT.md",
    REPO / "ako_runs/phase2_fused_sdpa/PHASE2_REPORT.md",
    REPO / "ako_runs/phase2_fused_sdpa/PHASE2_REPORT.built.md",
)

COMPLETED_CAMPAIGN_ROOTS = (
    FOLLOWUP / "robust_gate/audits/matmul_v4_instrument_v1",
    FOLLOWUP / "robust_gate/audits/fused_same_seed_stress_v2",
)

PROTOCOL_ROOTS = (
    FOLLOWUP / "convergence_v2",
    FOLLOWUP / "fused_epilogue_crossed_v1",
    FOLLOWUP / "reciprocal_v2",
    FOLLOWUP / "effort_frontier_v1",
)

PLAIN_COMPLETED_ROOTS = (
    FOLLOWUP / "fused_closure_v2",
    FOLLOWUP / "fused_reachability_v2",
    FOLLOWUP / "fused_frontier_closure_v3",
)

LEGACY_FIX_ROOT = FOLLOWUP / "legacy_cuda_harness_fix"
MATMUL = COMPLETED_CAMPAIGN_ROOTS[0]
SAME_SEED = COMPLETED_CAMPAIGN_ROOTS[1]

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".torch_ext",
    ".torch_extensions",
    "build_artifacts",
    "evidence",
}
EXCLUDED_NAMES = {".ninja_deps", ".ninja_log", "build.ninja"}
EXCLUDED_SUFFIXES = {
    ".a",
    ".cubin",
    ".dll",
    ".dylib",
    ".exe",
    ".fatbin",
    ".o",
    ".obj",
    ".ptx",
    ".pyc",
    ".pyo",
    ".so",
}
TEMPORARY_SUFFIXES = (".partial", ".part", ".tmp", ".swp", "~")
ALLOWED_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_NAMES = {".gitignore"}

EXPECTED_MATMUL_SUMMARY_SHA256 = (
    "125e66bbd76012e49695b1226ff39943d52bb61da9a44842b549a53b1f678ebd"
)
EXPECTED_MATMUL_MARGIN_V2_SHA256 = (
    "884a7cd513ae50f6ec58859da9185737fcc8fbb54e900503e4522c324d33e3af"
)
EXPECTED_PLAIN_TREE_ENTRIES = {
    "ako_runs/controlled_followup/fused_closure_v2": 545,
    "ako_runs/controlled_followup/fused_reachability_v2": 1698,
    "ako_runs/controlled_followup/fused_frontier_closure_v3": 259,
}
EXPECTED_PLAIN_REACHABILITY_RESULT_ENTRIES = 1682


class EvidenceError(RuntimeError):
    """The selection, receipt graph, or archive violates the evidence contract."""


@dataclass(frozen=True)
class NestedEvidence:
    label: str
    index: str
    bundle: str
    index_sha256: str
    bundle_sha256: str
    entries: int
    kind: str = "manifest_index"


NESTED_EVIDENCE = (
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
    ),
    NestedEvidence(
        "fused_reachability_row_sum_stress_v1",
        "ako_runs/controlled_followup/robust_gate/audits/fused_reachability_row_sum_stress_v1/evidence/complete_v1.index.json",
        "ako_runs/controlled_followup/robust_gate/audits/fused_reachability_row_sum_stress_v1/evidence/complete_v1.tar.gz",
        "00f985e92bbfdc5697d88b5e5334093d1e5c0124a82715c070c077bc6cb87cbe",
        "8acc2e8927f95761781688647cddcb5ef3e6dba9df2e9a8d650aced96aa41028",
        25,
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
    ),
    NestedEvidence(
        "fused_same_seed_stress_v2",
        "ako_runs/controlled_followup/robust_gate/audits/fused_same_seed_stress_v2/evidence/complete_v1.index.json",
        "ako_runs/controlled_followup/robust_gate/audits/fused_same_seed_stress_v2/evidence/complete_v1.tar.gz",
        "194ec1255fb19cbf85180a1d6b0d7e03631287e85987cc9eaa2a59afc6c3f7c1",
        "9f4b8be60744da896fc14bcdf03fb4cc49780656c85b4e7b41c7a3ea742a450c",
        70,
    ),
    NestedEvidence(
        "reciprocal_v2_preregistration",
        "ako_runs/controlled_followup/reciprocal_v2/evidence/prereg_v1.index.json",
        "ako_runs/controlled_followup/reciprocal_v2/evidence/prereg_v1.tar.gz",
        "f2ed15eb5d6bbe76ca52d949c59d354cc6205bdc53cf43fa9e96a93827139946",
        "2de75d240cfee932d2ced6d435cc33c8a05efb380d0cee50303739498e51f615",
        62,
    ),
    NestedEvidence(
        "reciprocal_v2_production_preregistration_v2",
        "ako_runs/controlled_followup/reciprocal_v2/evidence/prereg_v2.index.json",
        "ako_runs/controlled_followup/reciprocal_v2/evidence/prereg_v2.tar.gz",
        "7efe2b3e82d1bdbb621243bbaa6aaaffc101a27442197c50c52288685cd14037",
        "8bfa1d6d06b986836defeddac276e91bb75701a902412f40af6de765e5e868df",
        77,
    ),
    NestedEvidence(
        "convergence_v2_preregistration_v2",
        "ako_runs/controlled_followup/convergence_v2/evidence/prereg_v2.index.json",
        "ako_runs/controlled_followup/convergence_v2/evidence/prereg_v2.tar.gz",
        "31e6a293c8091b67a1e880930e387ce684fa4db03451cb09a2ee66d0208fe92d",
        "399d9928d0df31dfce5cc52443283d5339a8263b480f5d6f848b7413aebf6b85",
        40,
        "convergence_index",
    ),
    NestedEvidence(
        "effort_frontier_v1_preregistration",
        "ako_runs/controlled_followup/effort_frontier_v1/evidence/effort_frontier_prereg_v1.index.json",
        "ako_runs/controlled_followup/effort_frontier_v1/evidence/effort_frontier_prereg_v1.tar.gz",
        "d7858996de38be5078b7aceeb54c5e6099df0fecd3fa7ba811eb7b4e1af655cb",
        "50d1642c1a7c22f38ba95b08d9b77fc7793e022d99cade3f052030beab267ad6",
        21,
        "effort_frontier_index",
    ),
)

NESTED_FILES = {
    str((REPO / name).resolve())
    for specification in NESTED_EVIDENCE
    for name in (specification.index, specification.bundle)
}


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


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


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected a JSON object: {path}")
    return value


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"path escapes repository: {path}") from exc


def _validate_member_name(name: str) -> None:
    if not isinstance(name, str) or not name or "\\" in name:
        raise EvidenceError(f"unsafe archive path: {name!r}")
    value = PurePosixPath(name)
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise EvidenceError(f"unsafe archive path: {name!r}")
    if value.as_posix() != name:
        raise EvidenceError(f"non-canonical archive path: {name!r}")


def _excluded_reason(path: Path, extra_excluded_parts: frozenset[str] = frozenset()) -> str | None:
    try:
        parts = path.resolve().relative_to(REPO.resolve()).parts
    except ValueError:
        return "path outside repository"
    if any(part in EXCLUDED_PARTS or part in extra_excluded_parts for part in parts):
        return "cache, build, evidence, or explicitly excluded directory"
    if path.name in EXCLUDED_NAMES:
        return "build-control file"
    if path.name.endswith(".lock"):
        return "active/runtime lock"
    if path.name.endswith(TEMPORARY_SUFFIXES) or ".tmp." in path.name:
        return "partial or temporary file"
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return "compiled/build product"
    return None


def _eligible(path: Path, extra_excluded_parts: frozenset[str] = frozenset()) -> bool:
    return (
        _excluded_reason(path, extra_excluded_parts) is None
        and (path.name in ALLOWED_NAMES or path.suffix.lower() in ALLOWED_SUFFIXES)
    )


def _check_no_incomplete(root: Path, extra_excluded_parts: frozenset[str]) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        reason = _excluded_reason(path, extra_excluded_parts)
        if reason == "partial or temporary file":
            raise EvidenceError(
                f"incomplete/temporary file under selected root: {relative(path)}"
            )


class Selection:
    def __init__(self) -> None:
        self.categories: dict[Path, set[str]] = defaultdict(set)
        self.validations: dict[str, Any] = {}

    def add(
        self,
        path: Path,
        category: str,
        expected_hash: str | None = None,
        *,
        allow_nested_evidence: bool = False,
    ) -> None:
        if path.is_symlink() or not path.is_file():
            raise EvidenceError(f"required regular file missing or symlinked: {path}")
        resolved = path.resolve()
        relative(resolved)
        reason = _excluded_reason(resolved)
        if reason is not None and not (
            allow_nested_evidence and str(resolved) in NESTED_FILES
        ):
            raise EvidenceError(f"refusing {reason}: {relative(resolved)}")
        if not allow_nested_evidence and not (
            resolved.name in ALLOWED_NAMES or resolved.suffix.lower() in ALLOWED_SUFFIXES
        ):
            raise EvidenceError(f"unsupported evidence file type: {relative(resolved)}")
        if expected_hash is not None:
            if not HASH_RE.fullmatch(expected_hash):
                raise EvidenceError(f"invalid expected SHA-256 for {relative(resolved)}")
            observed = sha256_file(resolved)
            if observed != expected_hash:
                raise EvidenceError(
                    f"frozen hash mismatch for {relative(resolved)}: "
                    f"{observed} != {expected_hash}"
                )
        self.categories[resolved].add(category)

    def add_tree(
        self,
        root: Path,
        category: str,
        *,
        excluded_parts: Iterable[str] = (),
    ) -> None:
        if root.is_symlink() or not root.is_dir():
            raise EvidenceError(f"required campaign root missing or symlinked: {root}")
        relative(root)
        extra = frozenset(excluded_parts)
        _check_no_incomplete(root, extra)
        count = 0
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                lexical_parts = path.relative_to(REPO).parts
                if not any(
                    part in EXCLUDED_PARTS or part in extra for part in lexical_parts
                ):
                    raise EvidenceError(f"symlink under selected root: {path}")
                continue
            if path.is_file() and _eligible(path, extra):
                self.add(path, category)
                count += 1
        if count == 0:
            raise EvidenceError(f"selected campaign root is empty: {relative(root)}")

    def paths(self) -> list[Path]:
        return sorted(self.categories, key=relative)


def _tar_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members = archive.getmembers()
    names = [member.name for member in members]
    for name in names:
        _validate_member_name(name)
    if len(names) != len(set(names)):
        raise EvidenceError("archive contains duplicate member names")
    if any(not member.isfile() for member in members):
        raise EvidenceError("archive contains a non-regular member")
    return {member.name: member for member in members}


def _validated_entries(value: Any, context: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise EvidenceError(f"{context}: entries are not a list")
    entries: dict[str, dict[str, Any]] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise EvidenceError(f"{context}: entry is not an object")
        name = entry.get("path")
        size = entry.get("size", entry.get("bytes"))
        digest = entry.get("sha256")
        if not isinstance(name, str):
            raise EvidenceError(f"{context}: entry path is malformed")
        _validate_member_name(name)
        if (
            name == MANIFEST_MEMBER
            or name in entries
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or not HASH_RE.fullmatch(digest)
        ):
            raise EvidenceError(f"{context}: malformed entry {entry!r}")
        entries[name] = {"size": size, "sha256": digest, **entry}
    return entries


def _verify_payload_members(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    entries: dict[str, dict[str, Any]],
) -> None:
    if set(members) != set(entries):
        raise EvidenceError("archive membership differs from its manifest/index")
    for name, entry in entries.items():
        member = members[name]
        stream = archive.extractfile(member)
        if stream is None:
            raise EvidenceError(f"cannot read archive member: {name}")
        if member.size != entry["size"] or sha256_stream(stream) != entry["sha256"]:
            raise EvidenceError(f"archive member mismatch: {name}")


def verify_nested(specification: NestedEvidence) -> dict[str, Any]:
    index_path = REPO / specification.index
    bundle_path = REPO / specification.bundle
    if sha256_file(index_path) != specification.index_sha256:
        raise EvidenceError(f"nested index hash mismatch: {specification.label}")
    if sha256_file(bundle_path) != specification.bundle_sha256:
        raise EvidenceError(f"nested bundle hash mismatch: {specification.label}")
    index = load_json(index_path)

    if specification.kind == "convergence_index":
        if (
            index.get("archive_sha256") != specification.bundle_sha256
            or index.get("archive_size") != bundle_path.stat().st_size
            or index.get("file_count") != specification.entries
            or index.get("capture_mode") != "prereg"
        ):
            raise EvidenceError(
                f"convergence nested index contract mismatch: {specification.label}"
            )
        with tarfile.open(bundle_path, mode="r:gz") as archive:
            members = _tar_members(archive)
            manifest_member = members.pop(MANIFEST_MEMBER, None)
            if manifest_member is None:
                raise EvidenceError(
                    f"convergence nested manifest absent: {specification.label}"
                )
            stream = archive.extractfile(manifest_member)
            if stream is None:
                raise EvidenceError(
                    f"convergence nested manifest unreadable: {specification.label}"
                )
            raw_manifest = stream.read()
            try:
                embedded = json.loads(raw_manifest)
            except json.JSONDecodeError as exc:
                raise EvidenceError(
                    f"convergence nested manifest is invalid JSON: {specification.label}"
                ) from exc
            if (
                not isinstance(embedded, dict)
                or hashlib.sha256(raw_manifest).hexdigest()
                != index.get("evidence_manifest_sha256")
                or hashlib.sha256(
                    canonical_bytes(embedded.get("claim_state")) + b"\n"
                ).hexdigest()
                != index.get("state_sha256")
                or embedded.get("capture_mode") != "prereg"
            ):
                raise EvidenceError(
                    f"convergence nested manifest contract mismatch: {specification.label}"
                )
            entries = _validated_entries(
                embedded.get("files"), f"nested:{specification.label}"
            )
            if len(entries) != specification.entries:
                raise EvidenceError(
                    f"convergence nested entry count mismatch: {specification.label}"
                )
            _verify_payload_members(archive, members, entries)
        return {
            "label": specification.label,
            "index_path": specification.index,
            "index_sha256": specification.index_sha256,
            "bundle_path": specification.bundle,
            "bundle_sha256": specification.bundle_sha256,
            "nested_entry_count": specification.entries,
            "verified": True,
        }

    if specification.kind == "effort_frontier_index":
        indexed_manifest = index.get("manifest")
        if (
            not isinstance(indexed_manifest, dict)
            or canonical_sha256(indexed_manifest)
            != index.get("manifest_canonical_sha256")
            or index.get("bundle_filename") != bundle_path.name
            or index.get("bundle_sha256") != specification.bundle_sha256
            or index.get("bundle_size") != bundle_path.stat().st_size
            or index.get("entry_count") != specification.entries
            or index.get("stage") != "prereg"
        ):
            raise EvidenceError(
                f"effort-frontier nested index contract mismatch: {specification.label}"
            )
        entries = _validated_entries(
            indexed_manifest.get("entries"), f"nested:{specification.label}"
        )
        if len(entries) != specification.entries:
            raise EvidenceError(
                f"effort-frontier nested entry count mismatch: {specification.label}"
            )
        with tarfile.open(bundle_path, mode="r:gz") as archive:
            members = _tar_members(archive)
            manifest_member = members.pop(MANIFEST_MEMBER, None)
            if manifest_member is None:
                raise EvidenceError(
                    f"effort-frontier nested manifest absent: {specification.label}"
                )
            stream = archive.extractfile(manifest_member)
            if stream is None:
                raise EvidenceError(
                    f"effort-frontier nested manifest unreadable: {specification.label}"
                )
            try:
                embedded = json.loads(stream.read())
            except json.JSONDecodeError as exc:
                raise EvidenceError(
                    f"effort-frontier nested manifest is invalid JSON: {specification.label}"
                ) from exc
            if embedded != indexed_manifest:
                raise EvidenceError(
                    f"effort-frontier nested manifest differs: {specification.label}"
                )
            _verify_payload_members(archive, members, entries)
        return {
            "label": specification.label,
            "index_path": specification.index,
            "index_sha256": specification.index_sha256,
            "bundle_path": specification.bundle,
            "bundle_sha256": specification.bundle_sha256,
            "nested_entry_count": specification.entries,
            "verified": True,
        }

    if specification.kind == "legacy_v1":
        bundle_record = index.get("bundle")
        entries_value = index.get("entries")
        declared_path = bundle_record.get("path") if isinstance(bundle_record, dict) else None
        declared_hash = bundle_record.get("sha256") if isinstance(bundle_record, dict) else None
        declared_count = (
            index.get("counts", {}).get("selected_files")
            if isinstance(index.get("counts"), dict)
            else None
        )
        expects_manifest = False
        indexed_manifest = None
    elif specification.kind == "archived_index":
        entries_value = index.get("entries")
        declared_path = index.get("bundle")
        declared_hash = index.get("bundle_sha256")
        declared_count = index.get("entry_count")
        expects_manifest = True
        indexed_manifest = None
    else:
        indexed_manifest = index.get("manifest")
        if not isinstance(indexed_manifest, dict):
            raise EvidenceError(f"nested manifest missing: {specification.label}")
        if canonical_sha256(indexed_manifest) != index.get("manifest_sha256"):
            raise EvidenceError(f"nested manifest hash mismatch: {specification.label}")
        entries_value = indexed_manifest.get("entries")
        declared_path = index.get("bundle_path")
        declared_hash = index.get("bundle_sha256")
        declared_count = index.get("entry_count")
        expects_manifest = True

    entries = _validated_entries(entries_value, f"nested:{specification.label}")
    if (
        declared_path != specification.bundle
        or declared_hash != specification.bundle_sha256
        or declared_count != specification.entries
        or len(entries) != specification.entries
        or bundle_path.stat().st_size != index.get("bundle_size", bundle_path.stat().st_size)
    ):
        raise EvidenceError(f"nested index contract mismatch: {specification.label}")

    with tarfile.open(bundle_path, mode="r:gz") as archive:
        members = _tar_members(archive)
        if expects_manifest:
            manifest_member = members.pop(MANIFEST_MEMBER, None)
            if manifest_member is None:
                raise EvidenceError(f"nested manifest absent: {specification.label}")
            stream = archive.extractfile(manifest_member)
            if stream is None:
                raise EvidenceError(f"nested manifest unreadable: {specification.label}")
            try:
                embedded = json.loads(stream.read())
            except json.JSONDecodeError as exc:
                raise EvidenceError(
                    f"nested manifest is invalid JSON: {specification.label}"
                ) from exc
            if not isinstance(embedded, dict):
                raise EvidenceError(f"nested manifest is not an object: {specification.label}")
            if specification.kind == "archived_index":
                if embedded.get("entries") != entries_value:
                    raise EvidenceError(
                        f"archived nested manifest differs: {specification.label}"
                    )
            elif embedded != indexed_manifest:
                raise EvidenceError(f"nested embedded manifest differs: {specification.label}")
        _verify_payload_members(archive, members, entries)

    return {
        "label": specification.label,
        "index_path": specification.index,
        "index_sha256": specification.index_sha256,
        "bundle_path": specification.bundle,
        "bundle_sha256": specification.bundle_sha256,
        "nested_entry_count": specification.entries,
        "verified": True,
    }


def _resolve_bound_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise EvidenceError(f"malformed receipt-bound path under {relative(root)}")
    _validate_member_name(value)
    pure = PurePosixPath(value)
    path = REPO.joinpath(*pure.parts) if pure.parts[0] == "ako_runs" else root.joinpath(*pure.parts)
    resolved = path.resolve()
    relative(resolved)
    if resolved.is_symlink() or not resolved.is_file():
        raise EvidenceError(f"receipt-bound file missing: {path}")
    return resolved


def _validate_matmul_completion() -> dict[str, Any]:
    completion_path = MATMUL / "receipts/completion_receipt.json"
    summary_path = MATMUL / "results/summary.json"
    margin_path = MATMUL / "results/margin_report_v2.json"
    completion = load_json(completion_path)
    summary = load_json(summary_path)
    campaign_id = "controlled-followup-matmul-v4-instrument-audit-v1"
    if (
        completion.get("campaign_id") != campaign_id
        or summary.get("campaign_id") != campaign_id
        or completion.get("evidence_complete") is not True
        or summary.get("evidence_complete") is not True
        or completion.get("summary_sha256") != EXPECTED_MATMUL_SUMMARY_SHA256
        or sha256_file(summary_path) != EXPECTED_MATMUL_SUMMARY_SHA256
        or sha256_file(margin_path) != EXPECTED_MATMUL_MARGIN_V2_SHA256
        or summary.get("expected_records") != 66852
        or summary.get("observed_unique_records") != 66852
    ):
        raise EvidenceError("matmul-v4 completion/summary binding mismatch")
    raw_entries = completion.get("raw_files")
    summary_raw = summary.get("raw_files")
    if not isinstance(raw_entries, list) or raw_entries != summary_raw or len(raw_entries) != 6:
        raise EvidenceError("matmul-v4 raw-file receipt set mismatch")
    if sum(entry.get("records", -1) for entry in raw_entries if isinstance(entry, dict)) != 66852:
        raise EvidenceError("matmul-v4 raw record census mismatch")
    for entry in raw_entries:
        if not isinstance(entry, dict) or not HASH_RE.fullmatch(str(entry.get("sha256", ""))):
            raise EvidenceError("malformed matmul-v4 raw receipt")
        path = _resolve_bound_path(MATMUL, entry.get("path"))
        if sha256_file(path) != entry["sha256"]:
            raise EvidenceError(f"matmul-v4 raw hash mismatch: {relative(path)}")
    return {
        "state": "complete_receipt_bound",
        "evidence_complete": True,
        "observed_unique_records": 66852,
        "summary_sha256": EXPECTED_MATMUL_SUMMARY_SHA256,
        "margin_report_v2_sha256": EXPECTED_MATMUL_MARGIN_V2_SHA256,
    }


def _validate_same_seed_completion() -> dict[str, Any]:
    completion_path = SAME_SEED / "receipts/completion_receipt.json"
    collection_path = SAME_SEED / "receipts/collection_receipt.json"
    summary_path = SAME_SEED / "results/summary.json"
    completion = load_json(completion_path)
    collection = load_json(collection_path)
    summary = load_json(summary_path)
    campaign_id = "controlled-followup-fused-same-seed-stress-v2"
    summary_hash = sha256_file(summary_path)
    collection_hash = sha256_file(collection_path)
    coverage = summary.get("coverage")
    if (
        completion.get("campaign_id") != campaign_id
        or collection.get("campaign_id") != campaign_id
        or summary.get("campaign_id") != campaign_id
        or completion.get("evidence_complete") is not True
        or summary.get("evidence_complete") is not True
        or completion.get("summary_sha256") != summary_hash
        or completion.get("collection_receipt_sha256") != collection_hash
        or summary.get("collection_receipt_sha256") != collection_hash
        or collection.get("record_count") != 10240
        or collection.get("expected_records") != 10240
        or summary.get("effective_shared_seed_n") != 512
        or not isinstance(coverage, dict)
        or coverage.get("expected_records") != 10240
        or coverage.get("observed_unique_records") != 10240
        or coverage.get("missing_records") != 0
        or coverage.get("duplicate_records") != 0
        or coverage.get("unexpected_records") != 0
    ):
        raise EvidenceError("same-seed-v2 completion/summary binding mismatch")
    raw_path = _resolve_bound_path(SAME_SEED, collection.get("raw_path"))
    if (
        collection.get("raw_sha256") != summary.get("raw_sha256")
        or sha256_file(raw_path) != collection.get("raw_sha256")
    ):
        raise EvidenceError("same-seed-v2 raw stream hash mismatch")
    return {
        "state": "complete_receipt_and_archive_bound",
        "evidence_complete": True,
        "observed_unique_records": 10240,
        "effective_shared_seed_n": 512,
        "summary_sha256": summary_hash,
    }


def _validate_document_corrections() -> dict[str, Any]:
    receipt_path = FOLLOWUP / "provenance/historical_document_corrections_20260731.json"
    receipt = load_json(receipt_path)
    documents = receipt.get("documents")
    if (
        receipt.get("record_type") != "historical_document_correction_receipt"
        or not isinstance(documents, list)
        or len(documents) != 5
    ):
        raise EvidenceError("historical-document correction receipt is malformed")
    observed: list[dict[str, str]] = []
    for entry in documents:
        if not isinstance(entry, dict) or not HASH_RE.fullmatch(
            str(entry.get("post_edit_sha256", ""))
        ):
            raise EvidenceError("historical-document correction entry is malformed")
        path = _resolve_bound_path(REPO, entry.get("path"))
        digest = sha256_file(path)
        if digest != entry["post_edit_sha256"]:
            raise EvidenceError(f"historical correction hash mismatch: {relative(path)}")
        observed.append({"path": relative(path), "post_edit_sha256": digest})
    return {
        "state": "complete_receipt_bound",
        "receipt_path": relative(receipt_path),
        "documents": observed,
    }


def _validate_blocked_preflight_value(
    receipt: dict[str, Any], context: str
) -> None:
    action = str(receipt.get("action", ""))
    if (
        receipt.get("launch_ready") is True
        or receipt.get("ready") is True
        or action.startswith("launch_permitted")
    ):
        raise EvidenceError(
            f"preflight-shaped artifact contains a positive launch signal: {context}"
        )
    blocked = (
        receipt.get("launch_ready") is False
        or receipt.get("ready") is False
        or action.startswith("launch_forbidden")
    )
    if not blocked:
        raise EvidenceError(
            f"preflight-shaped artifact does not prove a blocked launch: {context}"
        )
    if (
        "zero_gpu_processes_started" in receipt
        and receipt.get("zero_gpu_processes_started") is not True
    ):
        raise EvidenceError(
            f"preflight does not affirm zero GPU processes started: {context}"
        )


def _blocked_preflight_files(root: Path) -> list[Path]:
    selected = []
    for path in sorted(root.rglob("*preflight*.json")):
        if not path.is_file() or not _eligible(path):
            continue
        receipt = load_json(path)
        _validate_blocked_preflight_value(receipt, relative(path))
        selected.append(path.resolve())
    return selected


def _eligible_result_count(root: Path) -> int:
    results = root / "results"
    if not results.is_dir():
        return 0
    preflight = set(_blocked_preflight_files(root))
    return sum(
        1
        for path in results.rglob("*")
        if path.is_file() and _eligible(path) and path.resolve() not in preflight
    )


def campaign_states() -> dict[str, Any]:
    convergence_summary = load_json(FOLLOWUP / "convergence_v2/manifests/summary.json")
    convergence_preflight = load_json(
        FOLLOWUP
        / "convergence_v2/receipts/launch_preflight_post_gpu_binding_20260731.json"
    )
    convergence_checks = {
        item.get("name"): item.get("passed")
        for item in convergence_preflight.get("checks", [])
        if isinstance(item, dict)
    }
    crossed = load_json(FOLLOWUP / "fused_epilogue_crossed_v1/campaign.json")
    reciprocal = load_json(FOLLOWUP / "reciprocal_v2/manifests/summary.json")
    effort = load_json(FOLLOWUP / "effort_frontier_v1/manifest.json")
    model_lock = load_json(FOLLOWUP / "effort_frontier_v1/locks/model_resolution_lock.json")
    if (
        convergence_summary.get("launch_state") != "blocked_pending_validate_launch"
        or convergence_preflight.get("ready") is not False
        or not str(convergence_preflight.get("action", "")).startswith(
            "launch_forbidden"
        )
        or convergence_checks.get("gpu_uuid_binding") is not True
        or convergence_checks.get("immutable_model_resolution") is not False
        or crossed.get("status") != "implemented_not_launched"
        or reciprocal.get("campaign_id") != "standard-matmul-reciprocal-transfer-v2-20260731"
        or reciprocal.get("total_cells") != 96
        or effort.get("status") != "preregistered_not_launched"
        or model_lock.get("state") != "unresolved"
    ):
        raise EvidenceError("new-campaign declared state differs from the preservation contract")

    return {
        "document_and_margin_corrections": _validate_document_corrections(),
        "matmul_v4_instrument_v1": _validate_matmul_completion(),
        "fused_same_seed_stress_v2": _validate_same_seed_completion(),
        "fused_closure_v2": {"state": "completed_plain_files_included"},
        "fused_reachability_v2": {
            "state": "completed_plain_files_and_fixed_nested_evidence_included"
        },
        "fused_frontier_closure_v3": {
            "state": "completed_plain_files_and_fixed_nested_evidence_included"
        },
        "convergence_v2": {
            "state": "protocol_frozen_launch_blocked_no_empirical_result",
            "blocked_preflight_receipts_included": len(
                _blocked_preflight_files(PROTOCOL_ROOTS[0])
            ),
            "unvalidated_result_files_excluded": _eligible_result_count(PROTOCOL_ROOTS[0]),
        },
        "fused_epilogue_crossed_v1": {
            "state": "implemented_not_launched_no_empirical_result",
            "blocked_preflight_receipts_included": len(
                _blocked_preflight_files(PROTOCOL_ROOTS[1])
            ),
            "unvalidated_result_files_excluded": _eligible_result_count(PROTOCOL_ROOTS[1]),
        },
        "reciprocal_v2": {
            "state": "preregistered_launch_blocked_no_empirical_result",
            "sealed_preregistration_included": True,
            "blocked_preflight_receipts_included": len(
                _blocked_preflight_files(PROTOCOL_ROOTS[2])
            ),
            "unvalidated_result_files_excluded": _eligible_result_count(PROTOCOL_ROOTS[2]),
        },
        "effort_frontier_v1": {
            "state": "preregistered_not_launched_no_empirical_result",
            "blocked_preflight_receipts_included": len(
                _blocked_preflight_files(PROTOCOL_ROOTS[3])
            ),
            "unvalidated_result_files_excluded": _eligible_result_count(PROTOCOL_ROOTS[3]),
        },
        "empirical_round2_program_complete": False,
    }


REQUIRED_SELECTED = (
    FOLLOWUP / "REVIEW_ROUND2_20260730.md",
    FOLLOWUP / "REVIEW_ROUND2_RESPONSE_20260731.md",
    FOLLOWUP / "RUN_20260731.md",
    MATMUL / "receipts/completion_receipt.json",
    MATMUL / "results/summary.json",
    MATMUL / "results/margin_report_v2.json",
    SAME_SEED / "receipts/completion_receipt.json",
    SAME_SEED / "results/raw/measurements.jsonl",
    SAME_SEED / "results/summary.json",
    FOLLOWUP / "fused_reachability_v2/results/confirmation_analysis_v1/confirmation_summary.json",
    FOLLOWUP / "convergence_v2/locks/protocol_freeze_receipt.json",
    FOLLOWUP / "fused_epilogue_crossed_v1/launch_lock.json",
    FOLLOWUP / "reciprocal_v2/receipts/source_freeze.json",
    FOLLOWUP / "effort_frontier_v1/manifest.json",
)


def selection() -> Selection:
    value = Selection()
    for path in DOCUMENTS:
        value.add(path, "controlling_documents")
    value.add_tree(LEGACY_FIX_ROOT, "legacy_cuda_harness_fix")
    for root in COMPLETED_CAMPAIGN_ROOTS:
        value.add_tree(root, "completed_round2_campaign")
    for root in PROTOCOL_ROOTS:
        # These campaigns have no validated empirical result. Preserve protocol,
        # preregistration, tests, and locks, but never sweep an active results tree.
        value.add_tree(root, "new_campaign_protocol", excluded_parts={"results"})
        for path in _blocked_preflight_files(root):
            value.add(path, "new_campaign_blocked_preflight")
    for root in PLAIN_COMPLETED_ROOTS:
        value.add_tree(root, "plain_completed_campaign_files")

    nested_validations = []
    for specification in NESTED_EVIDENCE:
        nested_validations.append(verify_nested(specification))
        value.add(
            REPO / specification.index,
            f"nested:{specification.label}",
            specification.index_sha256,
            allow_nested_evidence=True,
        )
        value.add(
            REPO / specification.bundle,
            f"nested:{specification.label}",
            specification.bundle_sha256,
            allow_nested_evidence=True,
        )

    for path in (Path(__file__), HERE / "README.md", HERE / "test_evidence_v3.py", HERE / "__init__.py"):
        value.add(path, "evidence_builder")

    selected = set(value.categories)
    missing = [relative(path) for path in REQUIRED_SELECTED if path.resolve() not in selected]
    if missing:
        raise EvidenceError(f"required coverage missing from selection: {missing}")
    plain_tree_entries = {
        relative(root): sum(
            1
            for path in selected
            if "plain_completed_campaign_files" in value.categories[path]
            and path.is_relative_to(root.resolve())
        )
        for root in PLAIN_COMPLETED_ROOTS
    }
    reachability_results = FOLLOWUP / "fused_reachability_v2/results"
    plain_reachability_result_entries = sum(
        1
        for path in selected
        if path.is_relative_to(reachability_results.resolve())
    )
    if plain_tree_entries != EXPECTED_PLAIN_TREE_ENTRIES:
        raise EvidenceError(
            "completed plain-tree census changed: "
            f"{plain_tree_entries} != {EXPECTED_PLAIN_TREE_ENTRIES}"
        )
    if plain_reachability_result_entries != EXPECTED_PLAIN_REACHABILITY_RESULT_ENTRIES:
        raise EvidenceError(
            "plain reachability result census changed: "
            f"{plain_reachability_result_entries} != "
            f"{EXPECTED_PLAIN_REACHABILITY_RESULT_ENTRIES}"
        )
    value.validations = {
        "nested_evidence": nested_validations,
        "plain_completed_tree_entries": plain_tree_entries,
        "plain_reachability_result_entries": plain_reachability_result_entries,
        "required_selected_paths": [relative(path) for path in REQUIRED_SELECTED],
    }
    return value


def _manifest_entries(value: Selection) -> list[dict[str, Any]]:
    return [
        {
            "path": relative(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "categories": sorted(value.categories[path]),
        }
        for path in value.paths()
    ]


def manifest(value: Selection, bundle_id: str = DEFAULT_NAME) -> dict[str, Any]:
    if not NAME_RE.fullmatch(bundle_id) or bundle_id in {".", ".."}:
        raise EvidenceError(f"unsafe evidence name: {bundle_id!r}")
    entries = _manifest_entries(value)
    states = campaign_states()
    return {
        "schema_version": 2,
        "record_type": "controlled_followup_round2_evidence_v3_manifest",
        "bundle_id": bundle_id,
        "bundle_scope": "round2 implementation plus all currently completed evidence",
        "claim_limit": (
            "Post-hoc file preservation only: not external preregistration and not "
            "evidence that launch-blocked campaigns ran."
        ),
        "campaign_states": states,
        "selection_policy": {
            "active_runtime_locks_excluded": True,
            "build_products_and_caches_excluded": True,
            "partial_and_temporary_files_fail_closed": True,
            "prospective_campaign_results_excluded_until_validated": True,
            "interim_evidence_v2_fused_postreview_v3_through_v8_excluded": True,
            "nested_archives_preserved_byte_for_byte": [
                specification.label for specification in NESTED_EVIDENCE
            ],
        },
        "validations": value.validations,
        "entry_count": len(entries),
        "total_payload_bytes": sum(row["size"] for row in entries),
        "category_counts": dict(
            sorted(Counter(category for row in entries for category in row["categories"]).items())
        ),
        "entries": entries,
    }


def normalized_info(name: str, size: int) -> tarfile.TarInfo:
    _validate_member_name(name)
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_tar(fileobj: BinaryIO, document: dict[str, Any]) -> None:
    with gzip.GzipFile(fileobj=fileobj, mode="wb", filename="", mtime=0, compresslevel=9) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.GNU_FORMAT) as archive:
            embedded = canonical_bytes(document) + b"\n"
            archive.addfile(normalized_info(MANIFEST_MEMBER, len(embedded)), io.BytesIO(embedded))
            for entry in document["entries"]:
                source = REPO / entry["path"]
                with source.open("rb") as stream:
                    archive.addfile(normalized_info(entry["path"], entry["size"]), stream)


def _tar_bytes(document: dict[str, Any]) -> bytes:
    """Return a deterministic archive; intended for small unit-test manifests."""
    output = io.BytesIO()
    _write_tar(output, document)
    return output.getvalue()


def write_bundle(path: Path, document: dict[str, Any]) -> None:
    with path.open("xb") as output:
        _write_tar(output, document)


def _validate_manifest(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    states = document.get("campaign_states")
    policy = document.get("selection_policy")
    if (
        document.get("schema_version") != 2
        or document.get("record_type") != "controlled_followup_round2_evidence_v3_manifest"
        or not NAME_RE.fullmatch(str(document.get("bundle_id", "")))
        or not isinstance(document.get("claim_limit"), str)
        or not isinstance(states, dict)
        or not isinstance(policy, dict)
        or not isinstance(document.get("validations"), dict)
    ):
        raise EvidenceError("embedded manifest header is malformed")
    entries = _validated_entries(document.get("entries"), "umbrella manifest")
    ordered_names = [entry.get("path") for entry in document["entries"]]
    if ordered_names != sorted(ordered_names):
        raise EvidenceError("umbrella manifest entries are not path-sorted")
    for raw_entry in document["entries"]:
        categories = raw_entry.get("categories")
        if (
            not isinstance(categories, list)
            or not categories
            or any(not isinstance(item, str) or not item for item in categories)
            or categories != sorted(set(categories))
        ):
            raise EvidenceError("umbrella manifest categories are malformed")
    expected_counts = dict(
        sorted(
            Counter(
                category
                for entry in document["entries"]
                for category in entry["categories"]
            ).items()
        )
    )
    if (
        document.get("entry_count") != len(entries)
        or document.get("total_payload_bytes") != sum(entry["size"] for entry in entries.values())
        or document.get("category_counts") != expected_counts
        or states.get("empirical_round2_program_complete") is not False
    ):
        raise EvidenceError("embedded manifest counts or state are inconsistent")
    return entries


def _release_path_exclusion(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if any(part in EXCLUDED_PARTS for part in parts):
        return "cache/build/evidence directory"
    leaf = PurePosixPath(name).name
    if leaf in EXCLUDED_NAMES or leaf.endswith(".lock"):
        return "runtime/build lock"
    if leaf.endswith(TEMPORARY_SUFFIXES) or ".tmp." in leaf:
        return "partial or temporary file"
    if PurePosixPath(name).suffix.lower() in EXCLUDED_SUFFIXES:
        return "compiled/build product"
    return None


def _expected_release_campaign_states(
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def entry_sha(path: Path) -> str:
        name = relative(path)
        digest = entries.get(name, {}).get("sha256")
        if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise EvidenceError(f"release campaign-state input is absent: {name}")
        return digest

    def blocked_count(root: Path) -> int:
        prefix = relative(root) + "/"
        return sum(
            1
            for name, entry in entries.items()
            if name.startswith(prefix)
            and "new_campaign_blocked_preflight" in entry.get("categories", [])
        )

    return {
        "document_and_margin_corrections": {
            "state": "complete_receipt_bound",
            "receipt_path": relative(
                FOLLOWUP / "provenance/historical_document_corrections_20260731.json"
            ),
            "documents": [
                {"path": relative(path), "post_edit_sha256": entry_sha(path)}
                for path in HISTORICAL_CORRECTION_DOCUMENTS
            ],
        },
        "matmul_v4_instrument_v1": {
            "state": "complete_receipt_bound",
            "evidence_complete": True,
            "observed_unique_records": 66852,
            "summary_sha256": EXPECTED_MATMUL_SUMMARY_SHA256,
            "margin_report_v2_sha256": EXPECTED_MATMUL_MARGIN_V2_SHA256,
        },
        "fused_same_seed_stress_v2": {
            "state": "complete_receipt_and_archive_bound",
            "evidence_complete": True,
            "observed_unique_records": 10240,
            "effective_shared_seed_n": 512,
            "summary_sha256": entry_sha(SAME_SEED / "results/summary.json"),
        },
        "fused_closure_v2": {"state": "completed_plain_files_included"},
        "fused_reachability_v2": {
            "state": "completed_plain_files_and_fixed_nested_evidence_included"
        },
        "fused_frontier_closure_v3": {
            "state": "completed_plain_files_and_fixed_nested_evidence_included"
        },
        "convergence_v2": {
            "state": "protocol_frozen_launch_blocked_no_empirical_result",
            "blocked_preflight_receipts_included": blocked_count(PROTOCOL_ROOTS[0]),
            "unvalidated_result_files_excluded": 0,
        },
        "fused_epilogue_crossed_v1": {
            "state": "implemented_not_launched_no_empirical_result",
            "blocked_preflight_receipts_included": blocked_count(PROTOCOL_ROOTS[1]),
            "unvalidated_result_files_excluded": 0,
        },
        "reciprocal_v2": {
            "state": "preregistered_launch_blocked_no_empirical_result",
            "sealed_preregistration_included": True,
            "blocked_preflight_receipts_included": blocked_count(PROTOCOL_ROOTS[2]),
            "unvalidated_result_files_excluded": 0,
        },
        "effort_frontier_v1": {
            "state": "preregistered_not_launched_no_empirical_result",
            "blocked_preflight_receipts_included": blocked_count(PROTOCOL_ROOTS[3]),
            "unvalidated_result_files_excluded": 0,
        },
        "empirical_round2_program_complete": False,
    }


def _validate_release_contract(
    document: dict[str, Any], entries: dict[str, dict[str, Any]]
) -> None:
    if document.get("campaign_states") != _expected_release_campaign_states(entries):
        raise EvidenceError("release campaign-state claims differ from bound evidence")
    policy = document["selection_policy"]
    required_true = (
        "active_runtime_locks_excluded",
        "build_products_and_caches_excluded",
        "partial_and_temporary_files_fail_closed",
        "prospective_campaign_results_excluded_until_validated",
        "interim_evidence_v2_fused_postreview_v3_through_v8_excluded",
    )
    if any(policy.get(key) is not True for key in required_true):
        raise EvidenceError("release selection policy is incomplete")
    expected_nested_labels = [item.label for item in NESTED_EVIDENCE]
    if policy.get("nested_archives_preserved_byte_for_byte") != expected_nested_labels:
        raise EvidenceError("release nested-evidence policy differs")

    validations = document["validations"]
    nested_rows = validations.get("nested_evidence")
    if not isinstance(nested_rows, list) or len(nested_rows) != len(NESTED_EVIDENCE):
        raise EvidenceError("release nested-evidence validations are incomplete")
    rows_by_label = {
        row.get("label"): row for row in nested_rows if isinstance(row, dict)
    }
    for specification in NESTED_EVIDENCE:
        expected_row = {
            "label": specification.label,
            "index_path": specification.index,
            "index_sha256": specification.index_sha256,
            "bundle_path": specification.bundle,
            "bundle_sha256": specification.bundle_sha256,
            "nested_entry_count": specification.entries,
            "verified": True,
        }
        if rows_by_label.get(specification.label) != expected_row:
            raise EvidenceError(
                f"release nested validation differs: {specification.label}"
            )
        for name, digest in (
            (specification.index, specification.index_sha256),
            (specification.bundle, specification.bundle_sha256),
        ):
            if entries.get(name, {}).get("sha256") != digest:
                raise EvidenceError(f"release nested entry differs: {name}")

    if validations.get("plain_completed_tree_entries") != EXPECTED_PLAIN_TREE_ENTRIES:
        raise EvidenceError("release plain completed-tree census differs")
    if (
        validations.get("plain_reachability_result_entries")
        != EXPECTED_PLAIN_REACHABILITY_RESULT_ENTRIES
    ):
        raise EvidenceError("release plain reachability-result census differs")
    expected_required = [relative(path) for path in REQUIRED_SELECTED]
    if validations.get("required_selected_paths") != expected_required:
        raise EvidenceError("release required-path validation differs")
    if any(name not in entries for name in expected_required):
        raise EvidenceError("release required path is absent")

    nested_paths = {
        name
        for specification in NESTED_EVIDENCE
        for name in (specification.index, specification.bundle)
    }
    prospective_result_prefixes = tuple(
        relative(root / "results") + "/" for root in PROTOCOL_ROOTS
    )
    prospective_root_prefixes = tuple(
        relative(root) + "/" for root in PROTOCOL_ROOTS
    )
    for name, entry in entries.items():
        if name not in nested_paths:
            reason = _release_path_exclusion(name)
            if reason is not None:
                raise EvidenceError(f"release contains {reason}: {name}")
        if any(
            name.startswith(
                f"ako_runs/controlled_followup/provenance/evidence_v2/fused_postreview_v{version}."
            )
            for version in range(3, 9)
        ):
            raise EvidenceError(f"release contains interim evidence-v2 archive: {name}")
        blocked_category = (
            "new_campaign_blocked_preflight" in entry.get("categories", [])
        )
        prospective = name.startswith(prospective_root_prefixes)
        preflight_shaped = (
            prospective
            and "preflight" in PurePosixPath(name).name
            and PurePosixPath(name).suffix == ".json"
        )
        if blocked_category != preflight_shaped:
            raise EvidenceError(
                f"release blocked-preflight category/path mismatch: {name}"
            )
        if name.startswith(prospective_result_prefixes) and not blocked_category:
            raise EvidenceError(f"release contains unvalidated prospective result: {name}")


def _verify_bundle(bundle: Path, document: dict[str, Any]) -> None:
    entries = _validate_manifest(document)
    with tarfile.open(bundle, mode="r:gz") as archive:
        members = _tar_members(archive)
        embedded_member = members.pop(MANIFEST_MEMBER, None)
        if embedded_member is None:
            raise EvidenceError("embedded manifest missing")
        stream = archive.extractfile(embedded_member)
        if stream is None or stream.read() != canonical_bytes(document) + b"\n":
            raise EvidenceError("embedded manifest bytes differ")
        for name, member in {MANIFEST_MEMBER: embedded_member, **members}.items():
            if (
                member.mode != 0o644
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
            ):
                raise EvidenceError(f"non-normalized archive metadata: {name}")
        _verify_payload_members(archive, members, entries)
        for name, entry in entries.items():
            if "new_campaign_blocked_preflight" not in entry.get("categories", []):
                continue
            stream = archive.extractfile(members[name])
            if stream is None:
                raise EvidenceError(f"blocked preflight is unreadable: {name}")
            try:
                receipt = json.loads(stream.read())
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"blocked preflight is invalid JSON: {name}") from exc
            if not isinstance(receipt, dict):
                raise EvidenceError(f"blocked preflight is not an object: {name}")
            _validate_blocked_preflight_value(receipt, name)


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


def build(name: str) -> dict[str, Any]:
    if not NAME_RE.fullmatch(name) or name in {".", ".."}:
        raise EvidenceError(f"unsafe evidence name: {name!r}")
    bundle = HERE / f"{name}.tar.gz"
    index_path = HERE / f"{name}.index.json"
    if bundle.exists() or index_path.exists():
        raise FileExistsError("evidence outputs are immutable; choose a new name")

    document = manifest(selection(), bundle_id=name)
    bundle_handle = tempfile.NamedTemporaryFile(
        prefix=f".{name}.", suffix=".tar.gz.tmp", dir=HERE, delete=False
    )
    bundle_tmp = Path(bundle_handle.name)
    bundle_handle.close()
    bundle_tmp.unlink()
    index_handle = tempfile.NamedTemporaryFile(
        prefix=f".{name}.", suffix=".index.json.tmp", dir=HERE, delete=False
    )
    index_tmp = Path(index_handle.name)
    index_handle.close()
    published_bundle = False
    try:
        write_bundle(bundle_tmp, document)
        _verify_bundle(bundle_tmp, document)
        _validate_release_contract(document, _validate_manifest(document))
        index = {
            "schema_version": 2,
            "record_type": "controlled_followup_round2_evidence_v3_index",
            "bundle_id": name,
            "bundle_path": relative(bundle),
            "bundle_sha256": sha256_file(bundle_tmp),
            "bundle_size": bundle_tmp.stat().st_size,
            "manifest_sha256": canonical_sha256(document),
            "entry_count": document["entry_count"],
            "campaign_states": document["campaign_states"],
        }
        index_tmp.write_bytes(canonical_bytes(index) + b"\n")
        _exclusive_publish(bundle_tmp, bundle)
        published_bundle = True
        _exclusive_publish(index_tmp, index_path)
    except Exception:
        if published_bundle and not index_path.exists():
            bundle.unlink(missing_ok=True)
        raise
    finally:
        bundle_tmp.unlink(missing_ok=True)
        index_tmp.unlink(missing_ok=True)
    return verify(index_path)


def resolve_index(value: Path) -> Path:
    path = value if value.is_absolute() else REPO / value
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"index missing or symlinked: {path}")
    relative(path)
    return path.resolve()


def verify(index_path: Path) -> dict[str, Any]:
    index_path = resolve_index(index_path)
    index = load_json(index_path)
    if index_path.read_bytes() != canonical_bytes(index) + b"\n":
        raise EvidenceError("index is not canonical JSON")
    bundle_id = index.get("bundle_id")
    if (
        index.get("schema_version") != 2
        or index.get("record_type") != "controlled_followup_round2_evidence_v3_index"
        or not isinstance(bundle_id, str)
        or not NAME_RE.fullmatch(bundle_id)
        or not HASH_RE.fullmatch(str(index.get("bundle_sha256", "")))
        or not HASH_RE.fullmatch(str(index.get("manifest_sha256", "")))
        or not isinstance(index.get("bundle_size"), int)
        or isinstance(index.get("bundle_size"), bool)
        or not isinstance(index.get("entry_count"), int)
        or isinstance(index.get("entry_count"), bool)
    ):
        raise EvidenceError("index header is malformed")
    expected_bundle_path = relative(HERE / f"{bundle_id}.tar.gz")
    if index.get("bundle_path") != expected_bundle_path:
        raise EvidenceError("index bundle path does not match bundle identity")
    bundle = (REPO / expected_bundle_path).resolve()
    relative(bundle)
    if bundle.is_symlink() or not bundle.is_file():
        raise EvidenceError("bundle missing or symlinked")
    if bundle.stat().st_size != index["bundle_size"] or sha256_file(bundle) != index["bundle_sha256"]:
        raise EvidenceError("bundle size or hash mismatch")
    with tarfile.open(bundle, mode="r:gz") as archive:
        members = _tar_members(archive)
        embedded = members.get(MANIFEST_MEMBER)
        stream = archive.extractfile(embedded) if embedded is not None else None
        if stream is None:
            raise EvidenceError("embedded manifest missing")
        try:
            document = json.loads(stream.read())
        except json.JSONDecodeError as exc:
            raise EvidenceError("embedded manifest is invalid JSON") from exc
    if not isinstance(document, dict):
        raise EvidenceError("embedded manifest is not an object")
    if (
        document.get("bundle_id") != bundle_id
        or canonical_sha256(document) != index["manifest_sha256"]
        or document.get("entry_count") != index["entry_count"]
        or document.get("campaign_states") != index.get("campaign_states")
    ):
        raise EvidenceError("index/manifest identity or state mismatch")
    _verify_bundle(bundle, document)
    _validate_release_contract(document, _validate_manifest(document))
    return {
        "ok": True,
        "entries": index["entry_count"],
        "bundle_id": bundle_id,
        "bundle_sha256": index["bundle_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dry_parser = sub.add_parser("dry-run")
    dry_parser.add_argument("--name", default=DEFAULT_NAME)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--name", default=DEFAULT_NAME)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "dry-run":
        document = manifest(selection(), bundle_id=args.name)
        _validate_release_contract(document, _validate_manifest(document))
        print(
            json.dumps(
                {
                    key: document[key]
                    for key in (
                        "bundle_id",
                        "entry_count",
                        "total_payload_bytes",
                        "category_counts",
                        "campaign_states",
                        "selection_policy",
                    )
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = build(args.name) if args.command == "build" else verify(args.index)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
