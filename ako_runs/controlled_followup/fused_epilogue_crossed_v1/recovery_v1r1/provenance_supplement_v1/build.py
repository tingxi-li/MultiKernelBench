#!/usr/bin/env python3
"""Build or independently verify the crossed-v1r1 provenance supplement."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

try:
    from . import common, freeze
except ImportError:  # pragma: no cover - direct-script CLI
    import common  # type: ignore
    import freeze  # type: ignore


OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
KNOWN_ROLES = {
    "final_summary",
    "parent_dependency",
    "parent_launch_lock_anchor",
    "parent_source",
    "primary_evidence_bundle",
    "primary_evidence_index",
    "recovery_dependency",
    "recovery_lock_anchor",
    "recovery_source",
    "supplement_lock",
    "supplement_source",
}


@dataclass(frozen=True)
class Snapshot:
    path: Path
    name: str
    data: bytes
    roles: tuple[str, ...]
    sha256: str

    @property
    def size(self) -> int:
        return len(self.data)


class Selection:
    def __init__(self) -> None:
        self._values: dict[str, Snapshot] = {}

    def add(self, path: Path, roles: list[str] | tuple[str, ...] | set[str], expected_sha256: str) -> None:
        name = common.repo_path(path)
        common.validate_member_name(name)
        normalized_roles = tuple(sorted(set(roles)))
        common.require(normalized_roles and set(normalized_roles) <= KNOWN_ROLES, f"invalid roles for {name}")
        common.validate_hash(expected_sha256, name)
        common.require(path.is_file() and not path.is_symlink(), f"selected input missing/symlinked: {name}")
        data = path.read_bytes()
        observed = common.bytes_sha256(data)
        common.require(observed == expected_sha256, f"selected input hash changed: {name}")
        if name in self._values:
            prior = self._values[name]
            common.require(prior.data == data and prior.sha256 == observed, f"conflicting selected bytes: {name}")
            normalized_roles = tuple(sorted(set(prior.roles) | set(normalized_roles)))
        self._values[name] = Snapshot(path, name, data, normalized_roles, observed)

    def values(self) -> list[Snapshot]:
        return [self._values[name] for name in sorted(self._values)]

    def names(self) -> set[str]:
        return set(self._values)

    def revalidate(self) -> None:
        for value in self.values():
            common.require(common.repo_path(value.path) == value.name, f"protected input path changed: {value.name}")
            common.require(
                value.path.is_file()
                and not value.path.is_symlink()
                and common.file_sha256(value.path) == value.sha256
                and value.path.stat().st_size == value.size,
                f"protected input changed during construction: {value.name}",
            )


def _sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def normalized_info(name: str, size: int) -> tarfile.TarInfo:
    common.validate_member_name(name)
    common.require(isinstance(size, int) and size >= 0, f"invalid member size: {name}")
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _assert_normalized(member: tarfile.TarInfo, label: str) -> None:
    common.require(
        member.isfile()
        and member.mode == 0o644
        and member.mtime == 0
        and member.uid == 0
        and member.gid == 0
        and member.uname == ""
        and member.gname == ""
        and member.linkname == ""
        and member.devmajor == 0
        and member.devminor == 0
        and all(key == "path" and value == member.name for key, value in member.pax_headers.items()),
        f"nondeterministic or nonregular member metadata: {label}:{member.name}",
    )


def _member_map(archive: tarfile.TarFile, label: str) -> dict[str, tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        common.validate_member_name(member.name)
        _assert_normalized(member, label)
    common.require(len({member.name for member in members}) == len(members), f"duplicate {label} members")
    return {member.name: member for member in members}


def _entry_map(entries: Any, label: str) -> dict[str, dict[str, Any]]:
    common.require(isinstance(entries, list), f"{label} entries are not a list")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        common.require(isinstance(entry, dict), f"{label} entry is not an object")
        name = entry.get("path")
        common.validate_member_name(name)
        common.require(name not in result, f"duplicate {label} entry: {name}")
        common.validate_hash(entry.get("sha256"), f"{label}:{name}")
        common.require(isinstance(entry.get("size"), int) and entry["size"] >= 0, f"invalid {label} size: {name}")
        result[name] = entry
    return result


def _binding() -> dict[str, str]:
    return {
        "campaign_id": common.CAMPAIGN_ID,
        "parent_git_commit": common.PARENT_GIT_COMMIT,
        "parent_launch_lock_sha256": common.PARENT_LOCK_SHA256,
        "parent_source_bundle_sha256": common.PARENT_SOURCE_BUNDLE_SHA256,
        "parent_dependency_bundle_sha256": common.PARENT_DEPENDENCY_BUNDLE_SHA256,
        "recovery_git_commit": common.RECOVERY_GIT_COMMIT,
        "recovery_id": common.RECOVERY_ID,
        "recovery_lock_sha256": common.RECOVERY_LOCK_SHA256,
        "recovery_source_bundle_sha256": common.RECOVERY_SOURCE_BUNDLE_SHA256,
        "recovery_dependency_bundle_sha256": common.RECOVERY_DEPENDENCY_BUNDLE_SHA256,
        "result_tag": common.RESULT_TAG,
    }


def _primary_binding() -> dict[str, str]:
    """Binding fields present in the already sealed primary evidence."""
    return {
        "campaign_id": common.CAMPAIGN_ID,
        "parent_launch_lock_sha256": common.PARENT_LOCK_SHA256,
        "parent_source_bundle_sha256": common.PARENT_SOURCE_BUNDLE_SHA256,
        "recovery_git_commit": common.RECOVERY_GIT_COMMIT,
        "recovery_id": common.RECOVERY_ID,
        "recovery_lock_sha256": common.RECOVERY_LOCK_SHA256,
        "recovery_source_bundle_sha256": common.RECOVERY_SOURCE_BUNDLE_SHA256,
        "result_tag": common.RESULT_TAG,
    }


def selection() -> tuple[Selection, dict[str, Any]]:
    lock = freeze.verify()
    common.validate_anchor_identities()
    selected = Selection()
    roles = lock["locked_path_roles"]
    for name, expected in lock["locked_file_sha256"].items():
        selected.add(common.repo_file(name), roles[name], expected)

    selected.add(common.PARENT_LOCK_PATH, {"parent_launch_lock_anchor"}, common.PARENT_LOCK_SHA256)
    selected.add(common.RECOVERY_LOCK_PATH, {"recovery_lock_anchor"}, common.RECOVERY_LOCK_SHA256)
    selected.add(common.FINAL_SUMMARY_PATH, {"final_summary"}, common.FINAL_SUMMARY_SHA256)
    selected.add(common.PRIMARY_INDEX_PATH, {"primary_evidence_index"}, common.PRIMARY_INDEX_SHA256)
    selected.add(common.PRIMARY_BUNDLE_PATH, {"primary_evidence_bundle"}, common.PRIMARY_BUNDLE_SHA256)
    selected.add(
        common.SUPPLEMENT_LOCK_PATH,
        {"supplement_lock"},
        common.file_sha256(common.SUPPLEMENT_LOCK_PATH),
    )
    for name, expected in lock["source_sha256"].items():
        selected.add(common.repo_file(name), {"supplement_source"}, expected)

    expected_count = 49 + 1 + 1 + 2 + 1 + len(lock["source_sha256"])
    common.require(len(selected.values()) == expected_count, "supplement selection count changed")
    common.require(set(lock["locked_file_sha256"]) <= selected.names(), "locked union is not fully selected")
    return selected, lock


def manifest(selected: Selection, lock: dict[str, Any]) -> dict[str, Any]:
    entries = [
        {
            "path": value.name,
            "roles": list(value.roles),
            "sha256": value.sha256,
            "size": value.size,
        }
        for value in selected.values()
    ]
    return {
        **_binding(),
        "coverage": {
            **lock["coverage"],
            "embedded_locked_paths": len(lock["locked_file_sha256"]),
            "experimental_results_modified": False,
            "supplement_payload_entries": len(entries),
        },
        "entries": entries,
        "experimental_results_modified": False,
        "final_summary": lock["final_summary"],
        "locked_file_bundle_sha256": lock["locked_file_bundle_sha256"],
        "parent": lock["parent"],
        "primary_evidence": lock["primary_evidence"],
        "record_type": "fused_crossed_v1r1_provenance_supplement_manifest",
        "recovery": lock["recovery"],
        "schema_version": 1,
        "supplement_id": common.SUPPLEMENT_ID,
        "supplement_lock": {
            "path": common.repo_path(common.SUPPLEMENT_LOCK_PATH),
            "sha256": common.file_sha256(common.SUPPLEMENT_LOCK_PATH),
            "source_bundle_sha256": lock["source_bundle_sha256"],
            "source_count": len(lock["source_sha256"]),
        },
        "supplement_scope": "provenance_self_containment_only",
    }


def _archive_payload_bytes(document: dict[str, Any], payloads: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    manifest_data = common.stable_json_bytes(document)
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            archive.addfile(
                normalized_info(common.SUPPLEMENT_MANIFEST_MEMBER, len(manifest_data)),
                io.BytesIO(manifest_data),
            )
            for name in sorted(payloads):
                data = payloads[name]
                archive.addfile(normalized_info(name, len(data)), io.BytesIO(data))
    return output.getvalue()


def _archive_bytes(document: dict[str, Any], selected: Selection) -> bytes:
    return _archive_payload_bytes(
        document,
        {value.name: value.data for value in selected.values()},
    )


def _validate_binding(value: dict[str, Any], label: str) -> None:
    expected = _binding()
    differences = [key for key, item in expected.items() if value.get(key) != item]
    common.require(not differences, f"{label} binding differs: {differences}")


def _validate_primary_binding(value: dict[str, Any], label: str) -> None:
    expected = _primary_binding()
    differences = [key for key, item in expected.items() if value.get(key) != item]
    common.require(not differences, f"{label} binding differs: {differences}")


def _verify_primary(index_data: bytes, bundle_data: bytes) -> tuple[dict[str, Any], dict[str, Any], bytes, set[str]]:
    index = common.parse_json_bytes(index_data, "embedded primary index")
    common.require(common.bytes_sha256(index_data) == common.PRIMARY_INDEX_SHA256, "embedded primary index hash differs")
    common.require(
        index.get("record_type") == "fused_crossed_v1r1_evidence_index"
        and index.get("entry_count") == common.PRIMARY_ENTRY_COUNT
        and index.get("bundle_sha256") == common.PRIMARY_BUNDLE_SHA256
        and index.get("bundle_size") == common.PRIMARY_BUNDLE_SIZE
        and index.get("manifest_sha256") == common.PRIMARY_MANIFEST_CANONICAL_SHA256,
        "embedded primary index identity differs",
    )
    _validate_primary_binding(index, "embedded primary index")
    common.require(
        len(bundle_data) == common.PRIMARY_BUNDLE_SIZE
        and common.bytes_sha256(bundle_data) == common.PRIMARY_BUNDLE_SHA256,
        "embedded primary bundle size/hash differs",
    )

    with tarfile.open(fileobj=io.BytesIO(bundle_data), mode="r:gz") as archive:
        members = _member_map(archive, "embedded primary")
        manifest_member = members.pop(common.PRIMARY_MANIFEST_MEMBER, None)
        common.require(manifest_member is not None, "embedded primary manifest is absent")
        stream = archive.extractfile(manifest_member)
        common.require(stream is not None, "embedded primary manifest unreadable")
        raw_manifest = stream.read()
        common.require(common.bytes_sha256(raw_manifest) == common.PRIMARY_MANIFEST_RAW_SHA256, "embedded primary manifest raw hash differs")
        primary_manifest = common.parse_json_bytes(raw_manifest, "embedded primary manifest")
        common.require(
            common.canonical_sha256(primary_manifest) == common.PRIMARY_MANIFEST_CANONICAL_SHA256
            and primary_manifest.get("record_type") == "fused_crossed_v1r1_evidence_manifest"
            and primary_manifest.get("complete") is True,
            "embedded primary manifest identity differs",
        )
        _validate_primary_binding(primary_manifest, "embedded primary manifest")
        entries = _entry_map(primary_manifest.get("entries"), "embedded primary")
        common.require(len(entries) == common.PRIMARY_ENTRY_COUNT and set(entries) == set(members), "embedded primary membership differs")
        final_data: bytes | None = None
        final_name = common.repo_path(common.FINAL_SUMMARY_PATH)
        for name, entry in entries.items():
            member = members[name]
            common.require(member.size == entry["size"], f"embedded primary member size differs: {name}")
            stream = archive.extractfile(member)
            common.require(stream is not None, f"embedded primary member unreadable: {name}")
            if name == final_name:
                data = stream.read()
                digest, size = common.bytes_sha256(data), len(data)
                final_data = data
            else:
                digest, size = _sha256_stream(stream)
            common.require(digest == entry["sha256"] and size == entry["size"], f"embedded primary payload differs: {name}")
    common.require(final_data is not None, "embedded primary final summary is absent")
    common.require(
        primary_manifest.get("summary_path") == common.repo_path(common.FINAL_SUMMARY_PATH)
        and primary_manifest.get("summary_sha256") == common.FINAL_SUMMARY_SHA256,
        "embedded primary summary binding differs",
    )
    return index, primary_manifest, final_data, set(entries)


def _role_union(parent: dict[str, Any], recovery: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[str]]]:
    specifications = (
        (parent, "source_sha256", "source_bundle_sha256", "parent_source"),
        (parent, "dependency_sha256", "dependency_bundle_sha256", "parent_dependency"),
        (recovery, "source_sha256", "source_bundle_sha256", "recovery_source"),
        (recovery, "dependency_sha256", "dependency_bundle_sha256", "recovery_dependency"),
    )
    maps: list[dict[str, str]] = []
    roles: dict[str, set[str]] = {}
    for source, map_key, bundle_key, role in specifications:
        mapping = common.verify_lock_map(source, map_key, bundle_key, role)
        maps.append(mapping)
        for name in mapping:
            roles.setdefault(name, set()).add(role)
    return common.merge_hash_maps(*maps), {name: sorted(value) for name, value in sorted(roles.items())}


def _verify_outer(index: dict[str, Any], bundle_data: bytes) -> dict[str, Any]:
    common.require(
        index.get("schema_version") == 1
        and index.get("record_type") == "fused_crossed_v1r1_provenance_supplement_index"
        and index.get("supplement_id") == common.SUPPLEMENT_ID
        and index.get("experimental_results_modified") is False,
        "supplement index identity differs",
    )
    _validate_binding(index, "supplement index")
    bundle_name_claim = index.get("bundle_path")
    common.validate_member_name(bundle_name_claim)
    bundle_name_value = PurePosixPath(bundle_name_claim)
    common.require(
        bundle_name_value.parent.as_posix() == common.EVIDENCE_ROOT_RELATIVE
        and bundle_name_value.name.endswith(".tar.gz")
        and OUTPUT_NAME_RE.fullmatch(bundle_name_value.name[: -len(".tar.gz")]) is not None,
        "supplement index bundle path differs",
    )
    common.require(
        len(bundle_data) == index.get("bundle_size")
        and common.bytes_sha256(bundle_data) == index.get("bundle_sha256"),
        "supplement bundle size/hash differs",
    )
    with tarfile.open(fileobj=io.BytesIO(bundle_data), mode="r:gz") as archive:
        members = _member_map(archive, "supplement")
        manifest_member = members.pop(common.SUPPLEMENT_MANIFEST_MEMBER, None)
        common.require(manifest_member is not None, "supplement manifest is absent")
        stream = archive.extractfile(manifest_member)
        common.require(stream is not None, "supplement manifest unreadable")
        raw_manifest = stream.read()
        manifest_value = common.parse_json_bytes(raw_manifest, "supplement manifest")
        common.require(
            common.canonical_sha256(manifest_value) == index.get("manifest_canonical_sha256")
            and common.bytes_sha256(raw_manifest) == index.get("manifest_raw_sha256")
            and manifest_value.get("record_type") == "fused_crossed_v1r1_provenance_supplement_manifest"
            and manifest_value.get("supplement_id") == common.SUPPLEMENT_ID
            and manifest_value.get("supplement_scope") == "provenance_self_containment_only"
            and manifest_value.get("experimental_results_modified") is False,
            "supplement manifest identity differs",
        )
        _validate_binding(manifest_value, "supplement manifest")
        entries = _entry_map(manifest_value.get("entries"), "supplement")
        common.require(
            len(entries) == index.get("entry_count") and set(entries) == set(members),
            "supplement membership differs",
        )
        payloads: dict[str, bytes] = {}
        for name, entry in entries.items():
            member = members[name]
            common.require(member.size == entry["size"], f"supplement member size differs: {name}")
            stream = archive.extractfile(member)
            common.require(stream is not None, f"supplement member unreadable: {name}")
            data = stream.read()
            common.require(
                len(data) == entry["size"] and common.bytes_sha256(data) == entry["sha256"],
                f"supplement payload differs: {name}",
            )
            roles = entry.get("roles")
            common.require(
                isinstance(roles, list)
                and roles == sorted(set(roles))
                and roles
                and set(roles) <= KNOWN_ROLES,
                f"supplement roles differ: {name}",
            )
            payloads[name] = data

    common.require(
        _archive_payload_bytes(manifest_value, payloads) == bundle_data,
        "supplement archive is not the canonical single-member gzip/tar encoding",
    )

    supplement_lock_name = common.SUPPLEMENT_LOCK_RELATIVE
    recovery_lock_name = common.RECOVERY_LOCK_RELATIVE
    parent_lock_name = common.PARENT_LOCK_RELATIVE
    final_name = common.FINAL_SUMMARY_RELATIVE
    primary_index_name = common.PRIMARY_INDEX_RELATIVE
    primary_bundle_name = common.PRIMARY_BUNDLE_RELATIVE
    for name in (
        supplement_lock_name,
        recovery_lock_name,
        parent_lock_name,
        final_name,
        primary_index_name,
        primary_bundle_name,
    ):
        common.require(name in payloads, f"required supplement anchor absent: {name}")

    supplement_lock = common.parse_json_bytes(payloads[supplement_lock_name], "embedded supplement lock")
    parent_lock = common.parse_json_bytes(payloads[parent_lock_name], "embedded parent lock")
    recovery_lock = common.parse_json_bytes(payloads[recovery_lock_name], "embedded recovery lock")
    final_summary = common.parse_json_bytes(payloads[final_name], "embedded final summary")
    supplement_lock_sha256 = common.bytes_sha256(payloads[supplement_lock_name])
    common.require(
        payloads[supplement_lock_name] == common.stable_json_bytes(supplement_lock)
        and supplement_lock.get("record_type") == "fused_crossed_v1r1_provenance_supplement_lock"
        and supplement_lock.get("supplement_id") == common.SUPPLEMENT_ID,
        "embedded supplement lock bytes or identity differ",
    )
    common.require(common.bytes_sha256(payloads[parent_lock_name]) == common.PARENT_LOCK_SHA256, "embedded parent lock hash differs")
    common.require(common.bytes_sha256(payloads[recovery_lock_name]) == common.RECOVERY_LOCK_SHA256, "embedded recovery lock hash differs")
    common.require(common.bytes_sha256(payloads[final_name]) == common.FINAL_SUMMARY_SHA256, "embedded final summary hash differs")

    locked, locked_roles = _role_union(parent_lock, recovery_lock)
    parent_sources = common.verify_lock_map(
        parent_lock, "source_sha256", "source_bundle_sha256", "parent source"
    )
    parent_dependencies = common.verify_lock_map(
        parent_lock, "dependency_sha256", "dependency_bundle_sha256", "parent dependency"
    )
    recovery_sources = common.verify_lock_map(
        recovery_lock, "source_sha256", "source_bundle_sha256", "recovery source"
    )
    recovery_dependencies = common.verify_lock_map(
        recovery_lock, "dependency_sha256", "dependency_bundle_sha256", "recovery dependency"
    )
    common.require(len(locked) == 49, "embedded locked union count differs")
    for name, expected in locked.items():
        common.require(name in payloads and common.bytes_sha256(payloads[name]) == expected, f"embedded locked file differs: {name}")

    source_value = supplement_lock.get("source_sha256")
    common.require(isinstance(source_value, dict), "supplement source map is absent")
    source_map = common.merge_hash_maps(source_value)
    common.require(
        source_map == source_value
        and set(source_map) == set(freeze.SOURCE_RELATIVES)
        and len(source_map) == 7
        and common.canonical_sha256(source_map) == supplement_lock.get("source_bundle_sha256"),
        "supplement source lock differs",
    )
    for name, expected in source_map.items():
        common.require(name in payloads and common.bytes_sha256(payloads[name]) == expected, f"embedded supplement source differs: {name}")

    expected_roles = {name: set(values) for name, values in locked_roles.items()}
    expected_roles.setdefault(parent_lock_name, set()).add("parent_launch_lock_anchor")
    expected_roles[recovery_lock_name] = {"recovery_lock_anchor"}
    expected_roles[final_name] = {"final_summary"}
    expected_roles[primary_index_name] = {"primary_evidence_index"}
    expected_roles[primary_bundle_name] = {"primary_evidence_bundle"}
    expected_roles[supplement_lock_name] = {"supplement_lock"}
    for name in source_map:
        expected_roles[name] = {"supplement_source"}
    common.require(set(expected_roles) == set(entries), "supplement role membership differs")
    for name, roles in expected_roles.items():
        common.require(entries[name]["roles"] == sorted(roles), f"supplement role assignment differs: {name}")

    primary_index, primary_manifest, nested_final, primary_paths = _verify_primary(
        payloads[primary_index_name], payloads[primary_bundle_name]
    )
    common.require(payloads[final_name] == nested_final, "outer and primary final-summary bytes differ")
    _validate_primary_binding(final_summary, "embedded final summary")
    common.require(
        final_summary.get("record_type") == "fused_crossed_final_summary"
        and final_summary.get("complete") is True,
        "embedded final summary is incomplete",
    )
    parent_paths = set(parent_lock["source_sha256"]) | set(parent_lock["dependency_sha256"])
    recovery_paths = set(recovery_lock["source_sha256"]) | set(recovery_lock["dependency_sha256"])
    parent_missing = sorted(parent_paths - primary_paths)
    recovery_missing = sorted(recovery_paths - primary_paths)

    expected_coverage = {
        "locked_union_count": len(locked),
        "parent_dependency_count": len(parent_dependencies),
        "parent_missing_from_primary_count": len(parent_missing),
        "parent_missing_from_primary_paths": parent_missing,
        "parent_source_count": len(parent_sources),
        "recovery_dependency_count": len(recovery_dependencies),
        "recovery_missing_from_primary_count": len(recovery_missing),
        "recovery_source_count": len(recovery_sources),
    }
    common.require(
        expected_coverage["parent_source_count"] == 17
        and expected_coverage["parent_dependency_count"] == 18
        and expected_coverage["recovery_source_count"] == 13
        and expected_coverage["recovery_dependency_count"] == 13
        and expected_coverage["locked_union_count"] == 49
        and expected_coverage["parent_missing_from_primary_count"] == 23
        and not recovery_missing,
        "primary-gap closure proof differs",
    )

    expected_parent = {
        "dependency_bundle_sha256": common.PARENT_DEPENDENCY_BUNDLE_SHA256,
        "git_commit": common.PARENT_GIT_COMMIT,
        "launch_lock_path": common.PARENT_LOCK_RELATIVE,
        "launch_lock_sha256": common.PARENT_LOCK_SHA256,
        "source_bundle_sha256": common.PARENT_SOURCE_BUNDLE_SHA256,
    }
    expected_recovery = {
        "dependency_bundle_sha256": common.RECOVERY_DEPENDENCY_BUNDLE_SHA256,
        "git_commit": common.RECOVERY_GIT_COMMIT,
        "recovery_id": common.RECOVERY_ID,
        "recovery_lock_path": common.RECOVERY_LOCK_RELATIVE,
        "recovery_lock_sha256": common.RECOVERY_LOCK_SHA256,
        "source_bundle_sha256": common.RECOVERY_SOURCE_BUNDLE_SHA256,
    }
    expected_final = {
        "path": common.FINAL_SUMMARY_RELATIVE,
        "sha256": common.FINAL_SUMMARY_SHA256,
        "size": len(payloads[final_name]),
    }
    expected_primary = {
        "bundle_path": common.PRIMARY_BUNDLE_RELATIVE,
        "bundle_sha256": common.PRIMARY_BUNDLE_SHA256,
        "bundle_size": len(payloads[primary_bundle_name]),
        "entry_count": len(primary_paths),
        "index_path": common.PRIMARY_INDEX_RELATIVE,
        "index_sha256": common.PRIMARY_INDEX_SHA256,
        "index_size": len(payloads[primary_index_name]),
        "manifest_canonical_sha256": common.canonical_sha256(primary_manifest),
        "manifest_raw_sha256": common.PRIMARY_MANIFEST_RAW_SHA256,
        "uncompressed_entry_bytes": sum(
            int(entry["size"]) for entry in primary_manifest["entries"]
        ),
    }
    expected_lock = {
        "campaign_id": common.CAMPAIGN_ID,
        "coverage": expected_coverage,
        "final_summary": expected_final,
        "locked_file_bundle_sha256": common.canonical_sha256(locked),
        "locked_file_sha256": locked,
        "locked_path_roles": locked_roles,
        "parent": expected_parent,
        "primary_evidence": expected_primary,
        "record_type": "fused_crossed_v1r1_provenance_supplement_lock",
        "recovery": expected_recovery,
        "result_tag": common.RESULT_TAG,
        "schema_version": 1,
        "source_bundle_sha256": common.canonical_sha256(source_map),
        "source_sha256": source_map,
        "supplement_id": common.SUPPLEMENT_ID,
        "supplement_scope": "provenance_self_containment_only",
    }
    common.require(supplement_lock == expected_lock, "embedded supplement lock semantics differ")

    expected_roles = {name: set(values) for name, values in locked_roles.items()}
    for name, role in (
        (parent_lock_name, "parent_launch_lock_anchor"),
        (recovery_lock_name, "recovery_lock_anchor"),
        (final_name, "final_summary"),
        (primary_index_name, "primary_evidence_index"),
        (primary_bundle_name, "primary_evidence_bundle"),
        (supplement_lock_name, "supplement_lock"),
    ):
        expected_roles.setdefault(name, set()).add(role)
    for name in source_map:
        expected_roles.setdefault(name, set()).add("supplement_source")
    expected_entries = [
        {
            "path": name,
            "roles": sorted(expected_roles[name]),
            "sha256": common.bytes_sha256(payloads[name]),
            "size": len(payloads[name]),
        }
        for name in sorted(expected_roles)
    ]
    common.require(
        set(expected_roles) == set(payloads) and len(expected_entries) == 61,
        "supplement exact payload membership differs",
    )
    expected_manifest_coverage = {
        **expected_coverage,
        "embedded_locked_paths": len(locked),
        "experimental_results_modified": False,
        "supplement_payload_entries": len(expected_entries),
    }
    expected_manifest = {
        **_binding(),
        "coverage": expected_manifest_coverage,
        "entries": expected_entries,
        "experimental_results_modified": False,
        "final_summary": expected_final,
        "locked_file_bundle_sha256": common.canonical_sha256(locked),
        "parent": expected_parent,
        "primary_evidence": expected_primary,
        "record_type": "fused_crossed_v1r1_provenance_supplement_manifest",
        "recovery": expected_recovery,
        "schema_version": 1,
        "supplement_id": common.SUPPLEMENT_ID,
        "supplement_lock": {
            "path": common.SUPPLEMENT_LOCK_RELATIVE,
            "sha256": supplement_lock_sha256,
            "source_bundle_sha256": common.canonical_sha256(source_map),
            "source_count": len(source_map),
        },
        "supplement_scope": "provenance_self_containment_only",
    }
    common.require(manifest_value == expected_manifest, "supplement manifest semantics differ")

    expected_index = {
        **_binding(),
        "bundle_path": bundle_name_claim,
        "bundle_sha256": common.bytes_sha256(bundle_data),
        "bundle_size": len(bundle_data),
        "entry_count": len(expected_entries),
        "experimental_results_modified": False,
        "final_summary_sha256": common.FINAL_SUMMARY_SHA256,
        "locked_file_bundle_sha256": common.canonical_sha256(locked),
        "locked_file_count": len(locked),
        "manifest_canonical_sha256": common.canonical_sha256(expected_manifest),
        "manifest_raw_sha256": common.bytes_sha256(common.stable_json_bytes(expected_manifest)),
        "parent_gap_closed": len(parent_missing),
        "primary_bundle_sha256": common.PRIMARY_BUNDLE_SHA256,
        "primary_index_sha256": common.PRIMARY_INDEX_SHA256,
        "record_type": "fused_crossed_v1r1_provenance_supplement_index",
        "schema_version": 1,
        "supplement_id": common.SUPPLEMENT_ID,
        "supplement_lock_sha256": supplement_lock_sha256,
    }
    common.require(index == expected_index, "supplement index semantics differ")
    return {
        "bundle_sha256": index["bundle_sha256"],
        "entries": index["entry_count"],
        "locked_files": len(locked),
        "parent_gap_closed": len(parent_missing),
        "primary_entries": len(primary_paths),
        "ok": True,
    }


def _index(document: dict[str, Any], bundle_path: Path, bundle_data: bytes) -> dict[str, Any]:
    manifest_data = common.stable_json_bytes(document)
    return {
        **_binding(),
        "bundle_path": common.repo_path(bundle_path),
        "bundle_sha256": common.bytes_sha256(bundle_data),
        "bundle_size": len(bundle_data),
        "entry_count": len(document["entries"]),
        "experimental_results_modified": False,
        "final_summary_sha256": common.FINAL_SUMMARY_SHA256,
        "locked_file_bundle_sha256": document["locked_file_bundle_sha256"],
        "locked_file_count": document["coverage"]["locked_union_count"],
        "manifest_canonical_sha256": common.canonical_sha256(document),
        "manifest_raw_sha256": common.bytes_sha256(manifest_data),
        "parent_gap_closed": document["coverage"]["parent_missing_from_primary_count"],
        "primary_bundle_sha256": common.PRIMARY_BUNDLE_SHA256,
        "primary_index_sha256": common.PRIMARY_INDEX_SHA256,
        "record_type": "fused_crossed_v1r1_provenance_supplement_index",
        "schema_version": 1,
        "supplement_id": common.SUPPLEMENT_ID,
        "supplement_lock_sha256": document["supplement_lock"]["sha256"],
    }


def build(out_prefix: Path = common.DEFAULT_OUTPUT_PREFIX) -> dict[str, Any]:
    out_prefix = common.lexical_absolute(out_prefix)
    common.repo_path(out_prefix)
    common.require(
        out_prefix.parent == common.lexical_absolute(common.EVIDENCE_ROOT)
        and not out_prefix.suffix
        and OUTPUT_NAME_RE.fullmatch(out_prefix.name) is not None,
        "supplement output prefix must be suffix-free under evidence_v3",
    )
    bundle_path = out_prefix.with_suffix(".tar.gz")
    index_path = out_prefix.with_suffix(".index.json")
    common.require(
        not bundle_path.exists()
        and not bundle_path.is_symlink()
        and not index_path.exists()
        and not index_path.is_symlink(),
        "supplement output already exists",
    )
    selected, lock = selection()
    document = manifest(selected, lock)
    bundle_data = _archive_bytes(document, selected)
    index = _index(document, bundle_path, bundle_data)
    result = _verify_outer(index, bundle_data)
    selected.revalidate()
    common.exclusive_bytes(bundle_path, bundle_data)
    common.exclusive_bytes(index_path, common.stable_json_bytes(index))
    common.require(verify(index_path) == result, "published supplement verification changed")
    return result


def verify(index_path: Path | None = None) -> dict[str, Any]:
    index_path = common.require_no_symlink_components(
        index_path or common.DEFAULT_OUTPUT_PREFIX.with_suffix(".index.json")
    )
    common.require(index_path.is_file() and not index_path.is_symlink(), "supplement index missing/symlinked")
    try:
        index_data = index_path.read_bytes()
    except OSError as exc:
        raise common.SupplementError(f"cannot read supplement index {index_path}: {exc}") from exc
    index = common.parse_json_bytes(index_data, "supplement index")
    common.require(
        index_data == common.stable_json_bytes(index),
        "supplement index is not canonical stable JSON",
    )
    bundle_name = index.get("bundle_path")
    common.validate_member_name(bundle_name)
    bundle_path = common.require_no_symlink_components(
        index_path.parent / PurePosixPath(bundle_name).name
    )
    common.require(bundle_path.is_file() and not bundle_path.is_symlink(), "supplement bundle missing/symlinked")
    return _verify_outer(index, bundle_path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--out-prefix", type=Path, default=common.DEFAULT_OUTPUT_PREFIX)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--index",
        type=Path,
        default=common.DEFAULT_OUTPUT_PREFIX.with_suffix(".index.json"),
    )
    args = parser.parse_args()
    result = build(args.out_prefix) if args.command == "build" else verify(args.index)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
