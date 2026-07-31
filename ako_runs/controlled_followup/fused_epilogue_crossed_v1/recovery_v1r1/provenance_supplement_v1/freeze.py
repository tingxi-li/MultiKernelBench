#!/usr/bin/env python3
"""Create or verify the immutable provenance-supplement lock."""
from __future__ import annotations

import argparse
import json
import tarfile
from typing import Any

try:
    from . import common
except ImportError:  # pragma: no cover - direct-script CLI
    import common  # type: ignore


SOURCE_RELATIVES = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/provenance_supplement_v1/__init__.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/provenance_supplement_v1/README.md",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/provenance_supplement_v1/build.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/provenance_supplement_v1/common.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/provenance_supplement_v1/freeze.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/provenance_supplement_v1/tests/__init__.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/provenance_supplement_v1/tests/test_supplement.py",
)


def _hash_files(names: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        path = common.repo_file(name)
        common.require(path.is_file() and not path.is_symlink(), f"supplement source missing/symlinked: {name}")
        result[name] = common.file_sha256(path)
    return result


def _primary_manifest() -> dict[str, Any]:
    with tarfile.open(common.PRIMARY_BUNDLE_PATH, "r:gz") as archive:
        members = archive.getmembers()
        common.require(
            all(member.isfile() for member in members)
            and len({member.name for member in members}) == len(members),
            "primary archive has nonregular or duplicate members",
        )
        for member in members:
            common.validate_member_name(member.name)
        selected = [member for member in members if member.name == common.PRIMARY_MANIFEST_MEMBER]
        common.require(len(selected) == 1, "primary manifest member differs")
        stream = archive.extractfile(selected[0])
        common.require(stream is not None, "primary manifest unreadable")
        raw = stream.read()
    common.require(common.bytes_sha256(raw) == common.PRIMARY_MANIFEST_RAW_SHA256, "primary manifest raw bytes changed")
    value = common.parse_json_bytes(raw, "primary manifest")
    common.require(
        common.canonical_sha256(value) == common.PRIMARY_MANIFEST_CANONICAL_SHA256
        and value.get("complete") is True
        and value.get("record_type") == "fused_crossed_v1r1_evidence_manifest",
        "primary manifest identity changed",
    )
    entries = value.get("entries")
    common.require(isinstance(entries, list) and len(entries) == common.PRIMARY_ENTRY_COUNT, "primary manifest entry count changed")
    return value


def _role_maps(parent: dict[str, Any], recovery: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[str]]]:
    specifications = (
        (parent, "source_sha256", "source_bundle_sha256", "parent_source"),
        (parent, "dependency_sha256", "dependency_bundle_sha256", "parent_dependency"),
        (recovery, "source_sha256", "source_bundle_sha256", "recovery_source"),
        (recovery, "dependency_sha256", "dependency_bundle_sha256", "recovery_dependency"),
    )
    maps: list[dict[str, str]] = []
    roles: dict[str, set[str]] = {}
    for lock, map_key, bundle_key, role in specifications:
        mapping = common.verify_lock_map(lock, map_key, bundle_key, role)
        maps.append(mapping)
        for name in mapping:
            roles.setdefault(name, set()).add(role)
    merged = common.merge_hash_maps(*maps)
    normalized_roles = {name: sorted(roles[name]) for name in sorted(roles)}
    common.require(set(merged) == set(normalized_roles), "locked role coverage differs")
    return merged, normalized_roles


def expected_lock() -> dict[str, Any]:
    anchors = common.validate_anchor_identities()
    parent = anchors["parent_lock"]
    recovery = anchors["recovery_lock"]
    locked, roles = _role_maps(parent, recovery)
    common.require(len(parent["source_sha256"]) == 17, "parent source count changed")
    common.require(len(parent["dependency_sha256"]) == 18, "parent dependency count changed")
    common.require(len(recovery["source_sha256"]) == 13, "recovery source count changed")
    common.require(len(recovery["dependency_sha256"]) == 13, "recovery dependency count changed")
    common.require(len(locked) == 49, "locked union count changed")
    for name, expected in locked.items():
        path = common.repo_file(name)
        common.require(
            path.is_file() and not path.is_symlink() and common.file_sha256(path) == expected,
            f"locked input changed: {name}",
        )

    primary = _primary_manifest()
    primary_entries = primary["entries"]
    primary_paths = {entry.get("path") for entry in primary_entries if isinstance(entry, dict)}
    common.require(len(primary_paths) == len(primary_entries), "primary paths are invalid or duplicated")
    parent_paths = set(parent["source_sha256"]) | set(parent["dependency_sha256"])
    recovery_paths = set(recovery["source_sha256"]) | set(recovery["dependency_sha256"])
    parent_missing = sorted(parent_paths - primary_paths)
    recovery_missing = sorted(recovery_paths - primary_paths)
    common.require(len(parent_missing) == 23, "primary parent-lock gap count changed")
    common.require(not recovery_missing, "primary bundle is missing recovery-locked inputs")

    sources = _hash_files(SOURCE_RELATIVES)
    return {
        "campaign_id": common.CAMPAIGN_ID,
        "coverage": {
            "locked_union_count": len(locked),
            "parent_dependency_count": len(parent["dependency_sha256"]),
            "parent_missing_from_primary_count": len(parent_missing),
            "parent_missing_from_primary_paths": parent_missing,
            "parent_source_count": len(parent["source_sha256"]),
            "recovery_dependency_count": len(recovery["dependency_sha256"]),
            "recovery_missing_from_primary_count": len(recovery_missing),
            "recovery_source_count": len(recovery["source_sha256"]),
        },
        "final_summary": {
            "path": common.repo_path(common.FINAL_SUMMARY_PATH),
            "sha256": common.FINAL_SUMMARY_SHA256,
            "size": common.FINAL_SUMMARY_PATH.stat().st_size,
        },
        "locked_file_bundle_sha256": common.canonical_sha256(locked),
        "locked_file_sha256": locked,
        "locked_path_roles": roles,
        "parent": {
            "dependency_bundle_sha256": common.PARENT_DEPENDENCY_BUNDLE_SHA256,
            "git_commit": common.PARENT_GIT_COMMIT,
            "launch_lock_path": common.repo_path(common.PARENT_LOCK_PATH),
            "launch_lock_sha256": common.PARENT_LOCK_SHA256,
            "source_bundle_sha256": common.PARENT_SOURCE_BUNDLE_SHA256,
        },
        "primary_evidence": {
            "bundle_path": common.repo_path(common.PRIMARY_BUNDLE_PATH),
            "bundle_sha256": common.PRIMARY_BUNDLE_SHA256,
            "bundle_size": common.PRIMARY_BUNDLE_SIZE,
            "entry_count": common.PRIMARY_ENTRY_COUNT,
            "index_path": common.repo_path(common.PRIMARY_INDEX_PATH),
            "index_sha256": common.PRIMARY_INDEX_SHA256,
            "index_size": common.PRIMARY_INDEX_PATH.stat().st_size,
            "manifest_canonical_sha256": common.PRIMARY_MANIFEST_CANONICAL_SHA256,
            "manifest_raw_sha256": common.PRIMARY_MANIFEST_RAW_SHA256,
            "uncompressed_entry_bytes": sum(int(entry["size"]) for entry in primary_entries),
        },
        "record_type": "fused_crossed_v1r1_provenance_supplement_lock",
        "recovery": {
            "dependency_bundle_sha256": common.RECOVERY_DEPENDENCY_BUNDLE_SHA256,
            "git_commit": common.RECOVERY_GIT_COMMIT,
            "recovery_id": common.RECOVERY_ID,
            "recovery_lock_path": common.repo_path(common.RECOVERY_LOCK_PATH),
            "recovery_lock_sha256": common.RECOVERY_LOCK_SHA256,
            "source_bundle_sha256": common.RECOVERY_SOURCE_BUNDLE_SHA256,
        },
        "result_tag": common.RESULT_TAG,
        "schema_version": 1,
        "source_bundle_sha256": common.canonical_sha256(sources),
        "source_sha256": sources,
        "supplement_id": common.SUPPLEMENT_ID,
        "supplement_scope": "provenance_self_containment_only",
    }


def verify() -> dict[str, Any]:
    common.repo_path(common.SUPPLEMENT_LOCK_PATH)
    common.require(common.SUPPLEMENT_LOCK_PATH.is_file(), "supplement lock is missing")
    common.require(not common.SUPPLEMENT_LOCK_PATH.is_symlink(), "supplement lock is symlinked")
    try:
        raw = common.SUPPLEMENT_LOCK_PATH.read_bytes()
    except OSError as exc:
        raise common.SupplementError(f"cannot read supplement lock: {exc}") from exc
    observed = common.parse_json_bytes(raw, "supplement lock")
    common.require(raw == common.stable_json_bytes(observed), "supplement lock is not canonical stable JSON")
    common.require(observed == expected_lock(), "supplement lock differs from current protected bytes")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        common.require(not common.SUPPLEMENT_LOCK_PATH.exists(), "supplement lock already exists")
        common.exclusive_bytes(common.SUPPLEMENT_LOCK_PATH, common.stable_json_bytes(expected_lock()))
    value = verify()
    print(
        json.dumps(
            {
                "locked_files": len(value["locked_file_sha256"]),
                "parent_gap_closed": value["coverage"]["parent_missing_from_primary_count"],
                "source_bundle_sha256": value["source_bundle_sha256"],
                "supplement_lock_sha256": common.file_sha256(common.SUPPLEMENT_LOCK_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
