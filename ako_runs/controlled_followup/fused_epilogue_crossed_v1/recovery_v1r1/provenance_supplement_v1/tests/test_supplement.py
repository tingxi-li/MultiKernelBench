from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.provenance_supplement_v1 import (
    build,
    common,
    freeze,
)


HERE = Path(__file__).resolve().parents[1]


@contextmanager
def _mock_published_lock(monkeypatch: pytest.MonkeyPatch):
    expected_lock = freeze.expected_lock()
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        path = Path(temporary) / "supplement_lock.json"
        path.write_bytes(common.stable_json_bytes(expected_lock))
        monkeypatch.setattr(common, "SUPPLEMENT_LOCK_PATH", path)
        monkeypatch.setattr(common, "SUPPLEMENT_LOCK_RELATIVE", common.repo_path(path))
        monkeypatch.setattr(freeze, "verify", lambda: expected_lock)
        yield expected_lock


def test_anchor_identities_and_primary_pair_are_unchanged():
    state = common.validate_anchor_identities()
    assert state["final_summary"]["complete"] is True
    assert state["primary_index"]["entry_count"] == 1644
    primary_index, primary_manifest, final_data, paths = build._verify_primary(
        common.PRIMARY_INDEX_PATH.read_bytes(),
        common.PRIMARY_BUNDLE_PATH.read_bytes(),
    )
    assert primary_index["bundle_sha256"] == common.PRIMARY_BUNDLE_SHA256
    assert primary_manifest["summary_sha256"] == common.FINAL_SUMMARY_SHA256
    assert hashlib.sha256(final_data).hexdigest() == common.FINAL_SUMMARY_SHA256
    assert len(paths) == common.PRIMARY_ENTRY_COUNT


def test_expected_lock_closes_exact_primary_gap():
    value = freeze.expected_lock()
    coverage = value["coverage"]
    assert coverage["parent_source_count"] == 17
    assert coverage["parent_dependency_count"] == 18
    assert coverage["recovery_source_count"] == 13
    assert coverage["recovery_dependency_count"] == 13
    assert coverage["locked_union_count"] == 49
    assert coverage["parent_missing_from_primary_count"] == 23
    assert coverage["recovery_missing_from_primary_count"] == 0
    assert (
        "ako_runs/phase2_fused_sdpa/runner2.py"
        in coverage["parent_missing_from_primary_paths"]
    )
    assert value["primary_evidence"]["entry_count"] == 1644


def test_all_locked_workspace_bytes_match_the_union():
    value = freeze.expected_lock()
    assert len(value["locked_file_sha256"]) == 49
    for name, expected in value["locked_file_sha256"].items():
        assert common.file_sha256(common.repo_file(name)) == expected
    assert common.canonical_sha256(value["locked_file_sha256"]) == value[
        "locked_file_bundle_sha256"
    ]


@pytest.mark.parametrize(
    "name",
    ("../escape", "/absolute", "a/../b", "a\\b", "./a", "a//b", "a\x00b", ""),
)
def test_unsafe_member_names_are_rejected(name: str):
    with pytest.raises(common.SupplementError):
        common.validate_member_name(name)


def test_normalized_member_metadata():
    info = build.normalized_info("safe/member.json", 17)
    assert info.isfile()
    assert (info.size, info.mode, info.mtime, info.uid, info.gid) == (
        17,
        0o644,
        0,
        0,
        0,
    )
    assert info.uname == info.gname == ""


def test_exclusive_publication_refuses_overwrite():
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        path = Path(temporary) / "value.bin"
        common.exclusive_bytes(path, b"first")
        with pytest.raises(common.SupplementError):
            common.exclusive_bytes(path, b"second")
        assert path.read_bytes() == b"first"


def test_exclusive_publication_preserves_foreign_partial():
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        path = Path(temporary) / "value.bin"
        partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
        partial.write_bytes(b"foreign")
        with pytest.raises(common.SupplementError):
            common.exclusive_bytes(path, b"new")
        assert partial.read_bytes() == b"foreign"


def test_symlinked_selection_is_rejected():
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        link = Path(temporary) / "linked.json"
        link.symlink_to(common.FINAL_SUMMARY_PATH)
        with pytest.raises(common.SupplementError):
            build.Selection().add(link, {"final_summary"}, common.FINAL_SUMMARY_SHA256)


def test_symlinked_repository_path_component_is_rejected():
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        directory = Path(temporary)
        link = directory / "linked_directory"
        link.symlink_to(HERE, target_is_directory=True)
        name = f"{common.repo_path(directory)}/linked_directory/README.md"
        with pytest.raises(common.SupplementError):
            common.repo_file(name)


@pytest.mark.parametrize("module", ("build", "freeze"))
def test_direct_cli_help(module: str):
    completed = subprocess.run(
        [sys.executable, str(HERE / f"{module}.py"), "--help"],
        cwd=common.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_freeze_requires_published_lock():
    value = freeze.verify()
    assert value["supplement_id"] == common.SUPPLEMENT_ID
    assert len(value["source_sha256"]) == 7


def test_selection_and_manifest(monkeypatch: pytest.MonkeyPatch):
    with _mock_published_lock(monkeypatch):
        selected, lock = build.selection()
        assert len(selected.values()) == 61
        assert set(lock["locked_file_sha256"]) <= selected.names()
        document = build.manifest(selected, lock)
        assert document["experimental_results_modified"] is False
        assert document["coverage"]["embedded_locked_paths"] == 49
        entries = {entry["path"]: entry for entry in document["entries"]}
        parent_lock = common.repo_path(common.PARENT_LOCK_PATH)
        assert entries[parent_lock]["roles"] == [
            "parent_launch_lock_anchor",
            "recovery_dependency",
        ]


def test_archive_is_deterministic_and_self_verifying(monkeypatch: pytest.MonkeyPatch):
    with _mock_published_lock(monkeypatch):
        selected, lock = build.selection()
        document = build.manifest(selected, lock)
        first = build._archive_bytes(document, selected)
        second = build._archive_bytes(document, selected)
        assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()
        index = build._index(document, common.DEFAULT_OUTPUT_PREFIX.with_suffix(".tar.gz"), first)
        result = build._verify_outer(index, first)
        assert result == {
            "bundle_sha256": hashlib.sha256(first).hexdigest(),
            "entries": 61,
            "locked_files": 49,
            "parent_gap_closed": 23,
            "primary_entries": 1644,
            "ok": True,
        }
        altered = dict(index)
        altered["locked_file_count"] = 48
        with pytest.raises(common.SupplementError):
            build._verify_outer(altered, first)

        false_claim = copy.deepcopy(document)
        false_claim["coverage"]["parent_source_count"] = 999
        false_bundle = build._archive_bytes(false_claim, selected)
        false_index = build._index(
            false_claim,
            common.DEFAULT_OUTPUT_PREFIX.with_suffix(".tar.gz"),
            false_bundle,
        )
        with pytest.raises(common.SupplementError):
            build._verify_outer(false_index, false_bundle)

        trailing = first + b"UNVALIDATED_TRAILING_BYTES"
        trailing_index = dict(index)
        trailing_index["bundle_sha256"] = hashlib.sha256(trailing).hexdigest()
        trailing_index["bundle_size"] = len(trailing)
        with pytest.raises(common.SupplementError):
            build._verify_outer(trailing_index, trailing)

        changed_header = bytearray(first)
        changed_header[4:8] = (1).to_bytes(4, "little")
        changed_header_index = dict(index)
        changed_header_index["bundle_sha256"] = hashlib.sha256(changed_header).hexdigest()
        changed_header_index["bundle_size"] = len(changed_header)
        with pytest.raises(common.SupplementError):
            build._verify_outer(changed_header_index, bytes(changed_header))

        payloads = {value.name: value.data for value in selected.values()}
        lock_name = common.SUPPLEMENT_LOCK_RELATIVE
        parsed_lock = json.loads(payloads[lock_name])
        compact_lock = json.dumps(
            parsed_lock, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        payloads[lock_name] = compact_lock
        compact_document = copy.deepcopy(document)
        for entry in compact_document["entries"]:
            if entry["path"] == lock_name:
                entry["sha256"] = hashlib.sha256(compact_lock).hexdigest()
                entry["size"] = len(compact_lock)
        compact_document["supplement_lock"]["sha256"] = hashlib.sha256(
            compact_lock
        ).hexdigest()
        compact_bundle = build._archive_payload_bytes(compact_document, payloads)
        compact_index = build._index(
            compact_document,
            common.DEFAULT_OUTPUT_PREFIX.with_suffix(".tar.gz"),
            compact_bundle,
        )
        with pytest.raises(common.SupplementError):
            build._verify_outer(compact_index, compact_bundle)

        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            compact_index_path = Path(temporary) / "supplement.index.json"
            compact_index_path.write_bytes(
                json.dumps(index, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            with pytest.raises(common.SupplementError):
                build.verify(compact_index_path)


def test_primary_payload_tampering_is_rejected():
    data = bytearray(common.PRIMARY_BUNDLE_PATH.read_bytes())
    data[len(data) // 2] ^= 1
    with pytest.raises(common.SupplementError):
        build._verify_primary(common.PRIMARY_INDEX_PATH.read_bytes(), bytes(data))
