"""Shared identities and fail-closed helpers for the provenance supplement."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any


HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parent
PARENT = RECOVERY.parent
REPO_ROOT = HERE.parents[4]

PARENT_LOCK_RELATIVE = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/launch_lock.json"
)
RECOVERY_LOCK_RELATIVE = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/"
    "recovery_v1r1/recovery_lock.json"
)
SUPPLEMENT_LOCK_RELATIVE = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/"
    "recovery_v1r1/provenance_supplement_v1/supplement_lock.json"
)
FINAL_SUMMARY_RELATIVE = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/"
    "results/crossed_v1r1/final_summary.json"
)
EVIDENCE_ROOT_RELATIVE = "ako_runs/controlled_followup/provenance/evidence_v3"
PRIMARY_INDEX_RELATIVE = (
    f"{EVIDENCE_ROOT_RELATIVE}/fused_crossed_v1r1_complete_v1.index.json"
)
PRIMARY_BUNDLE_RELATIVE = (
    f"{EVIDENCE_ROOT_RELATIVE}/fused_crossed_v1r1_complete_v1.tar.gz"
)
DEFAULT_OUTPUT_PREFIX_RELATIVE = (
    f"{EVIDENCE_ROOT_RELATIVE}/fused_crossed_v1r1_provenance_supplement_v1"
)

EVIDENCE_ROOT = REPO_ROOT / EVIDENCE_ROOT_RELATIVE

SUPPLEMENT_ID = "fused-crossed-v1r1-provenance-supplement-v1"
CAMPAIGN_ID = "fused-epilogue-crossed-v1"
RESULT_TAG = "crossed_v1r1"
RECOVERY_ID = "fused-epilogue-crossed-v1r1-reporting-fix"
PARENT_GIT_COMMIT = "4ddfc88d5712959e12161da75ce991f8c9a20248"
RECOVERY_GIT_COMMIT = "3c5beb19c25fb083c80efd2e5a7839d201e200ec"

PARENT_LOCK_PATH = REPO_ROOT / PARENT_LOCK_RELATIVE
RECOVERY_LOCK_PATH = REPO_ROOT / RECOVERY_LOCK_RELATIVE
SUPPLEMENT_LOCK_PATH = REPO_ROOT / SUPPLEMENT_LOCK_RELATIVE
FINAL_SUMMARY_PATH = REPO_ROOT / FINAL_SUMMARY_RELATIVE
PRIMARY_INDEX_PATH = REPO_ROOT / PRIMARY_INDEX_RELATIVE
PRIMARY_BUNDLE_PATH = REPO_ROOT / PRIMARY_BUNDLE_RELATIVE
DEFAULT_OUTPUT_PREFIX = REPO_ROOT / DEFAULT_OUTPUT_PREFIX_RELATIVE

PARENT_LOCK_SHA256 = "fe9a0123cc1e3aa64f5d0026e09d89b918838174c7964c85c4580b0962767e8a"
PARENT_SOURCE_BUNDLE_SHA256 = "f8d7745416c7254d10cfdd144d80d91409f745d521d0debb5f5f8cc87c9de0b6"
PARENT_DEPENDENCY_BUNDLE_SHA256 = "9af9f8ceb36d077cfe079a01e5a9df125e83932542449ceca8f4225a74dd5c84"
RECOVERY_LOCK_SHA256 = "919827539cce29d8a5aa134cd88f4afe14241703615a1b1ba29d40aabde31b63"
RECOVERY_SOURCE_BUNDLE_SHA256 = "40cb91ec9739f9b2073347cce55ea01b89aebcd5c94526589b323d247ebfb251"
RECOVERY_DEPENDENCY_BUNDLE_SHA256 = "66386b8582dd077fad85792e1ceefbf266dfc469570ef2f8682f8709b93ae824"
PRIMARY_INDEX_SHA256 = "e5d9e174c5f21395b71b40d7933b0601d9f44f4d58292fdc84fc59b36bb43860"
PRIMARY_BUNDLE_SHA256 = "b48d5b1d8087d6c777e0530d602920258c405529f59ce3c8dd6a081fea337d91"
PRIMARY_BUNDLE_SIZE = 19_603_043
PRIMARY_ENTRY_COUNT = 1_644
PRIMARY_MANIFEST_CANONICAL_SHA256 = "745e1ea6bcf6e6de1e0c103c423867488966b65c9fb7c23657962eeb344b80ea"
PRIMARY_MANIFEST_RAW_SHA256 = "547b0f6a744ea5bcbd880a8d12d2def668ce9c2ffc86322e7ae291d7ff6a274c"
FINAL_SUMMARY_SHA256 = "98552e48aa67411ba738e160d454e5c387e1c33cb1c543aa53a20198e4c0a70d"

PRIMARY_MANIFEST_MEMBER = "EVIDENCE_MANIFEST.json"
SUPPLEMENT_MANIFEST_MEMBER = "SUPPLEMENT_MANIFEST.json"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class SupplementError(RuntimeError):
    """The supplement selection, binding, or archive is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SupplementError(message)


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


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplementError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def parse_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupplementError(f"cannot parse {label}: {exc}") from exc
    require(isinstance(parsed, dict), f"{label} is not a JSON object")
    return parsed


def lexical_absolute(path: Path) -> Path:
    """Return an absolute, normalized path without following symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def require_no_symlink_components(path: Path, *, anchor: Path | None = None) -> Path:
    """Reject every existing symlink component and return the lexical path."""
    candidate = lexical_absolute(path)
    root = lexical_absolute(anchor) if anchor is not None else Path(candidate.anchor)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise SupplementError(f"path escapes anchor {root}: {path}") from exc
    current = root
    require(not current.is_symlink(), f"symlinked path component: {current}")
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), f"symlinked path component: {current}")
    return candidate


def repo_path(path: Path) -> str:
    candidate = require_no_symlink_components(path, anchor=REPO_ROOT)
    try:
        relative = candidate.relative_to(lexical_absolute(REPO_ROOT)).as_posix()
    except ValueError as exc:
        raise SupplementError(f"path escapes repository: {path}") from exc
    validate_member_name(relative)
    return relative


def validate_member_name(name: str) -> None:
    require(
        isinstance(name, str)
        and name
        and "\\" not in name
        and "\x00" not in name,
        f"unsafe archive path: {name!r}",
    )
    value = PurePosixPath(name)
    require(
        not value.is_absolute()
        and value.as_posix() == name
        and all(part not in {"", ".", ".."} for part in value.parts),
        f"unsafe archive path: {name!r}",
    )


def repo_file(name: str) -> Path:
    validate_member_name(name)
    path = lexical_absolute(REPO_ROOT.joinpath(*PurePosixPath(name).parts))
    require(repo_path(path) == name, f"noncanonical repository path: {name}")
    return path


def validate_hash(value: Any, label: str) -> str:
    require(isinstance(value, str) and HASH_RE.fullmatch(value) is not None, f"invalid SHA-256 for {label}")
    return value


def exclusive_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes without replacing an existing artifact."""
    path = require_no_symlink_components(path, anchor=REPO_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    require(not temporary.exists() and not temporary.is_symlink(), f"unreconciled partial: {temporary}")
    created_temporary = False
    try:
        with temporary.open("xb") as handle:
            created_temporary = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if created_temporary:
            temporary.unlink(missing_ok=True)


def merge_hash_maps(*maps: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for mapping in maps:
        require(isinstance(mapping, dict), "locked hash map is not an object")
        for name, expected in mapping.items():
            validate_member_name(name)
            validate_hash(expected, name)
            if name in merged:
                require(merged[name] == expected, f"conflicting locked hash for {name}")
            merged[name] = expected
    return dict(sorted(merged.items()))


def verify_lock_map(lock: dict[str, Any], map_key: str, bundle_key: str, label: str) -> dict[str, str]:
    mapping = lock.get(map_key)
    require(isinstance(mapping, dict) and mapping, f"{label} {map_key} missing")
    normalized = merge_hash_maps(mapping)
    require(
        canonical_sha256(normalized) == lock.get(bundle_key),
        f"{label} {bundle_key} mismatch",
    )
    return normalized


def validate_anchor_identities() -> dict[str, dict[str, Any]]:
    anchors = {
        "parent_lock": (PARENT_LOCK_PATH, PARENT_LOCK_SHA256),
        "recovery_lock": (RECOVERY_LOCK_PATH, RECOVERY_LOCK_SHA256),
        "primary_index": (PRIMARY_INDEX_PATH, PRIMARY_INDEX_SHA256),
        "primary_bundle": (PRIMARY_BUNDLE_PATH, PRIMARY_BUNDLE_SHA256),
        "final_summary": (FINAL_SUMMARY_PATH, FINAL_SUMMARY_SHA256),
    }
    for label, (path, expected) in anchors.items():
        repo_path(path)
        require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked")
        require(file_sha256(path) == expected, f"{label} bytes changed")
    require(PRIMARY_BUNDLE_PATH.stat().st_size == PRIMARY_BUNDLE_SIZE, "primary bundle size changed")

    parent = read_json(PARENT_LOCK_PATH)
    recovery = read_json(RECOVERY_LOCK_PATH)
    primary_index = read_json(PRIMARY_INDEX_PATH)
    final = read_json(FINAL_SUMMARY_PATH)
    require(
        parent.get("campaign_id") == CAMPAIGN_ID
        and parent.get("source_bundle_sha256") == PARENT_SOURCE_BUNDLE_SHA256
        and parent.get("dependency_bundle_sha256") == PARENT_DEPENDENCY_BUNDLE_SHA256,
        "parent lock identity changed",
    )
    require(
        recovery.get("campaign_id") == CAMPAIGN_ID
        and recovery.get("result_tag") == RESULT_TAG
        and recovery.get("recovery_id") == RECOVERY_ID
        and recovery.get("parent_git_commit") == PARENT_GIT_COMMIT
        and recovery.get("source_bundle_sha256") == RECOVERY_SOURCE_BUNDLE_SHA256
        and recovery.get("dependency_bundle_sha256") == RECOVERY_DEPENDENCY_BUNDLE_SHA256,
        "recovery lock identity changed",
    )
    require(
        primary_index.get("record_type") == "fused_crossed_v1r1_evidence_index"
        and primary_index.get("entry_count") == PRIMARY_ENTRY_COUNT
        and primary_index.get("manifest_sha256") == PRIMARY_MANIFEST_CANONICAL_SHA256
        and primary_index.get("bundle_sha256") == PRIMARY_BUNDLE_SHA256
        and primary_index.get("bundle_size") == PRIMARY_BUNDLE_SIZE
        and primary_index.get("recovery_git_commit") == RECOVERY_GIT_COMMIT
        and primary_index.get("recovery_lock_sha256") == RECOVERY_LOCK_SHA256,
        "primary evidence index identity changed",
    )
    require(
        final.get("record_type") == "fused_crossed_final_summary"
        and final.get("complete") is True
        and final.get("campaign_id") == CAMPAIGN_ID
        and final.get("result_tag") == RESULT_TAG
        and final.get("recovery_id") == RECOVERY_ID
        and final.get("recovery_git_commit") == RECOVERY_GIT_COMMIT
        and final.get("parent_launch_lock_sha256") == PARENT_LOCK_SHA256
        and final.get("recovery_lock_sha256") == RECOVERY_LOCK_SHA256,
        "final summary identity changed or incomplete",
    )
    return {
        "parent_lock": parent,
        "recovery_lock": recovery,
        "primary_index": primary_index,
        "final_summary": final,
    }
