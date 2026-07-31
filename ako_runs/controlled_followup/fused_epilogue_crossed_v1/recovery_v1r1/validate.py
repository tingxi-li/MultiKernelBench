#!/usr/bin/env python3
"""Validate the append-only v1r1 recovery and its launch prerequisites."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import common
except ImportError:  # pragma: no cover - direct-script CLI
    import common  # type: ignore


class ValidationError(common.RecoveryError):
    pass


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=common.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ValidationError(
            completed.stderr.strip() or f"git {' '.join(arguments)} failed"
        )
    return completed.stdout.strip()


def _parent_validate_module():
    if str(common.PARENT) not in sys.path:
        sys.path.insert(0, str(common.PARENT))
    name = "fused_crossed_parent_validate_v1r1"
    spec = importlib.util.spec_from_file_location(name, common.PARENT / "validate.py")
    if spec is None or spec.loader is None:
        raise ValidationError("cannot load frozen parent validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_structure() -> dict[str, Any]:
    incident = common.verify_incident_receipt()
    lock = common.load_recovery_lock()
    sources = lock.get("source_sha256")
    dependencies = lock.get("dependency_sha256")
    common.require(isinstance(sources, dict) and sources, "recovery source map missing")
    common.require(isinstance(dependencies, dict) and dependencies, "recovery dependency map missing")
    for mapping, label in ((sources, "source"), (dependencies, "dependency")):
        for relative, expected in mapping.items():
            path = common.REPO_ROOT / relative
            common.require(
                path.is_file()
                and not path.is_symlink()
                and common.file_sha256(path) == expected,
                f"frozen recovery {label} changed: {relative}",
            )
    common.require(
        common.canonical_sha256(sources) == lock.get("source_bundle_sha256"),
        "recovery source bundle mismatch",
    )
    common.require(
        common.canonical_sha256(dependencies)
        == lock.get("dependency_bundle_sha256"),
        "recovery dependency bundle mismatch",
    )
    common.require(
        lock.get("incident_receipt_sha256") == common.file_sha256(common.INCIDENT_PATH),
        "recovery lock does not bind incident receipt",
    )
    return {
        "incident": incident,
        "lock": lock,
        "lock_sha256": common.file_sha256(common.LOCK_PATH),
    }


def validate_launch_ready(gpu: int, *, allow_busy: bool = False) -> dict[str, Any]:
    state = validate_structure()
    lock = state["lock"]
    tracked = list(lock["source_sha256"]) + [common.repo_path(common.LOCK_PATH)]
    status = git("status", "--porcelain", "--", *tracked)
    common.require(not status, "recovery sources/lock are not committed")
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    upstream_head = git("rev-parse", upstream)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, upstream_head],
        cwd=common.REPO_ROOT,
    )
    common.require(
        ancestor.returncode == 0,
        "recovery commit is not present on the configured upstream",
    )
    parent = _parent_validate_module().validate_launch_ready(
        gpu, allow_busy=allow_busy
    )
    binding = common.binding_for_commit(head)
    if common.REMOTE_RECEIPT_PATH.exists():
        receipt = common.read_json(common.REMOTE_RECEIPT_PATH)
        common.validate_binding(receipt, binding, "remote verification receipt")
        common.require(
            receipt.get("verified_remote_commit") == head
            and receipt.get("git_upstream") == upstream
            and receipt.get("git_upstream_commit") == upstream_head
            and receipt.get("verified_remote_ref") is True,
            "recovery remote verification receipt differs",
        )
    return {
        "binding": binding,
        "git_upstream": upstream,
        "git_upstream_commit": upstream_head,
        "gpu": parent["gpu"],
        "parent": parent,
        "recovery_git_commit": head,
        "recovery_lock_sha256": state["lock_sha256"],
    }


def ensure_remote_receipt(ready: dict[str, Any]) -> dict[str, Any]:
    binding = ready["binding"]
    value = common.add_binding(
        {
            "campaign_id": common.CAMPAIGN_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_upstream": ready["git_upstream"],
            "git_upstream_commit": ready["git_upstream_commit"],
            "parent_git_commit": common.PARENT_GIT_COMMIT,
            "record_type": "fused_crossed_v1r1_remote_verification_receipt",
            "schema_version": 1,
            "verified_remote_commit": ready["recovery_git_commit"],
            "verified_remote_ref": True,
        },
        binding,
    )
    if common.REMOTE_RECEIPT_PATH.exists():
        observed = common.read_json(common.REMOTE_RECEIPT_PATH)
        common.validate_binding(observed, binding, "remote verification receipt")
        common.require(
            all(observed.get(key) == value.get(key) for key in value if key != "created_utc"),
            "existing remote verification receipt differs",
        )
        return observed
    common.exclusive_json(common.REMOTE_RECEIPT_PATH, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-ready", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    state = validate_structure()
    print(
        "structural=PASS "
        f"recovery={common.RECOVERY_ID} lock={state['lock_sha256']}"
    )
    if args.launch_ready:
        ready = validate_launch_ready(args.gpu, allow_busy=args.allow_busy)
        print(
            f"launch_ready=PASS gpu={args.gpu} "
            f"commit={ready['recovery_git_commit']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
