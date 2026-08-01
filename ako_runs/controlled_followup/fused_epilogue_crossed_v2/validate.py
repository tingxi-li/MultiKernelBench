#!/usr/bin/env python3
"""Validate v2 structure and fail-closed live launch preconditions."""
from __future__ import annotations

import argparse
import subprocess

try:
    from .core import (
        CAMPAIGN_ID, DEPENDENCY_PATHS, LOCK_PATH, PROBE_LOCK_PATH, REPO_ROOT,
        SOURCE_PATHS, ProtocolError, file_sha256, gpu_snapshot, load_cells,
        make_unfrozen_contract, read_json, validate_gpu,
    )
except ImportError:  # direct script execution
    from core import (
    CAMPAIGN_ID,
    DEPENDENCY_PATHS,
    LOCK_PATH,
    PROBE_LOCK_PATH,
    REPO_ROOT,
    SOURCE_PATHS,
    ProtocolError,
    file_sha256,
    gpu_snapshot,
    load_cells,
    make_unfrozen_contract,
    read_json,
    validate_gpu,
    )


def git(*arguments: str) -> str:
    completed = subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    if completed.returncode != 0:
        raise ProtocolError(completed.stderr.strip() or f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def validate_lock(stage: str) -> tuple[dict, list[dict], dict]:
    campaign, cells, _resolution = make_unfrozen_contract()
    require_resolved = stage == "campaign"
    cells = load_cells(require_resolved=require_resolved)
    path = LOCK_PATH if require_resolved else PROBE_LOCK_PATH
    if not path.is_file():
        raise ProtocolError(f"{stage} lock is missing")
    lock = read_json(path)
    if lock.get("schema_version") != 2 or lock.get("campaign_id") != CAMPAIGN_ID or lock.get("lock_stage") != stage:
        raise ProtocolError(f"invalid {stage} lock identity")
    if __package__:
        from .freeze import make_lock
    else:
        from freeze import make_lock

    expected = make_lock(stage)
    for key, value in expected.items():
        if key != "created_utc" and lock.get(key) != value:
            raise ProtocolError(f"{stage} lock differs at {key}")
    return campaign, cells, lock


def remote_launch_paths(stage: str, lock: dict) -> tuple[str, ...]:
    lock_path = LOCK_PATH if stage == "campaign" else PROBE_LOCK_PATH
    return (
        *SOURCE_PATHS,
        *DEPENDENCY_PATHS,
        str(lock_path.relative_to(REPO_ROOT)),
        *lock.get("support_evidence_sha256", {}),
    )


def validate_remote_ready(stage: str) -> dict:
    campaign, cells, lock = validate_lock(stage)
    lock_path = LOCK_PATH if stage == "campaign" else PROBE_LOCK_PATH
    launch_paths = remote_launch_paths(stage, lock)
    status = git("status", "--porcelain", "--", *launch_paths)
    if status:
        raise ProtocolError(f"{stage} sources/lock are not committed")
    git("ls-files", "--error-unmatch", "--", *launch_paths)
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    head = git("rev-parse", "HEAD")
    upstream_head = git("rev-parse", upstream)
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", head, upstream_head], cwd=REPO_ROOT)
    if ancestor.returncode != 0:
        raise ProtocolError("launch commit is not present on the configured upstream")
    return {
        "campaign_id": CAMPAIGN_ID,
        "cell_count": len(cells),
        "git_commit": head,
        "git_upstream": upstream,
        "git_upstream_commit": upstream_head,
        "lock_sha256": file_sha256(lock_path),
        "stage": stage,
    }


def validate_launch_ready(stage: str, gpu: int, *, allow_busy: bool = False) -> dict:
    ready = validate_remote_ready(stage)
    campaign, _cells, _lock = validate_lock(stage)
    snapshot = gpu_snapshot(gpu)
    validate_gpu(snapshot, campaign)
    busy = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    occupants = [line for line in busy.stdout.splitlines() if line.strip()]
    if (busy.returncode != 0 or occupants) and not allow_busy:
        raise ProtocolError(f"GPU {gpu} is busy or compute-app query failed: {occupants}")
    return {
        **ready,
        "gpu": snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("structure", "probes", "campaign"), default="structure")
    parser.add_argument("--launch-ready", action="store_true")
    parser.add_argument("--remote-ready", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    campaign, cells, resolution = make_unfrozen_contract()
    print(f"structural=PASS campaign={campaign['campaign_id']} cells={len(cells)} support={resolution['status']}")
    if args.stage != "structure":
        validate_lock(args.stage)
    if args.remote_ready and not args.launch_ready:
        if args.stage == "structure":
            raise ProtocolError("remote/launch readiness requires --stage probes|campaign")
        remote = validate_remote_ready(args.stage)
        print(f"remote_ready=PASS stage={args.stage} commit={remote['git_commit']}")
    if args.launch_ready:
        if args.stage == "structure":
            raise ProtocolError("remote/launch readiness requires --stage probes|campaign")
        ready = validate_launch_ready(args.stage, args.gpu, allow_busy=args.allow_busy)
        print(f"launch_ready=PASS stage={args.stage} gpu={args.gpu} commit={ready['git_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
