#!/usr/bin/env python3
"""Validate structure or the complete fail-closed GPU launch preconditions."""
from __future__ import annotations

import argparse
import subprocess

from core import (
    LOCK_PATH,
    REPO_ROOT,
    SOURCE_PATHS,
    ProtocolError,
    file_sha256,
    gpu_snapshot,
    load_contract,
    validate_gpu,
)


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
    )
    if completed.returncode != 0:
        raise ProtocolError(completed.stderr.strip() or f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def validate_launch_ready(gpu: int, *, allow_busy: bool = False) -> dict:
    campaign, cells, lock = load_contract()
    # Verify the original robust adapter all the way through its manifest and
    # gate-spec hash chain, rather than merely trusting our dependency lock.
    import sys
    fused_grid = REPO_ROOT / "ako_runs/controlled_followup/fused_grid"
    if str(fused_grid) not in sys.path:
        sys.path.insert(0, str(fused_grid))
    import robust_adapter
    context = robust_adapter.load_repository()
    if file_sha256(context.adapter_path) != lock["frozen_gate"]["adapter_manifest_sha256"]:
        raise ProtocolError("robust adapter differs from the launch lock")

    tracked_inputs = list(SOURCE_PATHS) + [str(LOCK_PATH.relative_to(REPO_ROOT))]
    status = git("status", "--porcelain", "--", *tracked_inputs)
    if status:
        raise ProtocolError(
            "campaign sources/lock are not committed; preregistration must be committed before launch"
        )
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    head = git("rev-parse", "HEAD")
    upstream_head = git("rev-parse", upstream)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, upstream_head], cwd=REPO_ROOT
    )
    if ancestor.returncode != 0:
        raise ProtocolError("launch commit is not present on the configured upstream")

    snapshot = gpu_snapshot(gpu)
    validate_gpu(snapshot, campaign)
    busy = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    occupants = [line for line in busy.stdout.splitlines() if line.strip()]
    if (busy.returncode != 0 or occupants) and not allow_busy:
        raise ProtocolError(f"GPU {gpu} is busy or compute-app query failed: {occupants}")
    return {
        "campaign_id": campaign["campaign_id"],
        "cell_count": len(cells),
        "git_commit": head,
        "git_upstream": upstream,
        "git_upstream_commit": upstream_head,
        "gpu": snapshot,
        "source_bundle_sha256": lock["source_bundle_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-ready", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    campaign, cells, lock = load_contract()
    print(f"structural=PASS campaign={campaign['campaign_id']} cells={len(cells)} lock={file_sha256(LOCK_PATH)}")
    if args.launch_ready:
        ready = validate_launch_ready(args.gpu, allow_busy=args.allow_busy)
        print(f"launch_ready=PASS gpu={args.gpu} commit={ready['git_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

