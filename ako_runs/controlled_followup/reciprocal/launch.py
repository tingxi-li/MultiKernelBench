#!/usr/bin/env python3
"""Inspect or dependency-gate the reciprocal-transfer campaign.

There is deliberately no in-tree GPU runner yet: destination implementations,
retune plans, recipe resolution, and audit receipts must be frozen first.  A
future runner can be passed explicitly with ``--runner``; ``--execute`` will not
invoke it unless every content-addressed prerequisite validates.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import validate


HERE = Path(__file__).resolve().parent


def runner_command(runner: Path, kind: str) -> list[str]:
    return [
        str(runner.resolve()),
        "--manifest",
        str(validate.MANIFESTS[kind].resolve()),
        "--gate-spec",
        str(validate.GATE_SPEC.resolve()),
        "--gate-lock",
        str(validate.GATE_LOCK.resolve()),
        "--validation-summary",
        str(validate.VALIDATION_SUMMARY.resolve()),
        "--acceptance-receipt",
        str(validate.ACCEPTANCE_RECEIPT.resolve()),
        "--recipe-lock",
        str(validate.RECIPE_LOCK.resolve()),
    ]


def print_jobs(manifest: dict, limit: int) -> None:
    jobs = manifest["jobs"][:limit] if limit else manifest["jobs"]
    for job in jobs:
        print(
            f"{job['ordinal']:02d} {job['job_id']} "
            f"card={job['recipe_card_sha256'][:12]}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", choices=tuple(validate.MANIFESTS), default="primary"
    )
    parser.add_argument("--list", action="store_true", help="list deterministic jobs")
    parser.add_argument(
        "--dry-run", action="store_true", help="show dependency and runner plan"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="return nonzero while any launch dependency is unresolved",
    )
    parser.add_argument(
        "--execute", action="store_true", help="delegate to an explicitly supplied runner"
    )
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.execute and (args.list or args.dry_run or args.validate_only):
        parser.error("--execute cannot be combined with inspection modes")

    documents = validate.validate_static()
    manifest = documents["manifests"][args.manifest]
    blockers = validate.launch_blockers(manifest)
    print(
        f"validated {manifest['campaign_id']} {args.manifest}: "
        f"2 origins x 4 destinations x 2 modes = {manifest['job_count']} jobs"
    )
    if args.list:
        print_jobs(manifest, args.limit)
    for blocker in blockers:
        print(f"BLOCKED: {blocker}")

    if args.runner is not None:
        command = runner_command(args.runner, args.manifest)
        print("runner:", shlex.join(command))
    elif args.dry_run or args.execute:
        print("BLOCKED: no external --runner supplied; no GPU runner is bundled")

    if args.execute:
        if blockers or args.runner is None:
            print("REFUSED: launch prerequisites are not frozen; no process started")
            return 2
        if not args.runner.is_file() or not args.runner.stat().st_mode & 0o111:
            print(f"REFUSED: runner is not an executable file: {args.runner}")
            return 2
        return subprocess.run(runner_command(args.runner, args.manifest), check=False).returncode

    if args.validate_only and blockers:
        return 2
    if not (args.list or args.dry_run or args.validate_only):
        print("inspection only; pass --execute after all blockers are resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
