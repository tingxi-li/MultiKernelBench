#!/usr/bin/env python3
"""List or launch reciprocal-v2 cells through an explicit external runner."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import protocol
import validate


def _exclusive(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite of {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=protocol.STAGES, default="audit")
    parser.add_argument("--manifest", choices=protocol.KINDS, help=argparse.SUPPRESS)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.execute and args.list:
        parser.error("--execute and --list are mutually exclusive")
    if args.execute and args.limit:
        parser.error("--limit is inspection-only; production stages require the full census")
    stage = args.manifest or args.stage
    kind = "audit" if stage == "audit" else "primary"
    documents = validate.validate_static()
    manifest = documents["manifests"][kind]
    jobs = manifest["jobs"][: args.limit or None]
    if args.list:
        for row in jobs:
            print(f"{row['ordinal']:02d} {row['job_id']}")
        return 0
    blockers = validate.dependency_blockers(stage)
    hardware = validate.gpu_blocker()
    if hardware:
        blockers.append(hardware)
    if args.runner is None:
        blockers.append("no explicit --runner was supplied")
    elif not args.runner.is_file():
        blockers.append(f"runner does not exist: {args.runner}")
    else:
        try:
            runner_relative = protocol.repo_path(args.runner.resolve())
        except ValueError:
            blockers.append("runner must be a repository file bound by provenance")
        else:
            provenance = (
                protocol.load_json(protocol.PROVENANCE_LOCK)
                if protocol.PROVENANCE_LOCK.is_file()
                else {}
            )
            expected_runner = provenance.get("execution_runner", {})
            if (
                expected_runner.get("path") != runner_relative
                or expected_runner.get("sha256") != protocol.file_sha256(args.runner.resolve())
            ):
                blockers.append("runner is not content-addressed by prelaunch provenance")
    if blockers:
        for blocker in blockers:
            print("BLOCKED:", blocker)
        print("REFUSED: no GPU process started")
        return 2
    assert args.runner is not None
    if not args.execute:
        print("launch-ready; pass --execute to start the frozen cells")
        return 0
    payload = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "stage": stage,
        "manifest": protocol.repo_path(validate.MANIFESTS[kind]),
        "manifest_sha256": protocol.file_sha256(validate.MANIFESTS[kind]),
        "jobs": [row["job_id"] for row in jobs],
        "gate_lock": protocol.repo_path(protocol.GATE_LOCK),
        "resolution_lock": protocol.repo_path(protocol.RESOLUTION_LOCK),
        "implementation_registry": protocol.repo_path(protocol.IMPLEMENTATION_REGISTRY),
        "translator_isolation_lock": protocol.repo_path(protocol.TRANSLATOR_ISOLATION_LOCK),
        "source_freeze": protocol.repo_path(protocol.SOURCE_FREEZE),
        "source_freeze_sha256": protocol.file_sha256(protocol.SOURCE_FREEZE),
        "timing_physical_gpu": protocol.TIMING_PHYSICAL_GPU,
        "timing_gpu_uuid": protocol.TIMING_GPU_UUID,
        "confirmation_blocks": protocol.CONFIRM_REPS,
        "block_order_seed": protocol.BLOCK_ORDER_SEED,
        "block_order_sha256": protocol.block_order_sha256(),
        "block_orders": protocol.block_orders() if stage == "primary" else None,
    }
    launch_receipt_path = protocol.HERE / "results" / stage / "launch_receipt.json"
    launch_receipt = {
        "schema_version": 2,
        "record_type": "reciprocal_v2_launch_receipt",
        "campaign_id": protocol.CAMPAIGN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "payload": payload,
        "payload_sha256": protocol.canonical_sha256(payload),
        "runner_path": protocol.repo_path(args.runner.resolve()),
        "runner_sha256": protocol.file_sha256(args.runner.resolve()),
        "prelaunch_provenance_sha256": protocol.file_sha256(protocol.PROVENANCE_LOCK),
        "driver_preflight_passed": True,
        "full_census": True,
    }
    _exclusive(launch_receipt_path, launch_receipt)
    payload["launch_receipt"] = protocol.repo_path(launch_receipt_path)
    payload["launch_receipt_sha256"] = protocol.file_sha256(launch_receipt_path)
    completed = subprocess.run(
        [str(args.runner.resolve())],
        input=json.dumps(payload, sort_keys=True) + "\n",
        text=True,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
