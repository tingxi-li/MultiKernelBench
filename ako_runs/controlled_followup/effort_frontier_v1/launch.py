#!/usr/bin/env python3
"""Fail-closed four-GPU launcher for effort-frontier trajectories."""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    from . import campaign, validate
except ImportError:  # direct script execution
    import campaign  # type: ignore
    import validate  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--result-root", type=Path, default=campaign.HERE / "results/search_v1")
    args = parser.parse_args()
    jobs = campaign.trajectories()
    if args.list:
        for row in jobs:
            print(f"gpu{row['physical_gpu']} {row['trajectory_id']} seed={row['search_seed']}")
        return 0
    blockers = validate.launch_blockers()
    if blockers:
        for blocker in blockers:
            print("BLOCKED:", blocker)
        print("REFUSED: zero provider requests and zero GPU processes started")
        return 2
    if not args.execute:
        print("launch-ready; pass --execute to start 20 trajectories")
        return 0
    args.result_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.result_root / ".launcher.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"REFUSED: another launcher holds {lock_path}")
        return 2
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()}\n")
    lock_handle.flush()
    by_gpu = defaultdict(list)
    for row in jobs:
        by_gpu[row["physical_gpu"]].append(row)
    failures = 0
    # Five deterministic waves: exactly one trajectory per physical GPU per wave.
    for wave in range(5):
        processes = []
        for gpu in range(4):
            row = by_gpu[gpu][wave]
            command = [
                sys.executable,
                str(campaign.HERE / "controller.py"),
                "--trajectory",
                row["trajectory_id"],
                "--result-root",
                str(args.result_root),
            ]
            processes.append((row, subprocess.Popen(command)))
        for row, process in processes:
            status = process.wait()
            if status:
                failures += 1
                print(f"FAILED: {row['trajectory_id']} exit={status}")
        if failures:
            print("REFUSED: no later wave started after a trajectory failure")
            break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
