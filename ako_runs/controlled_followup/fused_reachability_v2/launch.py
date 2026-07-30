#!/usr/bin/env python3
"""Launch screen or confirmation processes on one physical GPU."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from protocol import (
    HERE,
    JOBS,
    LOCK,
    file_sha256,
    acquire_active_lock,
    gpu_snapshot,
    nvcc_fingerprint,
    read_json,
    record_filename,
    stable_write,
    safe_result_root,
    validate_gpu_snapshot,
    validate_process_record,
    verify_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "confirmation"), required=True)
    parser.add_argument("--lane", choices=("cuda_noptx", "cuda_unlimited"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--selection", default="")
    parser.add_argument("--allow-busy", action="store_true")
    return parser.parse_args()


def git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=HERE.parents[2], capture_output=True, text=True, timeout=30
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def ensure_idle(index: int, allow_busy: bool) -> None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    occupants = [line for line in completed.stdout.splitlines() if line.strip()]
    if (completed.returncode != 0 or occupants) and not allow_busy:
        raise RuntimeError(f"GPU {index} preflight failed or is busy: {occupants}")


def selected_jobs(args: argparse.Namespace, all_jobs: list[dict]) -> list[dict]:
    lane_jobs = [job for job in all_jobs if job["lane"] == args.lane]
    if args.phase == "screen":
        return lane_jobs
    if not args.selection:
        raise ValueError("confirmation requires --selection")
    selection_path = Path(args.selection)
    selection = read_json(selection_path)
    if selection.get("record_type") != "fused_reachability_v2_confirmation_selection":
        raise RuntimeError("confirmation selection is not gate-adjudicated")
    if selection.get("launch_lock_sha256") != file_sha256(LOCK):
        raise RuntimeError("confirmation selection binds a different launch lock")
    selected_ids = {
        row["job_id"] for row in selection["selected"]
        if row["lane"] == args.lane and row.get("robust_eligible") is True
    }
    result = [job for job in lane_jobs if job["job_id"] in selected_ids]
    for job in result:
        selected = next(row for row in selection["selected"] if row["job_id"] == job["job_id"])
        if selected.get("job_sha256") != verify_lock()["job_sha256"][job["job_id"]]:
            raise RuntimeError(f"confirmation selection job hash mismatch: {job['job_id']}")
    if not result:
        raise RuntimeError(f"selection contains no eligible {args.lane} jobs")
    return result


def main() -> int:
    args = parse_args()
    lock = verify_lock()
    ensure_idle(args.gpu, args.allow_busy)
    before = gpu_snapshot(args.gpu)
    validate_gpu_snapshot(before)
    jobs = selected_jobs(args, read_json(JOBS))
    reps = lock["launch_policy"][args.phase]["reps"]
    randomizer = random.Random(lock["launch_policy"][args.phase]["randomization_seed"])
    order = []
    for rep in range(reps):
        block = list(jobs)
        randomizer.shuffle(block)
        order.extend((job, rep) for job in block)

    root = safe_result_root(args.tag)
    active_lock = acquire_active_lock(root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "launch_receipt.json"
    contract = {
        "schema_version": 1,
        "campaign_id": lock["campaign_id"],
        "phase": args.phase,
        "lane": args.lane,
        "physical_gpu": args.gpu,
        "logical_device": "cuda:0",
        "tag": args.tag,
        "reps": reps,
        "jobs": [job["job_id"] for job in jobs],
        "execution_order": [f"{job['job_id']}:rep{rep}" for job, rep in order],
        "randomization": "complete blocks by rep; job order shuffled within block",
        "selection_sha256": file_sha256(Path(args.selection)) if args.selection else None,
        "launch_lock_sha256": file_sha256(LOCK),
        "source_bundle_sha256": lock["source_bundle_sha256"],
    }
    if receipt_path.exists():
        old = read_json(receipt_path)
        if old.get("contract") != contract:
            raise RuntimeError("existing result tag has a different launch contract")
    else:
        stable_write(
            receipt_path,
            {
                "record_type": "fused_reachability_v2_launch_receipt",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "contract": contract,
                "gpu_before": before,
                "nvcc": nvcc_fingerprint(),
                "host": platform.node(),
                "python": sys.version,
                "git_commit": git_value("rev-parse", "HEAD"),
                "git_status_porcelain": git_value("status", "--porcelain"),
                "source_sha256": lock["source_sha256"],
            },
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TORCH_EXTENSIONS_DIR"] = str(HERE / ".torch_ext" / f"gpu{args.gpu}")
    env["CUDA_HOME"] = "/usr/local/cuda-13.1"
    env["PATH"] = "/usr/local/cuda-13.1/bin:" + env.get("PATH", "")
    env.setdefault("MAX_JOBS", "4")
    attempts_path = root / "attempts.jsonl"
    for position, (job, rep) in enumerate(order, 1):
        output = raw / record_filename(job["job_id"], rep)
        if output.exists():
            validate_process_record(
                output,
                job=job,
                phase=args.phase,
                rep=rep,
                physical_gpu=args.gpu,
                lock=lock,
            )
            print(f"[resume {position}/{len(order)}] {output.name}", flush=True)
            continue
        command = [
            sys.executable,
            str(HERE / "run_one.py"),
            "--job-json",
            json.dumps(job, sort_keys=True, separators=(",", ":")),
            "--rep",
            str(rep),
            "--phase",
            args.phase,
            "--trials",
            "100",
            "--warmup-s",
            "2.0",
            "--seed",
            "0",
            "--dist",
            "rand",
            "--physical-gpu",
            str(args.gpu),
            "--out",
            str(output),
        ]
        print(f"[run {position}/{len(order)}] {job['job_id']} rep={rep}", flush=True)
        completed = subprocess.run(command, cwd=HERE.parents[2], env=env)
        with attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "utc": datetime.now(timezone.utc).isoformat(),
                "job_id": job["job_id"],
                "rep": rep,
                "returncode": completed.returncode,
                "record_exists": output.exists(),
                "record_sha256": file_sha256(output) if output.exists() else None,
            }, sort_keys=True, separators=(",", ":")) + "\n")
    validated = []
    for job, rep in order:
        output = raw / record_filename(job["job_id"], rep)
        validated.append(validate_process_record(
            output, job=job, phase=args.phase, rep=rep,
            physical_gpu=args.gpu, lock=lock,
        ))
    process_failures = sum(not row["ok"] for row in validated)
    legacy_gate_failures = sum(
        row["ok"] and row["legacy_error"]["gate_pass"] is not True for row in validated
    )
    stable_write(
        root / "run_status.json",
        {
            "campaign_id": lock["campaign_id"],
            "phase": args.phase,
            "lane": args.lane,
            "expected_records": len(order),
            "observed_records": len(list(raw.glob("*.json"))),
            "process_failures": process_failures,
            "legacy_gate_failures": legacy_gate_failures,
            "gpu_after": gpu_snapshot(args.gpu),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    active_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
