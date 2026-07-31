#!/usr/bin/env python3
"""Launch the timing screen or signed/positive confirmation on physical GPU 0."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from core import (
    HERE,
    LOCK_PATH,
    REPO_ROOT,
    confirmation_plan,
    file_sha256,
    load_contract,
    gpu_snapshot,
    nvcc_fingerprint,
    read_json,
    result_root,
    screen_plan,
    stable_write,
    timing_filename,
)
from validate import validate_launch_ready


def validate_timing_record(path: Path, *, cell: dict, phase: str, distribution: str, rep: int, gpu: int, eligibility_sha256: str, lock: dict) -> dict:
    record = read_json(path)
    expected = {
        "campaign_id": "fused-epilogue-crossed-v1",
        "cell": cell,
        "distribution": distribution,
        "eligibility_sha256": eligibility_sha256,
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "logical_device": "cuda:0",
        "phase": phase,
        "physical_gpu": gpu,
        "rep": rep,
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "trials": 100,
        "warmup_s": 2.0,
    }
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if mismatches or not isinstance(record.get("ok"), bool):
        raise RuntimeError(f"timing record contract mismatch {path}: {mismatches}")
    if record["ok"]:
        times = record.get("times_ms")
        if not isinstance(times, list) or len(times) != 100:
            raise RuntimeError(f"timing sample count mismatch: {path}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("screen", "confirmation"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eligibility", required=True, help="audit summary for screen; frozen selection for confirmation")
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    campaign, cells, lock = load_contract()
    if args.gpu != campaign["hardware"]["timing_gpu"]:
        raise RuntimeError("screen and confirmation timing must run on physical GPU 0")
    ready = validate_launch_ready(args.gpu, allow_busy=args.allow_busy)
    eligibility_path = Path(args.eligibility).resolve()
    eligibility = read_json(eligibility_path)
    if eligibility.get("campaign_id") != campaign["campaign_id"] or eligibility.get("launch_lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("foreign eligibility artifact")
    if args.phase == "screen":
        if eligibility.get("record_type") != "fused_crossed_audit_summary" or eligibility.get("complete") is not True:
            raise RuntimeError("screen requires a complete audit summary")
        ids = set(eligibility.get("timing_eligible_cell_ids", []))
        plan = screen_plan(cells, ids)
    else:
        if eligibility.get("record_type") != "fused_crossed_confirmation_selection" or eligibility.get("complete") is not True:
            raise RuntimeError("confirmation requires a complete frozen selection")
        ids = set(eligibility.get("selected_cell_ids", []))
        plan = confirmation_plan(ids)
    by_id = {cell["cell_id"]: cell for cell in cells}
    if not ids or not ids.issubset(by_id):
        raise RuntimeError("eligibility artifact has no valid candidate IDs")
    root = result_root(args.tag) / args.phase
    root.mkdir(parents=True, exist_ok=True)
    active = (root / "active.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"another {args.phase} launcher is active") from None
    eligibility_hash = file_sha256(eligibility_path)
    contract = {
        "campaign_id": campaign["campaign_id"],
        "eligibility_path": str(eligibility_path.relative_to(REPO_ROOT)),
        "eligibility_sha256": eligibility_hash,
        "execution_order": plan,
        "git_commit": ready["git_commit"],
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "logical_device": "cuda:0",
        "phase": args.phase,
        "physical_gpu": args.gpu,
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "tag": args.tag,
    }
    receipt_path = root / "launch_receipt.json"
    if receipt_path.exists():
        if read_json(receipt_path).get("contract") != contract:
            raise RuntimeError("existing timing receipt has another contract")
    else:
        stable_write(receipt_path, {"contract": contract, "created_utc": datetime.now(timezone.utc).isoformat(), "gpu": ready["gpu"], "host": platform.node(), "nvcc": nvcc_fingerprint(), "python": sys.version, "record_type": "fused_crossed_timing_receipt", "schema_version": 1})
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["CUDA_HOME"] = "/usr/local/cuda-13.1"
    env["PATH"] = "/usr/local/cuda-13.1/bin:" + env.get("PATH", "")
    env["TORCH_EXTENSIONS_DIR"] = str(HERE / ".torch_ext" / "gpu0")
    env.setdefault("MAX_JOBS", "4")
    attempts = root / "attempts.jsonl"
    for position, row in enumerate(plan, 1):
        cell = by_id[row["cell_id"]]
        output = raw / timing_filename(cell["cell_id"], row["distribution"], row["rep"])
        if output.exists():
            validate_timing_record(output, cell=cell, phase=args.phase, distribution=row["distribution"], rep=row["rep"], gpu=args.gpu, eligibility_sha256=eligibility_hash, lock=lock)
            print(f"[resume {position}/{len(plan)}] {output.name}", flush=True)
            continue
        command = [
            sys.executable, str(HERE / "run_one.py"),
            "--phase", args.phase,
            "--cell-json", json.dumps(cell, sort_keys=True, separators=(",", ":")),
            "--distribution", row["distribution"],
            "--rep", str(row["rep"]),
            "--physical-gpu", str(args.gpu),
            "--eligibility", str(eligibility_path),
            "--out", str(output),
        ]
        print(f"[run {position}/{len(plan)}] {row['cell_id']} {row['distribution']} rep={row['rep']}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
        with attempts.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"cell_id": cell["cell_id"], "distribution": row["distribution"], "record_exists": output.exists(), "record_sha256": file_sha256(output) if output.exists() else None, "rep": row["rep"], "returncode": completed.returncode, "utc": datetime.now(timezone.utc).isoformat()}, sort_keys=True, separators=(",", ":")) + "\n")
        if not output.exists():
            raise RuntimeError(f"timing child returned without evidence: {cell['cell_id']}")
    records = [
        validate_timing_record(
            raw / timing_filename(row["cell_id"], row["distribution"], row["rep"]),
            cell=by_id[row["cell_id"]], phase=args.phase, distribution=row["distribution"], rep=row["rep"], gpu=args.gpu, eligibility_sha256=eligibility_hash, lock=lock,
        )
        for row in plan
    ]
    stable_write(root / "run_status.json", {"campaign_id": campaign["campaign_id"], "complete": True, "expected_records": len(plan), "failed_processes": sum(record["ok"] is not True for record in records), "gpu_after": gpu_snapshot(args.gpu), "observed_records": len(records), "phase": args.phase})
    active.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
