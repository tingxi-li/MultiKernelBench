#!/usr/bin/env python3
"""Launch v1r1 screen or confirmation with recovery-bound child records."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import common, validate
except ImportError:  # pragma: no cover - direct-script CLI
    import common  # type: ignore
    import validate  # type: ignore


def _parent_modules():
    if str(common.PARENT) not in sys.path:
        sys.path.insert(0, str(common.PARENT))
    import core as parent_core  # type: ignore

    name = "fused_crossed_parent_launch_v1r1"
    spec = importlib.util.spec_from_file_location(name, common.PARENT / "launch.py")
    if spec is None or spec.loader is None:
        raise common.RecoveryError("cannot load frozen parent timing launcher")
    parent_launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_launch)
    return parent_core, parent_launch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("screen", "confirmation"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    common.require(args.tag == common.RESULT_TAG, "recovery result tag differs")
    common.require(args.gpu == 0, "all recovery timing must use physical GPU 0")
    ready = validate.validate_launch_ready(args.gpu, allow_busy=args.allow_busy)
    validate.ensure_remote_receipt(ready)
    binding = ready["binding"]
    common.validate_retained_tree(binding)
    parent_core, parent_launch = _parent_modules()
    campaign, cells, lock = parent_core.load_contract()
    eligibility_path = Path(args.eligibility).resolve()
    common.require(
        eligibility_path.is_relative_to(common.RESULT_ROOT.resolve()),
        "eligibility escapes v1r1 result root",
    )
    eligibility = common.read_json(eligibility_path)
    common.validate_binding(eligibility, binding, "timing eligibility")
    common.require(
        eligibility.get("campaign_id") == common.CAMPAIGN_ID
        and eligibility.get("launch_lock_sha256")
        == common.PARENT_LAUNCH_LOCK_SHA256,
        "foreign timing eligibility",
    )
    if args.phase == "screen":
        common.require(
            eligibility.get("record_type") == "fused_crossed_audit_summary"
            and eligibility.get("complete") is True,
            "screen requires a complete recovery-bound audit summary",
        )
        ids = set(eligibility.get("timing_eligible_cell_ids", []))
        plan = parent_core.screen_plan(cells, ids)
    else:
        common.require(
            eligibility.get("record_type")
            == "fused_crossed_confirmation_selection"
            and eligibility.get("complete") is True,
            "confirmation requires a complete recovery-bound selection",
        )
        ids = set(eligibility.get("selected_cell_ids", []))
        plan = parent_core.confirmation_plan(ids)
    by_id = {cell["cell_id"]: cell for cell in cells}
    common.require(ids and ids.issubset(by_id), "eligibility has no valid candidates")
    root = common.RESULT_ROOT / args.phase
    root.mkdir(parents=True, exist_ok=True)
    active = (root / "active.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise common.RecoveryError(f"another {args.phase} launcher is active") from None
    eligibility_hash = common.file_sha256(eligibility_path)
    contract = {
        "campaign_id": common.CAMPAIGN_ID,
        "eligibility_path": common.repo_path(eligibility_path),
        "eligibility_sha256": eligibility_hash,
        "execution_order": plan,
        "git_commit": ready["recovery_git_commit"],
        "launch_lock_sha256": common.PARENT_LAUNCH_LOCK_SHA256,
        "logical_device": "cuda:0",
        "phase": args.phase,
        "physical_gpu": args.gpu,
        "source_bundle_sha256": common.PARENT_SOURCE_BUNDLE_SHA256,
        "tag": args.tag,
    }
    receipt_path = root / "launch_receipt.json"
    receipt = common.add_binding(
        {
            "contract": contract,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": ready["gpu"],
            "host": platform.node(),
            "record_type": "fused_crossed_timing_receipt",
            "schema_version": 1,
        },
        binding,
    )
    if receipt_path.exists():
        observed = common.read_json(receipt_path)
        common.validate_binding(observed, binding, "timing receipt")
        common.require(observed.get("contract") == contract, "timing receipt differs")
    else:
        common.exclusive_json(receipt_path, receipt)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "CUDA_HOME": "/usr/local/cuda-13.1",
            "MKB_CROSSED_RECOVERY_COMMIT": ready["recovery_git_commit"],
            "MKB_CROSSED_RECOVERY_LOCK_SHA256": binding[
                "recovery_lock_sha256"
            ],
            "PATH": "/usr/local/cuda-13.1/bin:" + env.get("PATH", ""),
            "TORCH_EXTENSIONS_DIR": str(common.PARENT / ".torch_ext/gpu0"),
        }
    )
    env.setdefault("MAX_JOBS", "4")
    attempts_path = root / "attempts.jsonl"
    if attempts_path.exists():
        for index, row in enumerate(common.read_jsonl(attempts_path), 1):
            common.validate_binding(row, binding, f"attempt row {index}")
    for position, row in enumerate(plan, 1):
        cell = by_id[row["cell_id"]]
        output = raw / parent_core.timing_filename(
            cell["cell_id"], row["distribution"], row["rep"]
        )
        if output.exists():
            record = parent_launch.validate_timing_record(
                output,
                cell=cell,
                phase=args.phase,
                distribution=row["distribution"],
                rep=row["rep"],
                gpu=args.gpu,
                eligibility_sha256=eligibility_hash,
                lock=lock,
            )
            common.validate_binding(record, binding, f"timing record {output}")
            print(f"[resume {position}/{len(plan)}] {output.name}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            (
                "ako_runs.controlled_followup.fused_epilogue_crossed_v1."
                "recovery_v1r1.run_one"
            ),
            "--phase",
            args.phase,
            "--cell-json",
            json.dumps(cell, sort_keys=True, separators=(",", ":")),
            "--distribution",
            row["distribution"],
            "--rep",
            str(row["rep"]),
            "--physical-gpu",
            str(args.gpu),
            "--eligibility",
            str(eligibility_path),
            "--out",
            str(output),
        ]
        print(
            f"[run {position}/{len(plan)}] {row['cell_id']} "
            f"{row['distribution']} rep={row['rep']}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=common.REPO_ROOT, env=env)
        attempt = common.add_binding(
            {
                "cell_id": cell["cell_id"],
                "distribution": row["distribution"],
                "record_exists": output.exists(),
                "record_sha256": common.file_sha256(output) if output.exists() else None,
                "rep": row["rep"],
                "returncode": completed.returncode,
                "utc": datetime.now(timezone.utc).isoformat(),
            },
            binding,
        )
        with attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    attempt,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        common.require(output.exists(), f"timing child returned without evidence: {cell['cell_id']}")
    records = []
    for row in plan:
        cell = by_id[row["cell_id"]]
        path = raw / parent_core.timing_filename(
            cell["cell_id"], row["distribution"], row["rep"]
        )
        record = parent_launch.validate_timing_record(
            path,
            cell=cell,
            phase=args.phase,
            distribution=row["distribution"],
            rep=row["rep"],
            gpu=args.gpu,
            eligibility_sha256=eligibility_hash,
            lock=lock,
        )
        common.validate_binding(record, binding, f"timing record {path}")
        records.append(record)
    status = common.add_binding(
        {
            "campaign_id": common.CAMPAIGN_ID,
            "complete": True,
            "expected_records": len(plan),
            "failed_processes": sum(record["ok"] is not True for record in records),
            "gpu_after": parent_core.gpu_snapshot(args.gpu),
            "observed_records": len(records),
            "phase": args.phase,
        },
        binding,
    )
    status_path = root / "run_status.json"
    if status_path.exists():
        common.require(common.read_json(status_path) == status, "timing status differs")
    else:
        common.exclusive_json(status_path, status)
    common.validate_retained_tree(binding)
    active.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
