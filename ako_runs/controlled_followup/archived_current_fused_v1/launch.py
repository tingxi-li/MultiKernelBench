#!/usr/bin/env python3
"""Launch the frozen 15-block campaign on an idle physical GPU."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.archived_current_fused_v1 import protocol
else:  # pragma: no cover
    from . import protocol


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def command(argv: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv,
            cwd=protocol.REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def preflight(campaign: dict[str, Any], gpu: int) -> dict[str, Any]:
    performance = campaign["performance_protocol"]
    if gpu != performance["physical_gpu"]:
        raise protocol.CampaignError(
            f"physical GPU {gpu} differs from frozen GPU {performance['physical_gpu']}"
        )
    identity = command(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=index,uuid,name,driver_version,persistence_mode,compute_cap,temperature.gpu,clocks.sm,clocks.max.sm,memory.total,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if identity.get("returncode") != 0 or not identity.get("stdout"):
        raise protocol.CampaignError(f"GPU identity query failed: {identity}")
    fields = [field.strip() for field in identity["stdout"].split(",")]
    if len(fields) < 4:
        raise protocol.CampaignError(f"GPU identity query was malformed: {identity}")
    if fields[1] != performance["expected_gpu_uuid"]:
        raise protocol.CampaignError(
            f"GPU UUID {fields[1]} differs from frozen {performance['expected_gpu_uuid']}"
        )
    if fields[2] != performance["expected_device_name"]:
        raise protocol.CampaignError(
            f"GPU name {fields[2]} differs from frozen {performance['expected_device_name']}"
        )
    compute = command(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if compute.get("returncode") != 0:
        raise protocol.CampaignError(f"GPU process query failed: {compute}")
    if compute.get("stdout", "").strip():
        raise protocol.CampaignError(f"GPU is busy: {compute['stdout']}")
    nvcc = command([performance["nvcc_path"], "--version"])
    if nvcc.get("returncode") != 0:
        raise protocol.CampaignError(f"nvcc query failed: {nvcc}")
    if f"release {performance['expected_nvcc_release']}" not in nvcc.get("stdout", ""):
        raise protocol.CampaignError(f"nvcc release differs from frozen protocol: {nvcc}")
    return {
        "captured_utc": utc_now(),
        "gpu_identity": identity,
        "compute_processes": compute,
        "nvidia_smi_full": command(["nvidia-smi", "-i", str(gpu), "-q"]),
        "nvcc": nvcc,
        "nvcc_file_sha256": protocol.sha256_file(Path(performance["nvcc_path"])),
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "host": command(["uname", "-a"]),
        "python": sys.version,
    }


def validate_raw(
    path: Path,
    record: dict[str, Any],
    *,
    job: dict[str, Any],
    campaign: dict[str, Any],
    launch_sha256: str,
) -> dict[str, Any]:
    if record.get("record_type") != "archived_current_fused_v1_measurement":
        raise protocol.CampaignError(f"bad raw record type: {path}")
    if record.get("campaign_canonical_sha256") != protocol.canonical_sha256(campaign):
        raise protocol.CampaignError(f"foreign campaign raw record: {path}")
    if record.get("job") != job:
        raise protocol.CampaignError(f"raw job binding differs: {path}")
    if record.get("launch_receipt_file_sha256") != launch_sha256:
        raise protocol.CampaignError(f"raw launch binding differs: {path}")
    if path.read_bytes() != protocol.stable_json_bytes(record):
        raise protocol.CampaignError(f"raw record is not stable JSON: {path}")
    if record.get("ok"):
        summary = record.get("timing_summary", {})
        diagnostic = record.get("correctness_diagnostic", {})
        if summary.get("n") != campaign["performance_protocol"]["trials"]:
            raise protocol.CampaignError(f"raw timing census differs: {path}")
        if diagnostic.get("pass") is not True:
            raise protocol.CampaignError(f"successful raw lacks correctness pass: {path}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol.validate_tag(args.tag)
    lock_stream = (protocol.HERE / ".launch.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise protocol.CampaignError("another archived-current launch is active") from exc
    campaign, receipt, jobs, lock = protocol.verify_lock()
    result_root = (protocol.HERE / "results" / args.tag).resolve()
    launch_path = result_root / "launch_receipt.json"
    if args.dry_run:
        print(protocol.stable_json_bytes(jobs).decode("utf-8"), end="")
        return 0
    if result_root.exists() and not launch_path.exists() and any(result_root.iterdir()):
        raise protocol.CampaignError("result namespace exists without a launch receipt")
    if not launch_path.exists():
        try:
            platform = preflight(campaign, args.gpu)
        except Exception as exc:  # noqa: BLE001 - preserve an explicit blocker
            result_root.mkdir(parents=True, exist_ok=True)
            blocker_path = result_root / "blocker.json"
            if blocker_path.exists():
                raise
            protocol.atomic_json(
                blocker_path,
                {
                    "schema_version": 1,
                    "record_type": "archived_current_fused_v1_launch_blocker",
                    "created_utc": utc_now(),
                    "tag": args.tag,
                    "gpu": args.gpu,
                    "campaign_canonical_sha256": protocol.canonical_sha256(campaign),
                    "launch_lock_file_sha256": protocol.sha256_file(protocol.LOCK_PATH),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        launch = {
            "schema_version": 1,
            "record_type": "archived_current_fused_v1_launch",
            "created_utc": utc_now(),
            "tag": args.tag,
            "result_root": str(result_root.relative_to(protocol.REPO_ROOT)),
            "physical_gpu": args.gpu,
            "campaign_canonical_sha256": protocol.canonical_sha256(campaign),
            "source_receipt_file_sha256": protocol.sha256_file(protocol.SOURCE_RECEIPT_PATH),
            "source_receipt_canonical_sha256": protocol.canonical_sha256(receipt),
            "jobs_file_sha256": protocol.sha256_file(protocol.JOBS_PATH),
            "jobs_canonical_sha256": protocol.canonical_sha256(jobs),
            "launch_lock_file_sha256": protocol.sha256_file(protocol.LOCK_PATH),
            "launch_lock_canonical_sha256": protocol.canonical_sha256(lock),
            "performance_protocol": campaign["performance_protocol"],
            "expected_records": jobs["expected_records"],
            "plan": jobs["plan"],
            "platform_preflight": platform,
        }
        protocol.atomic_json(launch_path, launch)
    else:
        launch = protocol.read_json(launch_path)
        if launch.get("plan") != jobs["plan"] or launch.get("launch_lock_file_sha256") != protocol.sha256_file(protocol.LOCK_PATH):
            raise protocol.CampaignError("existing launch receipt differs from frozen plan")
        if launch_path.read_bytes() != protocol.stable_json_bytes(launch):
            raise protocol.CampaignError("existing launch receipt is not stable JSON")
    launch_sha256 = protocol.sha256_file(launch_path)
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    child_env["CUDA_HOME"] = "/usr/local/cuda-13.1"
    child_env["PATH"] = "/usr/local/cuda-13.1/bin:" + child_env.get("PATH", "")
    child_env["PYTHONPATH"] = str(protocol.REPO_ROOT) + (
        os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else ""
    )
    child_env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    outcomes: list[dict[str, Any]] = []
    failures = 0
    for ordinal, job in enumerate(jobs["plan"], 1):
        protocol.verify_lock()
        raw_path = result_root / protocol.raw_relative(job)
        if raw_path.exists():
            record = validate_raw(
                raw_path,
                protocol.read_json(raw_path),
                job=job,
                campaign=campaign,
                launch_sha256=launch_sha256,
            )
            disposition = "resumed"
            returncode = 0 if record.get("ok") else 1
        else:
            argv = [
                sys.executable,
                str(protocol.HERE / "run_one.py"),
                "--job-id",
                job["job_id"],
                "--launch-receipt",
                str(launch_path.resolve()),
                "--out",
                str(raw_path.resolve()),
            ]
            started = utc_now()
            completed = subprocess.run(
                argv,
                cwd=protocol.REPO_ROOT,
                env=child_env,
                check=False,
                capture_output=True,
                text=True,
            )
            returncode = completed.returncode
            log_path = result_root / "logs" / f"block{job['block']:02d}" / f"{job['subject_id']}.json"
            protocol.atomic_json(
                log_path,
                {
                    "argv": argv,
                    "started_utc": started,
                    "completed_utc": utc_now(),
                    "returncode": returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )
            if not raw_path.exists():
                protocol.atomic_json(
                    raw_path,
                    {
                        "schema_version": 1,
                        "record_type": "archived_current_fused_v1_measurement",
                        "campaign_id": campaign["campaign_id"],
                        "campaign_canonical_sha256": protocol.canonical_sha256(campaign),
                        "source_receipt_file_sha256": protocol.sha256_file(protocol.SOURCE_RECEIPT_PATH),
                        "launch_lock_file_sha256": protocol.sha256_file(protocol.LOCK_PATH),
                        "jobs_file_sha256": protocol.sha256_file(protocol.JOBS_PATH),
                        "launch_receipt_file_sha256": launch_sha256,
                        "job": job,
                        "subject": protocol.subject_map(campaign)[job["subject_id"]],
                        "ok": False,
                        "error": "measurement child exited without a raw record",
                        "child_stderr": completed.stderr,
                    },
                )
            record = validate_raw(
                raw_path,
                protocol.read_json(raw_path),
                job=job,
                campaign=campaign,
                launch_sha256=launch_sha256,
            )
            disposition = "launched"
        if not record.get("ok") or returncode != 0:
            failures += 1
        outcomes.append(
            {
                **job,
                "raw_path": str(raw_path.relative_to(protocol.REPO_ROOT)),
                "raw_sha256": protocol.sha256_file(raw_path),
                "ok": bool(record.get("ok")),
                "child_returncode": returncode,
                "disposition": disposition,
            }
        )
        protocol.atomic_json(
            result_root / "launch_status.json",
            {
                "schema_version": 1,
                "record_type": "archived_current_fused_v1_launch_status",
                "updated_utc": utc_now(),
                "completed": ordinal,
                "expected": jobs["expected_records"],
                "failures": failures,
                "outcomes": outcomes,
            },
        )
        print(
            f"[{ordinal:02d}/{jobs['expected_records']}] {job['job_id']} "
            f"ok={record.get('ok')} median={record.get('timing_summary', {}).get('median_ms')}",
            flush=True,
        )
    protocol.verify_lock()
    completion = {
        "schema_version": 1,
        "record_type": "archived_current_fused_v1_completion",
        "completed_utc": utc_now(),
        "launch_receipt_file_sha256": launch_sha256,
        "expected_records": jobs["expected_records"],
        "observed_records": len(outcomes),
        "failures": failures,
        "success": failures == 0 and len(outcomes) == jobs["expected_records"],
        "outcomes": outcomes,
        "platform_postflight": command(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=uuid,name,driver_version,persistence_mode,temperature.gpu,clocks.sm,clocks.max.sm,memory.total,memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
    }
    completion_path = result_root / "completion_receipt.json"
    if completion_path.exists():
        observed = protocol.read_json(completion_path)
        if observed.get("launch_receipt_file_sha256") != launch_sha256:
            raise protocol.CampaignError("foreign completion receipt")
    else:
        protocol.atomic_json(completion_path, completion)
    return 0 if completion["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

