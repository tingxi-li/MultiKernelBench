#!/usr/bin/env python3
"""Checked randomized-complete-block launcher for frontier closure v3."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.fused_frontier_closure_v3 import (
        core,
        eligibility,
        provenance,
    )
else:  # pragma: no cover
    from . import core, eligibility, provenance


TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _command(argv: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        done = subprocess.run(
            argv,
            cwd=core.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "argv": argv,
            "returncode": done.returncode,
            "stdout": done.stdout.strip(),
            "stderr": done.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def _gpu_preflight(campaign: dict[str, Any], physical_gpu: int) -> dict[str, Any]:
    fields = (
        "index",
        "uuid",
        "name",
        "driver_version",
        "persistence_mode",
        "compute_cap",
        "pstate",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "clocks.sm",
        "clocks.mem",
        "clocks.max.sm",
        "clocks.max.memory",
        "power.limit",
        "temperature.gpu",
    )
    query = _command(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if query.get("returncode") != 0:
        raise core.ClosureError(f"GPU query failed: {query}")
    values = [value.strip() for value in query["stdout"].split(",")]
    if len(values) != len(fields):
        raise core.ClosureError("GPU query field count differs")
    snapshot = dict(zip(fields, values))
    hardware = campaign["hardware"]
    if (
        snapshot["index"] != str(physical_gpu)
        or snapshot["uuid"] != hardware["required_uuid"]
        or snapshot["name"] != hardware["required_name"]
        or snapshot["compute_cap"] != hardware["required_compute_capability"]
    ):
        raise core.ClosureError(f"GPU identity differs: {snapshot}")
    processes = _command(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if processes.get("returncode") != 0 or processes.get("stdout", "").strip():
        raise core.ClosureError(f"GPU is busy or process query failed: {processes}")
    nvcc = _command(["/usr/local/cuda-13.1/bin/nvcc", "--version"])
    if nvcc.get("returncode") != 0 or "release 13.1" not in nvcc.get("stdout", ""):
        raise core.ClosureError(f"CUDA 13.1 nvcc unavailable: {nvcc}")
    return {
        "captured_utc": _utc_now(),
        "gpu": snapshot,
        "compute_processes_before_launch": processes,
        "nvcc": nvcc,
        "nvidia_smi_full": _command(["nvidia-smi", "--id=3", "-q"]),
        "python": sys.version,
        "host": _command(["uname", "-a"]),
        "git_head": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
    }


def _binding(
    campaign: dict[str, Any],
    source: dict[str, Any],
    eligibility_receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": core.canonical_sha256(campaign),
        "source_receipt_sha256": core.sha256_file(core.SOURCE_RECEIPT_PATH),
        "eligibility_receipt_sha256": core.sha256_file(
            core.ELIGIBILITY_RECEIPT_PATH
        ),
        "eligibility_receipt_canonical_sha256": core.canonical_sha256(
            eligibility_receipt
        ),
        "eligibility_scope": eligibility_receipt["scope"],
        "eligibility_claim_limit": eligibility_receipt["claim_limit"],
        "performance_protocol_sha256": core.protocol_sha256(campaign),
        "candidate_sha256": source["candidate_sha256"],
    }


def _raw_relative(item: dict[str, Any]) -> Path:
    return Path("raw") / f"block{item['block']:02d}" / f"{item['candidate_id']}.json"


def _validate_raw(
    path: Path,
    record: dict[str, Any],
    item: dict[str, Any],
    binding: dict[str, Any],
    launch_sha: str,
) -> dict[str, Any]:
    if record.get("record_type") != "fused_frontier_closure_v3_measurement":
        raise core.ClosureError(f"foreign raw record: {path}")
    expected = {
        "candidate_id": item["candidate_id"],
        "block": item["block"],
        "position": item["position"],
        "campaign_canonical_sha256": binding["campaign_canonical_sha256"],
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "eligibility_receipt_sha256": binding["eligibility_receipt_sha256"],
        "performance_protocol_sha256": binding["performance_protocol_sha256"],
        "candidate_definition_sha256": binding["candidate_sha256"][
            item["candidate_id"]
        ],
        "launch_receipt_sha256": launch_sha,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise core.ClosureError(f"raw {key} binding differs: {path}")
    if path.read_bytes() != core.stable_json_bytes(record):
        raise core.ClosureError(f"raw record is not stable JSON: {path}")
    return record


def _failure_record(
    campaign: dict[str, Any],
    item: dict[str, Any],
    binding: dict[str, Any],
    launch_sha: str,
    error: str,
    trace: str,
) -> dict[str, Any]:
    definition = core.candidates_by_id(campaign)[item["candidate_id"]]
    return {
        "schema_version": 1,
        "record_type": "fused_frontier_closure_v3_measurement",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": binding["campaign_canonical_sha256"],
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "eligibility_receipt_sha256": binding["eligibility_receipt_sha256"],
        "performance_protocol_sha256": binding["performance_protocol_sha256"],
        "launch_receipt_sha256": launch_sha,
        "candidate_id": item["candidate_id"],
        "candidate_definition": definition,
        "candidate_definition_sha256": binding["candidate_sha256"][
            item["candidate_id"]
        ],
        "block": item["block"],
        "position": item["position"],
        "protocol": campaign["performance_protocol"],
        "ok": False,
        "error": error,
        "traceback": trace,
        "completed_utc": _utc_now(),
        "recorded_by": "checked launcher after child exited without raw evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--tag", default="performance_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not TAG.fullmatch(args.tag):
        parser.error("unsafe result tag")
    campaign = core.load_campaign()
    source = provenance.verify_receipt()
    eligible = eligibility.verify_receipt()
    if not eligible["all_candidates_original_gate_eligible"]:
        raise core.ClosureError("not all candidates have original gate eligibility")
    if args.gpu != campaign["hardware"]["physical_gpu"]:
        parser.error("--gpu must equal preregistered physical GPU 3")
    binding = _binding(campaign, source, eligible)
    plan = core.block_plan(campaign)
    result_root = core.RESULTS_ROOT / args.tag
    launch_path = result_root / "launch_receipt.json"
    status_path = result_root / "launch_status.json"
    if args.dry_run:
        print(
            core.stable_json_bytes(
                {
                    "binding": binding,
                    "result_root": str(result_root.relative_to(core.REPO_ROOT)),
                    "plan": plan,
                }
            ).decode(),
            end="",
        )
        return 0
    if not launch_path.exists() and result_root.exists() and any(result_root.iterdir()):
        raise core.ClosureError("new result tag directory is not empty")
    platform = _gpu_preflight(campaign, args.gpu)
    if launch_path.exists():
        launch = core.read_json(launch_path)
        if (
            launch.get("binding") != binding
            or launch.get("plan") != plan
            or launch_path.read_bytes() != core.stable_json_bytes(launch)
        ):
            raise core.ClosureError("existing launch receipt differs")
    else:
        launch = {
            "schema_version": 1,
            "record_type": "fused_frontier_closure_v3_launch",
            "created_utc": _utc_now(),
            "tag": args.tag,
            "result_root": str(result_root.relative_to(core.REPO_ROOT)),
            "binding": binding,
            "protocol": campaign["performance_protocol"],
            "plan": plan,
            "expected_records": len(plan),
            "platform_preflight": platform,
        }
        core.atomic_json(launch_path, launch)
    launch_sha = core.sha256_file(launch_path)
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = "3"
    child_env["CUDA_HOME"] = "/usr/local/cuda-13.1"
    child_env["PATH"] = "/usr/local/cuda-13.1/bin:" + child_env.get("PATH", "")
    child_env["PYTHONPATH"] = str(core.REPO_ROOT) + (
        os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else ""
    )
    outcomes, failures = [], 0
    for ordinal, item in enumerate(plan, 1):
        path = result_root / _raw_relative(item)
        if path.exists():
            record = _validate_raw(
                path, core.read_json(path), item, binding, launch_sha
            )
            disposition, returncode = "resumed", 0 if record.get("ok") else 1
        else:
            argv = [
                sys.executable,
                str(core.HERE / "measure.py"),
                "--candidate",
                item["candidate_id"],
                "--block",
                str(item["block"]),
                "--position",
                str(item["position"]),
                "--launch-receipt",
                str(launch_path.resolve()),
                "--out",
                str(path.resolve()),
            ]
            started = _utc_now()
            try:
                done = subprocess.run(
                    argv,
                    cwd=core.REPO_ROOT,
                    env=child_env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                returncode = done.returncode
                log = {
                    "argv": argv,
                    "started_utc": started,
                    "completed_utc": _utc_now(),
                    "returncode": returncode,
                    "stdout": done.stdout,
                    "stderr": done.stderr,
                }
            except Exception as exc:  # noqa: BLE001
                returncode = 1
                log = {
                    "argv": argv,
                    "started_utc": started,
                    "completed_utc": _utc_now(),
                    "returncode": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            log_path = (
                result_root
                / "logs"
                / f"block{item['block']:02d}"
                / f"{item['candidate_id']}.json"
            )
            core.atomic_json(log_path, log)
            if not path.exists():
                core.atomic_json(
                    path,
                    _failure_record(
                        campaign,
                        item,
                        binding,
                        launch_sha,
                        "measurement child exited without raw record",
                        log.get("traceback", log.get("stderr", "")),
                    ),
                )
            record = _validate_raw(
                path, core.read_json(path), item, binding, launch_sha
            )
            disposition = "launched"
        if not record.get("ok") or returncode != 0:
            failures += 1
        outcome = {
            **item,
            "raw_path": str(path.relative_to(core.REPO_ROOT)),
            "raw_sha256": core.sha256_file(path),
            "ok": bool(record.get("ok")),
            "child_returncode": returncode,
            "disposition": disposition,
        }
        outcomes.append(outcome)
        core.atomic_json(
            status_path,
            {
                "schema_version": 1,
                "record_type": "fused_frontier_closure_v3_launch_status",
                "tag": args.tag,
                "launch_receipt_sha256": launch_sha,
                "updated_utc": _utc_now(),
                "completed_records": len(outcomes),
                "expected_records": len(plan),
                "failed_records": failures,
                "complete": len(outcomes) == len(plan),
                "outcomes": outcomes,
            },
        )
        median = record.get("timing_summary", {}).get("median_ms")
        suffix = f" median={median:.6f}ms" if isinstance(median, (int, float)) else ""
        print(
            f"measure {ordinal:03d}/{len(plan)} {item['candidate_id']} "
            f"block={item['block']} ok={record.get('ok')}{suffix}",
            flush=True,
        )
    expected_paths = {str((result_root / _raw_relative(item)).resolve()) for item in plan}
    observed_paths = {
        str(path.resolve()) for path in (result_root / "raw").rglob("*.json")
    }
    if observed_paths != expected_paths:
        raise core.ClosureError("raw path set differs from launch plan")
    print(f"launch complete records={len(plan)} failures={failures}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
