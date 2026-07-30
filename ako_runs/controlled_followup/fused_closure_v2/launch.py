#!/usr/bin/env python3
"""Launch the preregistered randomized-complete-block performance campaign."""

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
    from ako_runs.controlled_followup.fused_closure_v2 import core, provenance
else:  # pragma: no cover
    from . import core, provenance


TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _command(argv: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=core.REPO_ROOT,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001 - best-effort platform evidence
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def _preflight(physical_gpu: int) -> dict[str, Any]:
    gpu = _command(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-gpu=index,uuid,name,driver_version,persistence_mode,temperature.gpu,clocks.sm,clocks.max.sm,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu.get("returncode") != 0 or not gpu.get("stdout"):
        raise core.ClosureError(f"nvidia-smi GPU preflight failed: {gpu}")
    compute = _command(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if compute.get("returncode") != 0:
        raise core.ClosureError(f"nvidia-smi compute-process query failed: {compute}")
    if compute.get("stdout", "").strip():
        raise core.ClosureError(
            "preregistered GPU is busy; refusing to contaminate serialized blocks: "
            + compute["stdout"]
        )
    return {
        "captured_utc": _utc_now(),
        "gpu_query": gpu,
        "compute_processes_before_launch": compute,
        "nvidia_smi_full": _command(["nvidia-smi", "-i", str(physical_gpu), "-q"]),
        "nvcc": _command(["/usr/local/cuda-13.1/bin/nvcc", "--version"]),
        "python": sys.version,
        "host": _command(["uname", "-a"]),
        "git_head": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
                "TORCH_CUDA_ARCH_LIST",
                "TORCH_EXTENSIONS_DIR",
            )
        },
    }


def _load_gate_summary(
    campaign: dict[str, Any], source_receipt: dict[str, Any], path: Path
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise core.ClosureError(f"missing complete gate summary: {path}")
    summary = core.read_json(path)
    if path.read_bytes() != core.stable_json_bytes(summary):
        raise core.ClosureError("gate summary is not stable JSON")
    if summary.get("record_type") != "fused_closure_v2_gate_summary":
        raise core.ClosureError("foreign gate summary record type")
    binding = summary.get("binding", {})
    if binding.get("campaign_canonical_sha256") != core.canonical_sha256(campaign):
        raise core.ClosureError("gate summary campaign binding differs")
    if binding.get("source_receipt_sha256") != core.sha256_file(
        core.SOURCE_RECEIPT_PATH
    ):
        raise core.ClosureError("gate summary source receipt binding differs")
    if binding.get("candidate_sha256") != source_receipt["candidate_sha256"]:
        raise core.ClosureError("gate summary candidate bindings differ")
    if not summary.get("coverage_complete") or not summary.get(
        "performance_launch_allowed"
    ):
        raise core.ClosureError("gate adjudication is incomplete")
    if summary.get("observed_records") != summary.get("expected_records"):
        raise core.ClosureError("gate summary observed/expected counts differ")
    adjudications = summary.get("adjudications", [])
    if [item.get("candidate_id") for item in adjudications] != campaign[
        "candidate_order"
    ]:
        raise core.ClosureError("gate adjudication candidate order differs")
    expected_relatives = [
        str(Path("raw") / case_id / f"seed{seed_index:03d}.json")
        for case_id in campaign["gate"]["case_ids"]
        for seed_index in range(64)
    ]
    if set(summary.get("raw_bundle_sha256", {})) != set(expected_relatives):
        raise core.ClosureError("gate raw-bundle path set differs from frozen plan")
    raw_records = []
    for relative in expected_relatives:
        bundle_path = path.parent / relative
        if not bundle_path.is_file():
            raise core.ClosureError(f"gate raw bundle is missing: {bundle_path}")
        if core.sha256_file(bundle_path) != summary["raw_bundle_sha256"][relative]:
            raise core.ClosureError(f"gate raw bundle hash differs: {bundle_path}")
        bundle = core.read_json(bundle_path)
        if bundle_path.read_bytes() != core.stable_json_bytes(bundle):
            raise core.ClosureError(f"gate raw bundle is not stable JSON: {bundle_path}")
        if bundle.get("binding") != binding or not isinstance(bundle.get("records"), list):
            raise core.ClosureError(f"gate raw bundle binding differs: {bundle_path}")
        raw_records.extend(bundle["records"])
    if len(raw_records) != summary["expected_records"] or core.canonical_sha256(
        raw_records
    ) != summary.get("raw_records_canonical_sha256"):
        raise core.ClosureError("gate raw records no longer match gate summary")
    return summary, core.sha256_file(path)


def _binding(
    campaign: dict[str, Any],
    source_receipt: dict[str, Any],
    gate_summary: dict[str, Any],
    gate_summary_sha256: str,
) -> dict[str, Any]:
    return {
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": core.canonical_sha256(campaign),
        "source_receipt_sha256": core.sha256_file(core.SOURCE_RECEIPT_PATH),
        "performance_protocol_sha256": core.protocol_sha256(campaign),
        "candidate_sha256": source_receipt["candidate_sha256"],
        "gate_summary_sha256": gate_summary_sha256,
        "gate_summary_raw_records_canonical_sha256": gate_summary[
            "raw_records_canonical_sha256"
        ],
        "gate_eligibility": {
            item["candidate_id"]: item["same_contract_eligible"]
            for item in gate_summary["adjudications"]
        },
    }


def _raw_relative(item: dict[str, Any]) -> Path:
    return Path("raw") / f"block{item['block']:02d}" / f"{item['candidate_id']}.json"


def _validate_raw(
    path: Path,
    record: Any,
    *,
    item: dict[str, Any],
    binding: dict[str, Any],
    launch_sha256: str,
) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("record_type") != (
        "fused_closure_v2_performance_measurement"
    ):
        raise core.ClosureError(f"invalid performance record: {path}")
    exact = {
        "candidate_id": item["candidate_id"],
        "block": item["block"],
        "position": item["position"],
        "campaign_canonical_sha256": binding["campaign_canonical_sha256"],
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "performance_protocol_sha256": binding["performance_protocol_sha256"],
        "gate_summary_sha256": binding["gate_summary_sha256"],
        "candidate_definition_sha256": binding["candidate_sha256"][
            item["candidate_id"]
        ],
        "performance_launch_receipt_sha256": launch_sha256,
    }
    for key, expected in exact.items():
        if record.get(key) != expected:
            raise core.ClosureError(f"performance record {key} binding differs: {path}")
    if path.read_bytes() != core.stable_json_bytes(record):
        raise core.ClosureError(f"performance record is not stable JSON: {path}")
    return record


def _launcher_failure_record(
    *,
    campaign: dict[str, Any],
    item: dict[str, Any],
    binding: dict[str, Any],
    launch_sha256: str,
    error: str,
    trace: str,
) -> dict[str, Any]:
    definition = core.candidates_by_id(campaign)[item["candidate_id"]]
    return {
        "schema_version": 1,
        "record_type": "fused_closure_v2_performance_measurement",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": binding["campaign_canonical_sha256"],
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "performance_protocol_sha256": binding["performance_protocol_sha256"],
        "gate_summary_sha256": binding["gate_summary_sha256"],
        "performance_launch_receipt_sha256": launch_sha256,
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
        "process_completed_utc": _utc_now(),
        "recorded_by": "launch.py after child failed before writing raw evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--tag", default="performance_v1")
    parser.add_argument("--gate-tag", default="gate_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for value in (args.tag, args.gate_tag):
        if not TAG_PATTERN.fullmatch(value):
            parser.error("tags may contain only letters, digits, dot, underscore, hyphen")
    campaign = core.load_campaign()
    source_receipt = provenance.verify_receipt()
    if args.gpu != campaign["performance_protocol"]["physical_gpu"]:
        parser.error("--gpu must equal preregistered physical GPU 0")
    gate_path = core.RESULTS_ROOT / args.gate_tag / "gate_summary.json"
    gate_summary, gate_sha = _load_gate_summary(campaign, source_receipt, gate_path)
    binding = _binding(campaign, source_receipt, gate_summary, gate_sha)
    plan = core.block_plan(campaign)
    result_root = core.RESULTS_ROOT / args.tag
    launch_path = result_root / "launch_receipt.json"
    status_path = result_root / "launch_status.json"
    if args.dry_run:
        print(
            core.stable_json_bytes(
                {
                    "binding": binding,
                    "gate_summary": str(gate_path.relative_to(core.REPO_ROOT)),
                    "result_root": str(result_root.relative_to(core.REPO_ROOT)),
                    "plan": plan,
                }
            ).decode("utf-8"),
            end="",
        )
        return 0

    if not launch_path.exists() and result_root.exists() and any(result_root.iterdir()):
        raise core.ClosureError("new performance tag directory is not empty")
    platform = _preflight(args.gpu)
    if launch_path.exists():
        launch = core.read_json(launch_path)
        if (
            launch.get("binding") != binding
            or launch.get("plan") != plan
            or launch.get("result_root")
            != str(result_root.relative_to(core.REPO_ROOT))
            or launch_path.read_bytes() != core.stable_json_bytes(launch)
        ):
            raise core.ClosureError("existing performance launch differs from plan")
    else:
        launch = {
            "schema_version": 1,
            "record_type": "fused_closure_v2_performance_launch",
            "created_utc": _utc_now(),
            "tag": args.tag,
            "result_root": str(result_root.relative_to(core.REPO_ROOT)),
            "gate_summary_path": str(gate_path.relative_to(core.REPO_ROOT)),
            "binding": binding,
            "protocol": campaign["performance_protocol"],
            "plan": plan,
            "expected_records": len(plan),
            "platform_preflight": platform,
        }
        core.atomic_json(launch_path, launch)
    launch_sha = core.sha256_file(launch_path)
    expected_paths = {
        str((result_root / _raw_relative(item)).resolve()): item for item in plan
    }
    outcomes = []
    failures = 0
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = "0"
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = str(core.REPO_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    for ordinal, item in enumerate(plan, 1):
        path = result_root / _raw_relative(item)
        if path.exists():
            record = _validate_raw(
                path,
                core.read_json(path),
                item=item,
                binding=binding,
                launch_sha256=launch_sha,
            )
            disposition = "resumed"
            returncode = 0 if record.get("ok") else 1
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
                completed = subprocess.run(
                    argv,
                    cwd=core.REPO_ROOT,
                    env=child_env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                returncode = completed.returncode
                log = {
                    "argv": argv,
                    "started_utc": started,
                    "completed_utc": _utc_now(),
                    "returncode": returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            except Exception as exc:  # noqa: BLE001 - retained evidence
                returncode = 1
                log = {
                    "argv": argv,
                    "started_utc": started,
                    "completed_utc": _utc_now(),
                    "returncode": returncode,
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
                failure = _launcher_failure_record(
                    campaign=campaign,
                    item=item,
                    binding=binding,
                    launch_sha256=launch_sha,
                    error="measurement child exited without a raw record",
                    trace=log.get("traceback", log.get("stderr", "")),
                )
                core.atomic_json(path, failure)
            record = _validate_raw(
                path,
                core.read_json(path),
                item=item,
                binding=binding,
                launch_sha256=launch_sha,
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
        status = {
            "schema_version": 1,
            "record_type": "fused_closure_v2_launch_status",
            "tag": args.tag,
            "launch_receipt_sha256": launch_sha,
            "updated_utc": _utc_now(),
            "completed_records": len(outcomes),
            "expected_records": len(plan),
            "failed_records": failures,
            "complete": len(outcomes) == len(plan),
            "outcomes": outcomes,
        }
        core.atomic_json(status_path, status)
        median = record.get("timing_summary", {}).get("median_ms")
        suffix = f" median={median:.6f}ms" if isinstance(median, (int, float)) else ""
        print(
            f"measure {ordinal:03d}/{len(plan)} {item['candidate_id']} "
            f"block={item['block']} ok={record.get('ok')}{suffix}",
            flush=True,
        )

    observed_paths = {str(path.resolve()) for path in (result_root / "raw").rglob("*.json")}
    if observed_paths != set(expected_paths):
        extra = sorted(observed_paths - set(expected_paths))
        missing = sorted(set(expected_paths) - observed_paths)
        raise core.ClosureError(
            f"raw file set differs from plan; extra={extra[:5]} missing={missing[:5]}"
        )
    print(
        f"performance launch complete: records={len(outcomes)} failures={failures} "
        f"status={status_path}",
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
