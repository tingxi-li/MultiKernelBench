#!/usr/bin/env python3
"""Freeze, validate, and run the two fresh current-Ada timing stages."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import protocol
except ImportError:
    import protocol  # type: ignore


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
GPU_LOCK_ID = "multikernelbench-GPU-45af34ad-0c74-74d0-ef3a-652090d837ae-timing"
GPU_LOCK_PATH = Path("/tmp") / f"{GPU_LOCK_ID}.lock"
GPU_SNAPSHOT_FIELDS = {
    "index",
    "uuid",
    "name",
    "driver_version",
    "compute_cap",
    "pstate",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "clocks.sm",
    "clocks.mem",
    "power.limit",
    "temperature.gpu",
}
for extra in (
    REPO_ROOT / "ako_runs/phase1_matmul",
    REPO_ROOT / "ako_runs/phase2_fused_sdpa",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if protocol.read_json(path) != value:
            raise protocol.ProtocolError(f"immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def execution_bindings() -> dict[str, Any]:
    contract = protocol.load_contract()
    imported = protocol.derive_imported_frontier(contract)
    sources = {
        protocol.repo_path(path): protocol.file_sha256(path)
        for path in protocol.local_source_paths()
    }
    return {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_execution_lock",
        "campaign_id": protocol.CAMPAIGN_ID,
        "contract_sha256": protocol.file_sha256(protocol.CONTRACT_PATH),
        "local_source_sha256": sources,
        "local_source_bundle_sha256": protocol.canonical_sha256(sources),
        "material_registry_sha256": protocol.canonical_sha256(contract["material_registry"]),
        "imported_frontier_sha256": protocol.canonical_sha256(imported),
        "selection_plan_sha256": protocol.canonical_sha256(
            protocol.timing_plan("selection_confirm")
        ),
        "selection_expected_records": 240,
        "terminal_expected_records": 120,
        "physical_gpu": 0,
    }


def validate_execution_lock(path: Path = protocol.EXECUTION_LOCK_PATH) -> dict[str, Any]:
    observed = protocol.read_json(path)
    expected = execution_bindings()
    if set(observed) != {*expected, "authorization_basis", "authorized_at_utc"} or any(
        observed.get(key) != value for key, value in expected.items()
    ):
        raise protocol.ProtocolError("execution lock fields or bindings changed")
    if not isinstance(observed["authorization_basis"], str) or not observed["authorization_basis"].strip():
        raise protocol.ProtocolError("execution lock lacks an authorization basis")
    if not isinstance(observed["authorized_at_utc"], str) or not observed["authorized_at_utc"].strip():
        raise protocol.ProtocolError("execution lock lacks an authorization timestamp")
    return observed


def freeze(authorization_basis: str) -> dict[str, Any]:
    if not authorization_basis.strip():
        raise protocol.ProtocolError("freeze requires a non-empty authorization basis")
    if protocol.EXECUTION_LOCK_PATH.exists():
        retained = validate_execution_lock()
        if retained["authorization_basis"] != authorization_basis.strip():
            raise protocol.ProtocolError("existing execution lock has another authorization basis")
        return retained
    value = {
        **execution_bindings(),
        "authorization_basis": authorization_basis.strip(),
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_once(protocol.EXECUTION_LOCK_PATH, value)
    return value


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
    )
    if completed.returncode:
        raise protocol.ProtocolError(completed.stderr.strip() or f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def remote_ready_paths(contract: dict[str, Any], stage: str) -> list[str]:
    if stage not in {"selection_confirm", "terminal_confirm"}:
        raise protocol.ProtocolError(f"unknown remote-readiness stage: {stage}")
    paths = [
        protocol.repo_path(path) for path in protocol.local_source_paths()
    ] + [
        protocol.repo_path(protocol.EXECUTION_LOCK_PATH),
        protocol.repo_path(protocol.RESULTS_ROOT / "imported_frontier.json"),
        *contract["material_registry"]["roles"].values(),
    ]
    if stage == "terminal_confirm":
        selection_root = protocol.RESULTS_ROOT / "selection_confirm"
        paths.extend(
            [
                protocol.repo_path(selection_root / "launch_receipt.json"),
                protocol.repo_path(selection_root / "run_status.json"),
                protocol.repo_path(protocol.RESULTS_ROOT / "selection_lock.json"),
            ]
            + [
                protocol.repo_path(
                    selection_root / "raw" / protocol.timing_filename(row)
                )
                for row in protocol.timing_plan("selection_confirm")
            ]
        )
    if len(paths) != len(set(paths)):
        raise protocol.ProtocolError("remote-readiness path set contains duplicates")
    return paths


def validate_remote_ready(contract: dict[str, Any], stage: str) -> dict[str, str]:
    paths = remote_ready_paths(contract, stage)
    if _git("status", "--porcelain", "--", *paths):
        raise protocol.ProtocolError(
            f"{stage} source, lock, input evidence, or registered materials are not committed"
        )
    _git("ls-files", "--error-unmatch", "--", *paths)
    branch = _git("branch", "--show-current")
    remote = _git("config", "--get", f"branch.{branch}.remote")
    merge_ref = _git("config", "--get", f"branch.{branch}.merge")
    head = _git("rev-parse", "HEAD")
    completed = subprocess.run(
        ["git", "ls-remote", "--exit-code", remote, merge_ref],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode or not completed.stdout.strip() or completed.stdout.split()[0] != head:
        raise protocol.ProtocolError("launch commit is not the configured upstream head")
    return {
        "git_commit": head,
        "upstream_remote": remote,
        "upstream_ref": merge_ref,
    }


def _git_file_sha256(commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode:
        raise protocol.ProtocolError(f"receipt commit lacks {relative}")
    return hashlib.sha256(completed.stdout).hexdigest()


def validate_recorded_git_binding(value: dict[str, Any]) -> None:
    commit = value.get("git_commit")
    remote, merge_ref = value.get("upstream_remote"), value.get("upstream_ref")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(remote, str)
        or not remote
        or not isinstance(merge_ref, str)
        or not merge_ref.startswith("refs/heads/")
    ):
        raise protocol.ProtocolError("launch receipt has malformed git/upstream fields")
    _git("cat-file", "-e", f"{commit}^{{commit}}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=30,
    ).returncode:
        raise protocol.ProtocolError("launch receipt commit is not in current HEAD history")
    branch = _git("branch", "--show-current")
    if (
        _git("config", "--get", f"branch.{branch}.remote") != remote
        or _git("config", "--get", f"branch.{branch}.merge") != merge_ref
    ):
        raise protocol.ProtocolError("launch receipt upstream differs from current branch")
    input_path, input_hash = value.get("input_artifact_path"), value.get("input_artifact_sha256")
    if (
        not isinstance(input_path, str)
        or not isinstance(input_hash, str)
        or _git_file_sha256(commit, input_path) != input_hash
        or _git_file_sha256(commit, protocol.repo_path(protocol.EXECUTION_LOCK_PATH))
        != value.get("execution_lock_sha256")
    ):
        raise protocol.ProtocolError("launch receipt inputs are not bound in its git commit")
    if value.get("stage") == "terminal_confirm":
        selection = protocol.read_json(REPO_ROOT / input_path)
        hashes = selection.get("selection_stage_hashes")
        if not isinstance(hashes, list) or not hashes:
            raise protocol.ProtocolError("terminal input lacks selection evidence hashes")
        for row in hashes:
            if (
                not isinstance(row, dict)
                or set(row) != {"path", "sha256"}
                or not isinstance(row["path"], str)
                or not isinstance(row["sha256"], str)
                or _git_file_sha256(commit, row["path"]) != row["sha256"]
            ):
                raise protocol.ProtocolError(
                    "terminal launch commit does not contain sealed selection evidence"
                )


def live_toolchain() -> dict[str, str]:
    import tilelang
    import torch
    import triton

    nvcc = subprocess.run(
        ["/usr/local/cuda-13.1/bin/nvcc", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if nvcc.returncode:
        raise protocol.ProtocolError("frozen nvcc is unavailable")
    release = next(
        (word.rstrip(",") for line in nvcc.stdout.splitlines() for word in line.split() if word.startswith("V13.")),
        None,
    )
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "tilelang": str(tilelang.__version__),
        "nvcc_release": str(release),
    }


def validate_gpu_snapshot(snapshot: Any, hardware: dict[str, Any]) -> dict[str, str]:
    if not isinstance(snapshot, dict) or set(snapshot) != GPU_SNAPSHOT_FIELDS or any(
        not isinstance(value, str) for value in snapshot.values()
    ):
        raise protocol.ProtocolError("GPU snapshot schema changed")
    expected = {
        "index": str(hardware["physical_gpu"]),
        "uuid": hardware["gpu_uuid"],
        "name": hardware["gpu_name"],
        "driver_version": hardware["driver_version"],
        "compute_cap": hardware["compute_capability"],
    }
    if any(snapshot.get(key) != value for key, value in expected.items()):
        raise protocol.ProtocolError("GPU snapshot identity differs from the frozen hardware")
    return snapshot


def gpu_idle_snapshot(phase: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    processes = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 1)]
        try:
            pid = int(fields[0])
        except (IndexError, ValueError) as exc:
            raise protocol.ProtocolError(f"cannot parse GPU process row: {line!r}") from exc
        processes.append(
            {"pid": pid, "used_memory_mib": fields[1] if len(fields) == 2 else ""}
        )
    foreign = [row for row in processes if row["pid"] != os.getpid()]
    return {
        "checked_at_unix": time.time(),
        "compute_processes": processes,
        "foreign_compute_processes": foreign,
        "idle_except_self": completed.returncode == 0 and not foreign,
        "phase": phase,
        "returncode": completed.returncode,
        "self_pid": os.getpid(),
        "stderr": completed.stderr.strip(),
    }


def validate_idle_snapshot(value: Any, phase: str) -> dict[str, Any]:
    required = {
        "checked_at_unix",
        "compute_processes",
        "foreign_compute_processes",
        "idle_except_self",
        "phase",
        "returncode",
        "self_pid",
        "stderr",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("phase") != phase:
        raise protocol.ProtocolError(f"invalid {phase} GPU-idle receipt")
    if (
        value.get("returncode") != 0
        or value.get("idle_except_self") is not True
        or value.get("foreign_compute_processes") != []
        or not isinstance(value.get("compute_processes"), list)
        or not isinstance(value.get("self_pid"), int)
        or not isinstance(value.get("checked_at_unix"), (int, float))
        or not math.isfinite(float(value["checked_at_unix"]))
        or not isinstance(value.get("stderr"), str)
    ):
        raise protocol.ProtocolError(f"physical GPU 0 was not idle at {phase}")
    for row in value["compute_processes"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"pid", "used_memory_mib"}
            or not isinstance(row["pid"], int)
            or not isinstance(row["used_memory_mib"], str)
            or row["pid"] != value["self_pid"]
        ):
            raise protocol.ProtocolError(f"foreign or malformed GPU process at {phase}")
    return value


def require_gpu_idle(phase: str) -> dict[str, Any]:
    return validate_idle_snapshot(gpu_idle_snapshot(phase), phase)


def _acquire_gpu_lock():
    GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = GPU_LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise protocol.ProtocolError(f"host-global physical GPU 0 lock is held: {GPU_LOCK_PATH}") from None
    return handle


def _validate_inherited_gpu_lock() -> None:
    raw = os.environ.get("FINITE_FRONTIER_GPU0_LOCK_FD", "")
    try:
        descriptor = int(raw)
        inherited = os.fstat(descriptor)
        expected = GPU_LOCK_PATH.stat()
    except (OSError, ValueError) as exc:
        raise protocol.ProtocolError("timing child lacks the inherited GPU 0 lock") from exc
    if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        raise protocol.ProtocolError("timing child inherited another GPU lock")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise protocol.ProtocolError("timing child does not share the held GPU 0 lock") from exc


def ready(stage: str = "selection_confirm") -> dict[str, Any]:
    contract = protocol.load_contract()
    lock = validate_execution_lock()
    remote = validate_remote_ready(contract, stage)
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core as source_core
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import validate as source_validate

    source_ready = source_validate.validate_launch_ready("campaign", 0)
    gpu = source_core.gpu_snapshot(0)
    validate_gpu_snapshot(gpu, contract["manifest"]["hardware"])
    toolchain = live_toolchain()
    if toolchain != contract["manifest"]["toolchain"]:
        raise protocol.ProtocolError("live toolchain differs from the current-Ada binding")
    _stage_plan(stage)
    return {
        **remote,
        "campaign_id": protocol.CAMPAIGN_ID,
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "gpu": gpu,
        "instrument_ready": source_ready,
        "toolchain": toolchain,
        "lock": lock,
    }


def _input_artifact(stage: str) -> Path:
    return (
        protocol.RESULTS_ROOT / "imported_frontier.json"
        if stage == "selection_confirm"
        else protocol.RESULTS_ROOT / "selection_lock.json"
    )


def _stage_plan(
    stage: str, *, rederive: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = _input_artifact(stage)
    value = protocol.read_json(artifact)
    if stage == "selection_confirm":
        if rederive:
            protocol.validate_imported_frontier(value)
        else:
            lock = validate_execution_lock()
            if (
                value.get("record_type") != "finite_frontier_ada_imported_frontier"
                or value.get("complete") is not True
                or protocol.canonical_sha256(value) != lock["imported_frontier_sha256"]
            ):
                raise protocol.ProtocolError("timing child received a foreign imported frontier")
        plan = protocol.timing_plan(stage)
    else:
        if rederive:
            try:
                from . import analyze
            except ImportError:
                import analyze  # type: ignore

            derived = analyze.selection_summary()
            if value != derived or value.get("terminal_authorized") is not True:
                raise protocol.ProtocolError("terminal selection lock is not re-derived/authorized")
        elif (
            value.get("record_type") != "finite_frontier_ada_selection_lock"
            or value.get("campaign_id") != protocol.CAMPAIGN_ID
            or value.get("complete") is not True
            or value.get("terminal_authorized") is not True
            or value.get("execution_lock_sha256")
            != protocol.file_sha256(protocol.EXECUTION_LOCK_PATH)
        ):
            raise protocol.ProtocolError("timing child received a foreign terminal selection lock")
        plan = protocol.timing_plan(stage, value["winners"])
        if value.get("terminal_plan_sha256") != protocol.canonical_sha256(plan):
            raise protocol.ProtocolError("terminal selection lock plan binding changed")
    return value, plan


def _expected_record(
    stage: str,
    row: dict[str, Any],
    input_path: Path,
    input_value: dict[str, Any],
    launch_binding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": row["cell_id"],
        "contract_sha256": protocol.file_sha256(protocol.CONTRACT_PATH),
        "distribution": row["distribution"],
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "input_artifact_path": protocol.repo_path(input_path),
        "input_artifact_sha256": protocol.file_sha256(input_path),
        "label": row["label"],
        **launch_binding,
        "physical_gpu": 0,
        "plan_position": row["position"],
        "record_kind": row["record_kind"],
        "row": row,
        "row_sha256": protocol.canonical_sha256(row),
        "stage": stage,
    }


def validate_launch_receipt(
    value: Any,
    expected_contract: dict[str, Any],
    contract: dict[str, Any] | None = None,
    *,
    validate_git: bool = True,
) -> dict[str, Any]:
    contract = contract or protocol.load_contract()
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "record_type",
            "created_at_utc",
            "contract",
            "gpu_idle_preflight",
        }
        or value.get("schema_version") != 1
        or value.get("record_type") != "finite_frontier_ada_launch_receipt"
        or not isinstance(value.get("created_at_utc"), str)
        or not isinstance(value.get("contract"), dict)
    ):
        raise protocol.ProtocolError("invalid timing-stage launch receipt schema")
    try:
        created = datetime.fromisoformat(value["created_at_utc"])
    except ValueError as exc:
        raise protocol.ProtocolError("invalid timing-stage launch timestamp") from exc
    if created.utcoffset() is None:
        raise protocol.ProtocolError("timing-stage launch timestamp lacks a timezone")
    observed = value["contract"]
    required = {
        *expected_contract,
        "git_commit",
        "gpu",
        "upstream_remote",
        "upstream_ref",
        "toolchain",
    }
    if set(observed) != required or any(
        observed.get(key) != expected for key, expected in expected_contract.items()
    ):
        raise protocol.ProtocolError("timing-stage launch contract changed")
    validate_gpu_snapshot(observed["gpu"], contract["manifest"]["hardware"])
    if observed["toolchain"] != contract["manifest"]["toolchain"]:
        raise protocol.ProtocolError("launch receipt toolchain changed")
    if validate_git:
        validate_recorded_git_binding(observed)
    validate_idle_snapshot(value["gpu_idle_preflight"], "stage_pre")
    return value


def launch_record_binding(receipt_path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "launch_created_at_utc": receipt["created_at_utc"],
        "launch_receipt_path": protocol.repo_path(receipt_path),
        "launch_receipt_sha256": protocol.file_sha256(receipt_path),
        "launch_stage_preflight_unix": float(
            receipt["gpu_idle_preflight"]["checked_at_unix"]
        ),
    }


def require_next_plan_position(
    raw: Path, plan: list[dict[str, Any]], row: dict[str, Any]
) -> None:
    prior_names = {protocol.timing_filename(item) for item in plan[: row["position"]]}
    observed_names = {path.name for path in raw.iterdir()} if raw.is_dir() else set()
    if observed_names != prior_names:
        raise protocol.ProtocolError("timing child is not the next frozen plan position")


def validate_run_status(
    value: Any, stage: str, expected_records: int, receipt_sha256: str
) -> dict[str, Any]:
    static = {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_run_status",
        "campaign_id": protocol.CAMPAIGN_ID,
        "complete": True,
        "expected_records": expected_records,
        "observed_records": expected_records,
        "launch_receipt_sha256": receipt_sha256,
        "stage": stage,
    }
    if not isinstance(value, dict) or set(value) != {*static, "gpu_idle_postflight"} or any(
        value.get(key) != expected for key, expected in static.items()
    ):
        raise protocol.ProtocolError(f"incomplete {stage} run status")
    validate_idle_snapshot(value["gpu_idle_postflight"], "stage_post")
    return value


def validate_record_sequence(
    records: list[dict[str, Any]], plan: list[dict[str, Any]]
) -> None:
    if len(records) != len(plan):
        raise protocol.ProtocolError("record sequence census changed")
    previous_end = -math.inf
    process_ids = set()
    for position, (record, row) in enumerate(zip(records, plan)):
        if row.get("position") != position or record.get("plan_position") != position:
            raise protocol.ProtocolError("record position differs from deterministic plan")
        start, end, pid = float(record["t_start"]), float(record["t_end"]), record["process_pid"]
        if start < previous_end:
            raise protocol.ProtocolError("record timestamps violate deterministic plan order")
        if pid in process_ids:
            raise protocol.ProtocolError("two timing records reused one process")
        process_ids.add(pid)
        previous_end = end


def validate_timing_record(
    path: Path,
    expected: dict[str, Any],
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = contract or protocol.load_contract()
    record = protocol.read_json(path)
    mismatch = [key for key, value in expected.items() if record.get(key) != value]
    if mismatch or record.get("record_type") != "finite_frontier_ada_timing_record" or record.get("schema_version") != 1:
        raise protocol.ProtocolError(f"foreign timing record {path}: {mismatch}")
    if record.get("ok") is not True or record.get("legacy_error", {}).get("gate_pass") is not True:
        raise protocol.ProtocolError(f"failed timing record: {path}")
    summary = protocol.summarize_times(record.get("times_ms", []))
    if any(record.get(key) != value for key, value in summary.items()):
        raise protocol.ProtocolError(f"timing summary changed: {path}")
    implementation = record.get("implementation_sha256")
    metadata = record.get("build_metadata")
    if (
        not isinstance(implementation, str)
        or re.fullmatch(r"[0-9a-f]{64}", implementation) is None
        or not isinstance(metadata, dict)
        or metadata.get("implementation_sha256") != implementation
        or not isinstance(metadata.get("builder"), str)
        or not metadata["builder"]
        or not isinstance(metadata.get("artifacts"), dict)
        or not isinstance(metadata.get("n_kernels"), int)
        or metadata["n_kernels"] != 2
    ):
        raise protocol.ProtocolError(f"timing record lacks a bound build fingerprint: {path}")
    if record.get("toolchain") != contract["manifest"]["toolchain"]:
        raise protocol.ProtocolError(f"timing record toolchain changed: {path}")
    validate_gpu_snapshot(record.get("gpu_preflight"), contract["manifest"]["hardware"])
    pre = validate_idle_snapshot(record.get("gpu_idle_preflight"), "record_pre")
    post = validate_idle_snapshot(record.get("gpu_idle_postflight"), "record_post")
    start, end = record.get("t_start"), record.get("t_end")
    compile_s = record.get("compile_s")
    launch_created = expected.get("launch_created_at_utc")
    launch_preflight = expected.get("launch_stage_preflight_unix")
    try:
        launch_created_unix = (
            datetime.fromisoformat(launch_created).timestamp()
            if isinstance(launch_created, str)
            else None
        )
    except ValueError:
        launch_created_unix = None
    if (
        record.get("trials") != contract["manifest"]["timing"]["trials"]
        or record.get("warmup_s") != contract["manifest"]["timing"]["warmup_s"]
        or not isinstance(record.get("warmup_iterations_actual"), int)
        or record["warmup_iterations_actual"] < 1
        or not isinstance(compile_s, (int, float))
        or not math.isfinite(float(compile_s))
        or float(compile_s) < 0
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(float(start))
        or not math.isfinite(float(end))
        or not float(start) < float(end)
        or (
            launch_created is not None
            and (
                launch_created_unix is None
                or not isinstance(launch_preflight, (int, float))
                or float(start) < max(launch_created_unix, float(launch_preflight))
            )
        )
        or not pre["checked_at_unix"] <= float(start)
        or not float(start) <= post["checked_at_unix"] <= float(end)
        or not isinstance(record.get("process_pid"), int)
        or record["process_pid"] <= 0
        or not isinstance(record.get("parent_pid"), int)
        or record["parent_pid"] <= 0
        or pre["self_pid"] != record["process_pid"]
        or post["self_pid"] != record["process_pid"]
    ):
        raise protocol.ProtocolError(f"timing/build/process receipt changed: {path}")
    return record


def _time_one(args: argparse.Namespace) -> int:
    stage = args.stage
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise protocol.ProtocolError("timing child requires CUDA_VISIBLE_DEVICES=0")
    _validate_inherited_gpu_lock()
    input_path = Path(args.input_artifact).resolve()
    input_value, plan = _stage_plan(stage, rederive=False)
    row = json.loads(args.row_json)
    if row not in plan or input_path != _input_artifact(stage).resolve():
        raise protocol.ProtocolError("timing row/input is outside the canonical stage plan")
    output = Path(args.out).resolve()
    expected_output = protocol.RESULTS_ROOT / stage / "raw" / protocol.timing_filename(row)
    if output != expected_output.resolve() or output.exists():
        raise protocol.ProtocolError("unsafe or existing timing output")
    contract = protocol.load_contract()
    receipt_path = Path(args.launch_receipt).resolve()
    canonical_receipt = protocol.RESULTS_ROOT / stage / "launch_receipt.json"
    if (
        receipt_path != canonical_receipt.resolve()
        or re.fullmatch(r"[0-9a-f]{64}", args.launch_receipt_sha256) is None
        or protocol.file_sha256(receipt_path) != args.launch_receipt_sha256
    ):
        raise protocol.ProtocolError("timing child received a foreign launch receipt")
    expected_launch = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "gpu_lock_id": GPU_LOCK_ID,
        "input_artifact_path": protocol.repo_path(input_path),
        "input_artifact_sha256": protocol.file_sha256(input_path),
        "plan": plan,
        "plan_sha256": protocol.canonical_sha256(plan),
        "stage": stage,
        "timing": contract["manifest"]["timing"],
    }
    receipt = validate_launch_receipt(
        protocol.read_json(receipt_path), expected_launch, contract, validate_git=False
    )
    launch_binding = launch_record_binding(receipt_path, receipt)
    require_next_plan_position(protocol.RESULTS_ROOT / stage / "raw", plan, row)
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import candidates as source_candidates
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core as source_core

    _campaign, cells, _lock = source_core.load_contract()
    by_id = {cell["cell_id"]: cell for cell in cells}
    if row["cell_id"] not in by_id:
        raise protocol.ProtocolError("timing plan names an unknown source cell")
    gpu = source_core.gpu_snapshot(0)
    validate_gpu_snapshot(gpu, contract["manifest"]["hardware"])
    toolchain = live_toolchain()
    if toolchain != contract["manifest"]["toolchain"]:
        raise protocol.ProtocolError("timing child toolchain binding changed")
    idle_pre = require_gpu_idle("record_pre")
    record = {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_timing_record",
        **_expected_record(stage, row, input_path, input_value, launch_binding),
        "gpu_preflight": gpu,
        "gpu_idle_preflight": idle_pre,
        "parent_pid": os.getppid(),
        "process_pid": os.getpid(),
        "toolchain": toolchain,
        "trials": 100,
        "warmup_s": 2.0,
        "t_start": time.time(),
    }
    try:
        import common
        import common2
        import runner2
        import torch

        seed, distribution = (
            (0, "rand") if row["distribution"] == "positive" else (2026073101, "randn")
        )
        built = source_candidates.build(by_id[row["cell_id"]])
        x, weight, bias = common2.fused_inputs(seed=seed, dist=distribution)
        with torch.no_grad():
            reference = common2.fused_reference(x, weight, bias, arm="GBGS", dtype=torch.float32)
            x16 = x.half().contiguous()
            observed = built.run(x16, weight, bias)
            torch.cuda.synchronize()
        record["legacy_error"] = common.gate_stats(reference, observed.float())
        if record["legacy_error"].get("gate_pass") is not True:
            raise RuntimeError("fresh timing input failed the correctness check")
        del reference, observed
        torch.cuda.empty_cache()
        times, warmup_iterations = runner2.time_kernel3(
            built.run,
            x16,
            weight,
            bias,
            num_trials=100,
            warmup_s=2.0,
            flush_l2=True,
        )
        numeric = [float(value) for value in times]
        idle_post = require_gpu_idle("record_post")
        record.update(
            {
                "build_metadata": _compact(built.metadata),
                "compile_s": built.compile_s,
                "implementation_sha256": built.metadata["implementation_sha256"],
                "gpu_idle_postflight": idle_post,
                "ok": True,
                "times_ms": numeric,
                "warmup_iterations_actual": warmup_iterations,
                **protocol.summarize_times(numeric),
            }
        )
    except Exception as exc:
        if "gpu_idle_postflight" not in record:
            try:
                record["gpu_idle_postflight"] = gpu_idle_snapshot("record_post")
            except Exception as idle_exc:
                record["gpu_idle_postflight_error"] = (
                    f"{type(idle_exc).__name__}: {idle_exc}"
                )
        record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "ok": False,
                "traceback": traceback.format_exc(),
            }
        )
    record["t_end"] = time.time()
    _write_once(output, record)
    return 0 if record["ok"] else 1


def run_stage(stage: str) -> int:
    gpu_lock = _acquire_gpu_lock()
    try:
        readiness = ready(stage)
        stage_idle_pre = require_gpu_idle("stage_pre")
        contract = protocol.load_contract()
        input_value, plan = _stage_plan(stage)
        input_path = _input_artifact(stage).resolve()
        root = protocol.RESULTS_ROOT / stage
        root.mkdir(parents=True, exist_ok=True)
        active = (root / "active.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            active.close()
            raise protocol.ProtocolError(f"another {stage} launcher is active") from None
        try:
            receipt = root / "launch_receipt.json"
            expected_launch = {
                "campaign_id": protocol.CAMPAIGN_ID,
                "execution_lock_sha256": protocol.file_sha256(
                    protocol.EXECUTION_LOCK_PATH
                ),
                "gpu_lock_id": GPU_LOCK_ID,
                "input_artifact_path": protocol.repo_path(input_path),
                "input_artifact_sha256": protocol.file_sha256(input_path),
                "plan": plan,
                "plan_sha256": protocol.canonical_sha256(plan),
                "stage": stage,
                "timing": contract["manifest"]["timing"],
            }
            receipt_value = {
                "schema_version": 1,
                "record_type": "finite_frontier_ada_launch_receipt",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": {
                    **expected_launch,
                    "git_commit": readiness["git_commit"],
                    "gpu": readiness["gpu"],
                    "toolchain": readiness["toolchain"],
                    "upstream_ref": readiness["upstream_ref"],
                    "upstream_remote": readiness["upstream_remote"],
                },
                "gpu_idle_preflight": stage_idle_pre,
            }
            if receipt.exists():
                retained_receipt = validate_launch_receipt(
                    protocol.read_json(receipt), expected_launch, contract
                )
            else:
                _write_once(receipt, receipt_value)
                retained_receipt = validate_launch_receipt(
                    protocol.read_json(receipt), expected_launch, contract
                )
            launch_binding = launch_record_binding(receipt, retained_receipt)
            raw = root / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "CUDA_HOME": "/usr/local/cuda-13.1",
                    "FINITE_FRONTIER_GPU0_LOCK_FD": str(gpu_lock.fileno()),
                    "TORCH_EXTENSIONS_DIR": str(HERE / ".torch_ext"),
                }
            )
            env["PATH"] = "/usr/local/cuda-13.1/bin:" + env.get("PATH", "")
            env.setdefault("MAX_JOBS", "4")
            for position, row in enumerate(plan):
                if row.get("position") != position:
                    raise protocol.ProtocolError("timing plan position changed")
                output = raw / protocol.timing_filename(row)
                expected = _expected_record(
                    stage, row, input_path, input_value, launch_binding
                )
                if output.exists():
                    validate_timing_record(output, expected, contract)
                    continue
                command = [
                    sys.executable,
                    str(HERE / "launch.py"),
                    "time-one",
                    "--stage",
                    stage,
                    "--input-artifact",
                    str(input_path),
                    "--row-json",
                    json.dumps(row, sort_keys=True, separators=(",", ":")),
                    "--out",
                    str(output),
                    "--launch-receipt",
                    str(receipt),
                    "--launch-receipt-sha256",
                    launch_binding["launch_receipt_sha256"],
                ]
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    pass_fds=(gpu_lock.fileno(),),
                )
                print(
                    f"[{position + 1}/{len(plan)}] {row['label']} "
                    f"{row['distribution']} -> {completed.returncode}",
                    flush=True,
                )
                if completed.returncode:
                    raise protocol.ProtocolError(
                        "timing child failed; this recorded attempt is terminal and "
                        "requires a new successor tag"
                    )
            protocol.validate_raw_census(raw, plan)
            records = [
                validate_timing_record(
                    raw / protocol.timing_filename(row),
                    _expected_record(
                        stage, row, input_path, input_value, launch_binding
                    ),
                    contract,
                )
                for row in plan
            ]
            validate_record_sequence(records, plan)
            status_path = root / "run_status.json"
            if status_path.exists():
                validate_run_status(
                    protocol.read_json(status_path),
                    stage,
                    len(plan),
                    protocol.file_sha256(receipt),
                )
            else:
                _write_once(
                    status_path,
                    {
                        "schema_version": 1,
                        "record_type": "finite_frontier_ada_run_status",
                        "campaign_id": protocol.CAMPAIGN_ID,
                        "complete": True,
                        "expected_records": len(plan),
                        "observed_records": len(records),
                        "gpu_idle_postflight": require_gpu_idle("stage_post"),
                        "launch_receipt_sha256": protocol.file_sha256(receipt),
                        "stage": stage,
                    },
                )
        finally:
            active.close()
        return 0
    finally:
        gpu_lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--authorization-basis", required=True)
    ready_parser = subparsers.add_parser("ready")
    ready_parser.add_argument(
        "--stage",
        choices=("selection-confirm", "terminal-confirm"),
        default="selection-confirm",
    )
    subparsers.add_parser("selection-confirm")
    subparsers.add_parser("terminal-confirm")
    one = subparsers.add_parser("time-one")
    one.add_argument("--stage", choices=("selection_confirm", "terminal_confirm"), required=True)
    one.add_argument("--input-artifact", required=True)
    one.add_argument("--row-json", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--launch-receipt", required=True)
    one.add_argument("--launch-receipt-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        value = freeze(args.authorization_basis)
        print(f"execution_lock_sha256={protocol.file_sha256(protocol.EXECUTION_LOCK_PATH)}")
        return 0
    if args.command == "ready":
        value = ready(args.stage.replace("-", "_"))
        print(f"ready commit={value['git_commit']} gpu={value['gpu']['uuid']}")
        return 0
    if args.command == "time-one":
        return _time_one(args)
    return run_stage(args.command.replace("-", "_"))


if __name__ == "__main__":
    raise SystemExit(main())
