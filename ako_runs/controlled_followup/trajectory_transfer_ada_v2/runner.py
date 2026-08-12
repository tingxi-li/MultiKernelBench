#!/usr/bin/env python3
"""Run immutable admission, positive-only screening, and confirmation."""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    from . import protocol
except ImportError:  # direct script execution
    import protocol  # type: ignore


HERE = Path(__file__).resolve().parent
REPO = protocol.REPO_ROOT
RESULTS = HERE / "results"
MODULE = "ako_runs.controlled_followup.trajectory_transfer_ada_v2.runner"
GPU_LOCK = Path("/tmp") / f"multikernelbench-{protocol.read_json(protocol.CAMPAIGN_PATH)['hardware']['gpu_uuid']}-timing.lock"
GPU_LOCK_ENV = "TRAJECTORY_TRANSFER_GPU0_LOCK_FD"
CHILD_TIMEOUT_S = 900
RUNTIME_MODULE_PATHS = {
    "common": REPO / "ako_runs/phase1_matmul/common.py",
    "common2": REPO / "ako_runs/phase2_fused_sdpa/common2.py",
    "runner2": REPO / "ako_runs/phase2_fused_sdpa/runner2.py",
}
PROVENANCE_PATH = HERE / "prelaunch_provenance.json"


def _runtime_module_receipt() -> dict[str, dict[str, str]]:
    receipt = {}
    for name, path in RUNTIME_MODULE_PATHS.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise protocol.ProtocolError(f"runtime module is missing: {resolved}")
        receipt[name] = {
            "path": str(resolved.relative_to(REPO)),
            "sha256": protocol.file_sha256(resolved),
        }
    return receipt


def result_root(tag: str) -> Path:
    if not tag or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in tag):
        raise protocol.ProtocolError("unsafe result tag")
    return RESULTS / tag


def write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise protocol.ProtocolError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _git(*args: str) -> str:
    run = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30)
    if run.returncode:
        raise protocol.ProtocolError(run.stderr.strip() or "git command failed")
    return run.stdout.strip()


def _remote_binding() -> dict[str, str]:
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if "/" not in upstream:
        raise protocol.ProtocolError("configured upstream is malformed")
    remote, branch = upstream.split("/", 1)
    ref = f"refs/heads/{branch}"
    rows = _git("ls-remote", remote, ref).splitlines()
    if len(rows) != 1 or rows[0].split()[1:] != [ref]:
        raise protocol.ProtocolError("live upstream is absent or ambiguous")
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", upstream) or head != rows[0].split()[0]:
        raise protocol.ProtocolError("HEAD is not the cached and live upstream commit")
    return {"git_commit": head, "upstream": upstream, "live_remote": remote, "live_ref": ref}


def _gpu() -> dict[str, str]:
    run = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-gpu=index,uuid,name,driver_version,compute_cap", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if run.returncode or len(run.stdout.splitlines()) != 1:
        raise protocol.ProtocolError("physical GPU 0 identity query failed")
    fields = [field.strip() for field in run.stdout.strip().split(",")]
    if len(fields) != 5:
        raise protocol.ProtocolError("physical GPU 0 identity is malformed")
    value = dict(zip(("index", "uuid", "name", "driver_version", "compute_capability"), fields))
    expected = protocol.read_json(protocol.CAMPAIGN_PATH)["hardware"]
    if any(value[key] != str(expected[expected_key]) for key, expected_key in (
        ("index", "physical_gpu"), ("uuid", "gpu_uuid"), ("name", "gpu_name"),
        ("driver_version", "driver_version"), ("compute_capability", "compute_capability"),
    )):
        raise protocol.ProtocolError("physical GPU 0 differs from frozen campaign")
    return value


def _pids() -> set[int]:
    run = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if run.returncode:
        raise protocol.ProtocolError("GPU process query failed")
    try:
        return {int(line.strip()) for line in run.stdout.splitlines() if line.strip()}
    except ValueError as exc:
        raise protocol.ProtocolError("GPU process query is malformed") from exc


@contextmanager
def gpu_lock() -> Iterator[Any]:
    handle = GPU_LOCK.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise protocol.ProtocolError("host-wide physical-GPU0 lock is held") from None
    try:
        yield handle
    finally:
        handle.close()


def _inherited_lock() -> None:
    try:
        descriptor = int(os.environ.get(GPU_LOCK_ENV, ""))
        inherited, expected = os.fstat(descriptor), GPU_LOCK.stat()
    except (OSError, ValueError) as exc:
        raise protocol.ProtocolError("GPU child lacks the inherited lock") from exc
    if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        raise protocol.ProtocolError("GPU child inherited a foreign lock")


def _idle(*, allow_self: bool = False) -> dict[str, Any]:
    pids = _pids()
    allowed = {os.getpid()} if allow_self else set()
    if not pids <= allowed:
        raise protocol.ProtocolError(f"physical GPU 0 is occupied: {sorted(pids - allowed)}")
    return {"gpu": _gpu(), "compute_pids": sorted(pids), "checked_unix_ns": time.time_ns()}


def _source_paths() -> list[Path]:
    paths = [HERE / name for name in protocol.SOURCE_RELATIVES]
    paths.extend(
        path for path in (protocol.ADMISSION_MANIFEST_PATH, protocol.CAMPAIGN_LOCK_PATH)
        if path.is_file()
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise protocol.ProtocolError(f"source closure is incomplete: {missing}")
    return paths


def _source_hashes() -> dict[str, str]:
    return {str(path.relative_to(REPO)): protocol.file_sha256(path) for path in _source_paths()}


def _bootstrap_runtime_modules() -> dict[str, dict[str, str]]:
    """Pin legacy top-level imports before a child can initialize the toolchain."""
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3 import (
        protocol as native_protocol,
    )

    cuda_bin = str(Path(native_protocol.CUDA_HOME) / "bin")
    os.environ["CUDA_HOME"] = native_protocol.CUDA_HOME
    os.environ["CUDA_PATH"] = native_protocol.CUDA_HOME
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    os.environ["PATH"] = os.pathsep.join([cuda_bin, *[part for part in path_parts if part != cuda_bin]])
    roots = (
        REPO / "ako_runs/phase2_fused_sdpa",
        REPO / "ako_runs/phase1_matmul",
    )
    for root in reversed(roots):
        value = str(root.resolve())
        sys.path[:] = [item for item in sys.path if str(Path(item or ".").resolve()) != value]
        sys.path.insert(0, value)
    receipt = _runtime_module_receipt()
    for name, expected in RUNTIME_MODULE_PATHS.items():
        expected = expected.resolve()
        loaded = sys.modules.get(name)
        module = loaded if loaded is not None else importlib.import_module(name)
        actual = Path(str(getattr(module, "__file__", ""))).resolve()
        if actual != expected:
            raise protocol.ProtocolError(
                f"runtime module was shadowed: {name}: {actual} != {expected}"
            )
        if receipt[name]["sha256"] != protocol.file_sha256(actual):
            raise protocol.ProtocolError(f"runtime module changed during bootstrap: {name}")
    materials = protocol.read_json(protocol.MATERIALS_PATH).get("files", {})
    if any(materials.get(value["path"]) != value["sha256"] for value in receipt.values()):
        raise protocol.ProtocolError("runtime module bootstrap differs from material closure")
    return receipt


def _toolchain() -> dict[str, Any]:
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3 import (
        protocol as native_protocol,
    )
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3.runner import (
        live_toolchain,
    )

    return {
        **live_toolchain(),
        "cache_loader_sha256": native_protocol.cache_loader_hashes(),
    }


def provenance() -> dict[str, Any]:
    protocol.check_frozen()
    relative = [str(path.relative_to(REPO)) for path in _source_paths()]
    if _git("status", "--porcelain", "--", *relative):
        raise protocol.ProtocolError("frozen source closure is not clean")
    _git("ls-files", "--error-unmatch", "--", *relative)
    remote = _remote_binding()
    with gpu_lock():
        idle = _idle()
    value = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "campaign_lock_sha256": protocol.file_sha256(protocol.CAMPAIGN_LOCK_PATH),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_preflight": idle,
        "record_type": "trajectory_transfer_ada_v2_provenance",
        "schema_version": 1,
        "source_sha256": _source_hashes(),
        "toolchain": _toolchain(),
        **remote,
    }
    write_once(PROVENANCE_PATH, value)
    return value


def _validate_provenance(
    receipt: Any, current_idle: dict[str, Any], remote: dict[str, str] | None,
) -> dict[str, Any]:
    expected_keys = {
        "campaign_id", "campaign_lock_sha256", "created_utc", "git_commit",
        "gpu_preflight", "live_ref", "live_remote", "record_type",
        "schema_version", "source_sha256", "toolchain", "upstream",
    }
    preflight = receipt.get("gpu_preflight") if isinstance(receipt, dict) else None
    created = receipt.get("created_utc") if isinstance(receipt, dict) else None
    try:
        timestamp = datetime.fromisoformat(created) if isinstance(created, str) else None
    except ValueError:
        timestamp = None
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_keys
        or receipt.get("campaign_id") != protocol.CAMPAIGN_ID
        or receipt.get("record_type") != "trajectory_transfer_ada_v2_provenance"
        or receipt.get("schema_version") != 1
        or receipt.get("campaign_lock_sha256")
        != protocol.file_sha256(protocol.CAMPAIGN_LOCK_PATH)
        or receipt.get("source_sha256") != _source_hashes()
        or receipt.get("toolchain") != _toolchain()
        or timestamp is None
        or timestamp.utcoffset() != timezone.utc.utcoffset(None)
        or not isinstance(preflight, dict)
        or set(preflight) != {"gpu", "compute_pids", "checked_unix_ns"}
        or preflight.get("compute_pids") != []
        or preflight.get("gpu") != current_idle.get("gpu")
        or isinstance(preflight.get("checked_unix_ns"), bool)
        or not isinstance(preflight.get("checked_unix_ns"), int)
        or preflight["checked_unix_ns"] <= 0
        or current_idle.get("compute_pids") != []
        or not isinstance(receipt.get("git_commit"), str)
        or len(receipt["git_commit"]) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in receipt["git_commit"])
        or not isinstance(receipt.get("upstream"), str)
        or "/" not in receipt["upstream"]
        or not isinstance(receipt.get("live_remote"), str)
        or not receipt["live_remote"]
        or not isinstance(receipt.get("live_ref"), str)
        or not receipt["live_ref"].startswith("refs/heads/")
        or (remote is not None and any(receipt.get(key) != value for key, value in remote.items()))
    ):
        raise protocol.ProtocolError("prelaunch provenance is malformed, stale, or foreign")
    return receipt


def _provenance_binding() -> dict[str, str]:
    if not PROVENANCE_PATH.is_file():
        raise protocol.ProtocolError("prelaunch provenance is missing")
    return {
        "prelaunch_provenance_path": str(PROVENANCE_PATH.relative_to(REPO)),
        "prelaunch_provenance_sha256": protocol.file_sha256(PROVENANCE_PATH),
    }


def _validate_provenance_binding(value: dict[str, Any]) -> dict[str, str]:
    expected = _provenance_binding()
    if any(value.get(key) != item for key, item in expected.items()):
        raise protocol.ProtocolError("launch provenance binding is stale or foreign")
    return expected


def _validate_bound_gpu_preflight(
    value: Any, allowed_pids: set[int] | None = None,
) -> dict[str, Any]:
    hardware = protocol.read_json(protocol.CAMPAIGN_PATH)["hardware"]
    expected_gpu = {
        "index": str(hardware["physical_gpu"]),
        "uuid": hardware["gpu_uuid"],
        "name": hardware["gpu_name"],
        "driver_version": hardware["driver_version"],
        "compute_capability": hardware["compute_capability"],
    }
    gpu = value.get("gpu") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != {"gpu", "compute_pids", "checked_unix_ns"}
        or not isinstance(gpu, dict)
        or set(gpu) != set(expected_gpu)
        or any(gpu.get(key) != expected for key, expected in expected_gpu.items())
        or not isinstance(value.get("compute_pids"), list)
        or any(not isinstance(pid, int) or isinstance(pid, bool) for pid in value["compute_pids"])
        or (
            value["compute_pids"] != [] if allowed_pids is None
            else not set(value["compute_pids"]) <= allowed_pids
        )
        or isinstance(value.get("checked_unix_ns"), bool)
        or not isinstance(value.get("checked_unix_ns"), int)
        or value["checked_unix_ns"] <= 0
    ):
        raise protocol.ProtocolError("launch GPU preflight is malformed or foreign")
    return value


def _child_provenance(
    runtime_modules: dict[str, dict[str, str]], preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_provenance_binding(),
        "gpu_preflight": _validate_bound_gpu_preflight(preflight, {os.getpid()}),
        "process_pid": os.getpid(),
        "runtime_modules": runtime_modules,
    }


def _validate_child_provenance(value: Any, *, build: bool) -> None:
    base_keys = {
        "prelaunch_provenance_path", "prelaunch_provenance_sha256",
        "gpu_preflight", "process_pid", "runtime_modules",
    }
    expected_keys = base_keys | (
        {"gpu_postflight", "t_start_unix_ns", "t_end_unix_ns"} if build else set()
    )
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or not isinstance(value.get("process_pid"), int)
        or value["process_pid"] <= 0
        or value.get("runtime_modules") != _runtime_module_receipt()
    ):
        raise protocol.ProtocolError("artifact child provenance is malformed")
    _validate_provenance_binding(value)
    _validate_bound_gpu_preflight(value.get("gpu_preflight"), {value["process_pid"]})
    if build:
        _validate_bound_gpu_preflight(value.get("gpu_postflight"), {value["process_pid"]})
        start, end = value.get("t_start_unix_ns"), value.get("t_end_unix_ns")
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or start <= 0 or end < start
        ):
            raise protocol.ProtocolError("artifact child provenance timestamps are malformed")


def ready(*, require_remote: bool = True) -> dict[str, Any]:
    protocol.check_frozen()
    if not PROVENANCE_PATH.is_file():
        raise protocol.ProtocolError("prelaunch provenance is missing")
    current_idle = _idle()
    remote = _remote_binding() if require_remote else None
    receipt = _validate_provenance(protocol.read_json(PROVENANCE_PATH), current_idle, remote)
    return {"provenance": receipt, "idle": current_idle, "campaign": protocol.read_json(protocol.CAMPAIGN_PATH)}


def _manifest_row(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    rows = [row for row in manifest["rows"] if row.get("entry_id") == key or row.get("record_id") == key]
    if len(rows) != 1:
        raise protocol.ProtocolError(f"unknown or duplicate manifest row: {key}")
    return rows[0]


def _failure(path: Path, row: dict[str, Any], status: str, exc: BaseException) -> None:
    write_once(path, _failure_value(
        row, status, f"{type(exc).__name__}: {exc}", traceback.format_exc(),
    ))


def _failure_value(
    row: dict[str, Any], status: str, error: str, trace: str | None,
) -> dict[str, Any]:
    protocol.validate_terminal_outcome(status)
    return {
        "admission_row_sha256": protocol.canonical_sha256(row),
        "campaign_id": protocol.CAMPAIGN_ID,
        "entry_id": row["entry_id"],
        "error": error,
        "record_type": "trajectory_transfer_ada_v2_admission_failure",
        "schema_version": 1,
        "terminal_status": status,
        "traceback": trace,
    }


def _validate_failure(value: Any, row: dict[str, Any]) -> str:
    expected_keys = {
        "admission_row_sha256", "campaign_id", "entry_id", "error",
        "record_type", "schema_version", "terminal_status", "traceback",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("admission_row_sha256") != protocol.canonical_sha256(row)
        or value.get("campaign_id") != protocol.CAMPAIGN_ID
        or value.get("entry_id") != row["entry_id"]
        or not isinstance(value.get("error"), str)
        or not value["error"]
        or value.get("record_type") != "trajectory_transfer_ada_v2_admission_failure"
        or value.get("schema_version") != 1
        or not (value.get("traceback") is None or isinstance(value.get("traceback"), str))
    ):
        raise protocol.ProtocolError("admission failure is malformed or foreign")
    status = protocol.validate_terminal_outcome(value.get("terminal_status"))
    if status == "GATE_PASSED":
        raise protocol.ProtocolError("failure receipt may not claim GATE_PASSED")
    return status


def _retain_parent_failure(
    path: Path, row: dict[str, Any], status: str, message: str,
) -> None:
    if path.exists():
        return
    write_once(path, _failure_value(row, status, message, None))


def _terminal_census(
    manifest: dict[str, Any], root: Path,
) -> tuple[dict[str, str], list[str]]:
    from . import artifacts

    statuses: dict[str, str] = {}
    passed: list[str] = []
    for row in manifest["rows"]:
        paths = artifacts.entry_paths(row["entry_id"], root)
        verified, failed = paths["verify"].is_file(), paths["failure"].is_file()
        if verified == failed:
            raise protocol.ProtocolError(
                f"admission has zero or multiple terminal outcomes: {row['entry_id']}"
            )
        if verified:
            status = "GATE_PASSED"
            passed.append(row["entry_id"])
        else:
            failure = protocol.read_json(paths["failure"])
            status = _validate_failure(failure, row)
        statuses[row["entry_id"]] = status
    return statuses, passed


def _admission_terminal_evidence(
    manifest: dict[str, Any], root: Path,
) -> list[dict[str, str]]:
    from . import artifacts

    _require_safe_tree(root, "admission")
    rows = []
    for row in manifest["rows"]:
        paths = artifacts.entry_paths(row["entry_id"], root)
        for path in sorted(paths["root"].rglob("*")):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise protocol.ProtocolError("admission evidence contains a symlink or special file")
            if path.is_file():
                rows.append({
                    "entry_id": row["entry_id"],
                    "path": str(path.relative_to(REPO)),
                    "sha256": protocol.file_sha256(path),
                    "size": path.stat().st_size,
                })
    return rows


def _require_exact_files(root: Path, expected: set[Path], label: str) -> None:
    _require_safe_tree(root, label)
    observed = {path for path in root.rglob("*") if path.is_file()}
    expected_dirs = set()
    for path in expected:
        parent = path.parent
        while parent != root:
            expected_dirs.add(parent)
            parent = parent.parent
    observed_dirs = {path for path in root.rglob("*") if path.is_dir()}
    if observed != expected or observed_dirs != expected_dirs:
        raise protocol.ProtocolError(f"{label} file census differs from its manifest")


def _require_safe_tree(root: Path, label: str) -> None:
    if not root.is_dir() or root.is_symlink():
        raise protocol.ProtocolError(f"{label} root is not a real directory")
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise protocol.ProtocolError(f"{label} contains a symlink or special file")


def _validate_admission_census(
    manifest: dict[str, Any], root: Path, statuses: dict[str, str],
) -> None:
    from . import artifacts

    artifacts_root = root / "artifacts"
    _require_safe_tree(root.parent, "result tag")
    if not artifacts_root.is_dir() or artifacts_root.is_symlink():
        raise protocol.ProtocolError("admission artifacts root is not a real directory")
    entry_ids = {row["entry_id"] for row in manifest["rows"]}
    if (
        {path.name for path in root.iterdir() if path.is_dir()} != {"artifacts"}
        or {path.name for path in root.iterdir() if path.is_file()}
        != {"launch_receipt.json", "run_status.json"}
        or not artifacts_root.is_dir()
        or {path.name for path in artifacts_root.iterdir()} != entry_ids
        or any(not path.is_dir() for path in artifacts_root.iterdir())
    ):
        raise protocol.ProtocolError("admission file census differs from its manifest")
    for row in manifest["rows"]:
        paths = artifacts.entry_paths(row["entry_id"], root)
        if not paths["root"].is_dir() or paths["root"].is_symlink():
            raise protocol.ProtocolError(
                f"admission entry root is not a real directory: {row['entry_id']}"
            )
        children = set(paths["root"].iterdir())
        directories = {path.name for path in children if path.is_dir()}
        files = {path.name for path in children if path.is_file()}
        if statuses[row["entry_id"]] == "GATE_PASSED":
            valid = directories == {"cache", "runtime_tmp"} and files == {
                "gate.jsonl", "build_record.json", "verify_record.json",
            }
        else:
            valid = (
                directories <= {"cache", "runtime_tmp"}
                and "failure.json" in files
                and files <= {"failure.json", "gate.jsonl", "build_record.json"}
            )
        if not valid or any(not path.is_file() and not path.is_dir() for path in children):
            raise protocol.ProtocolError(
                f"admission entry file census is foreign: {row['entry_id']}"
            )


def admission_one(tag: str, entry_id: str) -> int:
    _inherited_lock()
    runtime_modules = _bootstrap_runtime_modules()
    from . import artifacts, implementations

    manifest = protocol.read_json(protocol.ADMISSION_MANIFEST_PATH)
    row = _manifest_row(manifest, entry_id)
    root = result_root(tag) / "admission"
    paths = artifacts.entry_paths(entry_id, root)
    if any(path.exists() for path in paths.values()):
        raise protocol.ProtocolError("admission child refuses an existing entry")
    artifacts.prepare_cache_environment(entry_id, "admit", root)
    started = time.time_ns()
    failure_status = "BUILD_FAILED"
    try:
        pre = _idle()
        with artifacts.capture_torch_extensions() as requests:
            built = implementations.build(row["spec"], row["mechanism_enabled"], row["route"])
            failure_status = "GATE_FAILED"
            gate = artifacts.gate_built(built, row)
        paths["gate"].parent.mkdir(parents=True, exist_ok=True)
        with paths["gate"].open("x", encoding="utf-8") as handle:
            for record in gate["records"]:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        if len(gate["records"]) != 512 or gate["summary"].get("full_gate_pass") is not True:
            raise protocol.ProtocolError("artifact failed the complete frozen correctness gate")
        failure_status = "AUDIT_FAILED"
        dynamic_audit = artifacts.dynamic_work_audit(built)
        value = artifacts.build_record(
            row, built, gate["summary"], str(paths["gate"].relative_to(REPO)),
            protocol.file_sha256(paths["gate"]),
            {**_child_provenance(runtime_modules, pre), "gpu_postflight": _idle(allow_self=True), "t_start_unix_ns": started, "t_end_unix_ns": time.time_ns()},
            requests, dynamic_audit, root,
        )
        write_once(paths["build"], value)
        artifacts.validate_build_record(protocol.read_json(paths["build"]), row, root)
        return 0
    except Exception as exc:
        _failure(paths["failure"], row, failure_status, exc)
        return 1


def verify_one(tag: str, entry_id: str) -> int:
    _inherited_lock()
    runtime_modules = _bootstrap_runtime_modules()
    from . import artifacts, implementations

    row = _manifest_row(protocol.read_json(protocol.ADMISSION_MANIFEST_PATH), entry_id)
    root = result_root(tag) / "admission"
    paths = artifacts.entry_paths(entry_id, root)
    build = artifacts.validate_build_record(protocol.read_json(paths["build"]), row, root)
    artifacts.prepare_cache_environment(entry_id, "load_only", root)
    try:
        with artifacts.load_only_guards(build) as evidence:
            built = implementations.build(row["spec"], row["mechanism_enabled"], row["route"])
            live_gate = artifacts.quick_gate_built(built)
        artifacts.validate_load_evidence(evidence, row, build)
        value = artifacts.verify_record(
            row, build, built, live_gate, evidence,
            _child_provenance(runtime_modules, _idle(allow_self=True)), root,
        )
        write_once(paths["verify"], value)
        artifacts.validate_verify_record(protocol.read_json(paths["verify"]), row, build, root)
        return 0
    except Exception as exc:
        _failure(paths["failure"], row, "LAUNCH_FAILED", exc)
        return 1


def _child(command: list[str], lock_fd: int, *, entry_id: str | None = None, root: Path | None = None) -> int:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "0", GPU_LOCK_ENV: str(lock_fd)})
    if entry_id is not None and root is not None:
        from . import artifacts
        child_action = next(
            (value for value in command if value in {"admission-one", "verify-one", "time-one"}),
            None,
        )
        if child_action is None:
            raise protocol.ProtocolError("cannot determine artifact child mode")
        mode = "admit" if child_action == "admission-one" else "load_only"
        env.update(artifacts.cache_environment(entry_id, mode, root))
    process = subprocess.Popen(command, cwd=REPO, env=env, pass_fds=(lock_fd,), start_new_session=True)
    try:
        return process.wait(timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 124


def admit(tag: str) -> int:
    from . import artifacts

    root = result_root(tag) / "admission"
    if root.exists():
        raise protocol.ProtocolError("admission tag already exists; use a successor tag")
    root.mkdir(parents=True)
    manifest = protocol.read_json(protocol.ADMISSION_MANIFEST_PATH)
    with gpu_lock() as lock:
        readiness = ready()
        write_once(root / "launch_receipt.json", {
            "campaign_id": protocol.CAMPAIGN_ID, "campaign_lock_sha256": protocol.file_sha256(protocol.CAMPAIGN_LOCK_PATH),
            "expected_entries": manifest["expected_rows"], "gpu_preflight": readiness["idle"],
            "manifest_sha256": manifest["manifest_sha256"], "record_type": "trajectory_transfer_ada_v2_admission_launch", "schema_version": 1,
            **_provenance_binding(),
        })
        for index, row in enumerate(manifest["rows"], 1):
            paths = artifacts.entry_paths(row["entry_id"], root)
            build_rc = _child([sys.executable, "-m", MODULE, "admission-one", "--tag", tag, "--entry-id", row["entry_id"]], lock.fileno(), entry_id=row["entry_id"], root=root)
            if build_rc == 0:
                verify_rc = _child([sys.executable, "-m", MODULE, "verify-one", "--tag", tag, "--entry-id", row["entry_id"]], lock.fileno(), entry_id=row["entry_id"], root=root)
                if verify_rc and not paths["failure"].is_file():
                    _retain_parent_failure(
                        paths["failure"], row, "LAUNCH_FAILED",
                        f"load-only verification child exited {verify_rc}",
                    )
            elif not paths["failure"].is_file():
                _retain_parent_failure(
                    paths["failure"], row, "LAUNCH_FAILED",
                    f"artifact admission child exited {build_rc}",
                )
            print(f"[{index}/{manifest['expected_rows']}] {row['entry_id']}", flush=True)
        statuses, passed = _terminal_census(manifest, root)
        write_once(root / "run_status.json", {
            "campaign_id": protocol.CAMPAIGN_ID, "complete": True, "expected_entries": manifest["expected_rows"],
            "gate_passed_entries": passed, "observed_terminal_entries": len(statuses),
            "launch_receipt_sha256": protocol.file_sha256(root / "launch_receipt.json"),
            "record_type": "trajectory_transfer_ada_v2_admission_status", "schema_version": 1,
            "terminal_evidence_sha256": protocol.canonical_sha256(
                _admission_terminal_evidence(manifest, root)
            ),
            "terminal_statuses": statuses,
        })
    return 0


def _admitted(tag: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from . import artifacts

    manifest = protocol.read_json(protocol.ADMISSION_MANIFEST_PATH)
    root = result_root(tag) / "admission"
    launch = protocol.read_json(root / "launch_receipt.json")
    expected_launch = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "campaign_lock_sha256": protocol.file_sha256(protocol.CAMPAIGN_LOCK_PATH),
        "expected_entries": manifest["expected_rows"],
        "gpu_preflight": launch.get("gpu_preflight"),
        "manifest_sha256": manifest["manifest_sha256"],
        **_provenance_binding(),
        "record_type": "trajectory_transfer_ada_v2_admission_launch",
        "schema_version": 1,
    }
    _validate_provenance_binding(launch)
    _validate_bound_gpu_preflight(launch.get("gpu_preflight"))
    if launch != expected_launch:
        raise protocol.ProtocolError("admission launch receipt is foreign")
    status = protocol.read_json(root / "run_status.json")
    statuses, passed_list = _terminal_census(manifest, root)
    _validate_admission_census(manifest, root, statuses)
    expected_status = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "complete": True,
        "expected_entries": manifest["expected_rows"],
        "gate_passed_entries": passed_list,
        "launch_receipt_sha256": protocol.file_sha256(root / "launch_receipt.json"),
        "observed_terminal_entries": len(statuses),
        "record_type": "trajectory_transfer_ada_v2_admission_status",
        "schema_version": 1,
        "terminal_evidence_sha256": protocol.canonical_sha256(
            _admission_terminal_evidence(manifest, root)
        ),
        "terminal_statuses": statuses,
    }
    if status != expected_status:
        raise protocol.ProtocolError("admission is incomplete")
    passed = set(passed_list)
    entries = {}
    from ako_runs.controlled_followup.fused_grid import robust_adapter
    gate_context = robust_adapter.load_repository()
    for row in manifest["rows"]:
        paths = artifacts.entry_paths(row["entry_id"], root)
        if row["entry_id"] in passed:
            build = artifacts.validate_build_record(protocol.read_json(paths["build"]), row, root)
            _validate_child_provenance(build.get("provenance"), build=True)
            verify = artifacts.validate_verify_record(protocol.read_json(paths["verify"]), row, build, root)
            _validate_child_provenance(verify.get("provenance"), build=False)
            artifacts.validate_full_gate(build, row, gate_context)
            artifacts.validate_dynamic_work_audit(build.get("dynamic_work_audit"))
            metadata = build.get("source_build_metadata", {})
            if (
                metadata.get("n_kernels") != 2
                or metadata.get("primitive_graph_sha256") != row["primitive_graph_sha256"]
                or metadata.get("coordinate_cell_id") != row["coordinate_cell_id"]
                or metadata.get("implementation_id") != row["implementation_id"]
                or metadata.get("route") != row["route"]
                or metadata.get("primitive_mapping", {}).get("primitive_map_sha256")
                != protocol.file_sha256(protocol.PRIMITIVE_MAP_PATH)
            ):
                raise protocol.ProtocolError(
                    f"admitted primitive recipe is unbound: {row['entry_id']}"
                )
            entries[row["entry_id"]] = build
        elif not paths["failure"].is_file():
            raise protocol.ProtocolError("admission terminal census is incomplete")
    _validate_treatment_pairs(manifest, entries)
    return manifest, entries


def _validate_treatment_pairs(
    manifest: dict[str, Any], entries: dict[str, dict[str, Any]],
) -> None:
    """Check the held treatment coordinate wherever both pair members passed."""
    for destination in protocol.DESTINATIONS:
        for adaptation in protocol.ADAPTATIONS:
            grids = (protocol.FIXED_GRID,) if adaptation == "donor_fixed" else protocol.GRID_IDS
            for grid in grids:
                pair = [
                    entries[row["entry_id"]]
                    for row in manifest["rows"]
                    if row["destination"] == destination
                    and row["adaptation"] == adaptation
                    and row["grid_id"] == grid
                    and row["entry_id"] in entries
                ]
                if len(pair) != 2:
                    continue
                metadata = [item.get("source_build_metadata", {}) for item in pair]
                paired = {item.get("paired_config_sha256") for item in metadata}
                gemm_sources = {item.get("held_gemm_source_sha256") for item in metadata}
                if (
                    len(paired) != 1
                    or None in paired
                    or len(gemm_sources) != 1
                    or not all(
                        isinstance(value, str)
                        and len(value) == 64
                        and all(character in "0123456789abcdef" for character in value)
                        for value in gemm_sources
                    )
                    or any(item.get("n_kernels") != 2 for item in metadata)
                ):
                    raise protocol.ProtocolError(
                        f"off/on structural controls differ: {destination}/{adaptation}/{grid}"
                    )
                if any(
                    not item.get("adapter_module_sha256")
                    or not item.get("adapter_dispatch_sha256")
                    or not item.get("generated_source_sha256")
                    or item.get("route")
                    not in {
                        row["route"]
                        for row in manifest["rows"]
                        if row["destination"] == destination
                        and row["adaptation"] == adaptation
                        and row["grid_id"] == grid
                    }
                    or item.get("primitive_graph_sha256")
                    not in {
                        row["primitive_graph_sha256"]
                        for row in manifest["rows"]
                        if row["destination"] == destination
                        and row["adaptation"] == adaptation
                        and row["grid_id"] == grid
                    }
                    or item.get("primitive_mapping", {}).get("primitive_map_sha256")
                    != protocol.file_sha256(protocol.PRIMITIVE_MAP_PATH)
                    for item in metadata
                ):
                    raise protocol.ProtocolError(
                        f"transfer recipe/source identity is incomplete: {destination}/{adaptation}/{grid}"
                    )


def _time_one(tag: str, phase: str, record_id: str, output: Path) -> int:
    _inherited_lock()
    runtime_modules = _bootstrap_runtime_modules()
    from . import artifacts, implementations

    stage = result_root(tag) / phase
    manifest = protocol.read_json(stage / "manifest.json")
    row = _manifest_row(manifest, record_id)
    admission = protocol.read_json(protocol.ADMISSION_MANIFEST_PATH)
    entry = _manifest_row(admission, row["entry_id"])
    expected = stage / "raw" / f"{record_id}.json"
    if output.resolve() != expected.resolve() or output.exists():
        raise protocol.ProtocolError("timing child output is unsafe or exists")
    root = result_root(tag) / "admission"
    build = _admitted_entry(entry, root)
    artifacts.prepare_cache_environment(entry["entry_id"], "load_only", root)
    started = time.time_ns()
    record = {
        "campaign_id": protocol.CAMPAIGN_ID, "record_type": "trajectory_transfer_ada_v2_timing_record",
        "row": row, "row_sha256": protocol.canonical_sha256(row), "schema_version": 1,
        "primitive_graph_sha256": entry["primitive_graph_sha256"],
        "primitive_map_sha256": protocol.file_sha256(protocol.PRIMITIVE_MAP_PATH),
        "coordinate_cell_id": entry["coordinate_cell_id"],
        "implementation_id": entry["implementation_id"],
        "structural_route": entry["route"],
        "runtime_modules": runtime_modules,
        "t_start_unix_ns": started,
    }
    try:
        import common
        import common2
        import runner2
        import torch

        with artifacts.load_only_guards(build) as evidence:
            built = implementations.build(entry["spec"], entry["mechanism_enabled"], entry["route"])
            dist, seed = ("rand", 0) if row["distribution"] == "positive" else ("randn", 2026073101)
            x, weight, bias = common2.fused_inputs(seed=seed, dist=dist)
            x16 = x.half().contiguous()
            with torch.no_grad():
                reference = common2.fused_reference(x, weight, bias, arm="GBGS", dtype=torch.float32)
                observed = built.run(x16, weight, bias)
                torch.cuda.synchronize()
            if common.gate_stats(reference, observed.float()).get("gate_pass") is not True:
                raise protocol.ProtocolError("live timing correctness check failed")
            times, warmups = runner2.time_kernel3(
                built.run, x16, weight, bias,
                num_trials=protocol.read_json(protocol.CAMPAIGN_PATH)["timing"]["trials"],
                warmup_s=protocol.read_json(protocol.CAMPAIGN_PATH)["timing"]["warmup_s"], flush_l2=True,
            )
        artifacts.validate_load_evidence(evidence, entry, build)
        numeric = [float(value) for value in times]
        if len(numeric) != 100 or any(not math.isfinite(value) or value <= 0 for value in numeric):
            raise protocol.ProtocolError("timing trials are malformed")
        record.update({
            "full_median_ms": statistics.median(numeric), "implementation_sha256": build["artifact_identity_sha256"],
            "load_evidence": evidence, "ok": True, "primary_tail_median_ms": statistics.median(numeric[60:100]),
            "times_ms": numeric, "warmup_iterations": int(warmups),
        })
    except Exception as exc:
        record.update({"error": f"{type(exc).__name__}: {exc}", "ok": False, "traceback": traceback.format_exc()})
    record["t_end_unix_ns"] = time.time_ns()
    write_once(output, record)
    return 0 if record["ok"] else 1


def _admitted_entry(row: dict[str, Any], root: Path) -> dict[str, Any]:
    """Validate only one parent-authorized artifact in a fresh timing child."""
    from . import artifacts

    paths = artifacts.entry_paths(row["entry_id"], root)
    build = artifacts.validate_build_record(protocol.read_json(paths["build"]), row, root)
    _validate_child_provenance(build.get("provenance"), build=True)
    verify = artifacts.validate_verify_record(protocol.read_json(paths["verify"]), row, build, root)
    _validate_child_provenance(verify.get("provenance"), build=False)
    return build


def _run_manifest(tag: str, phase: str, manifest: dict[str, Any]) -> int:
    stage = result_root(tag) / phase
    if stage.exists():
        raise protocol.ProtocolError(f"{phase} already exists; tag is burned")
    (stage / "raw").mkdir(parents=True)
    (stage / "position_receipts").mkdir()
    write_once(stage / "manifest.json", manifest)
    with gpu_lock() as lock:
        readiness = ready()
        write_once(stage / "launch_receipt.json", {
            "campaign_id": protocol.CAMPAIGN_ID, "expected_records": manifest["expected_records"],
            "gpu_preflight": readiness["idle"], "manifest_sha256": protocol.canonical_sha256(manifest),
            "record_type": f"trajectory_transfer_ada_v2_{phase}_launch", "schema_version": 1,
            "admission_launch_receipt_sha256": protocol.file_sha256(result_root(tag) / "admission/launch_receipt.json"),
            "admission_run_status_sha256": protocol.file_sha256(result_root(tag) / "admission/run_status.json"),
            **_provenance_binding(),
        })
        previous = 0
        for position, row in enumerate(manifest["rows"], 1):
            record_id = row["record_id"]
            output = stage / "raw" / f"{record_id}.json"
            launched = time.time_ns()
            rc = _child([sys.executable, "-m", MODULE, "time-one", "--tag", tag, "--phase", phase, "--record-id", record_id, "--output", str(output)], lock.fileno(), entry_id=row["entry_id"], root=result_root(tag) / "admission")
            completed = time.time_ns()
            idle = _idle()
            receipt = {
                "campaign_id": protocol.CAMPAIGN_ID, "child_completed_unix_ns": completed,
                "child_launched_unix_ns": launched, "previous_child_completed_unix_ns": previous,
                "position": position, "raw_path": str(output.relative_to(REPO)),
                "raw_sha256": protocol.file_sha256(output) if output.is_file() else None,
                "record_id": record_id, "record_type": "trajectory_transfer_ada_v2_position_receipt",
                "returncode": rc, "schema_version": 1, "gpu_idle_after_child": idle,
            }
            write_once(stage / "position_receipts" / f"{position:04d}__{record_id}.json", receipt)
            if rc or not output.is_file():
                raise protocol.ProtocolError(f"{phase} failed at position {position}; tag is burned")
            previous = completed
            print(f"[{position}/{manifest['expected_records']}] {record_id}", flush=True)
        raw_hashes = {
            row["record_id"]: protocol.file_sha256(stage / "raw" / f"{row['record_id']}.json")
            for row in manifest["rows"]
        }
        position_hashes = {
            row["record_id"]: protocol.file_sha256(
                stage / "position_receipts" / f"{position:04d}__{row['record_id']}.json"
            )
            for position, row in enumerate(manifest["rows"], 1)
        }
        write_once(stage / "run_status.json", {
            "campaign_id": protocol.CAMPAIGN_ID, "complete": True,
            "expected_records": manifest["expected_records"], "observed_records": len(manifest["rows"]),
            "launch_receipt_sha256": protocol.file_sha256(stage / "launch_receipt.json"),
            "position_receipt_bundle_sha256": protocol.canonical_sha256(position_hashes),
            "raw_record_bundle_sha256": protocol.canonical_sha256(raw_hashes),
            "record_type": f"trajectory_transfer_ada_v2_{phase}_status", "schema_version": 1,
        })
    return 0


def screen(tag: str) -> int:
    admission, entries = _admitted(tag)
    bounded = {
        row["entry_id"] for row in admission["rows"]
        if row["adaptation"] == "bounded_retune" and row["entry_id"] in entries
    }
    return _run_manifest(tag, "screen", protocol.make_screen_manifest(bounded))


def derive_selection(tag: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    stage = result_root(tag) / "screen"
    manifest = protocol.read_json(stage / "manifest.json")
    admission, admitted = _admitted(tag)
    bounded = {
        row["entry_id"] for row in admission["rows"]
        if row["adaptation"] == "bounded_retune" and row["entry_id"] in admitted
    }
    expected_manifest = protocol.make_screen_manifest(bounded)
    if manifest != expected_manifest:
        raise protocol.ProtocolError("screen manifest differs from admitted artifacts")
    medians: dict[str, list[float]] = {}
    records, hashes = _validate_phase_evidence(
        tag, "screen", manifest, admission, admitted,
    )
    by_id = {row["entry_id"]: row for row in admission["rows"]}
    for row, record in zip(manifest["rows"], records, strict=True):
        if record.get("ok") is not True or row.get("distribution") != "positive":
            raise protocol.ProtocolError("screen record failed or used withheld data")
        medians.setdefault(row["entry_id"], []).append(float(record["primary_tail_median_ms"]))
    selection: dict[str, dict[str, str]] = {}
    for destination in protocol.DESTINATIONS:
        selection[destination] = {}
        for state in protocol.MECHANISM_STATES:
            candidates = [entry_id for entry_id in medians if by_id[entry_id]["destination"] == destination and by_id[entry_id]["mechanism_state"] == state]
            if not candidates or any(len(medians[entry_id]) != protocol.SCREEN_REPLICATES for entry_id in candidates):
                raise protocol.ProtocolError(f"no complete screen candidates: {destination}/{state}")
            selection[destination][state] = min(candidates, key=lambda entry_id: (statistics.median(medians[entry_id]), by_id[entry_id]["config_id"]))
    protocol.validate_selection(selection, admission)
    value = {
        "campaign_id": protocol.CAMPAIGN_ID, "record_type": "trajectory_transfer_ada_v2_selection",
        "schema_version": 1,
        "screen_evidence_sha256": protocol.canonical_sha256(hashes),
        "screen_manifest_sha256": protocol.file_sha256(stage / "manifest.json"),
        "selection": selection, "selection_distribution": "positive", "withheld_distribution_used": False,
    }
    return value, hashes


def select(tag: str) -> dict[str, Any]:
    value, _hashes = derive_selection(tag)
    write_once(result_root(tag) / "selection.json", value)
    return value


def confirm(tag: str) -> int:
    selection_path = result_root(tag) / "selection.json"
    retained = protocol.read_json(selection_path)
    selection, _screen_hashes = derive_selection(tag)
    if retained != selection:
        raise protocol.ProtocolError("selection failed independent screen rederivation")
    _admission_manifest, admitted = _admitted(tag)
    manifest = protocol.make_confirmation_manifest(selection["selection"])
    missing = sorted({row["entry_id"] for row in manifest["rows"]} - set(admitted))
    if missing:
        raise protocol.ProtocolError(
            f"confirmation is unauthorized because required artifacts were excluded: {missing}"
        )
    return _run_manifest(tag, "confirmation", manifest)


def _validate_position(row: dict[str, Any], record: dict[str, Any], receipt: dict[str, Any], raw: Path, previous: int, position: int) -> int:
    expected_keys = {
        "campaign_id", "child_completed_unix_ns", "child_launched_unix_ns",
        "gpu_idle_after_child", "position", "previous_child_completed_unix_ns",
        "raw_path", "raw_sha256", "record_id", "record_type", "returncode",
        "schema_version",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("campaign_id") != protocol.CAMPAIGN_ID
        or receipt.get("record_type") != "trajectory_transfer_ada_v2_position_receipt"
        or receipt.get("schema_version") != 1
        or receipt.get("record_id") != row["record_id"] or receipt.get("position") != position
        or receipt.get("previous_child_completed_unix_ns") != previous or receipt.get("returncode") != 0
        or receipt.get("raw_sha256") != protocol.file_sha256(raw)
        or receipt.get("raw_path") != str(raw.relative_to(REPO))
        or not receipt.get("child_launched_unix_ns") <= record["t_start_unix_ns"] <= record["t_end_unix_ns"] <= receipt.get("child_completed_unix_ns")
        or receipt.get("gpu_idle_after_child", {}).get("compute_pids") != []
    ):
        raise protocol.ProtocolError(f"invalid position receipt: {position}")
    return receipt["child_completed_unix_ns"]


def _validate_phase_evidence(
    tag: str,
    phase: str,
    manifest: dict[str, Any],
    admission: dict[str, Any],
    admitted: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    from . import analyze, artifacts

    stage = result_root(tag) / phase
    launch_path, status_path = stage / "launch_receipt.json", stage / "run_status.json"
    launch, status = protocol.read_json(launch_path), protocol.read_json(status_path)
    expected_launch = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "expected_records": manifest["expected_records"],
        "gpu_preflight": launch.get("gpu_preflight"),
        "manifest_sha256": protocol.canonical_sha256(manifest),
        "admission_launch_receipt_sha256": protocol.file_sha256(result_root(tag) / "admission/launch_receipt.json"),
        "admission_run_status_sha256": protocol.file_sha256(result_root(tag) / "admission/run_status.json"),
        **_provenance_binding(),
        "record_type": f"trajectory_transfer_ada_v2_{phase}_launch",
        "schema_version": 1,
    }
    _validate_provenance_binding(launch)
    _validate_bound_gpu_preflight(launch.get("gpu_preflight"))
    if launch != expected_launch:
        raise protocol.ProtocolError(f"{phase} launch receipt is foreign")
    campaign = protocol.read_json(protocol.CAMPAIGN_PATH)
    runtime_modules = _runtime_module_receipt()
    by_id = {row["entry_id"]: row for row in admission["rows"]}
    records, hashes = [], []
    raw_hashes: dict[str, str] = {}
    position_hashes: dict[str, str] = {}
    previous = 0
    for position, row in enumerate(manifest["rows"], 1):
        raw = stage / "raw" / f"{row['record_id']}.json"
        pos = stage / "position_receipts" / f"{position:04d}__{row['record_id']}.json"
        record, receipt = protocol.read_json(raw), protocol.read_json(pos)
        analyze.validate_timing_record(campaign, row, record)
        previous = _validate_position(row, record, receipt, raw, previous, position)
        entry, build = by_id.get(row["entry_id"]), admitted.get(row["entry_id"])
        if entry is None or build is None:
            raise protocol.ProtocolError(f"{phase} uses an unadmitted entry")
        if (
            record.get("implementation_sha256") != build["artifact_identity_sha256"]
            or record.get("primitive_graph_sha256") != entry["primitive_graph_sha256"]
            or record.get("coordinate_cell_id") != entry["coordinate_cell_id"]
            or record.get("implementation_id") != entry["implementation_id"]
            or record.get("structural_route") != entry["route"]
            or record.get("runtime_modules") != runtime_modules
        ):
            raise protocol.ProtocolError(f"{phase} record differs from admitted artifact")
        artifacts.validate_load_evidence(record.get("load_evidence"), entry, build)
        records.append(record)
        raw_hashes[row["record_id"]] = protocol.file_sha256(raw)
        position_hashes[row["record_id"]] = protocol.file_sha256(pos)
        hashes.extend((
            {"path": str(raw.relative_to(REPO)), "sha256": raw_hashes[row["record_id"]]},
            {"path": str(pos.relative_to(REPO)), "sha256": position_hashes[row["record_id"]]},
        ))
    expected_phase_files = {
        stage / "manifest.json", launch_path, status_path,
        *{
            stage / "raw" / f"{row['record_id']}.json"
            for row in manifest["rows"]
        },
        *{
            stage / "position_receipts" / f"{position:04d}__{row['record_id']}.json"
            for position, row in enumerate(manifest["rows"], 1)
        },
    }
    _require_exact_files(stage, expected_phase_files, phase)
    expected_status = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "complete": True,
        "expected_records": manifest["expected_records"],
        "launch_receipt_sha256": protocol.file_sha256(launch_path),
        "observed_records": len(manifest["rows"]),
        "position_receipt_bundle_sha256": protocol.canonical_sha256(position_hashes),
        "raw_record_bundle_sha256": protocol.canonical_sha256(raw_hashes),
        "record_type": f"trajectory_transfer_ada_v2_{phase}_status",
        "schema_version": 1,
    }
    if status != expected_status:
        raise protocol.ProtocolError(f"{phase} status is not exact-evidence-derived")
    hashes.extend((
        {"path": str(launch_path.relative_to(REPO)), "sha256": protocol.file_sha256(launch_path)},
        {"path": str(status_path.relative_to(REPO)), "sha256": protocol.file_sha256(status_path)},
    ))
    return records, hashes


def load_completed(tag: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    protocol.check_frozen()
    root = result_root(tag)
    allowed = {"admission", "screen", "confirmation", "selection.json", "analysis.json"}
    if any(path.name not in allowed for path in root.iterdir()):
        raise protocol.ProtocolError("result tag contains a foreign top-level artifact")
    campaign = protocol.read_json(protocol.CAMPAIGN_PATH)
    admission, admitted = _admitted(tag)
    selection_path = result_root(tag) / "selection.json"
    selection = protocol.read_json(selection_path)
    derived_selection, screen_hashes = derive_selection(tag)
    if selection != derived_selection:
        raise protocol.ProtocolError("selection failed independent screen rederivation")
    stage = result_root(tag) / "confirmation"
    manifest = protocol.read_json(stage / "manifest.json")
    expected_manifest = protocol.make_confirmation_manifest(derived_selection["selection"])
    if manifest != expected_manifest:
        raise protocol.ProtocolError("confirmation manifest differs from checkpointed selection")
    records, hashes = _validate_phase_evidence(
        tag, "confirmation", manifest, admission, admitted,
    )
    admission_hashes = []
    admission_root = result_root(tag) / "admission"
    for name in ("launch_receipt.json", "run_status.json"):
        path = admission_root / name
        admission_hashes.append({
            "path": str(path.relative_to(REPO)), "sha256": protocol.file_sha256(path),
        })
    admission_hashes.extend(_admission_terminal_evidence(admission, admission_root))
    return campaign, manifest, records, {
        "admission_evidence_files": len(admission_hashes),
        "admission_evidence_sha256": protocol.canonical_sha256(admission_hashes),
        "campaign_lock_sha256": protocol.file_sha256(protocol.CAMPAIGN_LOCK_PATH),
        **_provenance_binding(),
        "confirmation_evidence_sha256": protocol.canonical_sha256(hashes),
        "confirmation_manifest_sha256": protocol.file_sha256(stage / "manifest.json"),
        "screen_evidence_sha256": protocol.canonical_sha256(screen_hashes),
        "selection_sha256": protocol.file_sha256(selection_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("provenance")
    commands.add_parser("ready")
    for name in ("admit", "screen", "select", "confirm"):
        command = commands.add_parser(name); command.add_argument("--tag", required=True)
    for name in ("admission-one", "verify-one"):
        command = commands.add_parser(name); command.add_argument("--tag", required=True); command.add_argument("--entry-id", required=True)
    command = commands.add_parser("time-one"); command.add_argument("--tag", required=True); command.add_argument("--phase", choices=("screen", "confirmation"), required=True); command.add_argument("--record-id", required=True); command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "provenance": print(provenance())
        elif args.command == "ready": print(ready())
        elif args.command == "admit": return admit(args.tag)
        elif args.command == "screen": return screen(args.tag)
        elif args.command == "select": print(select(args.tag))
        elif args.command == "confirm": return confirm(args.tag)
        elif args.command == "admission-one": return admission_one(args.tag, args.entry_id)
        elif args.command == "verify-one": return verify_one(args.tag, args.entry_id)
        else: return _time_one(args.tag, args.phase, args.record_id, args.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError, protocol.ProtocolError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
