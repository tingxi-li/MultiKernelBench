#!/usr/bin/env python3
"""Prepare, freeze, and run the native-strategy recurrence timing study."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
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
    import protocol


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = protocol.CONTRACT_PATH
MANIFEST_PATH = protocol.MANIFEST_PATH
MATERIAL_LOCK_PATH = protocol.MATERIAL_LOCK_PATH
PROVENANCE_PATH = protocol.PROVENANCE_PATH
EXECUTION_LOCK_PATH = protocol.EXECUTION_LOCK_PATH
RESULTS = protocol.RESULTS_ROOT
GLOBAL_GPU0_LOCK = Path("/tmp") / f"multikernelbench-{protocol.GPU0_UUID}-timing.lock"
RECORD_TIMEOUT_S = 900
SOURCE_NAMES = (
    ".gitignore", "README.md", "__init__.py", "protocol.py", "runner.py",
    "analyze.py", "artifacts.py", "test_protocol.py", "contract.json", "manifest.json",
)


def stable_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def exclusive_json(path: Path, value: Any) -> None:
    if path.exists():
        raise protocol.ProtocolError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_bytes(stable_bytes(value))
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = protocol.make_contract()
    manifest = protocol.make_manifest(contract)
    for path, value in ((CONTRACT_PATH, contract), (MANIFEST_PATH, manifest)):
        if path.exists():
            if protocol.read_json(path) != value:
                raise protocol.ProtocolError(f"prepared artifact is stale: {path}")
        else:
            exclusive_json(path, value)
    return contract, manifest


def source_map() -> dict[str, str]:
    paths = [HERE / name for name in SOURCE_NAMES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise protocol.ProtocolError(f"source closure is incomplete: {missing}")
    return {
        str(path.relative_to(protocol.REPO_ROOT)): protocol.file_sha256(path)
        for path in paths
    }


def make_material_lock(
    contract: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if contract is None or manifest is None:
        contract, manifest = prepare()
    sources = source_map()
    incident = protocol.predecessor_incident()
    value = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "contract_sha256": protocol.file_sha256(CONTRACT_PATH),
        "expected_raw_records": protocol.RAW_RECORDS,
        "instrument_evidence_index_sha256": contract["materials"]["instrument_evidence_index_sha256"],
        "instrument_launch_lock_sha256": contract["materials"]["instrument_launch_lock_sha256"],
        "instrument_source_bundle_sha256": contract["materials"]["instrument_source_bundle_sha256"],
        "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "materials_sha256": contract["materials_sha256"],
        "predecessor_incident_sha256": protocol.PREDECESSOR_INCIDENT_SHA256,
        "predecessor_result_closure_sha256": protocol.canonical_sha256(
            incident["artifact_closure"]["files"]
        ),
        "record_type": "native_trajectory_replication_ada_v2_material_lock",
        "schema_version": 1,
        "source_bundle_sha256": protocol.canonical_sha256(sources),
        "source_sha256": sources,
        "stage": "material",
    }
    value["lock_sha256"] = protocol.canonical_sha256(value)
    return value


def freeze_material() -> dict[str, Any]:
    expected = make_material_lock()
    if MATERIAL_LOCK_PATH.exists():
        if protocol.read_json(MATERIAL_LOCK_PATH) != expected:
            raise protocol.ProtocolError("material lock is stale")
    else:
        exclusive_json(MATERIAL_LOCK_PATH, expected)
    return expected


def validate_material_frozen(
    *, rehash_materials: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not all(path.is_file() for path in (CONTRACT_PATH, MANIFEST_PATH, MATERIAL_LOCK_PATH)):
        raise protocol.ProtocolError("prepare/material-freeze artifacts are incomplete")
    contract, manifest = protocol.read_json(CONTRACT_PATH), protocol.read_json(MANIFEST_PATH)
    if rehash_materials:
        protocol.validate_contract(contract)
        protocol.validate_manifest(contract, manifest)
    elif (
        manifest.get("contract_sha256") != protocol.canonical_sha256(contract)
        or manifest.get("plan_sha256") != protocol.canonical_sha256(manifest.get("rows"))
    ):
        raise protocol.ProtocolError("prepared artifacts lost their deterministic binding")
    lock = protocol.read_json(MATERIAL_LOCK_PATH)
    if lock != make_material_lock(contract, manifest):
        raise protocol.ProtocolError("material lock no longer matches current sources")
    return contract, manifest, lock


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=protocol.REPO_ROOT, capture_output=True,
        text=True, timeout=30,
    )
    if completed.returncode:
        raise protocol.ProtocolError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def upstream_remote_ref(upstream: str) -> tuple[str, str]:
    if "/" not in upstream:
        raise protocol.ProtocolError("configured upstream has no remote/branch pair")
    remote, branch = upstream.split("/", 1)
    if not remote or not branch:
        raise protocol.ProtocolError("configured upstream has no remote/branch pair")
    return remote, f"refs/heads/{branch}"


def _live_upstream_commit(upstream: str) -> tuple[str, str, str]:
    remote, remote_ref = upstream_remote_ref(upstream)
    rows = _git("ls-remote", remote, remote_ref).splitlines()
    if len(rows) != 1:
        raise protocol.ProtocolError("live upstream branch is missing or ambiguous")
    fields = rows[0].split()
    if len(fields) != 2 or fields[1] != remote_ref:
        raise protocol.ProtocolError("live upstream response is malformed")
    return remote, remote_ref, fields[0]


def _clean_tracked(paths: list[str]) -> None:
    if _git("status", "--porcelain", "--", *paths):
        raise protocol.ProtocolError("frozen execution closure is not clean")
    _git("ls-files", "--error-unmatch", "--", *paths)


def _compute_pids() -> set[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if completed.returncode:
        raise protocol.ProtocolError("physical GPU 0 compute-process query failed")
    try:
        return {int(row.strip()) for row in completed.stdout.splitlines() if row.strip()}
    except ValueError as exc:
        raise protocol.ProtocolError("physical GPU 0 compute-process query is malformed") from exc


def validate_gpu_snapshot(snapshot: Any, contract: dict[str, Any]) -> dict[str, str]:
    expected = {
        "compute_cap": contract["hardware"]["compute_capability"],
        "index": "0",
        "name": contract["hardware"]["gpu_name"],
        "uuid": contract["hardware"]["gpu_uuid"],
    }
    if not isinstance(snapshot, dict) or any(snapshot.get(key) != value for key, value in expected.items()):
        raise protocol.ProtocolError("physical GPU 0 identity differs from the frozen contract")
    if not isinstance(snapshot.get("driver_version"), str) or not snapshot["driver_version"]:
        raise protocol.ProtocolError("physical GPU 0 driver version is missing")
    return {
        key: snapshot[key]
        for key in ("index", "uuid", "name", "driver_version", "compute_cap")
    }


def _gpu0(contract: dict[str, Any], *, require_idle: bool) -> dict[str, str]:
    result = validate_gpu_snapshot(protocol.core.gpu_snapshot(0), contract)
    if require_idle and _compute_pids():
        raise protocol.ProtocolError("physical GPU 0 is occupied")
    return result


def _nvcc_fingerprint() -> dict[str, Any]:
    path = Path(protocol.CUDA_HOME) / "bin/nvcc"
    if not path.is_file():
        raise protocol.ProtocolError(f"frozen nvcc is missing: {path}")
    completed = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, timeout=20,
    )
    if completed.returncode:
        raise protocol.ProtocolError(completed.stderr.strip() or "frozen nvcc fingerprint failed")
    return {
        "path": str(path),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "stdout": completed.stdout.strip(),
    }


def live_toolchain() -> dict[str, Any]:
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import candidates
    import common
    import common2
    import runner2
    import tilelang
    import torch
    import triton

    executable = Path(sys.executable).resolve()
    modules = {
        name: module
        for name, module in (
            ("common", common),
            ("common2", common2),
            ("crossed_candidates", candidates),
            ("crossed_core", protocol.core),
            ("runner2", runner2),
            ("tilelang", tilelang),
            ("torch", torch),
            ("triton", triton),
        )
    }
    module_files = {}
    for name, module in modules.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not Path(raw_path).is_file():
            raise protocol.ProtocolError(f"toolchain module has no source file: {name}")
        path = Path(raw_path).resolve()
        module_files[name] = {"path": str(path), "sha256": protocol.file_sha256(path)}

    instrument_lock = protocol.read_json(protocol.CROSSED / "launch_lock.json")
    source_hashes = instrument_lock.get("source_sha256")
    dependency_hashes = instrument_lock.get("dependency_sha256")
    if not isinstance(source_hashes, dict) or not isinstance(dependency_hashes, dict):
        raise protocol.ProtocolError("crossed instrument lock lacks source fingerprints")
    for relative, expected in {**source_hashes, **dependency_hashes}.items():
        path = protocol.REPO_ROOT / relative
        if not path.is_file() or protocol.file_sha256(path) != expected:
            raise protocol.ProtocolError(f"runtime source differs from instrument lock: {relative}")
    if not executable.is_file():
        raise protocol.ProtocolError("Python executable is missing")
    return {
        "instrument_dependency_sha256": dependency_hashes,
        "instrument_source_sha256": source_hashes,
        "module_files": module_files,
        "nvcc": _nvcc_fingerprint(),
        "python_executable": str(executable),
        "python_executable_sha256": protocol.file_sha256(executable),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "tilelang_version": str(tilelang.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
    }


@contextmanager
def gpu0_lock() -> Iterator[Any]:
    handle = GLOBAL_GPU0_LOCK.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise protocol.ProtocolError("global physical-GPU0 timing lock is held") from None
    try:
        yield handle
    finally:
        handle.close()


def _validate_inherited_gpu0_lock() -> None:
    raw = os.environ.get("NATIVE_RECURRENCE_GPU0_LOCK_FD", "")
    try:
        descriptor = int(raw)
        inherited = os.fstat(descriptor)
        expected = GLOBAL_GPU0_LOCK.stat()
    except (OSError, ValueError) as exc:
        raise protocol.ProtocolError("timing child lacks the inherited GPU0 lock") from exc
    if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        raise protocol.ProtocolError("timing child inherited another GPU lock")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise protocol.ProtocolError("timing child does not share the held GPU0 lock") from exc


def make_provenance() -> dict[str, Any]:
    contract, _manifest, material_lock = validate_material_frozen()
    paths = list(source_map()) + [str(MATERIAL_LOCK_PATH.relative_to(protocol.REPO_ROOT))]
    _clean_tracked(paths)
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    head, cached = _git("rev-parse", "HEAD"), _git("rev-parse", upstream)
    remote, remote_ref, live = _live_upstream_commit(upstream)
    if head != cached or head != live:
        raise protocol.ProtocolError("material-freeze commit is not the cached and live upstream head")
    instrument = subprocess.run(
        [sys.executable, str(protocol.CROSSED / "validate.py"), "--stage", "campaign", "--remote-ready"],
        cwd=protocol.REPO_ROOT, capture_output=True, text=True,
    )
    if instrument.returncode:
        raise protocol.ProtocolError(instrument.stderr.strip() or instrument.stdout.strip())
    toolchain = live_toolchain()
    with gpu0_lock():
        gpu = _gpu0(contract, require_idle=True)
    return {
        "campaign_id": protocol.CAMPAIGN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
        "git_upstream": upstream,
        "git_upstream_commit": cached,
        "gpu0": gpu,
        "host": platform.node(),
        "live_remote_commit": live,
        "live_remote_name": remote,
        "live_remote_ref": remote_ref,
        "material_lock_sha256": protocol.file_sha256(MATERIAL_LOCK_PATH),
        "record_type": "native_trajectory_replication_ada_v2_prelaunch_provenance",
        "remote_push_verified": True,
        "schema_version": 1,
        "source_bundle_sha256": material_lock["source_bundle_sha256"],
        "stage": "material_commit_live_upstream_verified",
        "toolchain": toolchain,
    }


def provenance() -> None:
    exclusive_json(PROVENANCE_PATH, make_provenance())


def _validate_toolchain_receipt(value: Any) -> None:
    required = {
        "instrument_dependency_sha256", "instrument_source_sha256", "module_files",
        "nvcc", "python_executable", "python_executable_sha256",
        "python_implementation", "python_version", "tilelang_version",
        "torch_cuda_version", "torch_version", "triton_version",
    }
    instrument_lock = protocol.read_json(protocol.CROSSED / "launch_lock.json")
    modules = value.get("module_files") if isinstance(value, dict) else None
    nvcc = value.get("nvcc") if isinstance(value, dict) else None
    strings = (
        "python_executable", "python_implementation", "python_version",
        "tilelang_version", "torch_cuda_version", "torch_version", "triton_version",
    )
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("instrument_source_sha256") != instrument_lock.get("source_sha256")
        or value.get("instrument_dependency_sha256") != instrument_lock.get("dependency_sha256")
        or any(not isinstance(value.get(key), str) or not value[key] for key in strings)
        or not Path(value["python_executable"]).is_absolute()
        or not isinstance(value.get("python_executable_sha256"), str)
        or len(value["python_executable_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["python_executable_sha256"])
        or not isinstance(modules, dict)
        or set(modules) != {
            "common", "common2", "crossed_candidates", "crossed_core", "runner2",
            "tilelang", "torch", "triton",
        }
        or not isinstance(nvcc, dict)
        or set(nvcc) != {"path", "returncode", "stderr", "stdout"}
        or nvcc.get("path") != str(Path(protocol.CUDA_HOME) / "bin/nvcc")
        or nvcc.get("returncode") != 0
        or not isinstance(nvcc.get("stdout"), str)
        or not isinstance(nvcc.get("stderr"), str)
    ):
        raise protocol.ProtocolError("toolchain receipt is malformed or unbound")
    for name, binding in modules.items():
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str)
            or not Path(binding["path"]).is_absolute()
            or not isinstance(binding["sha256"], str)
            or len(binding["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in binding["sha256"])
        ):
            raise protocol.ProtocolError(f"toolchain module binding is malformed: {name}")


def _validate_provenance(
    value: dict[str, Any], contract: dict[str, Any], material_lock: dict[str, Any]
) -> None:
    commit = value.get("git_commit")
    created = value.get("created_utc")
    try:
        timestamp = datetime.fromisoformat(created) if isinstance(created, str) else None
    except ValueError:
        timestamp = None
    expected_gpu = {
        "compute_cap": contract["hardware"]["compute_capability"],
        "index": "0",
        "name": contract["hardware"]["gpu_name"],
        "uuid": contract["hardware"]["gpu_uuid"],
    }
    gpu = value.get("gpu0")
    toolchain = value.get("toolchain")
    upstream = value.get("git_upstream")
    try:
        expected_remote = upstream_remote_ref(upstream) if isinstance(upstream, str) else None
    except protocol.ProtocolError:
        expected_remote = None
    expected_fields = {
        "campaign_id", "created_utc", "git_commit", "git_upstream",
        "git_upstream_commit", "gpu0", "host", "live_remote_commit",
        "live_remote_name", "live_remote_ref", "material_lock_sha256",
        "record_type", "remote_push_verified", "schema_version",
        "source_bundle_sha256", "stage", "toolchain",
    }
    if (
        set(value) != expected_fields
        or value.get("campaign_id") != protocol.CAMPAIGN_ID
        or value.get("record_type") != "native_trajectory_replication_ada_v2_prelaunch_provenance"
        or value.get("schema_version") != 1
        or value.get("stage") != "material_commit_live_upstream_verified"
        or value.get("material_lock_sha256") != protocol.file_sha256(MATERIAL_LOCK_PATH)
        or value.get("source_bundle_sha256") != material_lock["source_bundle_sha256"]
        or value.get("remote_push_verified") is not True
        or value.get("git_commit") != value.get("git_upstream_commit")
        or value.get("git_commit") != value.get("live_remote_commit")
        or not isinstance(commit, str)
        or len(commit) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in commit)
        or timestamp is None
        or timestamp.utcoffset() != timezone.utc.utcoffset(None)
        or value.get("host") != platform.node()
        or not isinstance(gpu, dict)
        or set(gpu) != {*expected_gpu, "driver_version"}
        or any(gpu.get(key) != expected for key, expected in expected_gpu.items())
        or not isinstance(gpu.get("driver_version"), str)
        or not gpu["driver_version"]
        or expected_remote != (value.get("live_remote_name"), value.get("live_remote_ref"))
    ):
        raise protocol.ProtocolError("material-stage provenance is stale or foreign")
    _validate_toolchain_receipt(toolchain)


def make_execution_lock(
    contract: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    material_lock: dict[str, Any] | None = None,
    provenance_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if contract is None or manifest is None or material_lock is None:
        contract, manifest, material_lock = validate_material_frozen()
    if provenance_receipt is None and not PROVENANCE_PATH.is_file():
        raise protocol.ProtocolError("material-stage provenance is missing; commit and push stage one first")
    receipt = provenance_receipt or protocol.read_json(PROVENANCE_PATH)
    _validate_provenance(receipt, contract, material_lock)
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore
    admission_plan = artifacts.admission_plan(contract)
    value = {
        "artifact_admission_expected_entries": 12,
        "artifact_admission_plan_sha256": protocol.canonical_sha256(admission_plan),
        "campaign_id": protocol.CAMPAIGN_ID,
        "expected_position_receipts": protocol.RAW_RECORDS,
        "expected_raw_records": protocol.RAW_RECORDS,
        "gpu0_uuid": contract["hardware"]["gpu_uuid"],
        "manifest_plan_sha256": manifest["plan_sha256"],
        "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "material_lock_sha256": protocol.file_sha256(MATERIAL_LOCK_PATH),
        "prelaunch_provenance_sha256": protocol.file_sha256(PROVENANCE_PATH),
        "record_timeout_s": RECORD_TIMEOUT_S,
        "record_type": "native_trajectory_replication_ada_v2_execution_lock",
        "schema_version": 1,
        "source_bundle_sha256": material_lock["source_bundle_sha256"],
        "stage": "execution",
        "timing": contract["timing"],
        "toolchain": receipt["toolchain"],
    }
    value["lock_sha256"] = protocol.canonical_sha256(value)
    return value


def freeze_execution() -> dict[str, Any]:
    expected = make_execution_lock()
    if EXECUTION_LOCK_PATH.exists():
        if protocol.read_json(EXECUTION_LOCK_PATH) != expected:
            raise protocol.ProtocolError("execution lock is stale")
    else:
        exclusive_json(EXECUTION_LOCK_PATH, expected)
    return expected


def validate_execution_frozen(
    *, rehash_materials: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract, manifest, material_lock = validate_material_frozen(rehash_materials=rehash_materials)
    if not PROVENANCE_PATH.is_file() or not EXECUTION_LOCK_PATH.is_file():
        raise protocol.ProtocolError("provenance/execution freeze is incomplete")
    provenance_receipt = protocol.read_json(PROVENANCE_PATH)
    _validate_provenance(provenance_receipt, contract, material_lock)
    execution_lock = protocol.read_json(EXECUTION_LOCK_PATH)
    if execution_lock != make_execution_lock(contract, manifest, material_lock, provenance_receipt):
        raise protocol.ProtocolError("execution lock no longer matches the frozen closure")
    return contract, manifest, execution_lock, provenance_receipt


def source_cells() -> dict[str, dict[str, Any]]:
    return {
        cell["cell_id"]: cell
        for cell in protocol.core.load_cells(require_resolved=True)
    }


def validate_artifact_admission(contract: dict[str, Any]) -> dict[str, Any]:
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore

    if not artifacts.MANIFEST_PATH.is_file():
        raise protocol.ProtocolError("artifact admission is incomplete")
    cells = source_cells()
    manifest = artifacts.validate_manifest(
        protocol.read_json(artifacts.MANIFEST_PATH), cells
    )
    receipt = _validate_admission_receipt(
        protocol.read_json(protocol.REPO_ROOT / manifest["launch_receipt_path"]),
        _admission_expected(contract),
    )
    execution_lock = protocol.read_json(EXECUTION_LOCK_PATH)
    provenance_receipt = protocol.read_json(PROVENANCE_PATH)
    if (
        receipt["contract"]["toolchain"] != execution_lock["toolchain"]
        or receipt["contract"]["gpu_preflight"] != provenance_receipt["gpu0"]
    ):
        raise protocol.ProtocolError("artifact admission launch identity changed")
    status_path = artifacts.ADMISSION_ROOT / "run_status.json"
    status = protocol.read_json(status_path)
    expected_status = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "complete": True,
        "expected_records": 12,
        "launch_receipt_sha256": manifest["launch_receipt_sha256"],
        "observed_records": 12,
        "record_type": "native_trajectory_replication_ada_artifact_admission_status",
        "schema_version": 1,
    }
    if status != expected_status:
        raise protocol.ProtocolError("artifact admission status is incomplete or foreign")
    paths = artifacts.manifest_paths(manifest)
    retained = [
        path for path in artifacts.ADMISSION_ROOT.rglob("*")
        if "runtime_tmp" not in path.relative_to(artifacts.ADMISSION_ROOT).parts
    ]
    if any(path.is_symlink() for path in retained):
        raise protocol.ProtocolError("artifact admission contains a symlink")
    observed = {
        protocol.repo_path(path)
        for path in retained
        if path.is_file()
        and path.name != "active.lock"
    }
    if observed != set(paths):
        raise protocol.ProtocolError("artifact admission file census differs from its manifest")
    closure = [
        {
            "path": relative,
            "sha256": protocol.file_sha256(protocol.REPO_ROOT / relative),
            "size": (protocol.REPO_ROOT / relative).stat().st_size,
        }
        for relative in sorted(paths)
    ]
    return {
        "artifact_admission_closure_sha256": protocol.canonical_sha256(closure),
        "artifact_admission_manifest_path": protocol.repo_path(artifacts.MANIFEST_PATH),
        "artifact_admission_manifest_sha256": protocol.file_sha256(artifacts.MANIFEST_PATH),
        "cells": cells,
        "closure": closure,
        "manifest": manifest,
        "paths": paths,
    }


def artifact_cell_state(cell_id: str, state: dict[str, Any]) -> dict[str, Any]:
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore

    manifest = state["manifest"]
    binding = artifacts.binding(cell_id, manifest)
    rows = [row for row in manifest["admission_plan"] if row["cell_id"] == cell_id]
    if len(rows) != 1:
        raise protocol.ProtocolError(f"artifact admission has no unique row: {cell_id}")
    entry = artifacts.validate_entry(
        protocol.read_json(protocol.REPO_ROOT / binding["admitted_entry_path"]),
        rows[0], state["cells"][cell_id],
    )
    build = artifacts.validate_build_record(
        protocol.read_json(protocol.REPO_ROOT / entry["build_record_path"]), rows[0]
    )
    return {"binding": binding, "build": build, "entry": entry, "row": rows[0]}


def ready(*, require_idle: bool = True, require_artifacts: bool = False) -> dict[str, Any]:
    contract, manifest, execution_lock, receipt = validate_execution_frozen()
    artifact_state = validate_artifact_admission(contract) if require_artifacts else None
    paths = list(source_map()) + [
        str(path.relative_to(protocol.REPO_ROOT))
        for path in (MATERIAL_LOCK_PATH, PROVENANCE_PATH, EXECUTION_LOCK_PATH)
    ]
    if artifact_state is not None:
        paths.extend(artifact_state["paths"])
    _clean_tracked(paths)
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    head, cached = _git("rev-parse", "HEAD"), _git("rev-parse", upstream)
    remote, remote_ref, live = _live_upstream_commit(upstream)
    if head != cached or head != live:
        raise protocol.ProtocolError("execution-freeze commit is not the cached and live upstream head")
    if (remote, remote_ref) != (receipt.get("live_remote_name"), receipt.get("live_remote_ref")):
        raise protocol.ProtocolError("configured upstream changed between freeze stages")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(receipt.get("git_commit", "")), head],
        cwd=protocol.REPO_ROOT,
    )
    if ancestor.returncode:
        raise protocol.ProtocolError("material-stage commit is not an ancestor of execution-stage HEAD")
    gpu = _gpu0(contract, require_idle=require_idle)
    toolchain = live_toolchain()
    if gpu != receipt.get("gpu0") or toolchain != receipt.get("toolchain"):
        raise protocol.ProtocolError("GPU/toolchain identity changed after material-stage provenance")
    return {
        "contract": contract,
        "execution_lock": execution_lock,
        "git_commit": head,
        "gpu0": gpu,
        "manifest": manifest,
        "provenance": receipt,
        "toolchain": toolchain,
        "artifacts": artifact_state,
    }


def _admission_expected(contract: dict[str, Any]) -> dict[str, Any]:
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore

    plan = artifacts.admission_plan(contract)
    return {
        "artifact_policy": contract["artifact_admission"],
        "campaign_id": protocol.CAMPAIGN_ID,
        "execution_lock_sha256": protocol.file_sha256(EXECUTION_LOCK_PATH),
        "gpu_lock_path": str(GLOBAL_GPU0_LOCK),
        "plan": plan,
        "plan_sha256": protocol.canonical_sha256(plan),
        "source_bundle_sha256": protocol.read_json(EXECUTION_LOCK_PATH)["source_bundle_sha256"],
        "stage": "artifact_admission",
    }


def _validate_admission_receipt(value: Any, expected: dict[str, Any]) -> dict[str, Any]:
    contract = value.get("contract") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != {"contract", "created_utc", "record_type", "schema_version"}
        or value.get("record_type") != "native_trajectory_replication_ada_artifact_admission_launch"
        or value.get("schema_version") != 1
        or not isinstance(contract, dict)
        or set(contract) != {*expected, "git_commit", "gpu_preflight", "toolchain"}
        or any(contract.get(key) != item for key, item in expected.items())
        or not isinstance(contract.get("git_commit"), str)
        or len(contract["git_commit"]) not in (40, 64)
        or not isinstance(contract.get("gpu_preflight"), dict)
        or not isinstance(contract.get("toolchain"), dict)
    ):
        raise protocol.ProtocolError("artifact admission launch receipt is incomplete or foreign")
    return value


def _artifact_gate(built: Any) -> dict[str, Any]:
    import common
    import common2
    import torch

    x, weight, bias = common2.fused_inputs(seed=0, dist="rand")
    with torch.no_grad():
        reference = common2.fused_reference(x, weight, bias, arm="GBGS", dtype=torch.float32)
        observed = built.run(x.half().contiguous(), weight, bias)
        torch.cuda.synchronize()
    result = common.gate_stats(reference, observed.float())
    if result.get("gate_pass") is not True:
        raise protocol.ProtocolError("admitted artifact failed the frozen positive-input gate")
    return result


def _artifact_child_context(
    cell_id: str, receipt_path: Path, receipt_sha256: str, mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise protocol.ProtocolError("artifact child requires CUDA_VISIBLE_DEVICES=0")
    _validate_inherited_gpu0_lock()
    artifacts.validate_cache_environment(cell_id, mode)
    canonical = artifacts.ADMISSION_ROOT / "launch_receipt.json"
    if (
        receipt_path.resolve() != canonical.resolve()
        or protocol.file_sha256(receipt_path) != receipt_sha256
    ):
        raise protocol.ProtocolError("artifact child received a foreign launch receipt")
    contract, _manifest, execution_lock, provenance_receipt = validate_execution_frozen()
    receipt = _validate_admission_receipt(
        protocol.read_json(receipt_path), _admission_expected(contract)
    )
    if (
        receipt["contract"]["toolchain"] != execution_lock["toolchain"]
        or receipt["contract"]["gpu_preflight"] != provenance_receipt["gpu0"]
    ):
        raise protocol.ProtocolError("artifact child launch identity changed")
    plan_rows = [row for row in artifacts.admission_plan(contract) if row["cell_id"] == cell_id]
    cells = source_cells()
    if len(plan_rows) != 1 or cell_id not in cells:
        raise protocol.ProtocolError(f"cell is outside artifact admission: {cell_id}")
    toolchain = live_toolchain()
    if toolchain != execution_lock["toolchain"]:
        raise protocol.ProtocolError("artifact child toolchain changed")
    return plan_rows[0], cells[cell_id], contract, toolchain


def _artifact_provenance(
    cell_id: str, mode: str, receipt_path: Path, gpu: dict[str, Any],
    toolchain: dict[str, Any], pids_pre: list[int], pids_post: list[int],
) -> dict[str, Any]:
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore

    return {
        "cache_environment": artifacts.cache_environment(cell_id, mode),
        "compute_pids_postflight": pids_post,
        "compute_pids_preflight": pids_pre,
        "execution_lock_sha256": protocol.file_sha256(EXECUTION_LOCK_PATH),
        "gpu_postflight": _gpu0(protocol.load_contract(), require_idle=False),
        "gpu_preflight": gpu,
        "launch_receipt_path": protocol.repo_path(receipt_path),
        "launch_receipt_sha256": protocol.file_sha256(receipt_path),
        "parent_pid": os.getppid(),
        "process_pid": os.getpid(),
        "toolchain": toolchain,
    }


def artifact_one(
    command: str, cell_id: str, output: Path, receipt_path: Path, receipt_sha256: str,
) -> int:
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore

    mode = "admit" if command == "artifact-build-one" else "load_only"
    try:
        row, cell, contract, toolchain = _artifact_child_context(
            cell_id, receipt_path, receipt_sha256, mode
        )
        paths = artifacts.entry_paths(cell_id)
        expected_output = paths["build" if mode == "admit" else "verify"]
        if output.resolve() != expected_output.resolve() or output.exists():
            raise protocol.ProtocolError("unsafe or existing artifact child output")
        gpu = _gpu0(contract, require_idle=True)
        pids_pre = sorted(_compute_pids())
        if pids_pre:
            raise protocol.ProtocolError("physical GPU 0 is occupied before artifact work")
        from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import candidates

        if mode == "admit":
            with artifacts.capture_torch_extensions() as requests:
                built = candidates.build(cell)
                correctness = _artifact_gate(built)
            pids_post = sorted(_compute_pids())
            if not set(pids_post) <= {os.getpid()}:
                raise protocol.ProtocolError("another compute process overlapped artifact admission")
            value = artifacts.build_record(
                row, built, correctness,
                _artifact_provenance(
                    cell_id, mode, receipt_path, gpu, toolchain, pids_pre, pids_post
                ),
                requests,
            )
            exclusive_json(output, value)
            artifacts.validate_build_record(protocol.read_json(output), row)
        else:
            build = artifacts.validate_build_record(protocol.read_json(paths["build"]), row)
            with artifacts.load_only_guards(build) as evidence:
                built = candidates.build(cell)
                correctness = _artifact_gate(built)
            artifacts.validate_load_evidence(evidence, cell, build)
            pids_post = sorted(_compute_pids())
            if not set(pids_post) <= {os.getpid()}:
                raise protocol.ProtocolError("another compute process overlapped artifact verification")
            value = artifacts.verify_record(
                row, build, built, correctness, evidence,
                _artifact_provenance(
                    cell_id, mode, receipt_path, gpu, toolchain, pids_pre, pids_post
                ),
            )
            exclusive_json(output, value)
            artifacts.validate_verify_record(protocol.read_json(output), row, build, cell)
        return 0
    except Exception as exc:
        failure = artifacts.entry_root(cell_id) / "failure.json"
        if not failure.exists():
            exclusive_json(
                failure,
                {
                    "campaign_id": protocol.CAMPAIGN_ID,
                    "cell_id": cell_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "phase": mode,
                    "record_type": "native_trajectory_replication_ada_artifact_admission_failure",
                    "schema_version": 1,
                    "traceback": traceback.format_exc(),
                },
            )
        return 1


def _artifact_child_command(
    command: str, row: dict[str, Any], receipt: Path, gpu_lock: Any,
) -> int:
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore

    paths = artifacts.entry_paths(row["cell_id"])
    output = paths["build" if command == "artifact-build-one" else "verify"]
    env = os.environ.copy()
    if command == "artifact-build-one":
        env.pop("TRITON_CACHE_MANAGER", None)
    env.update(
        {
            "CUDA_HOME": protocol.CUDA_HOME,
            "CUDA_VISIBLE_DEVICES": "0",
            "NATIVE_RECURRENCE_GPU0_LOCK_FD": str(gpu_lock.fileno()),
            "PATH": str(Path(protocol.CUDA_HOME) / "bin") + ":" + env.get("PATH", ""),
        }
    )
    env.update(
        artifacts.prepare_cache_environment(
            row["cell_id"], "admit" if command == "artifact-build-one" else "load_only"
        )
    )
    env.setdefault("MAX_JOBS", "4")
    completed = subprocess.run(
        [
            sys.executable, str(HERE / "runner.py"), command,
            "--cell-id", row["cell_id"], "--output", str(output),
            "--launch-receipt", str(receipt),
            "--launch-receipt-sha256", protocol.file_sha256(receipt),
        ],
        cwd=protocol.REPO_ROOT, env=env, pass_fds=(gpu_lock.fileno(),),
        timeout=RECORD_TIMEOUT_S,
    )
    return completed.returncode


def admit_artifacts() -> int:
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore

    with gpu0_lock() as gpu_lock:
        state = ready(require_idle=True, require_artifacts=False)
        contract = state["contract"]
        plan = artifacts.admission_plan(contract)
        artifacts.ADMISSION_ROOT.mkdir(parents=True, exist_ok=True)
        active_path = artifacts.ADMISSION_ROOT / "active.lock"
        active = active_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            active.close()
            raise protocol.ProtocolError("another artifact-admission launcher is active") from None
        try:
            receipt_path = artifacts.ADMISSION_ROOT / "launch_receipt.json"
            expected = _admission_expected(contract)
            receipt = {
                "contract": {
                    **expected,
                    "git_commit": state["git_commit"],
                    "gpu_preflight": state["gpu0"],
                    "toolchain": state["toolchain"],
                },
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "record_type": "native_trajectory_replication_ada_artifact_admission_launch",
                "schema_version": 1,
            }
            if receipt_path.exists():
                retained = _validate_admission_receipt(protocol.read_json(receipt_path), expected)
                if retained["contract"] != receipt["contract"]:
                    raise protocol.ProtocolError("artifact admission launch environment changed")
            else:
                exclusive_json(receipt_path, receipt)
            cells = source_cells()
            for row in plan:
                paths = artifacts.entry_paths(row["cell_id"])
                failure = paths["root"] / "failure.json"
                if failure.exists():
                    raise protocol.ProtocolError(
                        f"retained artifact failure requires a new successor: {failure}"
                    )
                if not paths["build"].exists():
                    if paths["root"].exists():
                        raise protocol.ProtocolError(
                            f"partial artifact admission requires a new successor: {paths['root']}"
                        )
                    if _artifact_child_command("artifact-build-one", row, receipt_path, gpu_lock):
                        raise protocol.ProtocolError("artifact build failed; retain it and create a successor")
                build = artifacts.validate_build_record(protocol.read_json(paths["build"]), row)
                if not paths["verify"].exists() and _artifact_child_command(
                    "artifact-verify-one", row, receipt_path, gpu_lock
                ):
                    raise protocol.ProtocolError("artifact verification failed; retain it and create a successor")
                artifacts.validate_verify_record(
                    protocol.read_json(paths["verify"]), row, build, cells[row["cell_id"]]
                )
                entry = artifacts.make_entry(row, cells[row["cell_id"]])
                if paths["entry"].exists():
                    artifacts.validate_entry(protocol.read_json(paths["entry"]), row, cells[row["cell_id"]])
                else:
                    exclusive_json(paths["entry"], entry)
                print(f"[{row['position'] + 1}/12] admitted {row['cell_id']}", flush=True)
            manifest = artifacts.make_manifest(receipt_path, cells)
            if artifacts.MANIFEST_PATH.exists():
                artifacts.validate_manifest(protocol.read_json(artifacts.MANIFEST_PATH), cells)
            else:
                exclusive_json(artifacts.MANIFEST_PATH, manifest)
            status_path = artifacts.ADMISSION_ROOT / "run_status.json"
            status = {
                "campaign_id": protocol.CAMPAIGN_ID,
                "complete": True,
                "expected_records": 12,
                "launch_receipt_sha256": protocol.file_sha256(receipt_path),
                "observed_records": 12,
                "record_type": "native_trajectory_replication_ada_artifact_admission_status",
                "schema_version": 1,
            }
            if status_path.exists():
                if protocol.read_json(status_path) != status:
                    raise protocol.ProtocolError("artifact admission status changed")
            else:
                exclusive_json(status_path, status)
            _gpu0(contract, require_idle=True)
            return 0
        finally:
            active.close()


def _manifest_row(manifest: dict[str, Any], row_id: str) -> dict[str, Any]:
    rows = [row for row in manifest["rows"] if row["row_id"] == row_id]
    if len(rows) != 1:
        raise protocol.ProtocolError("unknown or duplicate manifest row")
    return rows[0]


def time_one(row_id: str, output: Path) -> int:
    _validate_inherited_gpu0_lock()
    contract, manifest, execution_lock, _receipt = validate_execution_frozen(rehash_materials=False)
    artifact_state = validate_artifact_admission(contract)
    row = _manifest_row(manifest, row_id)
    artifact_cell = artifact_cell_state(row["cell_id"], artifact_state)
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore
    artifacts.validate_cache_environment(row["cell_id"], "load_only")
    expected_output = (RESULTS / "raw" / protocol.raw_filename(row)).resolve()
    if output.resolve() != expected_output or output.exists():
        raise protocol.ProtocolError("timing output is outside its frozen position or already exists")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or os.environ.get("CUDA_HOME") != protocol.CUDA_HOME
    ):
        raise protocol.ProtocolError("timing child is not bound to frozen GPU0/CUDA_HOME")
    toolchain = live_toolchain()
    if toolchain != execution_lock["toolchain"]:
        raise protocol.ProtocolError("timing child toolchain/source identity changed")
    gpu = _gpu0(contract, require_idle=True)
    started = time.time_ns()
    record: dict[str, Any] = {
        "block": row["block"],
        "block_position": row["block_position"],
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": row["cell_id"],
        "cell_sha256_expected": next(
            material["cell_sha256"]
            for material in contract["materials"]["selected_prefixes"]
            if material["cell_id"] == row["cell_id"]
        ),
        "compute_pids_preflight": [],
        "distribution": row["distribution"],
        "execution_lock_sha256": protocol.file_sha256(EXECUTION_LOCK_PATH),
        "global_position": row["global_position"],
        "gpu_preflight": gpu,
        "artifact_admission_closure_sha256": artifact_state["artifact_admission_closure_sha256"],
        "artifact_identity_sha256_expected": artifact_cell["build"]["artifact_identity_sha256"],
        **artifact_cell["binding"],
        "label": row["label"],
        "manifest_row_sha256": protocol.canonical_sha256(row),
        "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "physical_gpu": 0,
        "process_pid": os.getpid(),
        "record_kind": row["record_kind"],
        "record_type": "native_trajectory_replication_ada_v2_timing_record",
        "row_id": row_id,
        "schema_version": 1,
        "source_bundle_sha256": execution_lock["source_bundle_sha256"],
        "predecessor_implementation_sha256_expected": row["predecessor_implementation_sha256"],
        "t_start_unix_ns": started,
        "toolchain": toolchain,
        "trials": protocol.TIMING_TRIALS,
        "warmup_s": protocol.WARMUP_S,
    }
    try:
        from ako_runs.controlled_followup.fused_epilogue_crossed_v2.candidates import build
        import common
        import common2
        import runner2
        import torch

        cell = artifact_state["cells"][row["cell_id"]]
        if protocol.core.canonical_sha256(cell) != record["cell_sha256_expected"]:
            raise protocol.ProtocolError("built cell/config differs from admitted material")
        with artifacts.load_only_guards(artifact_cell["build"]) as load_evidence:
            built = build(cell)
            if built.config != artifact_cell["build"]["config"] or built.metadata.get("n_kernels") != 2:
                raise protocol.ProtocolError("loaded cell/config differs from its admitted artifact")
            seed, dist = (0, "rand") if row["distribution"] == "positive" else (protocol.WITHHELD_SEED, "randn")
            x, weight, bias = common2.fused_inputs(seed=seed, dist=dist)
            x16 = x.half().contiguous()
            with torch.no_grad():
                reference = common2.fused_reference(x, weight, bias, arm="GBGS", dtype=torch.float32)
                observed = built.run(x16, weight, bias)
                torch.cuda.synchronize()
            live_gate = common.gate_stats(reference, observed.float())
            if live_gate.get("gate_pass") is not True:
                raise protocol.ProtocolError("fresh timing launch failed the live correctness check")
            del reference, observed
            torch.cuda.empty_cache()
            times, warmup_iterations = runner2.time_kernel3(
                built.run, x16, weight, bias,
                num_trials=protocol.TIMING_TRIALS,
                warmup_s=protocol.WARMUP_S,
                flush_l2=True,
            )
        artifacts.validate_load_evidence(load_evidence, cell, artifact_cell["build"])
        numeric = [float(value) for value in times]
        gpu_postflight = _gpu0(contract, require_idle=False)
        compute_pids_postflight = sorted(_compute_pids())
        record.update(
            {
                "compute_pids_postflight": compute_pids_postflight,
                "gpu_postflight": gpu_postflight,
            }
        )
        if not set(compute_pids_postflight) <= {os.getpid()}:
            raise protocol.ProtocolError("another compute process overlapped the timing record")
        record.update(
            {
                "build_metadata": {
                    "artifact_identity_sha256": artifact_cell["build"]["artifact_identity_sha256"],
                    "n_kernels": built.metadata["n_kernels"],
                    "predecessor_implementation_sha256_observed": built.metadata.get("implementation_sha256"),
                },
                "cell_sha256": protocol.core.canonical_sha256(cell),
                "compile_s": float(built.compile_s),
                "full_median_ms": statistics.median(numeric),
                "artifact_identity_sha256": artifact_cell["build"]["artifact_identity_sha256"],
                "live_correctness": live_gate,
                "load_evidence": load_evidence,
                "ok": True,
                "primary_tail_median_ms": statistics.median(numeric[protocol.TAIL_START:protocol.TAIL_STOP]),
                "times_ms": numeric,
                "warmup_iterations_actual": int(warmup_iterations),
            }
        )
    except Exception as exc:
        record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "ok": False,
                "traceback": traceback.format_exc(),
            }
        )
    record["t_end_unix_ns"] = time.time_ns()
    exclusive_json(output, record)
    return 0 if record["ok"] else 1


def _wait_child(command: list[str], env: dict[str, str], lock_fd: int) -> int:
    process = subprocess.Popen(
        command, cwd=protocol.REPO_ROOT, env=env, start_new_session=True,
        pass_fds=(lock_fd,),
    )
    try:
        return process.wait(timeout=RECORD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return 124


def execute() -> int:
    try:
        from . import artifacts
    except ImportError:
        import artifacts  # type: ignore

    with gpu0_lock() as gpu_lock:
        state = ready(require_idle=True, require_artifacts=True)
        contract, manifest = state["contract"], state["manifest"]
        artifact_state = state["artifacts"]
        (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
        (RESULTS / "position_receipts").mkdir(parents=True, exist_ok=True)
        launch_path = RESULTS / "launch_receipt.json"
        launch_contract = {
            "campaign_id": protocol.CAMPAIGN_ID,
            "artifact_admission_closure_sha256": artifact_state["artifact_admission_closure_sha256"],
            "artifact_admission_manifest_path": artifact_state["artifact_admission_manifest_path"],
            "artifact_admission_manifest_sha256": artifact_state["artifact_admission_manifest_sha256"],
            "execution_lock_sha256": protocol.file_sha256(EXECUTION_LOCK_PATH),
            "expected_raw_records": protocol.RAW_RECORDS,
            "git_commit": state["git_commit"],
            "gpu_preflight": state["gpu0"],
            "manifest_plan_sha256": manifest["plan_sha256"],
            "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
            "physical_gpu": 0,
            "toolchain": state["toolchain"],
        }
        if launch_path.exists():
            launch = protocol.read_json(launch_path)
            if launch.get("contract") != launch_contract:
                raise protocol.ProtocolError("existing launch receipt has another contract")
        else:
            launch = {
                "contract": launch_contract,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "record_type": "native_trajectory_replication_ada_v2_launch_receipt",
                "schema_version": 1,
            }
            exclusive_json(launch_path, launch)
        env = os.environ.copy()
        env.update(
            {
                "CUDA_HOME": protocol.CUDA_HOME,
                "CUDA_VISIBLE_DEVICES": "0",
                "NATIVE_RECURRENCE_GPU0_LOCK_FD": str(gpu_lock.fileno()),
                "PATH": str(Path(protocol.CUDA_HOME) / "bin") + ":" + env.get("PATH", ""),
            }
        )
        env.setdefault("MAX_JOBS", "4")
        from ako_runs.controlled_followup.native_trajectory_replication_ada_v2 import analyze

        execution_binding = {
            "artifact_admission_closure_sha256": artifact_state["artifact_admission_closure_sha256"],
            "artifact_admission_manifest_path": artifact_state["artifact_admission_manifest_path"],
            "artifact_admission_manifest_sha256": artifact_state["artifact_admission_manifest_sha256"],
            "execution_lock_sha256": protocol.file_sha256(EXECUTION_LOCK_PATH),
            "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
            "source_bundle_sha256": state["execution_lock"]["source_bundle_sha256"],
            "toolchain": state["toolchain"],
        }
        previous_completed = 0

        for row in manifest["rows"]:
            artifact_cell = artifact_cell_state(row["cell_id"], artifact_state)
            raw_path = RESULTS / "raw" / protocol.raw_filename(row)
            position_path = RESULTS / "position_receipts" / protocol.position_receipt_filename(row)
            if raw_path.exists() or position_path.exists():
                if not raw_path.is_file() or not position_path.is_file():
                    raise protocol.ProtocolError("partial immutable position evidence exists")
                record = protocol.read_json(raw_path)
                analyze.validate_timing_record(
                    contract, row, record, execution_binding=execution_binding,
                    artifact_binding=artifact_cell["binding"], admitted_build=artifact_cell["build"],
                )
                position = protocol.read_json(position_path)
                previous_completed = analyze.validate_position_receipt(
                    row, record, position, raw_path, previous_completed,
                )
                continue
            command = [
                sys.executable, str(HERE / "runner.py"), "time-one",
                "--row-id", row["row_id"], "--output", str(raw_path),
            ]
            child_env = env.copy()
            child_env.update(
                artifacts.prepare_cache_environment(row["cell_id"], "load_only")
            )
            launched = time.time_ns()
            returncode = _wait_child(command, child_env, gpu_lock.fileno())
            completed = time.time_ns()
            gpu_idle_after_child = None
            compute_pids_after_child = None
            gpu_postflight_error = None
            try:
                gpu_idle_after_child = _gpu0(contract, require_idle=True)
                compute_pids_after_child = []
            except protocol.ProtocolError as exc:
                gpu_postflight_error = str(exc)
            position = {
                "campaign_id": protocol.CAMPAIGN_ID,
                "child_completed_unix_ns": completed,
                "child_launched_unix_ns": launched,
                "global_position": row["global_position"],
                "compute_pids_after_child": compute_pids_after_child,
                "gpu_idle_after_child": gpu_idle_after_child,
                "gpu_postflight_error": gpu_postflight_error,
                "raw_path": str(raw_path.relative_to(protocol.REPO_ROOT)),
                "raw_sha256": protocol.file_sha256(raw_path) if raw_path.is_file() else None,
                "record_type": "native_trajectory_replication_ada_v2_position_receipt",
                "returncode": returncode,
                "row_id": row["row_id"],
                "schema_version": 1,
            }
            exclusive_json(position_path, position)
            print(f"[{row['global_position']}/{protocol.RAW_RECORDS}] {row['label']} {row['distribution']} -> {returncode}", flush=True)
            if returncode or not raw_path.is_file() or gpu_postflight_error is not None:
                raise protocol.ProtocolError(f"fresh timing child failed at position {row['global_position']}")
            analyze.validate_timing_record(
                contract, row, protocol.read_json(raw_path),
                execution_binding=execution_binding,
                artifact_binding=artifact_cell["binding"], admitted_build=artifact_cell["build"],
            )
            previous_completed = analyze.validate_position_receipt(
                row, protocol.read_json(raw_path), position, raw_path,
                previous_completed,
            )
        gpu_after = _gpu0(contract, require_idle=True)
        raw_hashes = {
            protocol.raw_filename(row): protocol.file_sha256(RESULTS / "raw" / protocol.raw_filename(row))
            for row in manifest["rows"]
        }
        position_hashes = {
            protocol.position_receipt_filename(row): protocol.file_sha256(RESULTS / "position_receipts" / protocol.position_receipt_filename(row))
            for row in manifest["rows"]
        }
        exclusive_json(
            RESULTS / "run_status.json",
            {
                "campaign_id": protocol.CAMPAIGN_ID,
                "artifact_admission_closure_sha256": artifact_state["artifact_admission_closure_sha256"],
                "artifact_admission_manifest_sha256": artifact_state["artifact_admission_manifest_sha256"],
                "complete": True,
                "expected_position_receipts": protocol.RAW_RECORDS,
                "expected_raw_records": protocol.RAW_RECORDS,
                "gpu_after": gpu_after,
                "launch_receipt_sha256": protocol.file_sha256(launch_path),
                "observed_position_receipts": len(position_hashes),
                "observed_raw_records": len(raw_hashes),
                "position_receipt_bundle_sha256": protocol.canonical_sha256(position_hashes),
                "raw_record_bundle_sha256": protocol.canonical_sha256(raw_hashes),
                "record_type": "native_trajectory_replication_ada_v2_run_status",
                "schema_version": 1,
            },
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare", "freeze-material", "provenance", "freeze-execution",
        "ready", "admit-artifacts", "execute",
    ):
        sub.add_parser(name)
    one = sub.add_parser("time-one")
    one.add_argument("--row-id", required=True)
    one.add_argument("--output", type=Path, required=True)
    for name in ("artifact-build-one", "artifact-verify-one"):
        child = sub.add_parser(name)
        child.add_argument("--cell-id", required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--launch-receipt", type=Path, required=True)
        child.add_argument("--launch-receipt-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            contract, manifest = prepare()
            print(f"prepared materials={len(contract['materials']['selected_prefixes'])} records={len(manifest['rows'])}")
        elif args.command == "freeze-material":
            print(f"material_lock_sha256={freeze_material()['lock_sha256']}")
        elif args.command == "provenance":
            provenance()
            print(f"wrote {PROVENANCE_PATH}")
        elif args.command == "freeze-execution":
            print(f"execution_lock_sha256={freeze_execution()['lock_sha256']}")
        elif args.command == "ready":
            print(f"ready commit={ready(require_artifacts=True)['git_commit']}")
        elif args.command == "admit-artifacts":
            return admit_artifacts()
        elif args.command == "execute":
            return execute()
        elif args.command == "time-one":
            return time_one(args.row_id, args.output)
        else:
            return artifact_one(
                args.command, args.cell_id, args.output,
                args.launch_receipt, args.launch_receipt_sha256,
            )
        return 0
    except (OSError, json.JSONDecodeError, protocol.ProtocolError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
