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
CONTRACT_PATH = HERE / "contract.json"
MANIFEST_PATH = HERE / "manifest.json"
MATERIAL_LOCK_PATH = HERE / "material_lock.json"
PROVENANCE_PATH = HERE / "prelaunch_provenance.json"
EXECUTION_LOCK_PATH = HERE / "execution_lock.json"
RESULTS = HERE / "results"
GLOBAL_GPU0_LOCK = Path("/tmp") / f"multikernelbench-{protocol.GPU0_UUID}-timing.lock"
RECORD_TIMEOUT_S = 900
SOURCE_NAMES = (
    ".gitignore", "README.md", "__init__.py", "protocol.py", "runner.py",
    "analyze.py", "test_protocol.py", "contract.json", "manifest.json",
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
    value = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "contract_sha256": protocol.file_sha256(CONTRACT_PATH),
        "expected_raw_records": protocol.RAW_RECORDS,
        "instrument_evidence_index_sha256": contract["materials"]["instrument_evidence_index_sha256"],
        "instrument_launch_lock_sha256": contract["materials"]["instrument_launch_lock_sha256"],
        "instrument_source_bundle_sha256": contract["materials"]["instrument_source_bundle_sha256"],
        "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "materials_sha256": contract["materials_sha256"],
        "record_type": "native_trajectory_replication_ada_v1_material_lock",
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


def _gpu0(contract: dict[str, Any], *, require_idle: bool) -> dict[str, str]:
    snapshot = protocol.core.gpu_snapshot(0)
    expected = {
        "compute_cap": contract["hardware"]["compute_capability"],
        "index": "0",
        "name": contract["hardware"]["gpu_name"],
        "uuid": contract["hardware"]["gpu_uuid"],
    }
    if any(snapshot.get(key) != value for key, value in expected.items()):
        raise protocol.ProtocolError("physical GPU 0 identity differs from the frozen contract")
    if not isinstance(snapshot.get("driver_version"), str) or not snapshot["driver_version"]:
        raise protocol.ProtocolError("physical GPU 0 driver version is missing")
    if require_idle and _compute_pids():
        raise protocol.ProtocolError("physical GPU 0 is occupied")
    return {
        key: snapshot[key]
        for key in ("index", "uuid", "name", "driver_version", "compute_cap")
    }


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
        "record_type": "native_trajectory_replication_ada_v1_prelaunch_provenance",
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
        or value.get("record_type") != "native_trajectory_replication_ada_v1_prelaunch_provenance"
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
    value = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "expected_position_receipts": protocol.RAW_RECORDS,
        "expected_raw_records": protocol.RAW_RECORDS,
        "gpu0_uuid": contract["hardware"]["gpu_uuid"],
        "manifest_plan_sha256": manifest["plan_sha256"],
        "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "material_lock_sha256": protocol.file_sha256(MATERIAL_LOCK_PATH),
        "prelaunch_provenance_sha256": protocol.file_sha256(PROVENANCE_PATH),
        "record_timeout_s": RECORD_TIMEOUT_S,
        "record_type": "native_trajectory_replication_ada_v1_execution_lock",
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


def ready(*, require_idle: bool = True) -> dict[str, Any]:
    contract, manifest, execution_lock, receipt = validate_execution_frozen()
    paths = list(source_map()) + [
        str(path.relative_to(protocol.REPO_ROOT))
        for path in (MATERIAL_LOCK_PATH, PROVENANCE_PATH, EXECUTION_LOCK_PATH)
    ]
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
    }


def _manifest_row(manifest: dict[str, Any], row_id: str) -> dict[str, Any]:
    rows = [row for row in manifest["rows"] if row["row_id"] == row_id]
    if len(rows) != 1:
        raise protocol.ProtocolError("unknown or duplicate manifest row")
    return rows[0]


def time_one(row_id: str, output: Path) -> int:
    _validate_inherited_gpu0_lock()
    contract, manifest, execution_lock, _receipt = validate_execution_frozen(rehash_materials=False)
    row = _manifest_row(manifest, row_id)
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
        "implementation_sha256_expected": row["implementation_sha256"],
        "label": row["label"],
        "manifest_row_sha256": protocol.canonical_sha256(row),
        "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "physical_gpu": 0,
        "process_pid": os.getpid(),
        "record_kind": row["record_kind"],
        "record_type": "native_trajectory_replication_ada_v1_timing_record",
        "row_id": row_id,
        "schema_version": 1,
        "source_bundle_sha256": execution_lock["source_bundle_sha256"],
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

        cells = {cell["cell_id"]: cell for cell in protocol.core.load_cells(require_resolved=True)}
        cell = cells[row["cell_id"]]
        if protocol.core.canonical_sha256(cell) != record["cell_sha256_expected"]:
            raise protocol.ProtocolError("built cell/config differs from admitted material")
        built = build(cell)
        implementation = built.metadata.get("implementation_sha256")
        if implementation != row["implementation_sha256"] or built.metadata.get("n_kernels") != 2:
            raise protocol.ProtocolError("built implementation differs from admitted two-kernel material")
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
                    "implementation_sha256": implementation,
                    "n_kernels": built.metadata["n_kernels"],
                },
                "cell_sha256": protocol.core.canonical_sha256(cell),
                "compile_s": float(built.compile_s),
                "full_median_ms": statistics.median(numeric),
                "implementation_sha256": implementation,
                "live_correctness": live_gate,
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
    with gpu0_lock() as gpu_lock:
        state = ready(require_idle=True)
        contract, manifest = state["contract"], state["manifest"]
        (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
        (RESULTS / "position_receipts").mkdir(parents=True, exist_ok=True)
        launch_path = RESULTS / "launch_receipt.json"
        launch_contract = {
            "campaign_id": protocol.CAMPAIGN_ID,
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
                "record_type": "native_trajectory_replication_ada_v1_launch_receipt",
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
                "TORCH_EXTENSIONS_DIR": str(HERE / ".cache" / "gpu0"),
            }
        )
        env.setdefault("MAX_JOBS", "4")
        from ako_runs.controlled_followup.native_trajectory_replication_ada_v1 import analyze

        execution_binding = {
            "execution_lock_sha256": protocol.file_sha256(EXECUTION_LOCK_PATH),
            "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
            "source_bundle_sha256": state["execution_lock"]["source_bundle_sha256"],
            "toolchain": state["toolchain"],
        }
        previous_completed = 0

        for row in manifest["rows"]:
            raw_path = RESULTS / "raw" / protocol.raw_filename(row)
            position_path = RESULTS / "position_receipts" / protocol.position_receipt_filename(row)
            if raw_path.exists() or position_path.exists():
                if not raw_path.is_file() or not position_path.is_file():
                    raise protocol.ProtocolError("partial immutable position evidence exists")
                record = protocol.read_json(raw_path)
                analyze.validate_timing_record(contract, row, record, execution_binding=execution_binding)
                position = protocol.read_json(position_path)
                previous_completed = analyze.validate_position_receipt(
                    row, record, position, raw_path, previous_completed,
                )
                continue
            command = [
                sys.executable, str(HERE / "runner.py"), "time-one",
                "--row-id", row["row_id"], "--output", str(raw_path),
            ]
            launched = time.time_ns()
            returncode = _wait_child(command, env, gpu_lock.fileno())
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
                "record_type": "native_trajectory_replication_ada_v1_position_receipt",
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
                "complete": True,
                "expected_position_receipts": protocol.RAW_RECORDS,
                "expected_raw_records": protocol.RAW_RECORDS,
                "gpu_after": gpu_after,
                "launch_receipt_sha256": protocol.file_sha256(launch_path),
                "observed_position_receipts": len(position_hashes),
                "observed_raw_records": len(raw_hashes),
                "position_receipt_bundle_sha256": protocol.canonical_sha256(position_hashes),
                "raw_record_bundle_sha256": protocol.canonical_sha256(raw_hashes),
                "record_type": "native_trajectory_replication_ada_v1_run_status",
                "schema_version": 1,
            },
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "freeze-material", "provenance", "freeze-execution", "ready", "execute"):
        sub.add_parser(name)
    one = sub.add_parser("time-one")
    one.add_argument("--row-id", required=True)
    one.add_argument("--output", type=Path, required=True)
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
            print(f"ready commit={ready()['git_commit']}")
        elif args.command == "execute":
            return execute()
        else:
            return time_one(args.row_id, args.output)
        return 0
    except (OSError, json.JSONDecodeError, protocol.ProtocolError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
