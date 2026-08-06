#!/usr/bin/env python3
"""Prepare, freeze, and execute the non-controlling Ada C2 pilot."""
from __future__ import annotations

import argparse
import fcntl
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import secrets
import select
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import protocol
except ImportError:  # direct script execution
    import protocol


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contract.json"
MANIFEST_PATH = HERE / "manifest.json"
LOCK_PATH = HERE / "execution_lock.json"
PROVENANCE_PATH = HERE / "prelaunch_provenance.json"
RESULTS = HERE / "results"
AUTH_FD_ENV = "DECISION_COMPLEXITY_PARENT_AUTH_FD"
GPU_LOCK_FDS_ENV = "DECISION_COMPLEXITY_GPU_LOCK_FDS"
WAVE_READY_FD_ENV = "DECISION_COMPLEXITY_WAVE_READY_FD"
WAVE_RELEASE_FD_ENV = "DECISION_COMPLEXITY_WAVE_RELEASE_FD"
CUDA_HOME = Path("/usr/local/cuda-13.1")
SOURCE_NAMES = (
    ".gitignore", "README.md", "__init__.py", "protocol.py", "runner.py",
    "analyze.py", "test_protocol.py", "test_execution.py", "contract.json", "manifest.json",
)


def toolchain_fingerprint() -> dict[str, Any]:
    packages = {}
    for distribution, module_name in (
        ("torch", "torch"),
        ("triton", "triton"),
        ("tilelang", "tilelang"),
    ):
        distribution_version = importlib.metadata.version(distribution)
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise protocol.ProtocolError(f"toolchain module is unavailable: {module_name}")
        packages[distribution] = {
            "distribution_version": distribution_version,
            "module_path": str(Path(spec.origin).resolve()),
        }
        if distribution == "torch":
            module = importlib.import_module(module_name)
            packages[distribution]["runtime_version"] = str(module.__version__)
            packages[distribution]["compiled_cuda_version"] = str(module.version.cuda)
        else:
            packages[distribution]["runtime_version"] = distribution_version
            packages[distribution]["runtime_version_source"] = (
                "installed_distribution_metadata_without_import_to_preserve_cold_cache"
            )
    old_cuda_home = os.environ.get("CUDA_HOME")
    try:
        os.environ["CUDA_HOME"] = str(CUDA_HOME)
        nvcc = protocol.core.nvcc_fingerprint()
    finally:
        if old_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = old_cuda_home
    if nvcc.get("path") != str(CUDA_HOME / "bin/nvcc") or nvcc.get("returncode") != 0:
        raise protocol.ProtocolError("frozen CUDA compiler is unavailable")
    return {
        "cuda": {"home": str(CUDA_HOME), "nvcc": nvcc},
        "packages": packages,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


def gpu_lock_paths() -> list[Path]:
    return [Path(value) for value in protocol.GPU_LOCK_PATHS]


@contextmanager
def all_gpu_locks():
    handles = []
    try:
        for path in gpu_lock_paths():
            handle = path.open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise protocol.ProtocolError(f"host GPU timing lock is held: {path}") from None
            handles.append(handle)
        yield tuple(handle.fileno() for handle in handles)
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _validate_inherited_gpu_locks() -> tuple[int, ...]:
    try:
        fds = tuple(int(value) for value in os.environ.get(GPU_LOCK_FDS_ENV, "").split(","))
    except ValueError as exc:
        raise protocol.ProtocolError("inherited GPU lock FD chain is malformed") from exc
    expected = [str(path.resolve()) for path in gpu_lock_paths()]
    observed = []
    if len(fds) != len(expected) or len(set(fds)) != len(fds):
        raise protocol.ProtocolError("inherited GPU lock FD chain is incomplete")
    for fd in fds:
        if fd <= 2:
            raise protocol.ProtocolError("inherited GPU lock FD chain is invalid")
        try:
            os.fstat(fd)
            observed.append(str(Path(os.readlink(f"/proc/self/fd/{fd}")).resolve()))
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            raise protocol.ProtocolError("inherited GPU lock FD is not live") from exc
    if observed != expected:
        raise protocol.ProtocolError("inherited GPU lock FD paths are foreign")
    return fds


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


def _prepared() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = protocol.make_contract()
    return contract, protocol.make_manifest(contract)


def prepare() -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _prepared()
    for path, value in zip((CONTRACT_PATH, MANIFEST_PATH), expected):
        if path.exists():
            if protocol.read_json(path) != value:
                raise protocol.ProtocolError(f"prepared artifact is stale: {path}")
        else:
            exclusive_json(path, value)
    return expected


def source_map() -> dict[str, str]:
    paths = [HERE / name for name in SOURCE_NAMES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise protocol.ProtocolError(f"execution source closure is incomplete: {missing}")
    return {
        str(path.relative_to(protocol.REPO_ROOT)): protocol.file_sha256(path)
        for path in paths
    }


def make_lock(
    contract: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if contract is None or manifest is None:
        contract, manifest = prepare()
    sources = source_map()
    toolchain = toolchain_fingerprint()
    value = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "contract_sha256": protocol.file_sha256(CONTRACT_PATH),
        "execution_control": {
            "authorization": "one_use_parent_pipe_plus_four_child_wave_barrier_v2",
            "gpu_lock_paths": [str(path) for path in gpu_lock_paths()],
            "parent_receipts": "pid_timestamp_hash_idle_and_common_release_witness_v2",
            "schedule": contract["execution_schedule"],
        },
        "hardware": contract["hardware"],
        "instrument_evidence_index_sha256": contract["materials"]["instrument_evidence_index_sha256"],
        "instrument_launch_lock_sha256": contract["materials"]["instrument_launch_lock_sha256"],
        "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "materials_sha256": contract["materials_sha256"],
        "record_type": "decision_complexity_ada_v1_execution_lock",
        "requested_trajectories": 24,
        "schema_version": 1,
        "source_sha256": sources,
        "source_bundle_sha256": protocol.canonical_sha256(sources),
        "timing": contract["timing"],
        "toolchain": toolchain,
        "toolchain_sha256": protocol.canonical_sha256(toolchain),
    }
    value["lock_sha256"] = protocol.canonical_sha256(value)
    return value


def freeze() -> dict[str, Any]:
    expected = make_lock()
    if LOCK_PATH.exists():
        if protocol.read_json(LOCK_PATH) != expected:
            raise protocol.ProtocolError("execution lock is stale")
    else:
        exclusive_json(LOCK_PATH, expected)
    return expected


def validate_frozen(*, rehash_materials: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not all(path.is_file() for path in (CONTRACT_PATH, MANIFEST_PATH, LOCK_PATH)):
        raise protocol.ProtocolError("prepare/freeze artifacts are incomplete")
    contract, manifest = protocol.read_json(CONTRACT_PATH), protocol.read_json(MANIFEST_PATH)
    if rehash_materials:
        protocol.validate_contract(contract)
        protocol.validate_manifest(contract, manifest)
    else:
        if manifest.get("contract_sha256") != protocol.canonical_sha256(contract):
            raise protocol.ProtocolError("manifest lost contract binding")
    lock = protocol.read_json(LOCK_PATH)
    if lock != make_lock(contract, manifest):
        raise protocol.ProtocolError("execution lock no longer matches current sources")
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
        raise protocol.ProtocolError("configured upstream branch is missing or ambiguous at the live remote")
    fields = rows[0].split()
    if len(fields) != 2 or fields[1] != remote_ref:
        raise protocol.ProtocolError("live upstream response is malformed")
    return remote, remote_ref, fields[0]


def _live_gpu(gpu: int, contract: dict[str, Any], *, require_idle: bool) -> dict[str, Any]:
    snapshot = protocol.core.gpu_snapshot(gpu)
    expected = {
        "index": str(gpu),
        "uuid": contract["hardware"]["gpu_uuids"][gpu],
        "name": contract["hardware"]["name"],
        "compute_cap": contract["hardware"]["compute_capability"],
    }
    for key, value in expected.items():
        if snapshot.get(key) != value:
            raise protocol.ProtocolError(f"GPU {gpu} {key} mismatch")
    if not isinstance(snapshot.get("driver_version"), str) or not snapshot["driver_version"]:
        raise protocol.ProtocolError(f"GPU {gpu} driver version is missing")
    if require_idle:
        _gpu_occupancy(gpu)
    return {
        key: snapshot[key]
        for key in ("index", "uuid", "name", "driver_version", "compute_cap")
    }


def _gpu_occupancy(gpu: int) -> dict[str, Any]:
    started = time.time_ns()
    query = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    completed = time.time_ns()
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if query.returncode or rows:
        raise protocol.ProtocolError(f"GPU {gpu} is occupied or occupancy query failed")
    return {
        "compute_apps": [],
        "query_completed_unix_ns": completed,
        "query_started_unix_ns": started,
        "returncode": query.returncode,
    }


def _idle_gpu_evidence(gpu: int, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "gpu": _live_gpu(gpu, contract, require_idle=False),
        "occupancy": _gpu_occupancy(gpu),
    }


def validate_idle_gpu_evidence(
    evidence: dict[str, Any], gpu: int, contract: dict[str, Any]
) -> None:
    identity = evidence.get("gpu") if isinstance(evidence, dict) else None
    occupancy = evidence.get("occupancy") if isinstance(evidence, dict) else None
    expected = {
        "compute_cap": contract["hardware"]["compute_capability"],
        "index": str(gpu),
        "name": contract["hardware"]["name"],
        "uuid": contract["hardware"]["gpu_uuids"][gpu],
    }
    started = occupancy.get("query_started_unix_ns") if isinstance(occupancy, dict) else None
    completed = occupancy.get("query_completed_unix_ns") if isinstance(occupancy, dict) else None
    if (
        not isinstance(identity, dict)
        or set(identity) != {*expected, "driver_version"}
        or any(identity.get(key) != value for key, value in expected.items())
        or not isinstance(identity.get("driver_version"), str)
        or not identity["driver_version"]
        or not isinstance(occupancy, dict)
        or occupancy.get("compute_apps") != []
        or occupancy.get("returncode") != 0
        or not isinstance(started, int)
        or not isinstance(completed, int)
        or started <= 0
        or completed < started
    ):
        raise protocol.ProtocolError(f"GPU {gpu} idle evidence is invalid")


def _authorized_popen(
    command: list[str], env: dict[str, str], authorization: dict[str, Any],
    inherited_lock_fds: tuple[int, ...],
    *,
    wave_fds: tuple[int, int] | None = None,
) -> tuple[subprocess.Popen, dict[str, Any]]:
    if len(inherited_lock_fds) != len(protocol.GPU_UUIDS):
        raise protocol.ProtocolError("child launch lacks the four-GPU lock FD chain")
    extra_fds: tuple[int, ...] = ()
    if wave_fds is not None:
        ready_fd, release_fd = wave_fds
        if ready_fd <= 2 or release_fd <= 2 or ready_fd == release_fd:
            raise protocol.ProtocolError("wave barrier FD pair is invalid")
        extra_fds = wave_fds
    read_fd, write_fd = os.pipe()
    child_env = env.copy()
    child_env[AUTH_FD_ENV] = str(read_fd)
    child_env[GPU_LOCK_FDS_ENV] = ",".join(map(str, inherited_lock_fds))
    if wave_fds is not None:
        child_env[WAVE_READY_FD_ENV] = str(ready_fd)
        child_env[WAVE_RELEASE_FD_ENV] = str(release_fd)
    requested = time.time_ns()
    try:
        process = subprocess.Popen(
            command, cwd=protocol.REPO_ROOT, env=child_env,
            pass_fds=(read_fd, *inherited_lock_fds, *extra_fds),
            start_new_session=True,
        )
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    os.close(read_fd)
    payload = {
        **authorization,
        "authorization_nonce": secrets.token_hex(32),
        "authorized_unix_ns": time.time_ns(),
        "campaign_id": protocol.CAMPAIGN_ID,
        "child_pid": process.pid,
        "execution_lock_sha256": protocol.file_sha256(LOCK_PATH),
        "gpu_lock_paths": [str(path) for path in gpu_lock_paths()],
        "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "parent_pid": os.getpid(),
        "spawn_requested_unix_ns": requested,
    }
    data = protocol.canonical_bytes(payload)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(write_fd, data[offset:])
    except OSError as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise protocol.ProtocolError("child exited before consuming parent authorization") from exc
    finally:
        os.close(write_fd)
    return process, payload


def _consume_parent_authorization(
    scope: str, expected: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    value = os.environ.pop(AUTH_FD_ENV, None)
    try:
        fd = int(value or "")
    except ValueError as exc:
        raise protocol.ProtocolError("parent authorization FD is missing") from exc
    if fd <= 2:
        raise protocol.ProtocolError("parent authorization FD is invalid")
    chunks = []
    try:
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(map(len, chunks)) > 65536:
                raise protocol.ProtocolError("parent authorization payload is oversized")
    except OSError as exc:
        raise protocol.ProtocolError("parent authorization FD is unreadable") from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    raw = b"".join(chunks)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise protocol.ProtocolError("parent authorization payload is malformed") from exc
    fixed = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "child_pid": os.getpid(),
        "execution_lock_sha256": protocol.file_sha256(LOCK_PATH),
        "gpu_lock_paths": [str(path) for path in gpu_lock_paths()],
        "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "parent_pid": os.getppid(),
        "scope": scope,
    }
    if (
        not isinstance(payload, dict)
        or raw != protocol.canonical_bytes(payload)
        or any(payload.get(key) != item for key, item in {**fixed, **expected}.items())
        or not isinstance(payload.get("authorization_nonce"), str)
        or len(payload["authorization_nonce"]) != 64
        or not isinstance(payload.get("spawn_requested_unix_ns"), int)
        or not isinstance(payload.get("authorized_unix_ns"), int)
        or payload["authorized_unix_ns"] < payload["spawn_requested_unix_ns"]
    ):
        raise protocol.ProtocolError("parent authorization is stale or foreign")
    return payload, protocol.canonical_sha256(payload)


def _close_fd(fd: int | None) -> None:
    if isinstance(fd, int):
        try:
            os.close(fd)
        except OSError:
            pass


def _read_canonical_pipe(fd: int, label: str) -> dict[str, Any]:
    chunks = []
    try:
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(map(len, chunks)) > 65536:
                raise protocol.ProtocolError(f"{label} payload is oversized")
    except OSError as exc:
        raise protocol.ProtocolError(f"{label} pipe is unreadable") from exc
    finally:
        _close_fd(fd)
    raw = b"".join(chunks)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise protocol.ProtocolError(f"{label} payload is malformed") from exc
    if not isinstance(value, dict) or raw != protocol.canonical_bytes(value):
        raise protocol.ProtocolError(f"{label} payload is not canonical")
    return value


def _write_canonical_pipe(fd: int, value: dict[str, Any], label: str) -> None:
    data = protocol.canonical_bytes(value)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
    except OSError as exc:
        raise protocol.ProtocolError(f"{label} pipe closed early") from exc
    finally:
        _close_fd(fd)


def _release_wave(
    children: list[dict[str, Any]], position: int, timeout_s: int,
) -> dict[str, Any]:
    if (
        len(children) != protocol.REPLICATES
        or {child["row"]["block_position"] for child in children} != {position}
    ):
        raise protocol.ProtocolError("wave barrier has incomplete membership")
    pending = {child["ready_fd"]: child for child in children}
    ready_receipts: dict[str, dict[str, Any]] = {}
    deadline = time.monotonic() + timeout_s
    try:
        while pending:
            remaining = deadline - time.monotonic()
            readable, _, _ = select.select(list(pending), [], [], max(0.0, remaining))
            if not readable:
                raise protocol.ProtocolError("wave barrier timed out before all children were ready")
            for fd in readable:
                child = pending.pop(fd)
                child["ready_fd"] = None
                receipt = _read_canonical_pipe(fd, "wave ready")
                row, authorization = child["row"], child["authorization"]
                expected = {
                    "authorization_sha256": protocol.canonical_sha256(authorization),
                    "campaign_id": protocol.CAMPAIGN_ID,
                    "child_pid": child["process"].pid,
                    "trajectory_id": row["trajectory_id"],
                    "wave_position": position,
                }
                ready_unix_ns = receipt.get("ready_unix_ns")
                if (
                    set(receipt) != {*expected, "ready_unix_ns"}
                    or any(receipt.get(key) != value for key, value in expected.items())
                    or not isinstance(ready_unix_ns, int)
                    or ready_unix_ns <= authorization["authorized_unix_ns"]
                    or row["trajectory_id"] in ready_receipts
                ):
                    raise protocol.ProtocolError("wave child ready receipt is stale or foreign")
                ready_receipts[row["trajectory_id"]] = receipt
        released = time.time_ns()
        if released <= max(receipt["ready_unix_ns"] for receipt in ready_receipts.values()):
            raise protocol.ProtocolError("wave release clock did not follow all ready receipts")
        witness = {
            "campaign_id": protocol.CAMPAIGN_ID,
            "ready_receipts": dict(sorted(ready_receipts.items())),
            "release_nonce": secrets.token_hex(32),
            "release_unix_ns": released,
            "wave_position": position,
        }
        for child in children:
            fd = child["release_fd"]
            child["release_fd"] = None
            _write_canonical_pipe(fd, witness, "wave release")
        return witness
    finally:
        for child in children:
            _close_fd(child.get("ready_fd"))
            _close_fd(child.get("release_fd"))
            child["ready_fd"] = None
            child["release_fd"] = None


def _enter_wave_barrier(
    row: dict[str, Any], authorization_sha256: str,
    expected_trajectory_ids: set[str], trajectory_started_unix_ns: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    try:
        ready_fd = int(os.environ.pop(WAVE_READY_FD_ENV, ""))
        release_fd = int(os.environ.pop(WAVE_RELEASE_FD_ENV, ""))
    except ValueError as exc:
        raise protocol.ProtocolError("trajectory lacks its inherited wave barrier") from exc
    if ready_fd <= 2 or release_fd <= 2 or ready_fd == release_fd:
        _close_fd(ready_fd)
        _close_fd(release_fd)
        raise protocol.ProtocolError("trajectory inherited an invalid wave barrier")
    ready_unix_ns = time.time_ns()
    ready = {
        "authorization_sha256": authorization_sha256,
        "campaign_id": protocol.CAMPAIGN_ID,
        "child_pid": os.getpid(),
        "ready_unix_ns": ready_unix_ns,
        "trajectory_id": row["trajectory_id"],
        "wave_position": row["block_position"],
    }
    try:
        _write_canonical_pipe(ready_fd, ready, "wave ready")
        witness = _read_canonical_pipe(release_fd, "wave release")
    except BaseException:
        _close_fd(ready_fd)
        _close_fd(release_fd)
        raise
    received = time.time_ns()
    release = witness.get("release_unix_ns")
    receipts = witness.get("ready_receipts")
    nonce = witness.get("release_nonce")
    if (
        set(witness) != {
            "campaign_id", "ready_receipts", "release_nonce",
            "release_unix_ns", "wave_position",
        }
        or witness.get("campaign_id") != protocol.CAMPAIGN_ID
        or witness.get("wave_position") != row["block_position"]
        or not isinstance(receipts, dict)
        or set(receipts) != expected_trajectory_ids
        or receipts.get(row["trajectory_id"]) != ready
        or not isinstance(nonce, str)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
        or not isinstance(release, int)
        or not trajectory_started_unix_ns < ready_unix_ns < release <= received
    ):
        raise protocol.ProtocolError("wave release witness is stale or foreign")
    return ready, witness, received


def make_provenance() -> dict[str, Any]:
    contract, _manifest, lock = validate_frozen()
    paths = list(source_map()) + [str(LOCK_PATH.relative_to(protocol.REPO_ROOT))]
    if _git("status", "--porcelain", "--", *paths):
        raise protocol.ProtocolError("pilot source closure is not committed")
    _git("ls-files", "--error-unmatch", "--", *paths)
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    head, upstream_head = _git("rev-parse", "HEAD"), _git("rev-parse", upstream)
    remote, remote_ref, live_remote_commit = _live_upstream_commit(upstream)
    if head != upstream_head or head != live_remote_commit:
        raise protocol.ProtocolError("pilot commit is not the cached and live configured upstream head")
    instrument = subprocess.run(
        [sys.executable, str(protocol.CROSSED / "validate.py"), "--stage", "campaign", "--remote-ready"],
        cwd=protocol.REPO_ROOT, capture_output=True, text=True,
    )
    if instrument.returncode:
        raise protocol.ProtocolError(instrument.stderr.strip() or instrument.stdout.strip())
    with all_gpu_locks():
        gpu_preflights = [_idle_gpu_evidence(gpu, contract) for gpu in range(4)]
    toolchain = toolchain_fingerprint()
    return {
        "campaign_id": protocol.CAMPAIGN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_lock_sha256": protocol.file_sha256(LOCK_PATH),
        "git_commit": head,
        "git_upstream": upstream,
        "git_upstream_commit": upstream_head,
        "live_remote_commit": live_remote_commit,
        "live_remote_name": remote,
        "live_remote_ref": remote_ref,
        "gpu_lock_paths": [str(path) for path in gpu_lock_paths()],
        "gpu_preflights": gpu_preflights,
        "gpus": [item["gpu"] for item in gpu_preflights],
        "host": platform.node(),
        "toolchain": toolchain,
        "toolchain_sha256": protocol.canonical_sha256(toolchain),
        "record_type": "decision_complexity_ada_v1_prelaunch_provenance",
        "remote_push_verified": True,
        "schema_version": 1,
        "source_bundle_sha256": lock["source_bundle_sha256"],
    }


def provenance() -> None:
    exclusive_json(PROVENANCE_PATH, make_provenance())


def _ready_locked() -> dict[str, Any]:
    contract, manifest, lock = validate_frozen()
    if not PROVENANCE_PATH.is_file():
        raise protocol.ProtocolError("prelaunch provenance is missing; commit and push the frozen pilot first")
    receipt = protocol.read_json(PROVENANCE_PATH)
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    remote, remote_ref, live_remote_commit = _live_upstream_commit(upstream)
    if (
        receipt.get("campaign_id") != protocol.CAMPAIGN_ID
        or receipt.get("execution_lock_sha256") != protocol.file_sha256(LOCK_PATH)
        or receipt.get("source_bundle_sha256") != lock["source_bundle_sha256"]
        or receipt.get("remote_push_verified") is not True
        or receipt.get("git_commit") != _git("rev-parse", "HEAD")
        or receipt.get("git_upstream") != upstream
        or receipt.get("git_upstream_commit") != _git("rev-parse", upstream)
        or receipt.get("live_remote_name") != remote
        or receipt.get("live_remote_ref") != remote_ref
        or receipt.get("live_remote_commit") != live_remote_commit
        or receipt.get("git_commit") != live_remote_commit
        or receipt.get("gpu_lock_paths") != [str(path) for path in gpu_lock_paths()]
        or receipt.get("record_type") != "decision_complexity_ada_v1_prelaunch_provenance"
        or receipt.get("schema_version") != 1
        or receipt.get("toolchain") != lock.get("toolchain")
        or receipt.get("toolchain_sha256") != lock.get("toolchain_sha256")
        or toolchain_fingerprint() != lock.get("toolchain")
    ):
        raise protocol.ProtocolError("prelaunch provenance is stale or foreign")
    retained_preflights = receipt.get("gpu_preflights")
    if not isinstance(retained_preflights, list) or len(retained_preflights) != 4:
        raise protocol.ProtocolError("prelaunch provenance lacks four idle-GPU receipts")
    for gpu, evidence in enumerate(retained_preflights):
        validate_idle_gpu_evidence(evidence, gpu, contract)
    gpu_preflights = [_idle_gpu_evidence(gpu, contract) for gpu in range(4)]
    gpus = [item["gpu"] for item in gpu_preflights]
    if receipt.get("gpus") != gpus:
        raise protocol.ProtocolError("live four-GPU identity differs from provenance")
    return {
        "contract": contract, "manifest": manifest, "lock": lock,
        "prelaunch_gpu_evidence": gpu_preflights, "provenance": receipt,
    }


def ready() -> dict[str, Any]:
    with all_gpu_locks():
        return _ready_locked()


def _row(manifest: dict[str, Any], trajectory_id: str) -> dict[str, Any]:
    rows = [row for row in manifest["rows"] if row["trajectory_id"] == trajectory_id]
    if len(rows) != 1:
        raise protocol.ProtocolError("unknown or duplicate trajectory")
    return rows[0]


def _attempt_record_base(
    row: dict[str, Any], candidate: str, attempt_index: int, gpu: int,
    authorization: dict[str, Any], authorization_sha256: str,
) -> dict[str, Any]:
    return {
        "attempt_index": attempt_index,
        "authorization_sha256": authorization_sha256,
        "cache_isolation": protocol.attempt_cache_receipt(row, candidate, attempt_index),
        "campaign_id": protocol.CAMPAIGN_ID,
        "candidate_cell_id": candidate,
        "child_pid": authorization["child_pid"],
        "execution_lock_sha256": protocol.file_sha256(LOCK_PATH),
        "gpu_slot": gpu,
        "gpu_uuid": row["gpu_uuid"],
        "gpu_lock_paths": [str(path) for path in gpu_lock_paths()],
        "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "parent_pid": authorization["parent_pid"],
        "toolchain_sha256": protocol.read_json(LOCK_PATH)["toolchain_sha256"],
        "trajectory_id": row["trajectory_id"],
    }


def _attempt_environment(
    base: dict[str, str], row: dict[str, Any], candidate: str,
    attempt_index: int, gpu: int, cache_root: Path,
) -> dict[str, str]:
    env = base.copy()
    cache = protocol.attempt_cache_receipt(row, candidate, attempt_index)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_HOME": str(CUDA_HOME),
            "PATH": str(CUDA_HOME / "bin") + ":" + env.get("PATH", ""),
            "PHASE2_TL_CACHE": cache["phase2_tl_cache"],
            "DECISION_COMPLEXITY_CACHE_SCOPE_SHA256": cache["scope_sha256"],
            "TMPDIR": str(cache_root),
        }
    )
    env.update(
        {
            name: str(cache_root / relative)
            for name, relative in cache["directory_environment"].items()
        }
    )
    return env


def _charged_child_failure(
    row: dict[str, Any], candidate: str, attempt_index: int, gpu: int,
    physical_gpu_preflight: dict[str, Any], active_s: float,
    terminal_status: str, error: str, authorization: dict[str, Any],
    authorization_sha256: str, child_exit_code: int | None = None,
) -> dict[str, Any]:
    record = {
        **_attempt_record_base(
            row, candidate, attempt_index, gpu,
            authorization, authorization_sha256,
        ),
        "active_s": active_s,
        "error": error,
        "physical_gpu_preflight": physical_gpu_preflight,
        "record_type": "decision_complexity_ada_v1_attempt",
        "schema_version": 1,
        "terminal_status": terminal_status,
    }
    if child_exit_code is not None:
        record["child_exit_code"] = child_exit_code
    return record


def _validate_fresh_cache_environment(
    row: dict[str, Any], candidate: str, attempt_index: int
) -> None:
    cache = protocol.attempt_cache_receipt(row, candidate, attempt_index)
    paths = {
        name: Path(os.environ.get(name, ""))
        for name in cache["directory_environment"]
    }
    if (
        os.environ.get("PHASE2_TL_CACHE") != cache["phase2_tl_cache"]
        or os.environ.get("DECISION_COMPLEXITY_CACHE_SCOPE_SHA256")
        != cache["scope_sha256"]
        or Path(os.environ.get("TMPDIR", "")).resolve()
        != next(iter(paths.values())).parent.resolve()
    ):
        raise protocol.ProtocolError("attempt cache policy is missing or foreign")
    if (
        any(not str(path) or path.exists() for path in paths.values())
        or len({path.parent for path in paths.values()}) != 1
        or any(
            path.name != cache["directory_environment"][name]
            for name, path in paths.items()
        )
    ):
        raise protocol.ProtocolError("attempt cache root is not fresh and isolated")


def attempt(row: dict[str, Any], candidate_id: str, attempt_index: int, gpu: int, output: Path) -> int:
    contract, manifest, _lock = validate_frozen(rehash_materials=False)
    if _row(manifest, row["trajectory_id"]) != row or candidate_id != row["execution_contract"]["candidate_order"][attempt_index - 1]:
        raise protocol.ProtocolError("attempt is not the next frozen trajectory coordinate")
    canonical_output = RESULTS / "trajectories" / row["trajectory_id"] / "attempts" / f"attempt{attempt_index:02d}.json"
    launch_path = RESULTS / "launch_receipt.json"
    authorization, authorization_sha256 = _consume_parent_authorization(
        "attempt",
        {
            "attempt_index": attempt_index,
            "candidate_cell_id": candidate_id,
            "canonical_output": str(canonical_output.resolve()),
            "gpu_slot": gpu,
            "gpu_uuid": row["gpu_uuid"],
            "launch_receipt_sha256": protocol.file_sha256(launch_path),
            "staging_output": str(output.resolve()),
            "trajectory_id": row["trajectory_id"],
        },
    )
    _validate_inherited_gpu_locks()
    if (
        not isinstance(authorization.get("trajectory_authorization_sha256"), str)
        or len(authorization["trajectory_authorization_sha256"]) != 64
        or output.resolve() == canonical_output.resolve()
    ):
        raise protocol.ProtocolError("attempt authorization is not trajectory-scoped")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise protocol.ProtocolError("attempt CUDA_VISIBLE_DEVICES does not bind its physical GPU slot")
    _validate_fresh_cache_environment(row, candidate_id, attempt_index)
    cache_parent = Path(os.environ[next(iter(protocol.CACHE_DIRECTORIES))]).parent.resolve()
    if output.parent.resolve() != cache_parent:
        raise protocol.ProtocolError("attempt staging output is outside its authorized cache root")
    physical_gpu_preflight = _live_gpu(gpu, contract, require_idle=False)
    materials = {item["cell_id"]: item for item in contract["materials"]["candidates"]}
    expected, target_expected = materials[candidate_id], materials[protocol.TARGET_CELL_ID]
    base = _attempt_record_base(
        row, candidate_id, attempt_index, gpu,
        authorization, authorization_sha256,
    )
    started, stage = time.perf_counter(), "build"
    record: dict[str, Any] = {
        **base,
        "physical_gpu_preflight": physical_gpu_preflight,
        "record_type": "decision_complexity_ada_v1_attempt",
        "schema_version": 1,
    }
    try:
        from ako_runs.controlled_followup.fused_epilogue_crossed_v2.candidates import build
        import common
        import common2
        import runner2
        import torch

        cells = {cell["cell_id"]: cell for cell in protocol.core.load_cells(require_resolved=True)}
        candidate, target = build(cells[candidate_id]), build(cells[protocol.TARGET_CELL_ID])
        stage = "audit"
        for label, built, material in (("candidate", candidate, expected), ("target", target, target_expected)):
            observed = built.metadata.get("implementation_sha256")
            if observed != material["implementation_sha256"]:
                raise protocol.ProtocolError(f"{label} implementation hash drift")
        stage = "launch"
        x, weight, bias = common2.fused_inputs(seed=protocol.WITHHELD_SEED, dist="randn")
        x16 = x.half().contiguous()
        with torch.no_grad():
            reference = common2.fused_reference(x, weight, bias, arm="GBGS", dtype=torch.float32)
            candidate_output = candidate.run(x16, weight, bias)
            target_output = target.run(x16, weight, bias)
            torch.cuda.synchronize()
        stage = "gate"
        live_gate = {
            "candidate": common.gate_stats(reference, candidate_output.float()),
            "target": common.gate_stats(reference, target_output.float()),
        }
        if any(value.get("gate_pass") is not True for value in live_gate.values()):
            record.update({"terminal_status": "GATE_FAILED", "live_gate": live_gate})
        else:
            del reference, candidate_output, target_output
            torch.cuda.empty_cache()
            stage = "timing"
            pair_order = protocol.timing_pair_order(row, attempt_index)
            built_by_label = {"candidate": candidate, "target": target}
            times: dict[str, list[float]] = {}
            warmups: dict[str, int] = {}
            for label in pair_order:
                values, count = runner2.time_kernel3(
                    built_by_label[label].run, x16, weight, bias,
                    num_trials=protocol.TIMING_TRIALS,
                    warmup_s=protocol.WARMUP_S, flush_l2=True,
                )
                times[label] = [float(value) for value in values]
                warmups[label] = count
            candidate_tail = statistics.median(times["candidate"][protocol.TAIL_START:protocol.TAIL_STOP])
            target_tail = statistics.median(times["target"][protocol.TAIL_START:protocol.TAIL_STOP])
            record.update(
                {
                    "candidate_times_ms": times["candidate"],
                    "candidate_tail_median_ms": candidate_tail,
                    "gate_evidence_reused": {"path": expected["gate_path"], "sha256": expected["gate_sha256"]},
                    "implementation_sha256": expected["implementation_sha256"],
                    "live_gate": live_gate,
                    "measurement_order": pair_order,
                    "ratio_to_target": candidate_tail / target_tail,
                    "target_implementation_sha256": target_expected["implementation_sha256"],
                    "target_gate_evidence_reused": {
                        "path": target_expected["gate_path"],
                        "sha256": target_expected["gate_sha256"],
                    },
                    "target_tail_median_ms": target_tail,
                    "target_times_ms": times["target"],
                    "terminal_status": "GATE_PASSED",
                    "warmup_iterations": warmups,
                }
            )
    except Exception as exc:
        status = "BUILD_FAILED" if stage == "build" else "AUDIT_FAILED" if stage == "audit" else "GATE_FAILED" if stage == "gate" else "LAUNCH_FAILED"
        record.update({"error": f"{type(exc).__name__}: {exc}", "terminal_status": status, "traceback": traceback.format_exc()})
    record["active_s"] = time.perf_counter() - started
    exclusive_json(output, record)
    return 0


def trajectory(trajectory_id: str, gpu: int, launch_receipt: Path) -> int:
    if __package__:
        from . import analyze
    else:
        import analyze

    contract, manifest, lock = validate_frozen(rehash_materials=False)
    row = _row(manifest, trajectory_id)
    if row["gpu_slot"] != gpu:
        raise protocol.ProtocolError("trajectory launched on the wrong GPU slot")
    canonical_launch = RESULTS / "launch_receipt.json"
    if launch_receipt.resolve() != canonical_launch.resolve():
        raise protocol.ProtocolError("trajectory launch receipt path is not canonical")
    launch = protocol.read_json(launch_receipt)
    if (
        launch.get("campaign_id") != protocol.CAMPAIGN_ID
        or launch.get("execution_lock_sha256") != protocol.file_sha256(LOCK_PATH)
        or launch.get("manifest_sha256") != protocol.file_sha256(MANIFEST_PATH)
        or launch.get("record_type") != "decision_complexity_ada_v1_launch_receipt"
        or launch.get("requested_trajectories") != 24
        or launch.get("schema_version") != 1
        or launch.get("toolchain_sha256") != lock["toolchain_sha256"]
    ):
        raise protocol.ProtocolError("trajectory launch receipt is foreign")
    launch_sha256 = protocol.file_sha256(launch_receipt)
    trajectory_authorization, trajectory_authorization_sha256 = _consume_parent_authorization(
        "trajectory",
        {
            "gpu_slot": gpu,
            "gpu_uuid": row["gpu_uuid"],
            "launch_receipt_sha256": launch_sha256,
            "launch_sequence": row["block_position"] * protocol.REPLICATES + row["replicate"] + 1,
            "trajectory_id": trajectory_id,
            "wave_member_order": row["replicate"],
            "wave_position": row["block_position"],
        },
    )
    inherited_lock_fds = _validate_inherited_gpu_locks()
    root = RESULTS / "trajectories" / trajectory_id
    completion = root / "completion.json"
    if completion.exists():
        raise protocol.ProtocolError("authorized trajectory already has immutable completion")
    trajectory_started_unix_ns = time.time_ns()
    trajectory_gpu_preflight = _idle_gpu_evidence(gpu, contract)
    wave_ids = {
        item["trajectory_id"] for item in manifest["rows"]
        if item["block_position"] == row["block_position"]
    }
    wave_ready, wave_witness, wave_release_received_unix_ns = _enter_wave_barrier(
        row, trajectory_authorization_sha256, wave_ids,
        trajectory_started_unix_ns,
    )
    attempts = []
    previous_completed = 0
    active_attempt: list[subprocess.Popen | None] = [None]

    def stop_active_attempt(signum, _frame):
        process = active_attempt[0]
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_active_attempt)
    signal.signal(signal.SIGINT, stop_active_attempt)
    for attempt_index, candidate in enumerate(row["execution_contract"]["candidate_order"], 1):
        output = root / "attempts" / f"attempt{attempt_index:02d}.json"
        parent_path = root / "attempt_parent_receipts" / f"attempt{attempt_index:02d}.json"
        if output.exists() or parent_path.exists():
            if not output.is_file() or not parent_path.is_file():
                raise protocol.ProtocolError("partial immutable attempt evidence exists")
        else:
            gpu_preflight = _idle_gpu_evidence(gpu, contract)
            started = time.perf_counter()
            with tempfile.TemporaryDirectory(
                prefix=f"decision-complexity-gpu{gpu}-attempt{attempt_index:02d}-"
            ) as temporary:
                child_output = Path(temporary) / "attempt.json"
                command = [
                    sys.executable, str(HERE / "runner.py"), "attempt",
                    "--trajectory-id", trajectory_id, "--candidate", candidate,
                    "--attempt-index", str(attempt_index), "--gpu", str(gpu),
                    "--output", str(child_output),
                ]
                env = _attempt_environment(
                    os.environ, row, candidate, attempt_index, gpu, Path(temporary)
                )
                process, authorization = _authorized_popen(
                    command,
                    env,
                    {
                        "attempt_index": attempt_index,
                        "candidate_cell_id": candidate,
                        "canonical_output": str(output.resolve()),
                        "gpu_slot": gpu,
                        "gpu_uuid": row["gpu_uuid"],
                        "launch_receipt_sha256": launch_sha256,
                        "scope": "attempt",
                        "staging_output": str(child_output.resolve()),
                        "trajectory_authorization_sha256": trajectory_authorization_sha256,
                        "trajectory_id": trajectory_id,
                    },
                    inherited_lock_fds,
                )
                active_attempt[0] = process
                authorization_sha256 = protocol.canonical_sha256(authorization)
                try:
                    code = process.wait(timeout=contract["attempt_timeout_s"])
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    code = 124
                    record = _charged_child_failure(
                        row, candidate, attempt_index, gpu,
                        gpu_preflight["gpu"], time.perf_counter() - started,
                        "TIMEOUT", "attempt child exceeded the frozen timeout",
                        authorization, authorization_sha256,
                        code,
                    )
                else:
                    if code or not child_output.is_file():
                        record = _charged_child_failure(
                            row, candidate, attempt_index, gpu,
                            gpu_preflight["gpu"], time.perf_counter() - started,
                            "LAUNCH_FAILED", "attempt child exited unsuccessfully",
                            authorization, authorization_sha256, code,
                        )
                    else:
                        record = protocol.read_json(child_output)
                completed = time.time_ns()
                active_attempt[0] = None
                gpu_postflight = _idle_gpu_evidence(gpu, contract)
                exclusive_json(output, record)
                exclusive_json(
                    parent_path,
                    {
                        "attempt_index": attempt_index,
                        "authorization": authorization,
                        "authorization_sha256": authorization_sha256,
                        "campaign_id": protocol.CAMPAIGN_ID,
                        "candidate_cell_id": candidate,
                        "child_authorized_unix_ns": authorization["authorized_unix_ns"],
                        "child_completed_unix_ns": completed,
                        "child_launched_unix_ns": authorization["spawn_requested_unix_ns"],
                        "child_pid": authorization["child_pid"],
                        "gpu_postflight": gpu_postflight,
                        "gpu_preflight": gpu_preflight,
                        "gpu_slot": gpu,
                        "gpu_uuid": row["gpu_uuid"],
                        "parent_pid": authorization["parent_pid"],
                        "raw_path": str(output.relative_to(protocol.REPO_ROOT)),
                        "raw_sha256": protocol.file_sha256(output),
                        "record_type": "decision_complexity_ada_v1_attempt_parent_receipt",
                        "returncode": code,
                        "schema_version": 1,
                        "trajectory_authorization_sha256": trajectory_authorization_sha256,
                        "trajectory_id": trajectory_id,
                    },
                )
        record = protocol.read_json(output)
        analyze.validate_attempt(contract, row, record)
        parent = protocol.read_json(parent_path)
        previous_completed = analyze.validate_attempt_parent_receipt(
            contract, row, record, parent, output, previous_completed,
            trajectory_authorization_sha256, trajectory_authorization["child_pid"],
        )
        attempts.append({
            "parent_path": str(parent_path.relative_to(protocol.REPO_ROOT)),
            "parent_sha256": protocol.file_sha256(parent_path),
            "path": str(output.relative_to(protocol.REPO_ROOT)),
            "sha256": protocol.file_sha256(output),
        })
        if record["terminal_status"] == "GATE_PASSED" and record["ratio_to_target"] <= contract["terminal_ratio"]:
            break
    summary = analyze.derive_trajectory(contract, row, [protocol.read_json(protocol.REPO_ROOT / item["path"]) for item in attempts])
    trajectory_gpu_postflight = _idle_gpu_evidence(gpu, contract)
    trajectory_ended_unix_ns = time.time_ns()
    exclusive_json(completion, {
        **summary,
        "attempt_receipts": attempts,
        "launch_receipt_sha256": launch_sha256,
        "record_type": "decision_complexity_ada_v1_trajectory_completion",
        "schema_version": 1,
        "toolchain_sha256": lock["toolchain_sha256"],
        "trajectory_authorization_sha256": trajectory_authorization_sha256,
        "trajectory_child_pid": trajectory_authorization["child_pid"],
        "trajectory_ended_unix_ns": trajectory_ended_unix_ns,
        "trajectory_gpu_postflight": trajectory_gpu_postflight,
        "trajectory_gpu_preflight": trajectory_gpu_preflight,
        "trajectory_parent_pid": trajectory_authorization["parent_pid"],
        "trajectory_started_unix_ns": trajectory_started_unix_ns,
        "wave_ready_receipt": wave_ready,
        "wave_release_received_unix_ns": wave_release_received_unix_ns,
        "wave_release_unix_ns": wave_witness["release_unix_ns"],
        "wave_witness": wave_witness,
    })
    return 0


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _refuse_retained_execution_evidence() -> None:
    retained = [path for path in RESULTS.iterdir() if path.name != "active.lock"]
    if retained:
        raise protocol.ProtocolError(
            "retained result evidence is non-resumable; use a new successor "
            "campaign/result tag and result root"
        )


def execute() -> int:
    with all_gpu_locks() as inherited_lock_fds:
        state = _ready_locked()
        contract, manifest, lock = state["contract"], state["manifest"], state["lock"]
        RESULTS.mkdir(parents=True, exist_ok=True)
        active = (RESULTS / "active.lock").open("a+")
        try:
            fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            active.close()
            raise protocol.ProtocolError("another pilot launcher is active") from None
        try:
            _refuse_retained_execution_evidence()
            receipt = RESULTS / "launch_receipt.json"
            launch = {
                "campaign_id": protocol.CAMPAIGN_ID,
                "campaign_gpu_preflight": state["prelaunch_gpu_evidence"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "execution_lock_sha256": protocol.file_sha256(LOCK_PATH),
                "gpu_lock_paths": [str(path) for path in gpu_lock_paths()],
                "launcher_pid": os.getpid(),
                "manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
                "prelaunch_provenance_sha256": protocol.file_sha256(PROVENANCE_PATH),
                "record_type": "decision_complexity_ada_v1_launch_receipt",
                "requested_trajectories": 24,
                "schema_version": 1,
                "toolchain_sha256": lock["toolchain_sha256"],
            }
            exclusive_json(receipt, launch)
            launch_sha256 = protocol.file_sha256(receipt)
            for position in range(len(protocol.ARMS)):
                rows = sorted(
                    (row for row in manifest["rows"] if row["block_position"] == position),
                    key=lambda row: row["replicate"],
                )
                children = []
                processes: list[subprocess.Popen] = []
                try:
                    for row in rows:
                        root = RESULTS / "trajectories" / row["trajectory_id"]
                        completion = root / "completion.json"
                        parent_path = RESULTS / "trajectory_parent_receipts" / f"{row['trajectory_id']}.json"
                        if completion.exists() or parent_path.exists():
                            raise protocol.ProtocolError(
                                "retained trajectory evidence is non-resumable; use a new "
                                "successor campaign/result tag and result root"
                            )
                        preflight = _idle_gpu_evidence(row["gpu_slot"], contract)
                        command = [
                            sys.executable, str(HERE / "runner.py"), "trajectory",
                            "--trajectory-id", row["trajectory_id"],
                            "--gpu", str(row["gpu_slot"]),
                            "--launch-receipt", str(receipt),
                        ]
                        ready_read, ready_write = os.pipe()
                        release_read, release_write = os.pipe()
                        try:
                            process, authorization = _authorized_popen(
                                command,
                                os.environ,
                                {
                                    "gpu_slot": row["gpu_slot"],
                                    "gpu_uuid": row["gpu_uuid"],
                                    "launch_receipt_sha256": launch_sha256,
                                    "launch_sequence": position * protocol.REPLICATES + row["replicate"] + 1,
                                    "scope": "trajectory",
                                    "trajectory_id": row["trajectory_id"],
                                    "wave_member_order": row["replicate"],
                                    "wave_position": position,
                                },
                                inherited_lock_fds,
                                wave_fds=(ready_write, release_read),
                            )
                        except BaseException:
                            for fd in (ready_read, ready_write, release_read, release_write):
                                _close_fd(fd)
                            raise
                        _close_fd(ready_write)
                        _close_fd(release_read)
                        processes.append(process)
                        children.append({
                            "authorization": authorization,
                            "completion": completion,
                            "parent_path": parent_path,
                            "preflight": preflight,
                            "process": process,
                            "ready_fd": ready_read,
                            "release_fd": release_write,
                            "row": row,
                        })
                    wave_witness = _release_wave(
                        children, position,
                        contract["execution_schedule"]["wave_ready_timeout_s"],
                    )
                    codes = []
                    for child in children:
                        row = child["row"]
                        process = child["process"]
                        authorization = child["authorization"]
                        preflight = child["preflight"]
                        completion = child["completion"]
                        parent_path = child["parent_path"]
                        code = process.wait()
                        completed = time.time_ns()
                        postflight = _idle_gpu_evidence(row["gpu_slot"], contract)
                        completion_value = protocol.read_json(completion) if completion.is_file() else {}
                        codes.append(code)
                        exclusive_json(
                            parent_path,
                            {
                                "authorization": authorization,
                                "authorization_sha256": protocol.canonical_sha256(authorization),
                                "campaign_id": protocol.CAMPAIGN_ID,
                                "child_authorized_unix_ns": authorization["authorized_unix_ns"],
                                "child_completed_unix_ns": completed,
                                "child_launched_unix_ns": authorization["spawn_requested_unix_ns"],
                                "child_pid": authorization["child_pid"],
                                "completion_path": str(completion.relative_to(protocol.REPO_ROOT)),
                                "completion_sha256": protocol.file_sha256(completion) if completion.is_file() else None,
                                "gpu_postflight": postflight,
                                "gpu_preflight": preflight,
                                "gpu_slot": row["gpu_slot"],
                                "gpu_uuid": row["gpu_uuid"],
                                "launch_receipt_sha256": launch_sha256,
                                "launch_sequence": position * protocol.REPLICATES + row["replicate"] + 1,
                                "parent_pid": authorization["parent_pid"],
                                "record_type": "decision_complexity_ada_v1_trajectory_parent_receipt",
                                "returncode": code,
                                "schema_version": 1,
                                "trajectory_id": row["trajectory_id"],
                                "trajectory_ended_unix_ns": completion_value.get("trajectory_ended_unix_ns"),
                                "trajectory_started_unix_ns": completion_value.get("trajectory_started_unix_ns"),
                                "wave_member_order": row["replicate"],
                                "wave_position": position,
                                "wave_ready_receipt": wave_witness["ready_receipts"][row["trajectory_id"]],
                                "wave_release_received_unix_ns": completion_value.get("wave_release_received_unix_ns"),
                                "wave_release_unix_ns": wave_witness["release_unix_ns"],
                                "wave_witness": wave_witness,
                            },
                        )
                    if any(codes):
                        raise protocol.ProtocolError(f"trajectory wave {position} failed: {codes}")
                except BaseException:
                    for child in children:
                        _close_fd(child.get("ready_fd"))
                        _close_fd(child.get("release_fd"))
                    _terminate(processes)
                    raise
            completion_hashes = {
                row["trajectory_id"]: protocol.file_sha256(
                    RESULTS / "trajectories" / row["trajectory_id"] / "completion.json"
                )
                for row in manifest["rows"]
            }
            parent_hashes = {
                row["trajectory_id"]: protocol.file_sha256(
                    RESULTS / "trajectory_parent_receipts" / f"{row['trajectory_id']}.json"
                )
                for row in manifest["rows"]
            }
            campaign_postflight = [_idle_gpu_evidence(gpu, contract) for gpu in range(4)]
            exclusive_json(
                RESULTS / "run_status.json",
                {
                    "campaign_gpu_postflight": campaign_postflight,
                    "campaign_id": protocol.CAMPAIGN_ID,
                    "complete": True,
                    "launch_receipt_sha256": launch_sha256,
                    "observed_trajectories": 24,
                    "record_type": "decision_complexity_ada_v1_run_status",
                    "schema_version": 1,
                    "toolchain_sha256": lock["toolchain_sha256"],
                    "trajectory_completion_bundle_sha256": protocol.canonical_sha256(completion_hashes),
                    "trajectory_parent_bundle_sha256": protocol.canonical_sha256(parent_hashes),
                },
            )
        finally:
            active.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "freeze", "provenance", "ready", "execute"):
        sub.add_parser(name)
    trajectory_parser = sub.add_parser("trajectory")
    trajectory_parser.add_argument("--trajectory-id", required=True)
    trajectory_parser.add_argument("--gpu", type=int, required=True)
    trajectory_parser.add_argument("--launch-receipt", type=Path, required=True)
    attempt_parser = sub.add_parser("attempt")
    attempt_parser.add_argument("--trajectory-id", required=True)
    attempt_parser.add_argument("--candidate", required=True)
    attempt_parser.add_argument("--attempt-index", type=int, required=True)
    attempt_parser.add_argument("--gpu", type=int, required=True)
    attempt_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            contract, manifest = prepare(); print(f"prepared trajectories={manifest['requested_trajectories']} materials={len(contract['materials']['candidates'])}")
        elif args.command == "freeze":
            print(f"execution_lock_sha256={freeze()['lock_sha256']}")
        elif args.command == "provenance":
            provenance(); print(f"wrote {PROVENANCE_PATH}")
        elif args.command == "ready":
            print(f"ready commit={ready()['provenance']['git_commit']}")
        elif args.command == "execute":
            return execute()
        elif args.command == "trajectory":
            return trajectory(args.trajectory_id, args.gpu, args.launch_receipt)
        else:
            _contract, manifest, _lock = validate_frozen(rehash_materials=False)
            return attempt(_row(manifest, args.trajectory_id), args.candidate, args.attempt_index, args.gpu, args.output)
        return 0
    except (OSError, json.JSONDecodeError, protocol.ProtocolError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
