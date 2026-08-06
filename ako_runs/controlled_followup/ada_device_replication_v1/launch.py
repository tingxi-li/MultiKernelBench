#!/usr/bin/env python3
"""Fail-closed outer launcher for the four-device Ada audit replication.

The frozen crossed-v2 runner remains an external instrument.  Four cyclic
waves give every device all four 76-cell shards under its own result tag while
preventing identical shards from compiling concurrently.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import protocol
except ImportError:
    import protocol  # type: ignore


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SEALED_ROOT = HERE.parent / "fused_epilogue_crossed_v2"
INSTRUMENT_RESULTS = SEALED_ROOT / "results"
OUTER_RESULTS = HERE / "results"
EXPECTED_ROLE_PATHS = {
    "instrument_launch_lock": "ako_runs/controlled_followup/fused_epilogue_crossed_v2/launch_lock.json",
    "source_lock": "ako_runs/controlled_followup/fused_epilogue_crossed_v2/evidence/crossed_v2r3_complete_v1.index.json",
    "gate_lock": "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json",
    "analyzer": "ako_runs/controlled_followup/fused_epilogue_crossed_v2/analyze.py",
    "runner": "ako_runs/controlled_followup/fused_epilogue_crossed_v2/campaign_runner.py",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise protocol.ProtocolError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise protocol.ProtocolError(f"JSON must be an object: {path}")
    return value


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise protocol.ProtocolError(f"path escapes repository: {path}") from exc


def schedule() -> list[dict[str, Any]]:
    rows = [
        {
            "wave": wave,
            "physical_gpu": gpu,
            "device_uuid": protocol.GPU_UUIDS[gpu],
            "campaign_tag": protocol.TAGS[gpu],
            "shard_index": (gpu + wave) % 4,
            "shard_count": 4,
        }
        for wave in range(4)
        for gpu in range(4)
    ]
    if any(
        {row["shard_index"] for row in rows if row["wave"] == wave} != set(range(4))
        for wave in range(4)
    ) or any(
        {row["shard_index"] for row in rows if row["physical_gpu"] == gpu} != set(range(4))
        for gpu in range(4)
    ):
        raise protocol.ProtocolError("cyclic four-device schedule is not a complete Latin square")
    return rows


def validate_gpu_inventory(lines: list[str]) -> list[dict[str, Any]]:
    """Require the frozen physical-index to UUID mapping before any child starts."""
    observed = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise protocol.ProtocolError("malformed nvidia-smi GPU inventory")
        try:
            index = int(parts[0])
        except ValueError as exc:
            raise protocol.ProtocolError("malformed physical GPU index") from exc
        observed.append(
            {
                "index": index,
                "uuid": parts[1],
                "name": parts[2],
                "compute_capability": parts[3],
            }
        )
    expected = [
        {
            "index": index,
            "uuid": uuid,
            "name": protocol.GPU_NAME,
            "compute_capability": protocol.COMPUTE_CAPABILITY,
        }
        for index, uuid in enumerate(protocol.GPU_UUIDS)
    ]
    if observed != expected:
        raise protocol.ProtocolError("physical GPU index/UUID/SKU mapping changed")
    return observed


def live_gpu_inventory() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,compute_cap",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise protocol.ProtocolError("cannot capture live physical GPU inventory")
    return validate_gpu_inventory(
        [line for line in completed.stdout.splitlines() if line.strip()]
    )


def make_contract() -> dict[str, Any]:
    roles = {
        role: EXPECTED_ROLE_PATHS[role]
        for role in protocol.DEPENDENCY_ROLES
    }
    return {
        "schema_version": 1,
        "campaign_id": protocol.CAMPAIGN_ID,
        "instrument_campaign_id": protocol.INSTRUMENT_CAMPAIGN_ID,
        "reference_result_tag": protocol.REFERENCE_RESULT_TAG,
        "state": protocol.STATE,
        "claim_scope": protocol.CLAIM_SCOPE,
        "timing_allowed": False,
        "devices": [
            {
                "uuid": uuid,
                "name": protocol.GPU_NAME,
                "compute_capability": protocol.COMPUTE_CAPABILITY,
            }
            for uuid in protocol.GPU_UUIDS
        ],
        "dependency_roles": roles,
        "dependency_sha256": {
            relative: file_sha256(REPO_ROOT / relative)
            for relative in roles.values()
        },
    }


def prepare(contract_path: Path, manifest_path: Path, *, write: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = make_contract()
    manifest = protocol.make_manifest(contract, REPO_ROOT)
    if write:
        _write_once(contract_path, contract)
        _write_once(manifest_path, manifest)
    else:
        if read_json(contract_path) != contract or read_json(manifest_path) != manifest:
            raise protocol.ProtocolError("prepared contract or manifest differs from deterministic projection")
    return contract, manifest


def _role_paths(contract: dict[str, Any]) -> dict[str, Path]:
    roles = contract["dependency_roles"]
    allowed = set(EXPECTED_ROLE_PATHS)
    if not set(roles) <= allowed or set(roles) != set(protocol.DEPENDENCY_ROLES):
        raise protocol.ProtocolError("dependency roles differ from the execution contract")
    for role, expected in EXPECTED_ROLE_PATHS.items():
        if role in roles and roles[role] != expected:
            raise protocol.ProtocolError(f"{role} does not name the frozen instrument path")
    return {role: REPO_ROOT / relative for role, relative in roles.items()}


def validated_inputs(contract_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], dict[str, Any]]:
    contract = read_json(contract_path)
    protocol.validate_contract(contract, REPO_ROOT)
    manifest = read_json(manifest_path)
    if manifest != protocol.make_manifest(contract, REPO_ROOT):
        raise protocol.ProtocolError("manifest differs from the contract projection")
    roles = _role_paths(contract)
    lock_path = roles.get("instrument_launch_lock", roles.get("source_lock"))
    if lock_path is None:
        raise protocol.ProtocolError("contract lacks the frozen instrument launch lock")
    lock = read_json(lock_path)
    if (
        lock.get("campaign_id") != protocol.INSTRUMENT_CAMPAIGN_ID
        or lock.get("lock_stage") != "campaign"
    ):
        raise protocol.ProtocolError("foreign instrument launch lock")
    source_index = read_json(roles["source_lock"])
    if (
        source_index.get("record_type") != "fused_crossed_v2_complete_evidence_index"
        or source_index.get("campaign_id") != protocol.INSTRUMENT_CAMPAIGN_ID
        or source_index.get("launch_lock_sha256") != file_sha256(lock_path)
        or source_index.get("source_bundle_sha256") != lock.get("source_bundle_sha256")
    ):
        raise protocol.ProtocolError("source evidence index does not bind the frozen instrument")
    source_hashes = lock.get("source_sha256", {})
    for role in ("runner", "analyzer"):
        relative = repo_path(roles[role])
        if source_hashes.get(relative) != contract["dependency_sha256"][relative]:
            raise protocol.ProtocolError(f"instrument lock does not bind {role}")
    gate_relative = contract["dependency_roles"]["gate_lock"]
    if lock.get("frozen_gate", {}).get("gate_spec_sha256") != contract["dependency_sha256"][gate_relative]:
        raise protocol.ProtocolError("instrument lock does not bind the gate bytes")
    return contract, manifest, roles, lock


def authorization_bindings(contract_path: Path, manifest_path: Path) -> dict[str, Any]:
    contract, manifest, roles, lock = validated_inputs(contract_path, manifest_path)
    return {
        "schema_version": 1,
        "record_type": "ada_device_replication_v1_execution_lock",
        "campaign_id": protocol.CAMPAIGN_ID,
        "authorized": True,
        "timing_allowed": False,
        "contract_path": repo_path(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "manifest_path": repo_path(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "protocol_sha256": file_sha256(HERE / "protocol.py"),
        "outer_launcher_sha256": file_sha256(HERE / "launch.py"),
        "outer_analyzer_sha256": file_sha256(HERE / "analyze.py"),
        "instrument_campaign_id": protocol.INSTRUMENT_CAMPAIGN_ID,
        "reference_result_tag": protocol.REFERENCE_RESULT_TAG,
        "instrument_launch_lock_sha256": file_sha256(
            roles.get("instrument_launch_lock", roles["source_lock"])
        ),
        "instrument_source_bundle_sha256": lock["source_bundle_sha256"],
        "instrument_runner_sha256": file_sha256(roles["runner"]),
        "instrument_analyzer_sha256": file_sha256(roles["analyzer"]),
        "schedule_sha256": canonical_sha256(schedule()),
        "requested_records": manifest["requested_records"],
    }


def validate_authorization(path: Path, contract_path: Path, manifest_path: Path) -> dict[str, Any]:
    observed = read_json(path)
    expected = authorization_bindings(contract_path, manifest_path)
    allowed = {*expected, "authorization_basis", "authorized_at_utc"}
    if set(observed) != allowed or any(observed.get(key) != value for key, value in expected.items()):
        raise protocol.ProtocolError("execution lock fields or frozen bindings differ")
    if not isinstance(observed["authorization_basis"], str) or not observed["authorization_basis"].strip():
        raise protocol.ProtocolError("execution lock lacks an authorization basis")
    if not isinstance(observed["authorized_at_utc"], str) or not observed["authorized_at_utc"].strip():
        raise protocol.ProtocolError("execution lock lacks an authorization timestamp")
    return observed


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
    )
    if completed.returncode:
        raise protocol.ProtocolError(completed.stderr.strip() or f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def validate_remote_ready(paths: list[Path]) -> dict[str, str]:
    relatives = [repo_path(path) for path in paths]
    if _git("status", "--porcelain", "--", *relatives):
        raise protocol.ProtocolError("successor or instrument launch inputs are not committed")
    _git("ls-files", "--error-unmatch", "--", *relatives)
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
    if completed.returncode or not completed.stdout.strip():
        raise protocol.ProtocolError("configured upstream cannot be verified independently")
    upstream_head = completed.stdout.split()[0]
    if upstream_head != head:
        raise protocol.ProtocolError("launch commit is not the configured upstream head")
    return {"git_commit": head, "upstream_remote": remote, "upstream_ref": merge_ref}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != value:
            raise protocol.ProtocolError(f"existing immutable receipt differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _command(row: dict[str, Any], runner: Path) -> list[str]:
    return [
        sys.executable,
        str(runner),
        "audit",
        "--tag",
        row["campaign_tag"],
        "--gpu",
        str(row["physical_gpu"]),
        "--shard-index",
        str(row["shard_index"]),
        "--shard-count",
        str(row["shard_count"]),
    ]


def _terminate_processes(processes: list[tuple[dict[str, Any], subprocess.Popen, Any]]) -> None:
    for _row, process, _lock in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for _row, process, _lock in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def _failure_receipt(wave: int, rows: list[dict[str, Any]], returncodes: list[int]) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    _write_once(
        OUTER_RESULTS / "waves" / "failures" / f"wave{wave}__{stamp}__pid{os.getpid()}.json",
        {
            "schema_version": 1,
            "record_type": "ada_device_replication_v1_wave_failure",
            "complete": False,
            "schedule": rows,
            "returncodes": returncodes,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def execute(contract_path: Path, manifest_path: Path, authorization_path: Path) -> int:
    contract, _manifest, roles, _lock = validated_inputs(contract_path, manifest_path)
    authorization = validate_authorization(authorization_path, contract_path, manifest_path)
    launch_paths = [
        contract_path,
        manifest_path,
        authorization_path,
        HERE / "protocol.py",
        HERE / "launch.py",
        HERE / "analyze.py",
        *roles.values(),
    ]
    remote = validate_remote_ready(launch_paths)
    gpu_inventory = live_gpu_inventory()
    receipt_path = OUTER_RESULTS / "launch_receipt.json"
    if not receipt_path.exists():
        collisions = [tag for tag in protocol.TAGS if (INSTRUMENT_RESULTS / tag).exists()]
        if collisions:
            raise protocol.ProtocolError(f"new replication tags already exist: {collisions}")
    launch_contract = {
        **authorization,
        **remote,
        "execution_lock_sha256": file_sha256(authorization_path),
        "gpu_inventory": gpu_inventory,
        "schedule": schedule(),
    }
    if receipt_path.exists():
        retained = read_json(receipt_path)
        if (
            retained.get("schema_version") != 1
            or retained.get("record_type") != "ada_device_replication_v1_launch_receipt"
            or retained.get("contract") != launch_contract
        ):
            raise protocol.ProtocolError("existing launch receipt has another contract")
    else:
        _write_once(
            receipt_path,
            {
                "schema_version": 1,
                "record_type": "ada_device_replication_v1_launch_receipt",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": launch_contract,
            },
        )

    OUTER_RESULTS.mkdir(parents=True, exist_ok=True)
    active = (OUTER_RESULTS / "active.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise protocol.ProtocolError("another Ada replication launcher is active") from None

    old_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in old_handlers:
        signal.signal(signum, interrupted)
    try:
        for wave in range(4):
            wave_rows = [row for row in schedule() if row["wave"] == wave]
            status_path = OUTER_RESULTS / "waves" / f"wave{wave}.json"
            if status_path.exists():
                status = read_json(status_path)
                if (
                    status.get("complete") is not True
                    or status.get("schedule") != wave_rows
                    or status.get("returncodes") != [0, 0, 0, 0]
                ):
                    raise protocol.ProtocolError(f"invalid retained wave status: {status_path}")
                continue
            processes: list[tuple[dict[str, Any], subprocess.Popen, Any]] = []
            try:
                for row in wave_rows:
                    gpu = row["physical_gpu"]
                    cache = HERE / ".cache" / f"gpu{gpu}"
                    environment = os.environ.copy()
                    environment.update(
                        {
                            "CUDA_VISIBLE_DEVICES": str(gpu),
                            "PYTHONDONTWRITEBYTECODE": "1",
                            "XDG_CACHE_HOME": str(cache),
                            "TRITON_CACHE_DIR": str(cache / "triton"),
                            "CUDA_CACHE_PATH": str(cache / "cuda"),
                        }
                    )
                    lock_path = (
                        OUTER_RESULTS
                        / "shard_locks"
                        / f"{row['campaign_tag']}__shard{row['shard_index']:02d}.lock"
                    )
                    lock_path.parent.mkdir(parents=True, exist_ok=True)
                    shard_lock = lock_path.open("a+", encoding="utf-8")
                    try:
                        fcntl.flock(
                            shard_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                    except BlockingIOError:
                        shard_lock.close()
                        raise protocol.ProtocolError(
                            f"device shard is already running: {lock_path.name}"
                        ) from None
                    process = subprocess.Popen(
                        _command(row, roles["runner"]),
                        cwd=REPO_ROOT,
                        env=environment,
                        pass_fds=(shard_lock.fileno(),),
                        start_new_session=True,
                    )
                    processes.append((row, process, shard_lock))
                returncodes = [process.wait() for _row, process, _lock in processes]
            except BaseException:
                _terminate_processes(processes)
                raise
            finally:
                for _row, _process, shard_lock in processes:
                    shard_lock.close()
            if any(returncodes):
                _failure_receipt(wave, wave_rows, returncodes)
                raise protocol.ProtocolError(
                    f"wave {wave} failed; it may be retried after all child locks release"
                )
            _write_once(
                status_path,
                {
                    "schema_version": 1,
                    "record_type": "ada_device_replication_v1_wave_status",
                    "complete": True,
                    "schedule": wave_rows,
                    "returncodes": returncodes,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        return 0
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        active.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=HERE / "contract.json")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--write", action="store_true")
    sub.add_parser("list")
    template = sub.add_parser("lock-template")
    template.add_argument("--authorization-basis", default="USER_AUTHORIZATION_REQUIRED")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--execution-lock", type=Path, default=HERE / "execution_lock.json")
    freeze.add_argument("--authorization-basis", required=True)
    freeze.add_argument("--write", action="store_true")
    ready = sub.add_parser("ready")
    ready.add_argument("--execution-lock", type=Path, default=HERE / "execution_lock.json")
    run = sub.add_parser("execute")
    run.add_argument("--execution-lock", type=Path, default=HERE / "execution_lock.json")
    args = parser.parse_args()
    if args.command == "prepare":
        contract, manifest = prepare(args.contract, args.manifest, write=args.write)
        print(
            f"prepared={args.write} contract_sha256={canonical_sha256(contract)} "
            f"records={manifest['requested_records']}"
        )
        return 0
    if args.command == "list":
        validated_inputs(args.contract, args.manifest)
        for row in schedule():
            print(
                f"wave={row['wave']} gpu={row['physical_gpu']} tag={row['campaign_tag']} "
                f"shard={row['shard_index']}/4"
            )
        return 0
    if args.command == "lock-template":
        value = {
            **authorization_bindings(args.contract, args.manifest),
            "authorized": False,
            "authorization_basis": args.authorization_basis,
            "authorized_at_utc": None,
        }
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze":
        value = {
            **authorization_bindings(args.contract, args.manifest),
            "authorization_basis": args.authorization_basis,
            "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if args.write:
            _write_once(args.execution_lock, value)
        else:
            observed = validate_authorization(
                args.execution_lock, args.contract, args.manifest
            )
            value = observed
        print(
            f"frozen={args.write} execution_lock_sha256="
            f"{file_sha256(args.execution_lock) if args.execution_lock.exists() else canonical_sha256(value)}"
        )
        return 0
    if args.command == "ready":
        _contract, _manifest, roles, _lock = validated_inputs(args.contract, args.manifest)
        validate_authorization(args.execution_lock, args.contract, args.manifest)
        remote = validate_remote_ready(
            [args.contract, args.manifest, args.execution_lock, HERE / "protocol.py", HERE / "launch.py", HERE / "analyze.py", *roles.values()]
        )
        inventory = live_gpu_inventory()
        print(f"launch_ready=PASS commit={remote['git_commit']} gpus={len(inventory)}")
        return 0
    return execute(args.contract, args.manifest, args.execution_lock)


if __name__ == "__main__":
    raise SystemExit(main())
