#!/usr/bin/env python3
"""Compile and independently verify the seven performance-blind artifacts."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import artifacts, launch, protocol
except ImportError:
    import artifacts  # type: ignore
    import launch  # type: ignore
    import protocol  # type: ignore


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def _expected_launch(contract: dict[str, Any]) -> dict[str, Any]:
    imported = protocol.RESULTS_ROOT / "imported_frontier.json"
    plan = artifacts.admission_plan(contract)
    return {
        "artifact_policy": contract["manifest"]["artifact_admission"],
        "campaign_id": protocol.CAMPAIGN_ID,
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "gpu_lock_id": launch.GPU_LOCK_ID,
        "input_artifact_path": protocol.repo_path(imported),
        "input_artifact_sha256": protocol.file_sha256(imported),
        "plan": plan,
        "plan_sha256": protocol.canonical_sha256(plan),
        "stage": "artifact_admission",
    }


def _row(cell_id: str) -> dict[str, Any]:
    rows = [row for row in artifacts.admission_plan() if row["cell_id"] == cell_id]
    if len(rows) != 1:
        raise protocol.ProtocolError(f"cell is outside artifact admission: {cell_id}")
    return rows[0]


def _cells() -> dict[str, dict[str, Any]]:
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core as source_core

    _campaign, cells, _lock = source_core.load_contract()
    return {cell["cell_id"]: cell for cell in cells}


def _receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    canonical = artifacts.ADMISSION_ROOT / "launch_receipt.json"
    if (
        path.resolve() != canonical.resolve()
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or protocol.file_sha256(path) != expected_sha256
    ):
        raise protocol.ProtocolError("admission child received a foreign launch receipt")
    return launch.validate_launch_receipt(
        protocol.read_json(path), _expected_launch(protocol.load_contract()), validate_git=False
    )


def _gate(built: Any) -> dict[str, Any]:
    import common
    import common2
    import torch

    x, weight, bias = common2.fused_inputs(seed=0, dist="rand")
    with torch.no_grad():
        reference = common2.fused_reference(
            x, weight, bias, arm="GBGS", dtype=torch.float32
        )
        observed = built.run(x.half().contiguous(), weight, bias)
        torch.cuda.synchronize()
    result = common.gate_stats(reference, observed.float())
    if result.get("gate_pass") is not True:
        raise protocol.ProtocolError("admitted artifact failed the frozen positive-input gate")
    return result


def _provenance(receipt_path: Path, gpu: dict[str, Any], toolchain: dict[str, Any], pre, post):
    return {
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "gpu_idle_postflight": post,
        "gpu_idle_preflight": pre,
        "gpu_preflight": gpu,
        "launch_receipt_path": protocol.repo_path(receipt_path),
        "launch_receipt_sha256": protocol.file_sha256(receipt_path),
        "parent_pid": os.getppid(),
        "process_pid": os.getpid(),
        "toolchain": toolchain,
    }


def _write_failure(cell_id: str, phase: str, exc: Exception) -> None:
    path = artifacts.entry_root(cell_id) / "failure.json"
    launch._write_once(
        path,
        {
            "schema_version": 1,
            "record_type": "finite_frontier_ada_artifact_admission_failure",
            "campaign_id": protocol.CAMPAIGN_ID,
            "cell_id": cell_id,
            "error": f"{type(exc).__name__}: {exc}",
            "phase": phase,
            "traceback": traceback.format_exc(),
        },
    )


def _child_preflight(args: argparse.Namespace, mode: str):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise protocol.ProtocolError("admission child requires CUDA_VISIBLE_DEVICES=0")
    launch._validate_inherited_gpu_lock()
    row = _row(args.cell_id)
    artifacts.validate_cache_environment(args.cell_id, mode)
    receipt_path = Path(args.launch_receipt).resolve()
    _receipt(receipt_path, args.launch_receipt_sha256)
    cells = _cells()
    cell = cells.get(args.cell_id)
    if cell is None:
        raise protocol.ProtocolError("admission cell is absent from the source campaign")
    contract = protocol.load_contract()
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core as source_core

    gpu = source_core.gpu_snapshot(0)
    launch.validate_gpu_snapshot(gpu, contract["manifest"]["hardware"])
    toolchain = launch.live_toolchain()
    if toolchain != contract["manifest"]["toolchain"]:
        raise protocol.ProtocolError("admission child toolchain changed")
    return row, cell, receipt_path, gpu, toolchain


def build_one(args: argparse.Namespace) -> int:
    row, cell, receipt_path, gpu, toolchain = _child_preflight(args, "admit")
    output = artifacts.entry_paths(args.cell_id)["build"]
    if Path(args.out).resolve() != output.resolve() or output.exists():
        raise protocol.ProtocolError("unsafe or existing artifact build output")
    try:
        from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import candidates

        pre = launch.require_gpu_idle("record_pre")
        built = candidates.build(cell)
        correctness = _gate(built)
        post = launch.require_gpu_idle("record_post")
        value = artifacts.build_record(
            row, built, correctness,
            _provenance(receipt_path, gpu, toolchain, pre, post),
        )
        launch._write_once(output, value)
        artifacts.validate_build_record(protocol.read_json(output), row)
        return 0
    except Exception as exc:
        _write_failure(args.cell_id, "build", exc)
        return 1


def verify_one(args: argparse.Namespace) -> int:
    row, cell, receipt_path, gpu, toolchain = _child_preflight(args, "load_only")
    paths = artifacts.entry_paths(args.cell_id)
    output = paths["verify"]
    if Path(args.out).resolve() != output.resolve() or output.exists():
        raise protocol.ProtocolError("unsafe or existing artifact verification output")
    try:
        from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import candidates

        build = artifacts.validate_build_record(protocol.read_json(paths["build"]), row)
        pre = launch.require_gpu_idle("record_pre")
        with artifacts.load_only_guards(build) as evidence:
            built = candidates.build(cell)
            artifacts.validate_load_evidence(evidence, cell, build["cache"])
            correctness = _gate(built)
            artifacts.validate_load_evidence(evidence, cell, build["cache"])
        post = launch.require_gpu_idle("record_post")
        value = artifacts.verify_record(
            row, build, built, correctness, evidence,
            _provenance(receipt_path, gpu, toolchain, pre, post),
        )
        launch._write_once(output, value)
        artifacts.validate_verify_record(protocol.read_json(output), row, build, cell)
        return 0
    except Exception as exc:
        _write_failure(args.cell_id, "verify", exc)
        return 1


def _child_command(command: str, row: dict[str, Any], receipt: Path, gpu_lock) -> int:
    paths = artifacts.entry_paths(row["cell_id"])
    output = paths["build" if command == "build-one" else "verify"]
    env = os.environ.copy()
    if command == "build-one":
        env.pop("TRITON_CACHE_MANAGER", None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_HOME": "/usr/local/cuda-13.1",
            "FINITE_FRONTIER_GPU0_LOCK_FD": str(gpu_lock.fileno()),
        }
    )
    env.update(
        artifacts.cache_environment(
            row["cell_id"], "admit" if command == "build-one" else "load_only"
        )
    )
    env["PATH"] = "/usr/local/cuda-13.1/bin:" + env.get("PATH", "")
    env.setdefault("MAX_JOBS", "4")
    completed = subprocess.run(
        [
            sys.executable,
            str(HERE / "admit.py"),
            command,
            "--cell-id",
            row["cell_id"],
            "--out",
            str(output),
            "--launch-receipt",
            str(receipt),
            "--launch-receipt-sha256",
            protocol.file_sha256(receipt),
        ],
        cwd=REPO_ROOT,
        env=env,
        pass_fds=(gpu_lock.fileno(),),
    )
    return completed.returncode


def run() -> int:
    gpu_lock = launch._acquire_gpu_lock()
    try:
        readiness = launch.ready("artifact_admission")
        stage_pre = launch.require_gpu_idle("stage_pre")
        contract = protocol.load_contract()
        plan = artifacts.admission_plan(contract)
        artifacts.ADMISSION_ROOT.mkdir(parents=True, exist_ok=True)
        active = (artifacts.ADMISSION_ROOT / "active.lock").open("a+", encoding="utf-8")
        try:
            import fcntl

            try:
                fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise protocol.ProtocolError("another artifact-admission launcher is active") from None
            receipt_path = artifacts.ADMISSION_ROOT / "launch_receipt.json"
            expected = _expected_launch(contract)
            value = {
                "schema_version": 1,
                "record_type": "finite_frontier_ada_launch_receipt",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": {
                    **expected,
                    "git_commit": readiness["git_commit"],
                    "gpu": readiness["gpu"],
                    "toolchain": readiness["toolchain"],
                    "upstream_ref": readiness["upstream_ref"],
                    "upstream_remote": readiness["upstream_remote"],
                },
                "gpu_idle_preflight": stage_pre,
            }
            if receipt_path.exists():
                launch.validate_launch_receipt(protocol.read_json(receipt_path), expected, contract)
            else:
                launch._write_once(receipt_path, value)
                launch.validate_launch_receipt(protocol.read_json(receipt_path), expected, contract)
            cells = _cells()
            for row in plan:
                paths = artifacts.entry_paths(row["cell_id"])
                failure = paths["root"] / "failure.json"
                if failure.exists():
                    raise protocol.ProtocolError(
                        f"retained admission failure requires a new successor: {failure}"
                    )
                if not paths["build"].exists():
                    if paths["root"].exists():
                        raise protocol.ProtocolError(
                            f"partial admission without a build receipt requires a new successor: {paths['root']}"
                        )
                    if _child_command("build-one", row, receipt_path, gpu_lock):
                        raise protocol.ProtocolError("artifact build failed; retain it and create a successor")
                build = artifacts.validate_build_record(protocol.read_json(paths["build"]), row)
                if not paths["verify"].exists():
                    if _child_command("verify-one", row, receipt_path, gpu_lock):
                        raise protocol.ProtocolError("artifact verification failed; retain it and create a successor")
                artifacts.validate_verify_record(
                    protocol.read_json(paths["verify"]), row, build, cells[row["cell_id"]]
                )
                entry = artifacts.make_entry(row, cells[row["cell_id"]])
                launch._write_once(paths["entry"], entry)
                artifacts.validate_entry(protocol.read_json(paths["entry"]), row, cells[row["cell_id"]])
                print(f"[{row['position'] + 1}/7] admitted {row['cell_id']}", flush=True)
            manifest = artifacts.make_manifest(receipt_path, cells)
            launch._write_once(artifacts.MANIFEST_PATH, manifest)
            artifacts.validate_manifest(protocol.read_json(artifacts.MANIFEST_PATH), cells)
            status_path = artifacts.ADMISSION_ROOT / "run_status.json"
            if status_path.exists():
                launch.validate_run_status(
                    protocol.read_json(status_path), "artifact_admission", 7,
                    protocol.file_sha256(receipt_path),
                )
            else:
                launch._write_once(
                    status_path,
                    {
                        "schema_version": 1,
                        "record_type": "finite_frontier_ada_run_status",
                        "campaign_id": protocol.CAMPAIGN_ID,
                        "complete": True,
                        "expected_records": 7,
                        "observed_records": 7,
                        "gpu_idle_postflight": launch.require_gpu_idle("stage_post"),
                        "launch_receipt_sha256": protocol.file_sha256(receipt_path),
                        "stage": "artifact_admission",
                    },
                )
            return 0
        finally:
            active.close()
    finally:
        gpu_lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run")
    for name in ("build-one", "verify-one"):
        child = subparsers.add_parser(name)
        child.add_argument("--cell-id", required=True)
        child.add_argument("--out", required=True)
        child.add_argument("--launch-receipt", required=True)
        child.add_argument("--launch-receipt-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-one":
        return build_one(args)
    if args.command == "verify-one":
        return verify_one(args)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
