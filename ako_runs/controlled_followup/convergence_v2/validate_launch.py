#!/usr/bin/env python3
"""Fail-closed convergence-v2 launch-readiness validator."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Mapping

try:
    from .campaign import CAMPAIGN_ID, MODELS, OPERATIONS, sha256_file, validate_manifest_rows
    from .hidden_gate import validate_gate_bindings
    from .model_resolution import load_model_resolution_lock
except ImportError:  # direct script execution
    from campaign import CAMPAIGN_ID, MODELS, OPERATIONS, sha256_file, validate_manifest_rows  # type: ignore
    from hidden_gate import validate_gate_bindings  # type: ignore
    from model_resolution import load_model_resolution_lock  # type: ignore


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{16,}$")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _verify_hash_document(base: Path, relative_path: str, required_state: str) -> None:
    document = _read_json(base / relative_path)
    if document.get("campaign_id") != CAMPAIGN_ID or document.get("state") != required_state:
        raise ValueError(f"{relative_path}: wrong campaign/state")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{relative_path}: empty files map")
    for name, expected in files.items():
        if SHA256_RE.fullmatch(str(expected)) is None or sha256_file(base / name) != expected:
            raise ValueError(f"{relative_path}: hash mismatch for {name}")


def _check_manifests(base: Path) -> None:
    core = [json.loads(line) for line in (base / "manifests/core_192.jsonl").read_text().splitlines()]
    prompt = [json.loads(line) for line in (base / "manifests/prompt_extension_128.jsonl").read_text().splitlines()]
    validate_manifest_rows(core, prompt)
    summary = _read_json(base / "manifests/summary.json")
    if summary.get("total_trajectories") != 320:
        raise ValueError("summary does not bind 320 trajectories")
    for name in ("core_192.jsonl", "prompt_extension_128.jsonl"):
        if summary.get("manifests", {}).get(name) != sha256_file(base / "manifests" / name):
            raise ValueError(f"manifest summary hash mismatch: {name}")


def _check_gpu_lock(base: Path, check_runtime: bool) -> list[str]:
    lock = _read_json(base / "locks/gpu_assignment_lock.json")
    if lock.get("state") != "resolved":
        raise ValueError("GPU assignment lock is unresolved")
    rows = lock.get("slots")
    if not isinstance(rows, list) or sorted(row.get("slot") for row in rows) != [0, 1, 2, 3]:
        raise ValueError("GPU lock must bind slots 0..3")
    uuids = [row.get("gpu_uuid") for row in rows]
    if len(set(uuids)) != 4 or any(not isinstance(uuid, str) or GPU_UUID_RE.fullmatch(uuid) is None for uuid in uuids):
        raise ValueError("GPU lock requires four distinct concrete NVIDIA UUIDs")
    if check_runtime:
        run = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if run.returncode != 0:
            raise ValueError("nvidia-smi UUID preflight failed")
        visible = {line.strip() for line in run.stdout.splitlines() if line.strip()}
        if set(uuids) != visible:
            raise ValueError("runtime GPU UUIDs differ from frozen assignment")
    return uuids


def _check_reference_lock(base: Path, gpu_uuids: list[str]) -> None:
    lock = _read_json(base / "locks/reference_latency_lock.json")
    if lock.get("state") != "frozen" or lock.get("gpu_uuid") not in gpu_uuids:
        raise ValueError("reference latency lock is not frozen on an assigned GPU")
    refs = lock.get("references")
    if not isinstance(refs, dict) or set(refs) != set(OPERATIONS):
        raise ValueError("reference latency lock operation census mismatch")
    for operation, ref in refs.items():
        latency = ref.get("latency_ms")
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"{operation}: invalid reference latency")
        if ref.get("terminal_gate_passed") is not True:
            raise ValueError(f"{operation}: reference did not pass terminal gate")
        for field in ("candidate_source_sha256", "measurement_receipt_sha256"):
            if SHA256_RE.fullmatch(str(ref.get(field, ""))) is None:
                raise ValueError(f"{operation}: invalid {field}")


def _check_remote_lock(base: Path) -> None:
    lock = _read_json(base / "locks/remote_preregistration_lock.json")
    if lock.get("state") != "resolved":
        raise ValueError("remote preregistration lock is unresolved")
    commit = str(lock.get("commit_sha", ""))
    if re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise ValueError("remote preregistration commit is invalid")
    if not str(lock.get("remote_ref", "")) or not str(lock.get("pushed_at_utc", "")).endswith("Z"):
        raise ValueError("remote preregistration receipt is incomplete")
    if SHA256_RE.fullmatch(str(lock.get("remote_receipt_sha256", ""))) is None:
        raise ValueError("remote preregistration evidence hash is invalid")


def validate_launch_state(
    base: Path,
    *,
    environ: Mapping[str, str] = os.environ,
    check_gpu_runtime: bool = True,
    check_provider_sdks: bool = True,
) -> dict:
    checks: list[dict[str, str | bool]] = []

    def check(name: str, action) -> None:
        try:
            action()
        except Exception as exc:  # report every blocker in one pass
            checks.append({"name": name, "passed": False, "detail": str(exc)})
        else:
            checks.append({"name": name, "passed": True, "detail": "ok"})

    check("trajectory_manifests", lambda: _check_manifests(base))
    check("prompt_contract_lock", lambda: _verify_hash_document(base, "locks/prompt_contract_lock.json", "frozen"))
    check("protocol_freeze_receipt", lambda: _verify_hash_document(base, "locks/protocol_freeze_receipt.json", "frozen_not_launch_ready"))
    check("gate_preregistration_receipt", lambda: _verify_hash_document(base, "gates/FREEZE_RECEIPT.json", "preregistrations_frozen_not_executed"))
    check("immutable_model_resolution", lambda: load_model_resolution_lock(base / "locks/model_resolution_lock.json"))
    gpu_holder: list[list[str]] = []

    def gpu_check() -> None:
        gpu_holder.append(_check_gpu_lock(base, check_gpu_runtime))

    check("gpu_uuid_binding", gpu_check)
    check("robust_hidden_gate_bindings", lambda: validate_gate_bindings(base / "locks/gate_bindings.json", set(OPERATIONS)))
    check("remote_preregistration", lambda: _check_remote_lock(base))
    check(
        "provider_credentials",
        lambda: (_ for _ in ()).throw(ValueError("OPENAI_API_KEY and ANTHROPIC_API_KEY must both be present"))
        if not environ.get("OPENAI_API_KEY") or not environ.get("ANTHROPIC_API_KEY")
        else None,
    )
    if check_provider_sdks:
        check(
            "provider_sdks",
            lambda: (_ for _ in ()).throw(ValueError("openai and anthropic SDKs must both be importable"))
            if importlib.util.find_spec("openai") is None or importlib.util.find_spec("anthropic") is None
            else None,
        )
    check(
        "frozen_reference_latencies",
        lambda: _check_reference_lock(base, gpu_holder[0] if gpu_holder else []),
    )
    ready = all(bool(item["passed"]) for item in checks)
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "ready": ready,
        "action": "launch_permitted" if ready else "launch_forbidden",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--skip-gpu-runtime", action="store_true", help="diagnostic only; cannot authorize launch")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_launch_state(args.base, check_gpu_runtime=not args.skip_gpu_runtime)
    if args.skip_gpu_runtime:
        result["ready"] = False
        result["action"] = "launch_forbidden_diagnostic_gpu_check_skipped"
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
