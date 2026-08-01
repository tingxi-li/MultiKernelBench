#!/usr/bin/env python3
"""Static and launch-readiness validation for the RQ5 campaign."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    from . import campaign, make_manifest
except ImportError:  # direct script execution
    import campaign  # type: ignore
    import make_manifest  # type: ignore


CONVERGENCE_V2 = campaign.HERE.parent / "convergence_v2"
try:
    from ako_runs.controlled_followup.convergence_v2.model_resolution import (
        ModelResolutionError,
        load_model_resolution_lock,
    )
except ImportError:  # direct execution from the campaign directory
    import sys

    sys.path.insert(0, str(campaign.REPO_ROOT))
    from ako_runs.controlled_followup.convergence_v2.model_resolution import (  # noqa: E402
        ModelResolutionError,
        load_model_resolution_lock,
    )


COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_static() -> dict[str, Any]:
    expected = campaign.stable_json_bytes(make_manifest.build())
    _require(campaign.MANIFEST.is_file(), "manifest is missing")
    _require(campaign.MANIFEST.read_bytes() == expected, "manifest is stale")
    manifest = campaign.load_json(campaign.MANIFEST)
    jobs = manifest.get("trajectories")
    _require(isinstance(jobs, list) and jobs == campaign.trajectories(), "trajectory drift")
    _require(len(jobs) == 20, "expected 20 programmable trajectories")
    _require(len({row["trajectory_id"] for row in jobs}) == 20, "trajectory collision")
    _require(len({row["search_seed"] for row in jobs}) == 20, "search-seed collision")
    for lane in campaign.PROGRAMMABLE_LANES:
        lane_jobs = [row for row in jobs if row["lane"] == lane]
        _require(len(lane_jobs) == 5, f"{lane}: expected five independent contexts")
        _require(
            [row["replicate"] for row in lane_jobs] == list(range(5)),
            f"{lane}: replicate drift",
        )
    counts = {gpu: sum(row["physical_gpu"] == gpu for row in jobs) for gpu in range(4)}
    _require(set(counts.values()) == {5}, f"GPU allocation is unbalanced: {counts}")
    _require(campaign.GATE_SPEC.is_file(), "fused-v2 gate spec is missing")
    _require(campaign.GATE_RECEIPT.is_file(), "fused-v2 acceptance receipt is missing")
    receipt = campaign.load_json(campaign.GATE_RECEIPT)
    _require(receipt.get("accepted_for_fused_grid_screening") is True,
             "fused-v2 gate acceptance is not affirmative")
    _require(receipt["frozen_gate"]["file_sha256"] == campaign.file_sha256(campaign.GATE_SPEC),
             "fused-v2 gate receipt no longer binds the gate bytes")
    return manifest


def _load_object(path: Path, label: str, blockers: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        blockers.append(f"{label} is missing: {campaign.repo_path(path)}")
        return None
    try:
        value = campaign.load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"{label} is unreadable: {exc}")
        return None
    if not isinstance(value, dict):
        blockers.append(f"{label} is not an object")
        return None
    return value


def model_resolution(blockers: list[str]):
    raw = _load_object(campaign.MODEL_LOCK, "model resolution lock", blockers)
    if raw is None:
        return None
    if raw.get("campaign_id") != campaign.CAMPAIGN_ID:
        blockers.append("model resolution lock campaign mismatch")
        return None
    try:
        resolutions = load_model_resolution_lock(
            campaign.MODEL_LOCK,
            expected_models=[campaign.MODEL],
            require_resolved=True,
        )
    except ModelResolutionError as exc:
        blockers.append(f"immutable model resolution is unavailable: {exc}")
        return None
    return resolutions[
        f"{campaign.MODEL['provider']}:{campaign.MODEL['requested_alias']}"
    ]


def executor_registry(blockers: list[str]) -> dict[str, Any] | None:
    registry = _load_object(
        campaign.EXECUTOR_REGISTRY,
        "executor treatment registry (see EXECUTOR_TREATMENT_ARTIFACT.md)",
        blockers,
    )
    if registry is None:
        return None
    if registry.get("schema_version") != 1:
        blockers.append("executor registry schema mismatch")
    if registry.get("campaign_id") != campaign.CAMPAIGN_ID:
        blockers.append("executor registry campaign mismatch")
    entries = registry.get("lanes")
    if not isinstance(entries, dict) or set(entries) != set(campaign.PROGRAMMABLE_LANES):
        blockers.append("executor registry does not bind exactly four programmable lanes")
        return None
    for lane, entry in entries.items():
        if not isinstance(entry, dict):
            blockers.append(f"{lane}: executor entry is not an object")
            continue
        for split in (
            "tuning_command",
            "terminal_holdout_command",
            "confirmation_command",
        ):
            command = entry.get(split)
            if not isinstance(command, list) or not command or not all(
                isinstance(value, str) and value for value in command
            ):
                blockers.append(f"{lane}: {split} is not a frozen argv list")
        timeout = entry.get("timeout_s")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 7200:
            blockers.append(f"{lane}: timeout_s must be an integer in [1, 7200]")
        sources = entry.get("source_hashes")
        if not isinstance(sources, dict) or not sources:
            blockers.append(f"{lane}: no executor source hashes")
            continue
        for relative, digest in sources.items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                blockers.append(f"{lane}: malformed executor source binding")
                continue
            path = (campaign.REPO_ROOT / relative).resolve()
            try:
                path.relative_to(campaign.REPO_ROOT.resolve())
            except ValueError:
                blockers.append(f"{lane}: executor source escapes repository: {relative}")
                continue
            if (
                not path.is_file()
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or digest != campaign.file_sha256(path)
            ):
                blockers.append(f"{lane}: stale executor source hash for {relative}")
    control = registry.get("control")
    if not isinstance(control, dict):
        blockers.append("executor registry lacks the frozen control executor")
    else:
        command = control.get("confirmation_command")
        if not isinstance(command, list) or not command or not all(
            isinstance(value, str) and value for value in command
        ):
            blockers.append("control: confirmation_command is not a frozen argv list")
        timeout = control.get("timeout_s")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 7200:
            blockers.append("control: timeout_s must be an integer in [1, 7200]")
        sources = control.get("source_hashes")
        if not isinstance(sources, dict) or not sources:
            blockers.append("control: no executor source hashes")
        else:
            for relative, digest in sources.items():
                if not isinstance(relative, str) or not isinstance(digest, str):
                    blockers.append("control: malformed executor source binding")
                    continue
                path = (campaign.REPO_ROOT / relative).resolve()
                try:
                    path.relative_to(campaign.REPO_ROOT.resolve())
                except ValueError:
                    blockers.append(f"control: executor source escapes repository: {relative}")
                    continue
                if (
                    not path.is_file()
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    or digest != campaign.file_sha256(path)
                ):
                    blockers.append(f"control: stale executor source hash for {relative}")
    return registry


def provenance_blockers(blockers: list[str]) -> None:
    lock = _load_object(campaign.PROVENANCE_LOCK, "prelaunch provenance", blockers)
    if lock is None:
        return
    if lock.get("campaign_id") != campaign.CAMPAIGN_ID:
        blockers.append("prelaunch provenance campaign mismatch")
    if not COMMIT_RE.fullmatch(str(lock.get("git_commit", ""))):
        blockers.append("prelaunch provenance lacks a full immutable commit")
    if lock.get("remote_push_verified") is not True:
        blockers.append("prelaunch provenance lacks remote-push verification")
    if not isinstance(lock.get("remote_name"), str) or not lock["remote_name"].strip():
        blockers.append("prelaunch provenance lacks the verified remote name")
    if not isinstance(lock.get("remote_ref"), str) or not lock["remote_ref"].strip():
        blockers.append("prelaunch provenance lacks the verified remote ref")
    if lock.get("git_status_porcelain") != "":
        blockers.append("prelaunch provenance was not captured from a clean worktree")
    if not str(lock.get("external_timestamp_utc", "")).endswith("Z"):
        blockers.append("prelaunch provenance lacks an external UTC timestamp")
    if lock.get("manifest_sha256") != campaign.file_sha256(campaign.MANIFEST):
        blockers.append("prelaunch provenance manifest hash mismatch")
    if lock.get("campaign_file_sha256") != campaign.campaign_provenance_hashes():
        blockers.append("prelaunch provenance campaign-file map mismatch")
    if not campaign.MODEL_LOCK.is_file() or lock.get("model_resolution_lock_sha256") != campaign.file_sha256(campaign.MODEL_LOCK):
        blockers.append("prelaunch provenance model-lock hash mismatch")
    if not campaign.EXECUTOR_REGISTRY.is_file() or lock.get("executor_registry_sha256") != campaign.file_sha256(campaign.EXECUTOR_REGISTRY):
        blockers.append("prelaunch provenance executor-registry hash mismatch")


def gpu_blockers() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return [f"NVIDIA driver unavailable: {exc}"]
    if completed.returncode != 0:
        return ["NVIDIA driver unavailable: " + (completed.stderr.strip() or "nvidia-smi failed")]
    rows = [line.split(",") for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 4:
        return [f"expected four GPUs, observed {len(rows)}"]
    blockers = []
    for index, row in enumerate(rows):
        values = [value.strip() for value in row]
        if len(values) != 4:
            blockers.append(f"GPU {index}: malformed nvidia-smi record")
            continue
        try:
            observed_index = int(values[0])
        except ValueError:
            blockers.append(f"GPU {index}: malformed index")
            continue
        if observed_index != index or values[1] != campaign.GPU_UUIDS[index]:
            blockers.append(f"GPU {index}: identity differs from frozen allocation")
        if values[2] != "NVIDIA RTX 6000 Ada Generation" or values[3] != "8.9":
            blockers.append(f"GPU {index}: campaign is Ada-only")
    return blockers


def launch_blockers() -> list[str]:
    validate_static()
    blockers: list[str] = []
    model_resolution(blockers)
    executor_registry(blockers)
    provenance_blockers(blockers)
    if not os.environ.get("OPENAI_API_KEY"):
        blockers.append("OPENAI_API_KEY is absent")
    blockers.extend(gpu_blockers())
    return blockers


def confirmation_blockers() -> list[str]:
    """Confirmation needs frozen search provenance/executors, but no API credential."""
    validate_static()
    blockers: list[str] = []
    model_resolution(blockers)
    executor_registry(blockers)
    provenance_blockers(blockers)
    blockers.extend(gpu_blockers())
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-ready", action="store_true")
    args = parser.parse_args()
    validate_static()
    print("static validation: PASS (4 lanes x 5 contexts x 3 checkpoints)")
    if not args.launch_ready:
        return 0
    blockers = launch_blockers()
    for blocker in blockers:
        print("BLOCKED:", blocker)
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
