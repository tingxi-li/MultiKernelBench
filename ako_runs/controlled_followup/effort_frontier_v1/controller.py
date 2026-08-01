#!/usr/bin/env python3
"""Run one append-only, checkpointed effort-frontier search trajectory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from . import campaign, events, validate
except ImportError:  # direct script execution
    import campaign  # type: ignore
    import events  # type: ignore
    import validate  # type: ignore

try:
    from ako_runs.controlled_followup.convergence_v2.provider_adapters import (
        OpenAIResponsesAdapter,
        ProviderRequest,
    )
except ImportError:  # direct execution from the campaign directory
    import sys

    sys.path.insert(0, str(campaign.REPO_ROOT))
    from ako_runs.controlled_followup.convergence_v2.provider_adapters import (  # noqa: E402
        OpenAIResponsesAdapter,
        ProviderRequest,
    )


def _trajectory(identifier: str) -> dict[str, Any]:
    matches = [row for row in campaign.trajectories() if row["trajectory_id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"unknown trajectory: {identifier}")
    return matches[0]


def _parse_proposal(text: str) -> tuple[str, str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("model response must be one JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"source", "rationale"}:
        raise ValueError("model response must contain exactly source and rationale")
    source, rationale = value["source"], value["rationale"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("model response lacks nonempty source")
    if not isinstance(rationale, str):
        raise ValueError("model response lacks rationale")
    return source, rationale


def _candidate_path(event_path: Path, relative: str, digest: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise RuntimeError("candidate path is not a relative trajectory path")
    root = event_path.parent.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("candidate path escapes trajectory directory") from exc
    if not path.is_file() or campaign.file_sha256(path) != digest:
        raise RuntimeError("candidate bytes differ from the event binding")
    return path


def _normalize_result(
    result: Any,
    *,
    split: str,
    candidate_sha256: str,
    lane: str,
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("evaluator response is not an object")
    expected = {
        "schema_version": 1,
        "campaign_id": campaign.CAMPAIGN_ID,
        "trajectory_id": trajectory["trajectory_id"],
        "lane": lane,
        "evaluation_split": split,
        "candidate_sha256": candidate_sha256,
        "gate_spec_sha256": campaign.file_sha256(campaign.GATE_SPEC),
        "physical_gpu": trajectory["physical_gpu"],
        "logical_device": "cuda:0",
        "gpu_uuid": trajectory["required_gpu_uuid"],
    }
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    if mismatches:
        raise ValueError(f"evaluator response binding mismatch: {mismatches}")
    build_ok = result.get("build_ok")
    lane_policy_pass = result.get("lane_policy_pass")
    gate_pass = result.get("gate_pass")
    if not all(
        isinstance(value, bool) for value in (build_ok, lane_policy_pass, gate_pass)
    ):
        raise ValueError(
            "evaluator response lacks boolean build_ok/lane_policy_pass/gate_pass"
        )
    if gate_pass and not build_ok:
        raise ValueError("evaluator cannot pass the gate when its build failed")
    gate_summary = result.get("gate_summary_sha256")
    if gate_pass and (
        not isinstance(gate_summary, str)
        or re.fullmatch(r"[0-9a-f]{64}", gate_summary) is None
    ):
        raise ValueError("gate-passing evaluator result lacks a gate-summary SHA-256")
    eligible = bool(build_ok and lane_policy_pass and gate_pass)
    latency = result.get("median_ms")
    if eligible and (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or latency <= 0
    ):
        raise ValueError("eligible evaluator result lacks a finite positive median_ms")
    if not eligible:
        latency = None
    return {
        "build_ok": build_ok,
        "lane_policy_pass": lane_policy_pass,
        "gate_pass": gate_pass,
        "eligible": eligible,
        "median_ms": latency,
        "gate_summary_sha256": gate_summary,
        "executor_metadata": result.get("executor_metadata", {}),
        "error": result.get("error"),
        "gpu_uuid": result["gpu_uuid"],
    }


def _run_evaluator(
    command: list[str],
    *,
    timeout_s: int,
    split: str,
    candidate: Path,
    lane: str,
    trajectory: dict[str, Any],
) -> tuple[dict[str, Any], float, float]:
    candidate_sha = campaign.file_sha256(candidate)
    request = {
        "schema_version": 1,
        "campaign_id": campaign.CAMPAIGN_ID,
        "trajectory_id": trajectory["trajectory_id"],
        "lane": lane,
        "evaluation_split": split,
        "candidate_path": str(candidate.resolve()),
        "candidate_sha256": candidate_sha,
        "gate_spec": str(campaign.GATE_SPEC.resolve()),
        "gate_spec_sha256": campaign.file_sha256(campaign.GATE_SPEC),
        "physical_gpu": trajectory["physical_gpu"],
        "required_gpu_uuid": trajectory["required_gpu_uuid"],
        "logical_device": "cuda:0",
    }
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(trajectory["physical_gpu"])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request, sort_keys=True) + "\n",
            text=True,
            capture_output=True,
            env=environment,
            check=False,
            timeout=timeout_s,
        )
        duration = time.perf_counter() - started
    except (OSError, subprocess.TimeoutExpired) as exc:
        duration = time.perf_counter() - started
        controller_started = time.perf_counter()
        result = {
            "build_ok": False,
            "lane_policy_pass": False,
            "gate_pass": False,
            "eligible": False,
            "median_ms": None,
            "error": f"evaluator infrastructure error: {exc}",
        }
        return result, duration, time.perf_counter() - controller_started
    controller_started = time.perf_counter()
    if completed.returncode != 0:
        result = {
            "build_ok": False,
            "lane_policy_pass": False,
            "gate_pass": False,
            "eligible": False,
            "median_ms": None,
            "error": completed.stderr[-4000:] or f"evaluator exit {completed.returncode}",
        }
        return result, duration, time.perf_counter() - controller_started
    try:
        raw = json.loads(completed.stdout)
        normalized = _normalize_result(
            raw,
            split=split,
            candidate_sha256=candidate_sha,
            lane=lane,
            trajectory=trajectory,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        normalized = {
            "build_ok": False,
            "lane_policy_pass": False,
            "gate_pass": False,
            "eligible": False,
            "median_ms": None,
            "error": f"invalid evaluator response: {exc}",
        }
    return normalized, duration, time.perf_counter() - controller_started


def _event_payload(effort: float, **kwargs: Any) -> dict[str, Any]:
    clock = events.clock_payload(effort)
    clock.update(kwargs)
    return clock


def _best_candidate(rows: list[dict[str, Any]], event_path: Path) -> tuple[float, Path, str] | None:
    candidates = []
    for row in rows:
        if row.get("event_type") != "tuning_evaluation" or row.get("eligible") is not True:
            continue
        digest = row.get("candidate_sha256")
        relative = row.get("candidate_relative_path")
        if not isinstance(digest, str):
            raise RuntimeError("eligible event lacks candidate hash")
        path = _candidate_path(event_path, relative, digest)
        candidates.append((float(row["median_ms"]), path, digest))
    return min(candidates, default=None, key=lambda item: (item[0], item[2]))


def _start_bindings(registry: dict[str, Any], resolution: Any) -> dict[str, Any]:
    return {
        "manifest_sha256": campaign.file_sha256(campaign.MANIFEST),
        "gate_spec_sha256": campaign.file_sha256(campaign.GATE_SPEC),
        "gate_receipt_sha256": campaign.file_sha256(campaign.GATE_RECEIPT),
        "model_resolution_lock_sha256": campaign.file_sha256(campaign.MODEL_LOCK),
        "executor_registry_sha256": campaign.file_sha256(campaign.EXECUTOR_REGISTRY),
        "prelaunch_provenance_sha256": campaign.file_sha256(campaign.PROVENANCE_LOCK),
        "immutable_model_revision": resolution.immutable_revision,
        "executor_source_hashes": registry,
    }


def run(identifier: str, result_root: Path) -> None:
    trajectory = _trajectory(identifier)
    try:
        validate.validate_static()
    except (validate.ValidationError, OSError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"launch refused: static protocol validation failed: {exc}") from exc
    blockers: list[str] = []
    resolution = validate.model_resolution(blockers)
    registry = validate.executor_registry(blockers)
    validate.provenance_blockers(blockers)
    blockers.extend(validate.gpu_blockers())
    if not os.environ.get("OPENAI_API_KEY"):
        blockers.append("OPENAI_API_KEY is absent")
    if blockers:
        raise RuntimeError("launch refused: " + "; ".join(blockers))
    assert resolution is not None and registry is not None
    adapter = OpenAIResponsesAdapter.from_env(resolution)
    lane = trajectory["lane"]
    entry = registry["lanes"][lane]
    event_path = result_root.resolve() / identifier / "events.jsonl"
    rows = events.load_events(event_path, identifier)
    bindings = _start_bindings(entry["source_hashes"], resolution)
    if not rows:
        events.append_event(
            event_path,
            identifier,
            _event_payload(
                0.0,
                event_type="trajectory_start",
                search_seed=trajectory["search_seed"],
                lane=lane,
                replicate=trajectory["replicate"],
                physical_gpu=trajectory["physical_gpu"],
                required_gpu_uuid=trajectory["required_gpu_uuid"],
                adapter_audit=adapter.audit_metadata(),
                bindings=bindings,
            ),
        )
        rows = events.load_events(event_path, identifier)
    elif rows[0].get("bindings") != bindings:
        raise RuntimeError("trajectory start bindings differ from current frozen inputs")
    if any(row.get("event_type") == "trajectory_complete" for row in rows):
        if rows[-1].get("event_type") != "trajectory_complete":
            raise RuntimeError("events appear after trajectory_complete")
        return

    system_prompt = (campaign.HERE / "prompts/system.md").read_text(encoding="utf-8")
    task_prompt = (campaign.HERE / "prompts/task.md").read_text(encoding="utf-8")
    while True:
        rows = events.load_events(event_path, identifier)
        last = rows[-1]
        effort = float(last["cumulative_active_effort_s"])
        if last["event_type"] in {
            "provider_request",
            "tuning_evaluation_start",
            "terminal_evaluation_start",
        }:
            raise RuntimeError(
                f"unreconciled external action at event {last['event_index']}; "
                "refusing an automatic retry that could duplicate billed/GPU work"
            )

        # A durable provider response can be evaluated safely after a local crash.
        if last["event_type"] == "provider_response" and last.get("parse_error") is None:
            candidate = _candidate_path(
                event_path, last["candidate_relative_path"], last["candidate_sha256"]
            )
            iteration = int(last["iteration"])
            events.append_event(
                event_path,
                identifier,
                _event_payload(
                    effort,
                    event_type="tuning_evaluation_start",
                    iteration=iteration,
                    candidate_relative_path=last["candidate_relative_path"],
                    candidate_sha256=last["candidate_sha256"],
                ),
            )
            result, duration, evaluation_controller_s = _run_evaluator(
                entry["tuning_command"],
                timeout_s=entry["timeout_s"],
                split="tuning",
                candidate=candidate,
                lane=lane,
                trajectory=trajectory,
            )
            effort += duration + evaluation_controller_s
            events.append_event(
                event_path,
                identifier,
                {
                    **events.clock_payload(
                        effort, gpu=duration, controller=evaluation_controller_s
                    ),
                    "event_type": "tuning_evaluation",
                    "iteration": iteration,
                    "candidate_relative_path": last["candidate_relative_path"],
                    "candidate_sha256": last["candidate_sha256"],
                    **result,
                },
            )
            continue

        # Freeze every completed-effort checkpoint crossed by the last action.
        frozen = {
            row["checkpoint_label"]
            for row in rows
            if row.get("event_type") == "checkpoint_freeze"
        }
        crossed = next(
            (
                (label, seconds)
                for label, seconds in zip(campaign.CHECKPOINT_LABELS, campaign.CHECKPOINTS_S)
                if effort >= seconds and label not in frozen
            ),
            None,
        )
        if crossed is not None:
            label, target = crossed
            selection_started = time.perf_counter()
            best = _best_candidate(rows, event_path)
            selection_s = time.perf_counter() - selection_started
            effort += selection_s
            terminal = None
            terminal_s = 0.0
            terminal_controller_s = 0.0
            if best is not None:
                relative = best[1].relative_to(event_path.parent.resolve()).as_posix()
                iteration = max(
                    int(row["iteration"])
                    for row in rows
                    if row.get("event_type") == "tuning_evaluation"
                    and row.get("candidate_sha256") == best[2]
                )
                events.append_event(
                    event_path,
                    identifier,
                    _event_payload(
                        effort,
                        controller=selection_s,
                        event_type="terminal_evaluation_start",
                        iteration=iteration,
                        checkpoint_label=label,
                        candidate_relative_path=relative,
                        candidate_sha256=best[2],
                    ),
                )
                terminal, terminal_s, terminal_controller_s = _run_evaluator(
                    entry["terminal_holdout_command"],
                    timeout_s=entry["timeout_s"],
                    split="terminal_holdout",
                    candidate=best[1],
                    lane=lane,
                    trajectory=trajectory,
                )
                effort += terminal_s + terminal_controller_s
            terminal_eligible = bool(terminal and terminal.get("eligible") is True)
            checkpoint_controller_s = terminal_controller_s if best is not None else selection_s
            events.append_event(
                event_path,
                identifier,
                {
                    **events.clock_payload(
                        effort, gpu=terminal_s, controller=checkpoint_controller_s
                    ),
                    "event_type": "checkpoint_freeze",
                    "iteration": max(
                        (int(row["iteration"]) for row in rows if "iteration" in row),
                        default=-1,
                    ),
                    "checkpoint_label": label,
                    "checkpoint_target_active_effort_s": target,
                    "checkpoint_overshoot_s": effort - target,
                    "candidate_relative_path": (
                        best[1].relative_to(event_path.parent.resolve()).as_posix()
                        if best
                        else None
                    ),
                    "candidate_sha256": best[2] if best else None,
                    "tuning_median_ms": best[0] if best else None,
                    "terminal_holdout": terminal,
                    "selection_eligible": terminal_eligible,
                    "terminal_feedback_exposed_to_future_search": False,
                },
            )
            continue

        if effort >= campaign.CHECKPOINTS_S[-1]:
            if frozen != set(campaign.CHECKPOINT_LABELS):
                continue
            events.append_event(
                event_path,
                identifier,
                _event_payload(
                    effort,
                    event_type="trajectory_complete",
                    checkpoint_count=len(frozen),
                ),
            )
            return

        preparation_started = time.perf_counter()
        iteration = 1 + max(
            (
                int(row["iteration"])
                for row in rows
                if row.get("event_type") == "provider_request"
            ),
            default=-1,
        )
        feedback = _best_candidate(rows, event_path)
        feedback_text = (
            "No fused-v2-legal tuning candidate yet."
            if feedback is None
            else f"Current best tuning median is {feedback[0]:.9g} ms."
        )
        request_id = hashlib.sha256(
            f"{trajectory['search_seed']}|{identifier}|{iteration}".encode()
        ).hexdigest()
        request = ProviderRequest(
            system_prompt=system_prompt,
            user_prompt=(
                task_prompt
                + f"\nAssigned lane: {lane}. Independent context: {request_id}.\n"
                + feedback_text
                + '\nReturn JSON only: {"source":"...","rationale":"..."}.'
            ),
            stochastic_request_id=request_id,
            max_output_tokens=16384,
        )
        preparation_s = time.perf_counter() - preparation_started
        effort += preparation_s
        events.append_event(
            event_path,
            identifier,
            _event_payload(
                effort,
                controller=preparation_s,
                event_type="provider_request",
                iteration=iteration,
                request_id=request_id,
                immutable_model_revision=resolution.immutable_revision,
                feedback_candidate_sha256=feedback[2] if feedback else None,
                feedback_tuning_median_ms=feedback[0] if feedback else None,
                terminal_feedback_included=False,
                prompt_sha256=campaign.canonical_sha256(
                    {
                        "system_prompt": request.system_prompt,
                        "user_prompt": request.user_prompt,
                    }
                ),
            ),
        )
        provider_started = time.perf_counter()
        response = adapter.generate(request)
        provider_s = time.perf_counter() - provider_started
        controller_started = time.perf_counter()
        if response.resolved_model_revision != resolution.immutable_revision:
            raise RuntimeError("provider response revision differs from immutable lock")
        if response.stochastic_request_id != request_id:
            raise RuntimeError("provider response request binding differs")
        relative = digest = rationale = None
        try:
            source, rationale = _parse_proposal(response.text)
            candidate = event_path.parent / "candidates" / f"candidate_{iteration:05d}.txt"
            campaign.atomic_write(candidate, source.encode())
            relative = candidate.relative_to(event_path.parent).as_posix()
            digest = campaign.file_sha256(candidate)
            parse_error = None
        except ValueError as exc:
            parse_error = str(exc)
        controller_s = time.perf_counter() - controller_started
        effort += provider_s + controller_s
        events.append_event(
            event_path,
            identifier,
            {
                **events.clock_payload(effort, provider=provider_s, controller=controller_s),
                "event_type": "provider_response",
                "iteration": iteration,
                "request_id": request_id,
                "response_id": response.response_id,
                "resolved_model_revision": response.resolved_model_revision,
                "usage": response.usage.as_event_payload(),
                "parse_error": parse_error,
                "candidate_relative_path": relative,
                "candidate_sha256": digest,
                "rationale": rationale,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.trajectory, args.result_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
