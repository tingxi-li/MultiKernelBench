"""Append-only, hash-chained and effort-accounted trajectory events."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    from . import campaign
except ImportError:  # direct script execution
    import campaign  # type: ignore


GENESIS = "0" * 64
ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "trajectory_id",
        "event_index",
        "previous_event_sha256",
        "event_sha256",
        "completed_at_utc",
    }
)
CLOCK_KEYS = (
    "provider_wait_s",
    "controller_compute_s",
    "gpu_evaluation_s",
    "human_intervention_s",
)
EVENT_TYPES = frozenset(
    {
        "trajectory_start",
        "provider_request",
        "provider_response",
        "tuning_evaluation_start",
        "tuning_evaluation",
        "terminal_evaluation_start",
        "checkpoint_freeze",
        "trajectory_complete",
    }
)
USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
        "raw_categories",
    }
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def event_hash(event: dict[str, Any]) -> str:
    payload = {key: value for key, value in event.items() if key != "event_sha256"}
    return campaign.canonical_sha256(payload)


def clock_payload(
    cumulative: float,
    *,
    provider: float = 0.0,
    controller: float = 0.0,
    gpu: float = 0.0,
    human: float = 0.0,
) -> dict[str, float]:
    values = {
        "cumulative_active_effort_s": cumulative,
        "provider_wait_s": provider,
        "controller_compute_s": controller,
        "gpu_evaluation_s": gpu,
        "human_intervention_s": human,
    }
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("effort-clock values must be finite and nonnegative")
    return values


def _validate_usage(usage: Any, location: str) -> None:
    if not isinstance(usage, dict) or set(usage) != USAGE_KEYS:
        raise ValueError(f"{location}: provider usage categories differ from protocol")
    raw = usage["raw_categories"]
    if not isinstance(raw, dict) or any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in raw.items()
    ):
        raise ValueError(f"{location}: invalid raw token categories")
    for key in USAGE_KEYS - {"raw_categories"}:
        value = usage[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{location}: invalid token count {key}")
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise ValueError(f"{location}: total_tokens is inconsistent")


def _validate_semantics(event: dict[str, Any], previous_effort: float, location: str) -> None:
    event_type = event.get("event_type")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"{location}: unknown event_type {event_type!r}")
    effort = event.get("cumulative_active_effort_s")
    if (
        isinstance(effort, bool)
        or not isinstance(effort, (int, float))
        or not math.isfinite(float(effort))
        or effort < previous_effort
    ):
        raise ValueError(f"{location}: effort clock regressed or is non-finite")
    components = []
    for key in CLOCK_KEYS:
        value = event.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"{location}: invalid effort category {key}")
        components.append(float(value))
    delta = float(effort) - previous_effort
    if not math.isclose(delta, sum(components), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(
            f"{location}: effort delta {delta} differs from categorized {sum(components)}"
        )
    human = float(event["human_intervention_s"])
    intervention = event.get("human_intervention")
    if human > 0:
        if (
            not isinstance(intervention, dict)
            or not isinstance(intervention.get("description"), str)
            or not intervention["description"].strip()
            or not isinstance(intervention.get("actor"), str)
            or not intervention["actor"].strip()
        ):
            raise ValueError(f"{location}: nonzero human effort lacks actor/description")
    elif intervention is not None:
        raise ValueError(f"{location}: human intervention metadata has zero duration")
    timestamp = event.get("completed_at_utc")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise ValueError(f"{location}: missing explicit UTC completion timestamp")
    if event_type == "provider_response":
        _validate_usage(event.get("usage"), location)
    elif "usage" in event:
        raise ValueError(f"{location}: usage is only permitted on provider_response")
    iteration_events = EVENT_TYPES - {"trajectory_start", "trajectory_complete"}
    if event_type in iteration_events:
        iteration = event.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError(f"{location}: invalid iteration")


def load_events(path: Path, trajectory_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    previous = GENESIS
    previous_effort = 0.0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            location = f"{path}:{line_number}"
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"{location}: event is not an object")
            expected = {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "trajectory_id": trajectory_id,
                "event_index": len(rows),
                "previous_event_sha256": previous,
            }
            mismatches = [key for key, value in expected.items() if event.get(key) != value]
            if mismatches:
                raise ValueError(f"{location}: event-chain mismatch {mismatches}")
            digest = event_hash(event)
            if event.get("event_sha256") != digest:
                raise ValueError(f"{location}: event hash mismatch")
            _validate_semantics(event, previous_effort, location)
            rows.append(event)
            previous = digest
            previous_effort = float(event["cumulative_active_effort_s"])
    return rows


def append_event(path: Path, trajectory_id: str, event: dict[str, Any]) -> dict[str, Any]:
    forbidden = ENVELOPE_KEYS & set(event)
    if forbidden:
        raise ValueError(f"caller may not override event envelope: {sorted(forbidden)}")
    rows = load_events(path, trajectory_id)
    complete = {
        "schema_version": 1,
        "campaign_id": campaign.CAMPAIGN_ID,
        "trajectory_id": trajectory_id,
        "event_index": len(rows),
        "previous_event_sha256": rows[-1]["event_sha256"] if rows else GENESIS,
        "completed_at_utc": _utc_now(),
        **event,
    }
    previous_effort = float(rows[-1]["cumulative_active_effort_s"]) if rows else 0.0
    _validate_semantics(complete, previous_effort, str(path))
    complete["event_sha256"] = event_hash(complete)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(complete, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return complete
