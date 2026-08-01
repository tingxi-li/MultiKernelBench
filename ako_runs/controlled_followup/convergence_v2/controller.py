#!/usr/bin/env python3
"""Append-only event journal and exogenous completed-evaluation clock."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

try:
    from .campaign import canonical_json
except ImportError:  # direct script execution
    from campaign import canonical_json  # type: ignore


TERMINAL_EVENT_TYPES = {"trajectory_succeeded", "trajectory_censored", "trajectory_failed_closed"}
ATTEMPT_TERMINAL_STATUSES = {"success", "build_failed", "gate_failed", "timeout", "runtime_failed"}
SENSITIVE_KEYS = {"api_key", "authorization", "access_token", "secret", "password"}


class JournalError(RuntimeError):
    pass


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _check_no_secrets(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                raise JournalError(f"sensitive key rejected at {path}.{key}")
            _check_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_no_secrets(item, f"{path}[{index}]")


def _event_hash(event_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(event_without_hash)).hexdigest()


class EventJournal:
    """Hash-chained JSONL with semantic idempotency and fail-closed parsing."""

    def __init__(self, path: Path, campaign_id: str, trajectory_id: str):
        self.path = path
        self.campaign_id = campaign_id
        self.trajectory_id = trajectory_id

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text().splitlines()
            if any(not line.strip() for line in lines):
                raise JournalError("blank/partial event line")
            events = [json.loads(line) for line in lines]
        except (OSError, json.JSONDecodeError) as exc:
            raise JournalError(f"event journal is unreadable: {exc}") from exc
        previous = "0" * 64
        event_ids: set[str] = set()
        terminal_seen = False
        for sequence, event in enumerate(events):
            if event.get("schema_version") != 1:
                raise JournalError(f"event {sequence}: wrong schema_version")
            if event.get("campaign_id") != self.campaign_id or event.get("trajectory_id") != self.trajectory_id:
                raise JournalError(f"event {sequence}: campaign/trajectory binding mismatch")
            if event.get("sequence") != sequence or event.get("previous_event_sha256") != previous:
                raise JournalError(f"event {sequence}: broken sequence/hash link")
            supplied = event.get("event_sha256")
            unhashed = {key: value for key, value in event.items() if key != "event_sha256"}
            if supplied != _event_hash(unhashed):
                raise JournalError(f"event {sequence}: event hash mismatch")
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in event_ids:
                raise JournalError(f"event {sequence}: missing/duplicate event_id")
            if terminal_seen:
                raise JournalError(f"event {sequence}: event after terminal outcome")
            _check_no_secrets(event.get("payload", {}))
            event_ids.add(event_id)
            previous = supplied
            terminal_seen = event.get("event_type") in TERMINAL_EVENT_TYPES
        return events

    def append(
        self,
        event_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        _check_no_secrets(payload)
        events = self.read()
        for existing in events:
            if existing["event_id"] == event_id:
                if existing["event_type"] == event_type and existing["payload"] == payload:
                    return existing
                raise JournalError(f"idempotency conflict for event_id={event_id}")
        if events and events[-1]["event_type"] in TERMINAL_EVENT_TYPES:
            raise JournalError("cannot append after terminal outcome")
        previous = events[-1]["event_sha256"] if events else "0" * 64
        event: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "trajectory_id": self.trajectory_id,
            "sequence": len(events),
            "event_id": event_id,
            "event_type": event_type,
            "recorded_at_utc": _utc_now(),
            "monotonic_ns": time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
            "previous_event_sha256": previous,
            "payload": payload,
        }
        event["event_sha256"] = _event_hash(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        descriptor = os.open(self.path, flags, 0o600)
        try:
            pending = canonical_json(event)
            while pending:
                written = os.write(descriptor, pending)
                if written <= 0:
                    raise JournalError("short write to event journal")
                pending = pending[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.read()  # verify the durable chain before returning
        return event

    def total_charged_compute_s(self) -> float:
        total = sum(
            float(event["payload"].get("charged_compute_s", 0.0))
            for event in self.read()
            if event["event_type"] == "evaluation_finished"
        )
        if not math.isfinite(total) or total < 0:
            raise JournalError("invalid charged compute total")
        return total

    def resume_or_censor(self) -> str:
        events = self.read()
        if events and events[-1]["event_type"] in TERMINAL_EVENT_TYPES:
            return "terminal"
        started: dict[str, dict[str, Any]] = {}
        finished: set[str] = set()
        for event in events:
            if event["event_type"] == "evaluation_started":
                started[event["payload"]["attempt_id"]] = event
            elif event["event_type"] == "evaluation_finished":
                finished.add(event["payload"]["attempt_id"])
        incomplete = sorted(set(started) - finished)
        if incomplete:
            fingerprint = hashlib.sha256("\0".join(incomplete).encode()).hexdigest()[:20]
            self.append(
                f"resume-censor-{fingerprint}",
                "trajectory_censored",
                {
                    "reason": "incomplete_attempt_on_resume",
                    "incomplete_attempt_ids": incomplete,
                    "right_censored": True,
                    "replacement_permitted": False,
                    "charged_compute_s": self.total_charged_compute_s(),
                },
            )
            return "censored"
        return "resumable"


class CompletedEvaluationClock:
    """Charges monotonic build/evaluation time, including unsuccessful attempts."""

    def __init__(self, journal: EventJournal, monotonic_ns: Callable[[], int] = time.monotonic_ns):
        self.journal = journal
        self._monotonic_ns = monotonic_ns
        self._active: dict[str, tuple[int, str]] = {}
        self._resume_state = journal.resume_or_censor()

    def begin(self, attempt_id: str, kind: str) -> None:
        if self._resume_state != "resumable":
            raise JournalError(f"trajectory cannot evaluate after resume state={self._resume_state}")
        if attempt_id in self._active:
            raise JournalError(f"attempt already active: {attempt_id}")
        started = self._monotonic_ns()
        self.journal.append(
            f"attempt-start-{attempt_id}",
            "evaluation_started",
            {"attempt_id": attempt_id, "kind": kind},
            monotonic_ns=started,
        )
        self._active[attempt_id] = (started, kind)

    def finish(self, attempt_id: str, status: str, *, details: dict[str, Any] | None = None) -> float:
        if status not in ATTEMPT_TERMINAL_STATUSES:
            raise JournalError(f"unsupported attempt status: {status}")
        if attempt_id not in self._active:
            raise JournalError(f"attempt is not active: {attempt_id}")
        finished = self._monotonic_ns()
        started, kind = self._active.pop(attempt_id)
        elapsed_s = (finished - started) / 1_000_000_000
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise JournalError("non-monotonic completed-evaluation clock")
        payload = {
            "attempt_id": attempt_id,
            "kind": kind,
            "status": status,
            "charged_compute_s": elapsed_s,
            "charged_even_if_unsuccessful": True,
            "details": details or {},
        }
        self.journal.append(
            f"attempt-finish-{attempt_id}",
            "evaluation_finished",
            payload,
            monotonic_ns=finished,
        )
        return elapsed_s


class BudgetedTrajectoryController:
    """Prevent work after budget exhaustion and expose within-budget eligibility."""

    def __init__(
        self,
        journal: EventJournal,
        completed_evaluation_budget_s: float,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ):
        if not math.isfinite(completed_evaluation_budget_s) or completed_evaluation_budget_s <= 0:
            raise JournalError("completed-evaluation budget must be finite and positive")
        self.journal = journal
        self.budget_s = completed_evaluation_budget_s
        self.clock = CompletedEvaluationClock(journal, monotonic_ns)

    def begin_attempt(self, attempt_id: str, kind: str) -> None:
        if self.journal.total_charged_compute_s() >= self.budget_s:
            censor_at_resource_cap(
                self.journal,
                reason="completed_evaluation_budget_exhausted",
                cap_name="completed_evaluation_s",
            )
            raise JournalError("completed-evaluation budget is exhausted")
        self.clock.begin(attempt_id, kind)

    def finish_attempt(
        self,
        attempt_id: str,
        status: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> dict[str, float | bool]:
        charged = self.clock.finish(attempt_id, status, details=details)
        total = self.journal.total_charged_compute_s()
        return {
            "charged_compute_s": charged,
            "cumulative_compute_s": total,
            "completed_within_budget": total <= self.budget_s,
        }


def record_provider_usage(journal: EventJournal, request_id: str, result: Any) -> dict[str, Any]:
    """Record category counts and response identity, never prompt text or credentials."""
    payload = {
        "request_id": request_id,
        "response_id": result.response_id,
        "resolved_model_revision": result.resolved_model_revision,
        "sampling_seed_supported": result.sampling_seed_supported,
        "stochastic_request_id": result.stochastic_request_id,
        "usage": result.usage.as_event_payload(),
    }
    return journal.append(f"provider-usage-{request_id}", "provider_usage", payload)


def censor_at_resource_cap(journal: EventJournal, *, reason: str, cap_name: str) -> dict[str, Any]:
    return journal.append(
        f"resource-cap-{cap_name}",
        "trajectory_censored",
        {
            "reason": reason,
            "cap_name": cap_name,
            "right_censored": True,
            "replacement_permitted": False,
            "charged_compute_s": journal.total_charged_compute_s(),
        },
    )
