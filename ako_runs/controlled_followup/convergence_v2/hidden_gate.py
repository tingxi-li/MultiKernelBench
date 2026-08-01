#!/usr/bin/env python3
"""Interfaces that keep tuning feedback separate from terminal holdouts."""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Protocol


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GateBindingError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class CandidateRef:
    candidate_id: str
    source_sha256: str
    artifact_sha256: str

    def validate(self) -> None:
        if not self.candidate_id:
            raise GateBindingError("candidate_id is required")
        for name, value in (("source", self.source_sha256), ("artifact", self.artifact_sha256)):
            if SHA256_RE.fullmatch(value) is None:
                raise GateBindingError(f"invalid {name} SHA-256")


@dataclasses.dataclass(frozen=True)
class TuningGateDecision:
    passed: bool
    failed_metric_names: tuple[str, ...]
    decision_id: str

    def searcher_view(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failed_metric_names": list(self.failed_metric_names),
            "decision_id": self.decision_id,
        }


@dataclasses.dataclass(frozen=True)
class TerminalGateDecision:
    passed: bool
    decision_id: str
    terminal_dataset_sha256: str


class HiddenTuningGateService(Protocol):
    """Implemented outside the searcher's worktree/process."""

    dataset_sha256: str
    principal: str

    def evaluate(self, operation: str, candidate: CandidateRef, seed: int) -> TuningGateDecision: ...


class TerminalHoldoutService(Protocol):
    """Only the terminal evaluator may hold an implementation of this protocol."""

    dataset_sha256: str
    principal: str

    def evaluate_once(self, operation: str, candidate: CandidateRef, seed: int) -> TerminalGateDecision: ...


class SearcherGateFacade:
    """The only gate object supplied to a search trajectory."""

    def __init__(self, service: HiddenTuningGateService):
        if service.principal != "search_controller":
            raise GateBindingError("tuning service must be bound to search_controller")
        self.__service = service

    def evaluate(self, operation: str, candidate: CandidateRef, seed: int) -> dict[str, object]:
        candidate.validate()
        decision = self.__service.evaluate(operation, candidate, seed)
        # The return shape intentionally has no metrics, thresholds, inputs, or dataset identity.
        return decision.searcher_view()


class TerminalEvaluator:
    def __init__(self, service: TerminalHoldoutService, *, tuning_dataset_sha256: str):
        if service.principal != "terminal_evaluator":
            raise GateBindingError("terminal service must be bound to terminal_evaluator")
        if service.dataset_sha256 == tuning_dataset_sha256:
            raise GateBindingError("terminal and tuning datasets must be distinct")
        self.__service = service

    def evaluate_once(self, operation: str, candidate: CandidateRef, seed: int) -> TerminalGateDecision:
        candidate.validate()
        return self.__service.evaluate_once(operation, candidate, seed)


def validate_gate_bindings(path: Path, required_operations: set[str]) -> dict[str, dict]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateBindingError(f"cannot read gate binding lock: {exc}") from exc
    if document.get("schema_version") != 1 or document.get("state") != "resolved":
        raise GateBindingError("gate binding lock is not resolved")
    bindings = document.get("operations")
    if not isinstance(bindings, dict) or set(bindings) != required_operations:
        raise GateBindingError("gate binding operations do not match campaign operations")
    for operation, binding in bindings.items():
        if binding.get("gate_state") != "completed_frozen":
            raise GateBindingError(f"{operation}: robust gate is not completed and frozen")
        tuning = binding.get("tuning", {})
        terminal = binding.get("terminal", {})
        for label, item, principal in (
            ("tuning", tuning, "search_controller"),
            ("terminal", terminal, "terminal_evaluator"),
        ):
            if item.get("principal") != principal:
                raise GateBindingError(f"{operation}: wrong {label} principal")
            if SHA256_RE.fullmatch(str(item.get("dataset_sha256", ""))) is None:
                raise GateBindingError(f"{operation}: invalid {label} dataset hash")
            if not str(item.get("service_endpoint", "")).startswith("unix://"):
                raise GateBindingError(f"{operation}: {label} endpoint must be a local unix socket")
        if tuning["dataset_sha256"] == terminal["dataset_sha256"]:
            raise GateBindingError(f"{operation}: tuning and terminal datasets are identical")
    return bindings

