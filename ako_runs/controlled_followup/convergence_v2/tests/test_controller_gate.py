from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ako_runs.controlled_followup.convergence_v2.controller import (
    BudgetedTrajectoryController,
    CompletedEvaluationClock,
    EventJournal,
    JournalError,
)
from ako_runs.controlled_followup.convergence_v2.hidden_gate import (
    CandidateRef,
    GateBindingError,
    SearcherGateFacade,
    TerminalEvaluator,
    TerminalGateDecision,
    TuningGateDecision,
)


class FakeTuning:
    dataset_sha256 = "1" * 64
    principal = "search_controller"
    threshold = 1e-9

    def evaluate(self, operation, candidate, seed):
        return TuningGateDecision(False, ("max_abs_err",), "decision-tuning")


class FakeTerminal:
    principal = "terminal_evaluator"

    def __init__(self, dataset_sha256="2" * 64):
        self.dataset_sha256 = dataset_sha256

    def evaluate_once(self, operation, candidate, seed):
        return TerminalGateDecision(True, "decision-terminal", self.dataset_sha256)


class Clock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class ControllerGateTest(unittest.TestCase):
    def test_journal_idempotency_charging_and_secret_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            journal = EventJournal(Path(temp) / "events.jsonl", "campaign", "trajectory")
            first = journal.append("start", "trajectory_started", {"gpu_slot": 0}, monotonic_ns=1)
            self.assertEqual(journal.append("start", "trajectory_started", {"gpu_slot": 0})["event_sha256"], first["event_sha256"])
            with self.assertRaises(JournalError):
                journal.append("start", "trajectory_started", {"gpu_slot": 1})
            clock = CompletedEvaluationClock(journal, Clock([1_000_000_000, 3_500_000_000]))
            clock.begin("a1", "build_and_gate")
            self.assertEqual(clock.finish("a1", "gate_failed"), 2.5)
            self.assertEqual(journal.total_charged_compute_s(), 2.5)
            with self.assertRaises(JournalError):
                journal.append("secret", "provider_usage", {"api_key": "never"})

    def test_incomplete_attempt_is_right_censored_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            journal = EventJournal(Path(temp) / "events.jsonl", "campaign", "trajectory")
            journal.append("attempt-start-a", "evaluation_started", {"attempt_id": "a", "kind": "build"})
            self.assertEqual(journal.resume_or_censor(), "censored")
            last = journal.read()[-1]
            self.assertEqual(last["event_type"], "trajectory_censored")
            self.assertTrue(last["payload"]["right_censored"])
            self.assertFalse(last["payload"]["replacement_permitted"])
            self.assertEqual(journal.resume_or_censor(), "terminal")

    def test_budget_crossing_is_charged_and_blocks_next_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            journal = EventJournal(Path(temp) / "events.jsonl", "campaign", "trajectory")
            controller = BudgetedTrajectoryController(journal, 1.0, Clock([0, 1_500_000_000]))
            controller.begin_attempt("a", "build")
            outcome = controller.finish_attempt("a", "build_failed")
            self.assertEqual(outcome["cumulative_compute_s"], 1.5)
            self.assertFalse(outcome["completed_within_budget"])
            with self.assertRaises(JournalError):
                controller.begin_attempt("b", "build")
            self.assertEqual(journal.read()[-1]["event_type"], "trajectory_censored")

    def test_hidden_gate_surface_and_terminal_separation(self) -> None:
        candidate = CandidateRef("candidate", "a" * 64, "b" * 64)
        visible = SearcherGateFacade(FakeTuning()).evaluate("matmul", candidate, 7)
        self.assertEqual(set(visible), {"passed", "failed_metric_names", "decision_id"})
        self.assertNotIn("threshold", visible)
        terminal = TerminalEvaluator(FakeTerminal(), tuning_dataset_sha256="1" * 64)
        self.assertTrue(terminal.evaluate_once("matmul", candidate, 8).passed)
        with self.assertRaises(GateBindingError):
            TerminalEvaluator(FakeTerminal("1" * 64), tuning_dataset_sha256="1" * 64)


if __name__ == "__main__":
    unittest.main()
