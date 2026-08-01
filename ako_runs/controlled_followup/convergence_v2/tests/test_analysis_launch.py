from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ako_runs.controlled_followup.convergence_v2.analyze import (
    SurvivalObservation,
    holm_adjust,
    kaplan_meier,
    logrank_test,
    restricted_mean_survival_time,
)
from ako_runs.controlled_followup.convergence_v2.validate_launch import validate_launch_state


BASE = Path(__file__).resolve().parents[1]


class AnalysisLaunchTest(unittest.TestCase):
    def test_survival_fixtures(self) -> None:
        rows = [
            SurvivalObservation("a", "g", 1.0, True),
            SurvivalObservation("b", "g", 2.0, True),
            SurvivalObservation("c", "g", 3.0, False),
        ]
        curve = kaplan_meier(rows)
        self.assertAlmostEqual(float(curve[1]["survival"]), 2 / 3)
        self.assertAlmostEqual(restricted_mean_survival_time(rows, 3.0), 2.0)
        slow = [SurvivalObservation(f"s{i}", "slow", 4.0, False) for i in range(4)]
        fast = [SurvivalObservation(f"f{i}", "fast", float(i + 1), True) for i in range(4)]
        result = logrank_test(fast, slow)
        self.assertLess(result["p_value"], 0.05)
        self.assertEqual(holm_adjust({"a": 0.03, "b": 0.08, "c": 0.2}), {"a": 0.09, "b": 0.16, "c": 0.2})

    def test_checked_in_state_is_launch_blocked_for_real_reasons(self) -> None:
        result = validate_launch_state(BASE, environ={}, check_gpu_runtime=False, check_provider_sdks=False)
        self.assertFalse(result["ready"])
        by_name = {row["name"]: row for row in result["checks"]}
        self.assertTrue(by_name["trajectory_manifests"]["passed"])
        self.assertTrue(by_name["prompt_contract_lock"]["passed"])
        self.assertFalse(by_name["immutable_model_resolution"]["passed"])
        self.assertFalse(by_name["robust_hidden_gate_bindings"]["passed"])
        self.assertFalse(by_name["provider_credentials"]["passed"])

    def test_resolved_fixture_can_pass_cpu_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "campaign"
            shutil.copytree(BASE, target)
            model = json.loads((target / "locks/model_resolution_lock.json").read_text())
            model["state"] = "resolved"
            for index, row in enumerate(model["resolutions"]):
                row.update({"immutable_revision": row["requested_alias"] + "-immutable-r1", "provider_attested_immutable": True, "resolved_at_utc": "2026-07-31T12:00:00Z", "resolution_evidence_sha256": str(index + 1) * 64})
            (target / "locks/model_resolution_lock.json").write_text(json.dumps(model))
            uuids = [f"GPU-{index:032x}" for index in range(4)]
            gpu = json.loads((target / "locks/gpu_assignment_lock.json").read_text())
            gpu["state"] = "resolved"
            for row, uuid in zip(gpu["slots"], uuids):
                row["gpu_uuid"] = uuid
            (target / "locks/gpu_assignment_lock.json").write_text(json.dumps(gpu))
            gates = json.loads((target / "locks/gate_bindings.json").read_text())
            gates["state"] = "resolved"
            for index, binding in enumerate(gates["operations"].values()):
                binding["gate_state"] = "completed_frozen"
                binding["tuning"] = {"principal": "search_controller", "dataset_sha256": format(index + 1, "064x"), "service_endpoint": f"unix:///tmp/tuning-{index}.sock"}
                binding["terminal"] = {"principal": "terminal_evaluator", "dataset_sha256": format(index + 11, "064x"), "service_endpoint": f"unix:///tmp/terminal-{index}.sock"}
            (target / "locks/gate_bindings.json").write_text(json.dumps(gates))
            remote = json.loads((target / "locks/remote_preregistration_lock.json").read_text())
            remote.update({"state": "resolved", "commit_sha": "a" * 40, "remote_ref": "origin/frozen", "pushed_at_utc": "2026-07-31T12:00:00Z", "remote_receipt_sha256": "b" * 64})
            (target / "locks/remote_preregistration_lock.json").write_text(json.dumps(remote))
            refs = json.loads((target / "locks/reference_latency_lock.json").read_text())
            refs.update({"state": "frozen", "gpu_uuid": uuids[0]})
            refs["references"] = {op: {"latency_ms": 1.0, "terminal_gate_passed": True, "candidate_source_sha256": "c" * 64, "measurement_receipt_sha256": "d" * 64} for op in refs["required_operations"]}
            (target / "locks/reference_latency_lock.json").write_text(json.dumps(refs))
            result = validate_launch_state(target, environ={"OPENAI_API_KEY": "present", "ANTHROPIC_API_KEY": "present"}, check_gpu_runtime=False, check_provider_sdks=False)
            self.assertTrue(result["ready"], result)


if __name__ == "__main__":
    unittest.main()
