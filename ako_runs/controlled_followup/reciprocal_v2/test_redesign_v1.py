from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent


class ReciprocalRedesignTest(unittest.TestCase):
    def test_redesign_is_single_translator_literal_controlling_and_budgeted(self):
        policy = json.loads((HERE / "redesign_v1.json").read_text())
        design = policy["default_design"]
        self.assertEqual(
            design["cell_count"],
            len(design["origins"])
            * len(design["destinations"])
            * len(design["transfer_modes"])
            * len(design["translators"]),
        )
        self.assertEqual(design["cell_count"], 24)
        self.assertEqual(policy["analysis_policy"]["controlling_interaction_estimand"],
                         "origin_by_destination_within_literal_arm")
        self.assertEqual(policy["analysis_policy"]["retuned_arm_role"], "descriptive_only")
        self.assertFalse(policy["analysis_policy"]["retuned_null_may_support_compiler_effect_claim"])
        budget = policy["total_budget"]
        expected_attempts = (
            budget["literal_cells"] * budget["literal_attempts_per_cell"]
            + budget["retuned_cells"] * budget["retuned_attempts_per_cell"]
        )
        self.assertEqual(budget["audit_candidate_attempts_ceiling"], expected_attempts)
        self.assertEqual(budget["screen_measurement_records_ceiling"], expected_attempts * 2)
        self.assertEqual(budget["primary_measurement_records"], 24 * 15)
        self.assertEqual(budget["post_audit_timing_records_ceiling"], 480 + 360)
        self.assertEqual(budget["total_enumerated_execution_records_ceiling"],
                         expected_attempts + 480 + 360)

    def test_historical_freeze_is_still_bound_and_launcher_refuses_without_runner(self):
        policy = json.loads((HERE / "redesign_v1.json").read_text())
        freeze = HERE / policy["historical_parent"]["source_freeze_path"]
        import hashlib

        self.assertEqual(hashlib.sha256(freeze.read_bytes()).hexdigest(),
                         policy["historical_parent"]["source_freeze_sha256"])
        protocol = types.SimpleNamespace(
            STAGES=("audit", "screen", "primary"), KINDS=("audit", "primary")
        )
        validate = types.SimpleNamespace(
            validate_static=lambda: {"manifests": {"audit": {"jobs": []}}},
            dependency_blockers=lambda _stage: [],
            gpu_blocker=lambda: None,
        )
        spec = importlib.util.spec_from_file_location("reciprocal_v2_frozen_launch", HERE / "launch.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        launch = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"protocol": protocol, "validate": validate}):
            spec.loader.exec_module(launch)
            output = io.StringIO()
            with mock.patch.object(sys, "argv", ["launch.py", "--stage", "audit"]), \
                 mock.patch.object(launch.subprocess, "run") as execute, redirect_stdout(output):
                self.assertEqual(launch.main(), 2)
            execute.assert_not_called()
            self.assertIn("no explicit --runner was supplied", output.getvalue())
            self.assertIn("REFUSED: no GPU process started", output.getvalue())


if __name__ == "__main__":
    unittest.main()
