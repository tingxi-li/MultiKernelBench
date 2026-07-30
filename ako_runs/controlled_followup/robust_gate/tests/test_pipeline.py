from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path

from ako_runs.controlled_followup.robust_gate.calibrate import (
    calibrate_gate_spec,
    round_up_125,
)
from ako_runs.controlled_followup.robust_gate.collect import collect_records
from ako_runs.controlled_followup.robust_gate.schema import SchemaError, load_json
from ako_runs.controlled_followup.robust_gate.validate import validate_records


HERE = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = copy.deepcopy(load_json(HERE / "manifest_matmul_v3.json"))
        self.manifest["split_counts"]["calibration"] = 2
        self.manifest["split_counts"]["validation"] = 2

    def _calibration(self):
        records = collect_records(
            self.manifest,
            op="matmul",
            gate_id="semantic_q32",
            split="calibration",
            candidate="anchor",
            cpu_smoke=True,
        )
        return records, calibrate_gate_spec(
            self.manifest,
            records,
            selected_ops={"matmul"},
            selected_gates={"semantic_q32"},
        )

    def test_round_up_125(self) -> None:
        self.assertEqual(round_up_125(0), 0)
        self.assertEqual(round_up_125(1.0), 1.0)
        self.assertEqual(round_up_125(1.0000001), 2.0)
        self.assertEqual(round_up_125(0.021), 0.05)
        with self.assertRaises(ValueError):
            round_up_125(math.nan)

    def test_matmul_end_to_end_pass_and_negative_failure(self) -> None:
        _calibration, gate_spec = self._calibration()
        scaled_threshold = gate_spec["gates"]["matmul/semantic_q32"]["thresholds"][
            "scaled_rmse"
        ]["value"]
        self.assertTrue(math.isfinite(scaled_threshold))
        self.assertLess(scaled_threshold, 1.0)
        exact = collect_records(
            self.manifest,
            op="matmul",
            gate_id="semantic_q32",
            split="validation",
            candidate="exact",
            candidate_name="cpu-exact",
            cpu_smoke=True,
        )
        passing = validate_records(self.manifest, gate_spec, exact)
        self.assertTrue(passing["success"])
        self.assertEqual(passing["groups"][0]["n_failed_records"], 0)

        zeros = collect_records(
            self.manifest,
            op="matmul",
            gate_id="semantic_q32",
            split="validation",
            candidate="zeros",
            candidate_name="deliberately-wrong",
            cpu_smoke=True,
        )
        failing = validate_records(self.manifest, gate_spec, zeros)
        self.assertFalse(failing["success"])
        self.assertGreater(failing["groups"][0]["n_failed_records"], 0)

    def test_calibration_rejects_duplicates_and_candidate_rows(self) -> None:
        records, _gate = self._calibration()
        with self.assertRaises(SchemaError):
            calibrate_gate_spec(self.manifest, records + [records[0]])
        bad = copy.deepcopy(records)
        bad[0]["role"] = "candidate"
        with self.assertRaises(SchemaError):
            calibrate_gate_spec(self.manifest, bad)

    def test_validation_requires_complete_locked_coverage(self) -> None:
        _records, gate_spec = self._calibration()
        exact = collect_records(
            self.manifest,
            op="matmul",
            gate_id="semantic_q32",
            split="validation",
            candidate="exact",
            cpu_smoke=True,
        )
        incomplete = validate_records(self.manifest, gate_spec, exact[:-1])
        self.assertFalse(incomplete["success"])
        self.assertFalse(incomplete["groups"][0]["coverage_complete"])


if __name__ == "__main__":
    unittest.main()
