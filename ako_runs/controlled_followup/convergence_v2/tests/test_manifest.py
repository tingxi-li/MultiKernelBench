from __future__ import annotations

import json
import unittest
from pathlib import Path

from ako_runs.controlled_followup.convergence_v2.campaign import build_manifests
from ako_runs.controlled_followup.convergence_v2.make_manifest import expected_outputs


BASE = Path(__file__).resolve().parents[1]


class ManifestTest(unittest.TestCase):
    def test_exact_census_models_and_balance(self) -> None:
        core, prompt = build_manifests()
        self.assertEqual((len(core), len(prompt), len(core + prompt)), (192, 128, 320))
        self.assertEqual(
            {(row["provider"], row["requested_model_alias"]) for row in core + prompt},
            {("openai", "gpt-5.6-sol"), ("anthropic", "claude-opus-4.8")},
        )
        cells: dict[tuple, list[dict]] = {}
        for row in core + prompt:
            key = (row["operation"], row["dsl"], row["model_key"], row["prompt_arm"])
            cells.setdefault(key, []).append(row)
        self.assertEqual(len(cells), 40)
        for rows in cells.values():
            self.assertEqual(sorted(row["replicate"] for row in rows), list(range(8)))
            self.assertEqual({slot: sum(row["gpu_slot"] == slot for row in rows) for slot in range(4)}, {0: 2, 1: 2, 2: 2, 3: 2})

    def test_compute_budget(self) -> None:
        core, prompt = build_manifests()
        seconds = sum(row["budget"]["completed_evaluation_s"] for row in core + prompt)
        self.assertEqual(seconds, 203_520)
        self.assertAlmostEqual(seconds / 3600, 56.53333333333333)

    def test_frozen_outputs_are_deterministic(self) -> None:
        for name, data in expected_outputs().items():
            self.assertEqual((BASE / "manifests" / name).read_bytes(), data)

    def test_gate_preregistrations_do_not_claim_completion(self) -> None:
        sum_gate = json.loads((BASE / "gates/sum_gate_preregistration.json").read_text())
        sdpa_gate = json.loads((BASE / "gates/sdpa_gate_preregistration.json").read_text())
        self.assertEqual(sum_gate["freeze_state"], "preregistered_not_executed")
        self.assertEqual(len(sum_gate["cases"]), 6)
        self.assertEqual(sum_gate["seed_plan"]["validation_seeds_per_case"], 512)
        self.assertIsNone(sum_gate["threshold_policy"].get("thresholds"))
        self.assertEqual(sdpa_gate["freeze_state"], "preregistered_not_executed")
        self.assertEqual(len(sdpa_gate["cases"]), 6)
        self.assertEqual(sdpa_gate["seed_plan"]["calibration_seeds_per_case"], 32)
        self.assertEqual(sdpa_gate["seed_plan"]["validation_seeds_per_case"], 64)


if __name__ == "__main__":
    unittest.main()

