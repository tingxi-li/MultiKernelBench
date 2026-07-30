from __future__ import annotations

import ast
import copy
import unittest

from ako_runs.controlled_followup.robust_gate.schema import file_sha256, load_json

from ..analyze import analyze_records
from ..runner import (
    MANIFEST_PATH,
    GATES,
    _seed_map,
    run_boundary,
    repo_path,
    threshold_failures,
    verify_campaign,
)


class ListWriter:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(row)


class StressTests(unittest.TestCase):
    def test_boundary_brackets_registered_rounding_gap(self) -> None:
        manifest, gate = verify_campaign(require_freeze=False)
        writer = ListWriter()
        run_boundary(manifest, gate, writer, "unit-test")
        self.assertEqual(len(writer.rows), 6)
        by_id = {(row["candidate"], row["gate_id"]): row for row in writer.rows}
        for gate_id in GATES:
            self.assertTrue(by_id[("below_raw_safety_4p5e7", gate_id)]["gate_pass"])
            gap = by_id[("rounding_gap_4p7e7", gate_id)]
            self.assertTrue(gap["gate_pass"])
            self.assertTrue(gap["raw_safety_exceeded"])
            above = by_id[("above_registered_5p1e7", gate_id)]
            self.assertFalse(above["gate_pass"])
            self.assertEqual(above["failure_metrics"], ["row_sum_error_max"])

    def test_fresh_namespace_and_exact_record_count(self) -> None:
        manifest, _ = verify_campaign(require_freeze=False)
        self.assertEqual(manifest["stress_split"]["seeds"], 256)
        self.assertEqual(manifest["workloads"][1]["expected_records"], 2048)
        original = load_json(
            repo_path(manifest["original_fused_v2"]["manifest"]["path"])
        )
        self.assertNotEqual(
            manifest["stress_split"]["namespace"], original["seed_namespace"]
        )

    def test_runner_does_not_fit_thresholds(self) -> None:
        tree = ast.parse(MANIFEST_PATH.with_name("runner.py").read_text(encoding="utf-8"))
        imports = []
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Call):
                calls.append(getattr(node.func, "id", getattr(node.func, "attr", "")))
        self.assertFalse(any(name.endswith("calibrate") for name in imports))
        self.assertNotIn("calibrate_gate", calls)

    def test_analyzer_requires_every_winner_gate_seed(self) -> None:
        original, gate = verify_campaign(require_freeze=False)
        manifest = copy.deepcopy(original)
        manifest["stress_split"]["seeds"] = 2
        boundary_writer = ListWriter()
        run_boundary(manifest, gate, boundary_writer, "unit-test")
        winners = []
        for winner in manifest["winners"]:
            for gate_id in GATES:
                frozen = gate["gates"][f"fused_softmax/{gate_id}"]
                metrics = {name: 0.0 for name in frozen["thresholds"]}
                for index in range(2):
                    winners.append(
                        {
                            "candidate": winner["job_id"],
                            "gate_id": gate_id,
                            "seed_index": index,
                            "case_id": manifest["case"]["id"],
                            "namespace": manifest["stress_split"]["namespace"],
                            "tensor_seeds": _seed_map(
                                manifest["stress_split"]["namespace"],
                                manifest["case"]["id"],
                                index,
                            ),
                            "job_sha256": winner["job_sha256"],
                            "stress_manifest_sha256": file_sha256(MANIFEST_PATH),
                            "original_gate_sha256": manifest["original_fused_v2"][
                                "gate_spec"
                            ]["sha256"],
                            "metrics": dict(metrics),
                            "threshold_failures": [],
                            "ok": True,
                            "gate_pass": True,
                            "raw_safety_exceeded": False,
                        }
                    )
        summary = analyze_records(
            manifest, gate, boundary_writer.rows, winners
        )
        self.assertTrue(summary["evidence_complete"])
        self.assertTrue(summary["all_winner_groups_success"])
        winners.pop()
        incomplete = analyze_records(manifest, gate, boundary_writer.rows, winners)
        self.assertFalse(incomplete["evidence_complete"])


if __name__ == "__main__":
    unittest.main()
