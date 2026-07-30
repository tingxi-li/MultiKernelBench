from __future__ import annotations

import ast
import copy
import unittest

from ako_runs.controlled_followup.robust_gate.schema import canonical_sha256, load_json

from ..analyze import analyze_records
from ..runner import (
    GATES,
    MANIFEST_PATH,
    Plan,
    _base,
    repo_path,
    seed_plan,
    threshold_failures,
    verify_campaign,
)


class ReachabilityStressTests(unittest.TestCase):
    def test_exact_six_screen_selected_candidates_are_bound(self) -> None:
        manifest, _gate, lock, _seeds = verify_campaign(require_freeze=False)
        self.assertEqual(len(manifest["selected_candidates"]), 6)
        self.assertEqual(len({row["job_id"] for row in manifest["selected_candidates"]}), 6)
        for row in manifest["selected_candidates"]:
            self.assertEqual(lock["job_sha256"][row["job_id"]], row["job_sha256"])

    def test_exact_256_fresh_seed_tuples_are_disjoint(self) -> None:
        manifest, _gate, _lock, seeds = verify_campaign(require_freeze=False)
        self.assertEqual(len(seeds), 256)
        values = [value for row in seeds for value in row["tensor_seeds"].values()]
        self.assertEqual(len(values), 768)
        self.assertEqual(len(set(values)), 768)
        self.assertEqual(seed_plan(manifest), seeds)
        self.assertEqual(canonical_sha256(seed_plan(manifest)), canonical_sha256(seeds))

    def test_no_threshold_fitting_or_performance_feedback(self) -> None:
        manifest = load_json(MANIFEST_PATH)
        self.assertTrue(manifest["correctness_only"])
        self.assertFalse(manifest["performance_selection_policy"]["feedback_authorized"])
        self.assertFalse(manifest["fixed_threshold_policy"]["threshold_fitting_allowed"])
        tree = ast.parse(MANIFEST_PATH.with_name("runner.py").read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
        self.assertFalse(any(name.endswith("calibrate") for name in names))

    def test_analyzer_requires_every_candidate_gate_seed(self) -> None:
        manifest, gate, _lock, full_seeds = verify_campaign(require_freeze=False)
        seeds = full_seeds[:2]
        seed_hash = canonical_sha256(seeds)
        bundle = "unit-test-source-bundle"
        build_sha = "unit-test-build-receipt"
        records = []
        for candidate in manifest["selected_candidates"]:
            plan = Plan(candidate, None, {})
            for seed in seeds:
                for gate_id in GATES:
                    frozen = gate["gates"][f"fused_softmax/{gate_id}"]
                    metrics = {name: 0.0 for name in frozen["thresholds"]}
                    row = _base(
                        manifest,
                        {"source_bundle_canonical_sha256": bundle},
                        seed_hash,
                        plan,
                        gate_id,
                        seed["seed_index"],
                        seed["tensor_seeds"],
                        build_sha,
                    )
                    threshold = frozen["thresholds"]["row_sum_error_max"]
                    row.update(
                        {
                            "ok": True,
                            "metrics": metrics,
                            "threshold_failures": threshold_failures(frozen, metrics),
                            "gate_pass": True,
                            "registered_row_sum_threshold": threshold["value"],
                            "raw_safety_cutoff": threshold["observed_anchor_max"] * threshold["safety_factor"],
                            "raw_safety_exceeded": False,
                        }
                    )
                    records.append(row)
        summary = analyze_records(
            manifest,
            gate,
            records,
            audit_source_bundle=bundle,
            build_receipt_sha256=build_sha,
            seed_rows=seeds,
        )
        self.assertTrue(summary["evidence_complete"])
        self.assertTrue(summary["all_candidate_gate_groups_success"])
        incomplete = analyze_records(
            manifest,
            gate,
            records[:-1],
            audit_source_bundle=bundle,
            build_receipt_sha256=build_sha,
            seed_rows=seeds,
        )
        self.assertFalse(incomplete["evidence_complete"])

    def test_registered_threshold_is_unchanged(self) -> None:
        manifest, gate, _lock, _seeds = verify_campaign(require_freeze=False)
        for gate_id in GATES:
            value = gate["gates"][f"fused_softmax/{gate_id}"]["thresholds"]["row_sum_error_max"]["value"]
            self.assertEqual(value, manifest["registered_gate_binding"]["registered_row_sum_threshold"])


if __name__ == "__main__":
    unittest.main()
