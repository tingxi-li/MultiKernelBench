from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from ako_runs.controlled_followup.robust_gate.schema import canonical_sha256, load_json

from ..analyze import analyze_records, exact_interval, holm_adjust, quantile
from ..runner import (
    GATES,
    MANIFEST_PATH,
    POLICY_PATH,
    AppendOnlyJsonl,
    Plan,
    _base,
    seed_plan,
    threshold_failures,
    verify_campaign,
    verify_prior_seed_replay,
)


EXPECTED_CANDIDATES = [
    "tilelang.g08",
    "triton.g05",
    "cuda_noptx.g04",
    "cuda_unlimited.g04",
    "cuda_noptx_streamed.g05",
    "cuda_noptx_streamed.g08",
    "cuda_noptx_streamed.g09",
    "cuda_unlimited_streamed.g05",
    "cuda_unlimited_streamed.g06",
    "cuda_unlimited_streamed.g07",
]


class SameSeedV2Tests(unittest.TestCase):
    def test_exact_roster_seed_count_and_record_census(self) -> None:
        manifest, _gate, _lock, seeds = verify_campaign(require_freeze=False)
        self.assertEqual(manifest["candidate_order"], EXPECTED_CANDIDATES)
        self.assertEqual(len(manifest["candidates"]), 10)
        self.assertEqual(len(seeds), 512)
        self.assertEqual(manifest["workload"]["expected_records"], 10 * 512 * 2)
        self.assertEqual(manifest["seed_plan"]["effective_seed_count"], 512)

    def test_exact_prior_seed_replay_and_known_vectors(self) -> None:
        manifest, _gate, _lock, seeds = verify_campaign(require_freeze=False)
        verify_prior_seed_replay(manifest, seeds)
        self.assertEqual(seeds[186]["namespace"], "MKB-fused-row-sum-gain16-stress-v1-20260730")
        self.assertEqual(
            seeds[186]["tensor_seeds"],
            {"bias": 438066528671610939, "weight": 2733515708989840296, "x": 7928387594506867311},
        )
        self.assertEqual(
            seeds[197]["tensor_seeds"],
            {"bias": 8435418904273984620, "weight": 167026983864710939, "x": 489350623665119054},
        )
        self.assertEqual(seeds[256]["namespace"], "MKB-fused-same-seed-gain16-stress-v2-20260731")
        values = [value for seed in seeds for value in seed["tensor_seeds"].values()]
        self.assertEqual(len(values), len(set(values)))

    def test_v1_is_preserved_noncontrolling(self) -> None:
        manifest = load_json(MANIFEST_PATH)
        status = load_json(MANIFEST_PATH.with_name("legacy_v1_status.json"))
        self.assertFalse(status["controlling"])
        self.assertEqual(status["disposition"], "preserved_invalid_unlaunched_pilot")
        self.assertIn("MKB-fused-reachability-row-sum", status["reason"])
        self.assertEqual(status["measurement_files_observed"], [])
        self.assertEqual(manifest["legacy_v1_disposition_binding"]["canonical_sha256"], canonical_sha256(status))

    def test_inference_is_bound_and_does_not_inflate_gate_views(self) -> None:
        manifest = load_json(MANIFEST_PATH)
        policy = load_json(POLICY_PATH)
        self.assertEqual(policy["paired_contrasts"], manifest["paired_contrasts"])
        self.assertEqual(len(policy["paired_contrasts"]), 6)
        self.assertIn("512", policy["effective_unit"])
        self.assertIn("not independent", policy["effective_unit"])
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(quantile([0.0, 1.0, 2.0, 3.0], 0.5), 1.5)
        interval = exact_interval(0, 512)
        self.assertEqual(interval[0], 0.0)
        self.assertGreater(interval[1], 0.0)

    def test_no_threshold_fitting_or_performance_feedback(self) -> None:
        manifest = load_json(MANIFEST_PATH)
        self.assertFalse(manifest["fixed_threshold_policy"]["threshold_fitting_allowed"])
        self.assertFalse(manifest["fixed_threshold_policy"]["threshold_mutation_authorized"])
        self.assertFalse(manifest["performance_selection_feedback_authorized"])
        tree = ast.parse(MANIFEST_PATH.with_name("runner.py").read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any(name.endswith("calibrate") for name in imported))

    def test_append_only_partial_resume_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "measurements.jsonl"
            writer = AppendOnlyJsonl(output)
            row = {"candidate_id": "a", "gate_id": "g", "seed_index": 0}
            writer.write(row)
            writer.retain()
            resumed = AppendOnlyJsonl(output)
            self.assertIn(("a", "g", 0), resumed.keys)
            with self.assertRaises(ValueError):
                resumed.write(row)
            resumed.write({"candidate_id": "a", "gate_id": "g", "seed_index": 1})
            resumed.finish(2)
            self.assertTrue(output.is_file())
            self.assertFalse(output.with_name(output.name + ".partial").exists())

    def test_analyzer_requires_complete_bound_census_and_pairs_seeds(self) -> None:
        manifest, gate_spec, _lock, full_seeds = verify_campaign(require_freeze=False)
        seeds = full_seeds[:2]
        source_bundle = "unit-source-bundle"
        build_sha = "unit-build-receipt"
        records = []
        for candidate in manifest["candidates"]:
            plan = Plan(candidate, None, {})
            for seed in seeds:
                for gate_id in GATES:
                    gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
                    metrics = {name: 0.0 for name in gate["thresholds"]}
                    row = _base(
                        manifest,
                        {"source_bundle_canonical_sha256": source_bundle},
                        canonical_sha256(seeds),
                        plan,
                        gate_id,
                        seed,
                        build_sha,
                    )
                    row_sum = gate["thresholds"]["row_sum_error_max"]
                    row.update(
                        {
                            "ok": True,
                            "metrics": metrics,
                            "threshold_ratios": {name: 0.0 for name, rule in gate["thresholds"].items() if rule["value"] > 0},
                            "threshold_failures": threshold_failures(gate, metrics),
                            "gate_pass": True,
                            "registered_row_sum_threshold": row_sum["value"],
                            "raw_safety_cutoff": row_sum["observed_anchor_max"] * row_sum["safety_factor"],
                            "raw_safety_exceeded": False,
                        }
                    )
                    records.append(row)
        summary = analyze_records(
            manifest,
            gate_spec,
            records,
            source_bundle=source_bundle,
            build_receipt_sha256=build_sha,
            seed_rows=seeds,
        )
        self.assertTrue(summary["evidence_complete"])
        self.assertEqual(summary["effective_shared_seed_n"], 2)
        self.assertEqual(len(summary["paired_primary_joint_all_gates"]), 6)
        self.assertTrue(all(row["shared_seed_n"] == 2 for row in summary["paired_primary_joint_all_gates"]))
        incomplete = analyze_records(
            manifest,
            gate_spec,
            records[:-1],
            source_bundle=source_bundle,
            build_receipt_sha256=build_sha,
            seed_rows=seeds,
        )
        self.assertFalse(incomplete["evidence_complete"])


if __name__ == "__main__":
    unittest.main()

