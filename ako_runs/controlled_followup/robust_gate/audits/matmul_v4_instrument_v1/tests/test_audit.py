from __future__ import annotations

import ast
import copy
import unittest

import torch

from ako_runs.controlled_followup.robust_gate.distributions import make_matmul_inputs
from ako_runs.controlled_followup.robust_gate.metrics import compute_metrics
from ako_runs.controlled_followup.robust_gate.oracles import contract_reference, semantic_reference

from ..analyze import analyze_records, expected_records
from ..bindings import (
    MANIFEST_PATH,
    AuditBindingError,
    assert_registered_gate,
    audit_manifest,
    audit_manifest_hashes,
    load_json,
    repo_path,
    threshold_failures,
    verify_original_v4,
)
from ..runner import (
    GATE_ORDER,
    MIXED_CONTRACT,
    _wrong_output,
    compute_matmul_metrics_precomputed,
    tensor_seeds,
)


class BindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = audit_manifest()
        self.gate = verify_original_v4(self.manifest)

    def test_original_hashes_and_fixed_gate(self) -> None:
        self.assertEqual(
            self.manifest["original_v4"]["gate_spec"]["canonical_sha256"],
            "838c21d696febb2e712d03eff120326375fd34c5169a3d9c4978a2f8b6160aca",
        )
        mutated = copy.deepcopy(self.gate)
        mutated["gates"]["matmul/semantic_q32"]["thresholds"]["max_abs_err"][
            "value"
        ] += 1.0
        with self.assertRaises(AuditBindingError):
            assert_registered_gate(self.manifest, mutated)

    def test_namespaces_are_fresh_and_disjoint(self) -> None:
        original_manifest = load_json(
            repo_path(self.manifest["original_v4"]["manifest"]["path"])
        )
        namespaces = {
            self.manifest["splits"]["real_contact"]["namespace"],
            self.manifest["splits"]["synthetic"]["namespace"],
            self.manifest["splits"]["structural_smoke"]["namespace"],
            *[
                block["namespace"]
                for block in self.manifest["splits"]["replication_blocks"]
            ],
        }
        self.assertEqual(len(namespaces), 7)
        self.assertNotIn(original_manifest["seed_namespace"], namespaces)

    def test_runner_has_no_threshold_fitting_import_or_call(self) -> None:
        runner_path = MANIFEST_PATH.with_name("runner.py")
        tree = ast.parse(runner_path.read_text(encoding="utf-8"))
        imports = []
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
        self.assertFalse(any(name.endswith("calibrate") for name in imports))
        self.assertNotIn("calibrate_gate", calls)


class MetricTests(unittest.TestCase):
    def test_precomputed_metrics_equal_canonical_metrics(self) -> None:
        manifest = audit_manifest()
        shape = manifest["cpu_test_shape"]
        for case in manifest["cases"]:
            seeds = tensor_seeds("metric-parity-v1", case["id"], 3)
            inputs = make_matmul_inputs(shape, case, seeds, "cpu")
            reference = semantic_reference("matmul", inputs)
            candidate = inputs["a"].float() @ inputs["b"].float()
            scale = inputs["a"].double().abs() @ inputs["b"].double().abs()
            self.assertEqual(
                compute_metrics("matmul", reference, candidate, inputs),
                compute_matmul_metrics_precomputed(reference, candidate, scale),
            )

    def test_paired_cancellation_exclusions_are_exact_and_pass(self) -> None:
        manifest = audit_manifest()
        gate = verify_original_v4(manifest)
        case = next(case for case in manifest["cases"] if case["id"] == "paired_cancellation")
        seeds = tensor_seeds("paired-exclusion-unit-v1", case["id"], 0)
        inputs = make_matmul_inputs(manifest["cpu_test_shape"], case, seeds, "cpu")
        scale = inputs["a"].double().abs() @ inputs["b"].double().abs()
        for gate_id in GATE_ORDER:
            reference = (
                contract_reference("matmul", inputs, MIXED_CONTRACT)
                if gate_id == "conformance_mixed"
                else semantic_reference("matmul", inputs)
            )
            self.assertEqual(torch.count_nonzero(reference).item(), 0)
            for control_id in ("zeros", "row_roll1", "column_roll1"):
                candidate = _wrong_output(control_id, gate_id, reference, inputs)
                metrics = compute_matmul_metrics_precomputed(reference, candidate, scale)
                self.assertEqual(
                    threshold_failures(gate["gates"][f"matmul/{gate_id}"], metrics), []
                )


class AnalyzerTests(unittest.TestCase):
    def _mini_campaign(self):
        manifest = copy.deepcopy(audit_manifest())
        for split in ("real_contact", "synthetic", "structural_smoke"):
            manifest["splits"][split]["seeds_per_case"] = 1
        for block in manifest["splits"]["replication_blocks"]:
            block["seeds_per_case"] = 1
        hashes = audit_manifest_hashes(manifest)
        gate_spec = verify_original_v4(audit_manifest())
        records = []
        for descriptor in expected_records(manifest).values():
            structural = descriptor["role"] == "structural_control"
            reject = descriptor["expected_outcome"] == "reject"
            frozen_gate = gate_spec["gates"][f"matmul/{descriptor['gate_id']}"]
            metrics = {name: 0.0 for name in frozen_gate["thresholds"]}
            if reject:
                metrics["max_abs_err"] = (
                    frozen_gate["thresholds"]["max_abs_err"]["value"] * 2.0
                )
            failures = threshold_failures(frozen_gate, metrics)
            record = {
                **descriptor,
                "record_type": "matmul_v4_fixed_threshold_audit_measurement",
                "campaign_id": manifest["campaign_id"],
                "audit_manifest_sha256": hashes["raw_sha256"],
                "audit_manifest_canonical_sha256": hashes["canonical_sha256"],
                "original_gate_sha256": manifest["original_v4"]["gate_spec"]["sha256"],
                "original_gate_canonical_sha256": manifest["original_v4"]["gate_spec"][
                    "canonical_sha256"
                ],
                "mode": "production",
                "shape": manifest["shape"],
                "tensor_seeds": tensor_seeds(
                    descriptor["namespace"], descriptor["case_id"], descriptor["seed_index"]
                ),
                "ok": not structural,
                "gate_pass": False if (structural or reject) else True,
                "metrics": metrics,
                "threshold_failures": failures,
            }
            if structural:
                record["error_category"] = "metric_preflight"
            if descriptor["role"] == "exact_zero_exclusion":
                record["reference_exact_zero"] = True
            records.append(record)
        return manifest, gate_spec, records

    def test_complete_casewise_campaign(self) -> None:
        manifest, gate, records = self._mini_campaign()
        summary = analyze_records(manifest, gate, records)
        self.assertTrue(summary["evidence_complete"])
        self.assertTrue(summary["endpoints"]["all_preregistered_endpoints_success"])
        self.assertEqual(summary["expected_records"], 282)

    def test_exact_exclusion_failure_is_not_counted_as_rejection_success(self) -> None:
        manifest, gate, records = self._mini_campaign()
        exclusion = next(row for row in records if row["role"] == "exact_zero_exclusion")
        frozen_gate = gate["gates"][f"matmul/{exclusion['gate_id']}"]
        exclusion["metrics"]["max_abs_err"] = (
            frozen_gate["thresholds"]["max_abs_err"]["value"] * 2.0
        )
        exclusion["gate_pass"] = False
        exclusion["threshold_failures"] = threshold_failures(
            frozen_gate, exclusion["metrics"]
        )
        summary = analyze_records(manifest, gate, records)
        self.assertTrue(summary["evidence_complete"])
        self.assertFalse(summary["endpoints"]["synthetic_discrimination_success"])


if __name__ == "__main__":
    unittest.main()
