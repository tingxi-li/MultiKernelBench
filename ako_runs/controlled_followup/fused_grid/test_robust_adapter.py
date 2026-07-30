"""CPU-only tests for the fused-grid robust-gate adapter and launcher."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_robust_manifest  # noqa: E402
import robust_adapter as adapter  # noqa: E402
import robust_launch  # noqa: E402
from robust_gate.oracles import native_mixed_reference  # noqa: E402


class RobustAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = adapter.load_repository()
        cls.cpu_shape = cls.context.robust_manifest["operations"]["fused_softmax"][
            "cpu_test_shape"
        ]
        cls.contract = cls.context.gate_spec["gates"][
            "fused_softmax/conformance_mixed"
        ]["contract"]

    def plan(self, execute, *, real_job=False):
        job = self.context.jobs[0] if real_job else None
        return adapter.CandidatePlan(
            candidate=(
                adapter.candidate_name(self.context, job)
                if job is not None
                else "test:candidate"
            ),
            job=job,
            job_sha256=(
                self.context.adapter["grid"]["job_sha256"][job["job_id"]]
                if job is not None
                else None
            ),
            config=None,
            build_metadata={"test": True},
            execute=execute,
        )

    def evaluate(self, plan, *, split="tuning", case_id=None, index=0):
        case_id = case_id or self.context.cases[0]
        return adapter.evaluate_case_seed(
            self.context,
            [plan],
            case_id=case_id,
            split=split,
            seed_index=index,
            device="cpu",
            shape=self.cpu_shape,
        )[0][plan.candidate]

    def test_generated_manifest_and_grid_are_exact(self):
        expected = make_robust_manifest.expected_bytes()
        observed = adapter.DEFAULT_ADAPTER_MANIFEST.read_bytes()
        self.assertEqual(observed, expected)
        self.assertEqual(len(self.context.jobs), 76)
        self.assertEqual(self.context.adapter["robust_gate"]["split_counts"], {
            "tuning": 8,
            "validation": 64,
        })
        self.assertEqual(adapter.seed_indices(self.context, "tuning"), tuple(range(8)))
        self.assertEqual(adapter.seed_indices(self.context, "validation"), tuple(range(64)))

    def test_candidate_runs_once_and_scores_both_frozen_gates(self):
        calls = []

        def execute(inputs, _prepared):
            calls.append(1)
            return native_mixed_reference("fused_softmax", inputs, self.contract)

        rows = self.evaluate(self.plan(execute))
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [row["gate_id"] for row in rows],
            ["semantic_mixed", "conformance_mixed"],
        )
        self.assertTrue(all(row["ok"] for row in rows))
        self.assertTrue(all(row["manifest_sha256"] == self.context.manifest_sha256 for row in rows))
        self.assertEqual(rows[0]["tensor_seeds"], rows[1]["tensor_seeds"])
        self.assertEqual(
            rows[0]["output_shape"], [self.cpu_shape["M"], self.cpu_shape["N"]]
        )

    def test_computable_wrong_output_fails_gate_despite_ok(self):
        def execute(inputs, _prepared):
            return torch.zeros(
                (self.cpu_shape["M"], self.cpu_shape["N"]), dtype=torch.float32
            )

        rows = self.evaluate(self.plan(execute))
        self.assertTrue(all(row["ok"] for row in rows))
        self.assertTrue(all(not row["gate_pass"] for row in rows))
        self.assertTrue(all(row["threshold_failures"] for row in rows))

    def test_runtime_failure_is_retained_for_both_gates(self):
        def execute(_inputs, _prepared):
            raise RuntimeError("deliberate candidate failure")

        rows = self.evaluate(self.plan(execute))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not row["ok"] for row in rows))
        self.assertTrue(all(not row["gate_pass"] for row in rows))
        self.assertTrue(all("deliberate candidate failure" in row["error"] for row in rows))

    def test_partial_validation_fails_closed(self):
        def execute(inputs, _prepared):
            return native_mixed_reference("fused_softmax", inputs, self.contract)

        plan = self.plan(execute)
        rows = self.evaluate(plan, split="validation")
        summary = adapter.summarize_validation(
            self.context,
            rows,
            candidates=[plan.candidate],
            case_ids=[self.context.cases[0]],
            indices=[0],
        )
        self.assertFalse(summary["success"])
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(summary["launch_coverage"]["complete"])
        self.assertEqual(summary["grid_jobs_sha256"], self.context.adapter["grid"]["jobs_sha256"])
        self.assertTrue(all(not group["coverage_complete"] for group in summary["groups"]))

    def test_atomic_bundle_binding_detects_foreign_content(self):
        def execute(inputs, _prepared):
            return native_mixed_reference("fused_softmax", inputs, self.contract)

        plan = self.plan(execute, real_job=True)
        rows = self.evaluate(plan)
        binding = robust_launch.bundle_binding(
            self.context, plan.job, "tuning", self.context.cases[0], 0
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            robust_launch.write_bundle(path, binding, rows)
            state, _ = robust_launch.read_bundle_state(path, binding)
            self.assertIn(state, ("complete", "failed"))
            foreign = dict(binding)
            foreign["grid_job_sha256"] = "0" * 64
            state, _ = robust_launch.read_bundle_state(path, foreign)
            self.assertEqual(state, "foreign")


if __name__ == "__main__":
    unittest.main()
