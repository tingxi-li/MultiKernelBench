#!/usr/bin/env python3
"""CPU-only structural tests for the reciprocal-transfer scaffold."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bind_gate
import launch
import make_manifests
import validate


def fake_gate_documents(root: Path) -> tuple[Path, Path, Path, Path, dict]:
    gate_path = root / "gate_spec.json"
    summary_path = root / "summary.json"
    receipt_path = root / "receipt.json"
    raw_path = root / "raw.jsonl"
    spec = {
        "schema_version": "1.0",
        "campaign_id": "cpu-test-gate",
        "manifest_sha256": "a" * 64,
        "calibration_records_sha256": "b" * 64,
        "gates": {
            "matmul.primary": {
                "op": "matmul",
                "gate_id": "primary",
                "anchors": ["anchor:a", "anchor:b"],
                "required_cases": ["signed", "legacy"],
                "calibration_records": 2560,
                "required_validation_seeds_per_case": 512,
            }
        },
    }
    gate_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
    summary = {
        "schema_version": "1.0",
        "campaign_id": spec["campaign_id"],
        "manifest_sha256": spec["manifest_sha256"],
        "gate_spec_sha256": validate.canonical_sha256(spec),
        "success_rule": "all_metrics_all_cases_all_seeds_and_complete_coverage",
        "success": True,
        "failures": [],
        "groups": [
            {
                "op": "matmul",
                "gate_id": "primary",
                "success": True,
                "coverage_complete": True,
                "n_failed_records": 0,
                "missing_records": 0,
                "n_records": 1024,
                "observed_failure_rate": 0.0,
            }
        ],
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    raw_path.write_bytes(b"{}\n" * 1024)
    receipt = {
        "schema_version": "1.0",
        "campaign_id": spec["campaign_id"],
        "acceptance_rule": summary["success_rule"],
        "accepted_for_reciprocal_recipe_transfer": True,
        "manifest_canonical_sha256": spec["manifest_sha256"],
        "frozen_gate": {
            "path": "gate_spec.json",
            "canonical_sha256": validate.canonical_sha256(spec),
            "file_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        },
        "validation": {
            "records": 1024,
            "failed_records": 0,
            "collection_failures": 0,
            "coverage_complete": True,
            "summary": {
                "path": "summary.json",
                "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "success": True,
            },
            "raw_evidence": [
                {
                    "gate_id": "primary",
                    "path": "raw.jsonl",
                    "records": 1024,
                    "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                }
            ],
        },
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return gate_path, summary_path, receipt_path, raw_path, spec


def patch_gate_paths(
    root: Path, gate_path: Path, summary_path: Path, receipt_path: Path
):
    return mock.patch.multiple(
        validate,
        REPO_ROOT=root.resolve(),
        GATE_SPEC=gate_path.resolve(),
        VALIDATION_SUMMARY=summary_path.resolve(),
        ACCEPTANCE_RECEIPT=receipt_path.resolve(),
    )


class ReciprocalScaffoldTest(unittest.TestCase):
    def test_generated_documents_are_current(self):
        make_manifests.check_documents()
        expected = make_manifests.build_documents()
        self.assertEqual(len(expected), 5)
        for path, data in expected.items():
            self.assertEqual(path.read_bytes(), data)

    def test_cards_are_source_grounded(self):
        docs = validate.validate_static()
        tile = docs["cards"]["tilelang_phase1_confirmed"]
        triton = docs["cards"]["triton_grouped_autotuned"]
        self.assertEqual(
            tile["literal_treatment"]["configs"],
            [make_manifests.tilelang_confirmed_config()],
        )
        self.assertEqual(tile["literal_treatment"]["configs"][0]["kc"], 2048)
        self.assertEqual(
            triton["literal_treatment"]["configs"],
            make_manifests.triton_native_configs(),
        )
        self.assertEqual(len(triton["literal_treatment"]["configs"]), 13)
        self.assertTrue(
            all(
                point["group_m"] == 8
                for point in triton["literal_treatment"]["configs"]
            )
        )

    def test_primary_and_audit_are_complete_factorials(self):
        docs = validate.validate_static()
        expected = [
            (origin, destination, mode)
            for origin in make_manifests.ORIGINS
            for destination in make_manifests.DESTINATIONS
            for mode in make_manifests.TRANSFER_MODES
        ]
        for kind in ("primary", "audit"):
            manifest = docs["manifests"][kind]
            got = [
                (
                    job["recipe_origin"],
                    job["destination_dsl"],
                    job["transfer_mode"],
                )
                for job in manifest["jobs"]
            ]
            self.assertEqual(got, expected)
            self.assertEqual(len(got), 16)

    def test_generated_dependencies_target_the_accepted_v4_gate(self):
        docs = validate.validate_static()
        expected = make_manifests.GATE_SPEC_RELATIVE
        self.assertEqual((HERE / expected).resolve(), validate.GATE_SPEC.resolve())
        for card in docs["cards"].values():
            self.assertEqual(card["gate_dependency"]["gate_spec"], expected)
            self.assertEqual(
                card["gate_dependency"]["validation_summary"],
                make_manifests.GATE_VALIDATION_SUMMARY_RELATIVE,
            )
            self.assertEqual(
                card["gate_dependency"]["acceptance_receipt"],
                make_manifests.GATE_ACCEPTANCE_RECEIPT_RELATIVE,
            )
        for manifest in docs["manifests"].values():
            self.assertEqual(
                manifest["robust_gate_dependency"]["gate_spec"], expected
            )

    def test_legacy_v3_gate_is_preserved_and_not_launchable(self):
        legacy = validate.GATE_SPEC.parent.parent / "gate_spec.json"
        self.assertEqual(
            hashlib.sha256(legacy.read_bytes()).hexdigest(),
            "48b987bd052b81c20d9b47cdba8b4693c2e1130ae46d4d5ab5147212b1b1ce34",
        )
        with self.assertRaises(validate.ValidationError):
            bind_gate.build_lock(
                legacy,
                validate.canonical_sha256(validate.load_json(legacy)),
                validate.VALIDATION_SUMMARY,
                validate.ACCEPTANCE_RECEIPT,
            )

    def test_missing_gate_is_an_explicit_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blockers = validate.gate_dependency_blockers(
                root / "gate_spec.json",
                root / "gate_lock.json",
                root / "summary.json",
                root / "receipt.json",
            )
        self.assertEqual(len(blockers), 4)
        self.assertTrue(any("not frozen" in blocker for blocker in blockers))
        self.assertTrue(any("not bound" in blocker for blocker in blockers))

    def test_content_addressed_fake_gate_can_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate_path, summary_path, receipt_path, _, spec = fake_gate_documents(root)
            lock_path = root / "gate_lock.json"
            gate_hash = validate.canonical_sha256(spec)
            with patch_gate_paths(root, gate_path, summary_path, receipt_path):
                lock = bind_gate.build_lock(
                    gate_path, gate_hash, summary_path, receipt_path
                )
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with patch_gate_paths(root, gate_path, summary_path, receipt_path):
                self.assertEqual(
                    validate.gate_dependency_blockers(
                        gate_path, lock_path, summary_path, receipt_path
                    ),
                    [],
                )

    def test_gate_binding_requires_validator_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate_path, summary_path, receipt_path, _, spec = fake_gate_documents(root)
            digest = validate.canonical_sha256(spec)
            with patch_gate_paths(root, gate_path, summary_path, receipt_path):
                lock = bind_gate.build_lock(
                    gate_path, digest, summary_path, receipt_path
                )
                self.assertEqual(lock["gate_spec_sha256"], digest)
                with self.assertRaises(validate.ValidationError):
                    bind_gate.build_lock(
                        gate_path, "f" * 64, summary_path, receipt_path
                    )

                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["success"] = False
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(validate.ValidationError):
                    bind_gate.build_lock(
                        gate_path, digest, summary_path, receipt_path
                    )

    def test_execute_refuses_before_any_subprocess(self):
        with mock.patch.object(launch.subprocess, "run") as run:
            with redirect_stdout(io.StringIO()):
                status = launch.main(["--execute", "--runner", "/bin/true"])
        self.assertEqual(status, 2)
        run.assert_not_called()

    def test_retuned_jobs_require_frozen_plans(self):
        manifest = validate.validate_static()["manifests"]["audit"]
        for job in manifest["jobs"]:
            required = job["required_files"]
            if job["transfer_mode"] == "retuned":
                self.assertTrue(any(path.startswith("retune_plans/") for path in required))
            else:
                self.assertFalse(any(path.startswith("retune_plans/") for path in required))


if __name__ == "__main__":
    unittest.main()
