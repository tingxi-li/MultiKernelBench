#!/usr/bin/env python3
"""CPU-only tests for the missing-split validation summary normalizer."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import normalize_validation_summary as normalizer


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.adapter_path = root / "adapter.json"
        self.summary_path = root / "summary.json"
        self.records_path = root / "records.jsonl"
        self.cases = ["case_a", "case_b"]
        self.gates = ["semantic_mixed", "conformance_mixed"]
        self.jobs = [
            {
                "dsl": "dsl_a", "geom": "fused", "grid_id": f"g{i:02d}",
                "grid_index": i, "job_id": f"dsl_a.g{i:02d}",
                "set": f"BM={64 + i * 32}", "variant": "GBGS",
            }
            for i in range(2)
        ]
        self.job_hashes = {
            job["job_id"]: normalizer.canonical_sha256(job) for job in self.jobs
        }
        sources = {"impl.py": "1" * 64}
        self.adapter = {
            "schema_version": 1,
            "operation": normalizer.OPERATION,
            "grid": {
                "job_count": len(self.jobs),
                "job_sha256": self.job_hashes,
                "manifest_sha256": "2" * 64,
                "jobs_sha256": "3" * 64,
            },
            "robust_gate": {
                "campaign_id": "synthetic-robust-v1",
                "manifest_canonical_sha256": "4" * 64,
                "gate_spec_sha256": "5" * 64,
                "gate_spec_canonical_sha256": "6" * 64,
                "case_ids": self.cases,
                "gate_keys": [f"{normalizer.OPERATION}/{gate}" for gate in self.gates],
                "split_counts": {"tuning": 1, "validation": 2},
            },
            "source_sha256": sources,
            "source_bundle_sha256": normalizer.canonical_sha256(sources),
        }
        self.write_json(self.adapter_path, self.adapter)
        adapter_hash = normalizer.sha256_bytes(self.adapter_path.read_bytes())
        records = []
        groups = []
        for job in self.jobs:
            job_hash = self.job_hashes[job["job_id"]]
            candidate = f"fused-grid:{job['job_id']}:{job_hash[:12]}"
            for gate in self.gates:
                groups.append({
                    "op": normalizer.OPERATION,
                    "gate_id": gate,
                    "candidate": candidate,
                    "grid_job_id": job["job_id"],
                    "grid_job_sha256": job_hash,
                    "n_records": 4,
                    "n_failed_records": 0,
                    "missing_records": 0,
                    "coverage_complete": True,
                    "success": True,
                    "observed_failure_rate": 0.0,
                })
                for case in self.cases:
                    for seed in range(2):
                        records.append({
                            "schema_version": "1.0",
                            "record_type": "robust_gate_measurement",
                            "campaign_id": "synthetic-robust-v1",
                            "manifest_sha256": "4" * 64,
                            "op": normalizer.OPERATION,
                            "gate_id": gate,
                            "case_id": case,
                            "split": "validation",
                            "seed_index": seed,
                            "candidate": candidate,
                            "role": "candidate",
                            "adapter_manifest_sha256": adapter_hash,
                            "source_bundle_sha256": self.adapter[
                                "source_bundle_sha256"
                            ],
                            "source_sha256": self.adapter["source_bundle_sha256"],
                            "grid_manifest_sha256": "2" * 64,
                            "grid_jobs_sha256": "3" * 64,
                            "grid_job_id": job["job_id"],
                            "grid_job_sha256": job_hash,
                            "grid_job": job,
                            "gate_spec_sha256": "5" * 64,
                            "gate_spec_canonical_sha256": "6" * 64,
                            "ok": True,
                            "gate_pass": True,
                            "threshold_failures": [],
                        })
        self.records = records
        self.write_records(records)
        count = len(records)
        self.summary = {
            "schema_version": "1.0",
            "campaign_id": "synthetic-robust-v1",
            "manifest_sha256": "4" * 64,
            "adapter_manifest_sha256": adapter_hash,
            "source_bundle_sha256": self.adapter["source_bundle_sha256"],
            "grid_manifest_sha256": "2" * 64,
            "grid_jobs_sha256": "3" * 64,
            "robust_manifest_sha256": "4" * 64,
            "gate_spec_sha256": "5" * 64,
            "gate_spec_canonical_sha256": "6" * 64,
            "status": "PASS",
            "success": True,
            "failures": [],
            "absent_candidates": [],
            "groups": groups,
            "launch_coverage": {
                "complete": True,
                "expected_records": count,
                "observed_records": count,
                "missing_records": 0,
                "unexpected_records": 0,
                "duplicate_records": 0,
                "missing_examples": [],
                "unexpected_examples": [],
            },
        }
        self.write_json(self.summary_path, self.summary)

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.write_bytes(normalizer.stable_bytes(value))

    def write_records(self, records: list[dict[str, object]]) -> None:
        self.records_path.write_bytes(
            b"".join(normalizer.canonical_bytes(record) + b"\n" for record in records)
        )

    def normalize(self):
        return normalizer.normalize(
            self.summary_path,
            self.records_path,
            self.adapter_path,
            receipt_parent=self.root,
        )


class NormalizeValidationSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_only_split_is_added_and_inputs_are_hashed(self):
        derived, receipt = self.fixture.normalize()
        source = copy.deepcopy(derived)
        self.assertEqual(source.pop("split"), "validation")
        self.assertEqual(source, self.fixture.summary)
        self.assertEqual(
            receipt["source_summary"]["sha256"],
            normalizer.sha256_bytes(self.fixture.summary_path.read_bytes()),
        )
        self.assertEqual(
            receipt["source_records"]["sha256"],
            normalizer.sha256_bytes(self.fixture.records_path.read_bytes()),
        )
        self.assertEqual(receipt["verification"]["observed_records"], 16)
        self.assertEqual(receipt["verification"]["group_count"], 4)

    def test_record_split_or_provenance_tampering_fails_closed(self):
        records = copy.deepcopy(self.fixture.records)
        records[0]["split"] = "tuning"
        self.fixture.write_records(records)
        with self.assertRaisesRegex(normalizer.NormalizeError, "split mismatch"):
            self.fixture.normalize()

        records[0]["split"] = "validation"
        records[0]["grid_job_sha256"] = "f" * 64
        self.fixture.write_records(records)
        with self.assertRaisesRegex(normalizer.NormalizeError, "grid job hash mismatch"):
            self.fixture.normalize()

    def test_missing_or_duplicate_record_fails_cartesian_audit(self):
        self.fixture.write_records(self.fixture.records[:-1])
        with self.assertRaisesRegex(normalizer.NormalizeError, "line count mismatch"):
            self.fixture.normalize()

        records = self.fixture.records[:-1] + [self.fixture.records[0]]
        self.fixture.write_records(records)
        with self.assertRaisesRegex(normalizer.NormalizeError, "duplicate validation"):
            self.fixture.normalize()

    def test_nonpass_or_already_tagged_summary_is_refused(self):
        summary = copy.deepcopy(self.fixture.summary)
        summary["success"] = False
        self.fixture.write_json(self.fixture.summary_path, summary)
        with self.assertRaisesRegex(normalizer.NormalizeError, "summary success mismatch"):
            self.fixture.normalize()

        summary = copy.deepcopy(self.fixture.summary)
        summary["split"] = "validation"
        self.fixture.write_json(self.fixture.summary_path, summary)
        with self.assertRaisesRegex(normalizer.NormalizeError, "already has a split"):
            self.fixture.normalize()

    def test_outputs_are_idempotent_but_never_overwritten(self):
        derived, _receipt = self.fixture.normalize()
        out = self.fixture.root / "derived.json"
        data = normalizer.stable_bytes(derived)
        self.assertEqual(normalizer.atomic_write_exact(out, data), "wrote")
        self.assertEqual(normalizer.atomic_write_exact(out, data), "verified")
        with self.assertRaisesRegex(normalizer.NormalizeError, "refusing to replace"):
            normalizer.atomic_write_exact(out, b"different\n")


if __name__ == "__main__":
    unittest.main()
