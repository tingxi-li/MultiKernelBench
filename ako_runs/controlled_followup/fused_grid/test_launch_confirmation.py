#!/usr/bin/env python3
"""CPU-only tests for the robust confirmation launcher."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_screen
import launch_confirmation as confirm
from test_analyze_screen import SyntheticCampaign


class ConfirmationLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticCampaign(Path(self.tmp.name))
        analysis = self.fixture.analyze(robust=True)
        self.document = analyze_screen.build_confirmation(analysis)
        self.confirmation_path = Path(self.tmp.name) / "confirm.json"
        self.confirmation_path.write_bytes(
            analyze_screen.stable_json_bytes(self.document)
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def load_bound(self) -> confirm.BoundConfirmation:
        with mock.patch.object(confirm, "verify_current_screening_sources"):
            return confirm.load_bound_confirmation(
                self.confirmation_path,
                self.fixture.manifest_path,
                self.fixture.receipt_path,
                self.fixture.raw,
                self.fixture.robust_path,
                self.fixture.adapter_path,
            )

    def test_exact_reconstruction_is_accepted(self):
        bound = self.load_bound()
        self.assertEqual(bound.document, self.document)
        self.assertEqual(bound.raw_sha256, confirm.sha256_file(self.confirmation_path))

    def test_hand_edited_selection_fails_even_with_rehashed_jobs(self):
        edited = copy.deepcopy(self.document)
        edited["jobs"][0]["set"] += ",x_unregistered=1"
        edited["jobs_sha256"] = analyze_screen.canonical_sha256(edited["jobs"])
        self.confirmation_path.write_bytes(analyze_screen.stable_json_bytes(edited))
        with self.assertRaisesRegex(confirm.ConfirmationError, "reconstructed"):
            self.load_bound()

    def test_canonical_jobs_hash_is_required(self):
        edited = copy.deepcopy(self.document)
        edited["jobs_sha256"] = "0" * 64
        with self.assertRaisesRegex(confirm.ConfirmationError, "canonical jobs hash"):
            confirm.validate_confirmation_document(edited)

    def test_plan_has_five_fresh_process_entries_per_job(self):
        first = confirm.plan_jobs(self.document["jobs"], 5, 20260730)
        second = confirm.plan_jobs(self.document["jobs"], 5, 20260730)
        self.assertEqual(
            [(job["confirmation_id"], rep) for job, rep in first],
            [(job["confirmation_id"], rep) for job, rep in second],
        )
        self.assertEqual(len(first), len(self.document["jobs"]) * 5)
        by_id: dict[str, set[int]] = {}
        for job, rep in first:
            by_id.setdefault(job["confirmation_id"], set()).add(rep)
        self.assertTrue(all(reps == set(range(5)) for reps in by_id.values()))

    def test_resume_requires_exact_confirmation_binding(self):
        bound = self.load_bound()
        args = argparse.Namespace(
            dist="rand",
            seed=0,
            trials=100,
            warmup_s=2.0,
            gpu=0,
            order_seed=20260730,
            reps=5,
            timeout=1800,
        )
        protocol_hash = confirm.validate_protocol(bound, args)
        provenance = {
            "campaign_id": bound.document["campaign_id"],
            "confirmation_sha256": bound.raw_sha256,
            "confirmation_jobs_sha256": bound.document["jobs_sha256"],
            "confirmation_provenance": bound.document["provenance"],
            "protocol_sha256": protocol_hash,
            "source_bundle_sha256": "a" * 64,
            "git_commit": "test-commit",
            "launch_args": {"gpu": 0, "order_seed": 20260730, "reps": 5},
        }
        job = bound.document["jobs"][0]
        record = {
            "ok": True,
            "dsl": job["dsl"],
            "variant": job["variant"],
            "rep": 0,
            "timing": {"median_ms": 1.0},
            "error": {"gate_pass": True},
            "confirmation_provenance": confirm.record_binding(provenance, job, 0),
        }
        path = Path(self.tmp.name) / "record.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(
            confirm.record_state(path, provenance, job, 0)[0], "complete_ok"
        )
        foreign = copy.deepcopy(provenance)
        foreign["source_bundle_sha256"] = "b" * 64
        self.assertEqual(confirm.record_state(path, foreign, job, 0)[0], "foreign")

    def test_legacy_gate_failure_is_retained_as_failed(self):
        bound = self.load_bound()
        job = bound.document["jobs"][0]
        provenance = {
            "campaign_id": bound.document["campaign_id"],
            "confirmation_sha256": bound.raw_sha256,
            "confirmation_jobs_sha256": bound.document["jobs_sha256"],
            "confirmation_provenance": bound.document["provenance"],
            "protocol_sha256": "c" * 64,
            "source_bundle_sha256": "d" * 64,
            "git_commit": "test-commit",
            "launch_args": {"gpu": 0, "order_seed": 20260730, "reps": 5},
        }
        record = {
            "ok": True,
            "dsl": job["dsl"],
            "variant": job["variant"],
            "rep": 0,
            "timing": {"median_ms": 0.25},
            "error": {"gate_pass": False},
            "confirmation_provenance": confirm.record_binding(provenance, job, 0),
        }
        path = Path(self.tmp.name) / "gate-fail.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(
            confirm.record_state(path, provenance, job, 0)[0], "complete_failed"
        )


if __name__ == "__main__":
    unittest.main()
