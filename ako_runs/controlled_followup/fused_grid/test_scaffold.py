#!/usr/bin/env python3
"""CPU-only structural checks for the fused-grid scaffold."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import launch
import make_manifest


class FusedGridScaffoldTest(unittest.TestCase):
    def test_source_grid_is_exactly_shared(self):
        grid, source_hash = make_manifest.source_grid()
        self.assertEqual(len(grid), 19)
        self.assertEqual(len(source_hash), 64)
        self.assertEqual(len({tuple(sorted(point.items())) for point in grid}), 19)

    def test_generated_documents(self):
        jobs, manifest = make_manifest.build_documents()
        self.assertEqual(len(jobs), 76)
        self.assertEqual(manifest["grid_point_count"], 19)
        self.assertEqual(manifest["job_count"], 76)
        for dsl in make_manifest.DSLS:
            lane = [job for job in jobs if job["dsl"] == dsl]
            self.assertEqual(len(lane), 19)
            self.assertTrue(all(job["variant"] == "GBGS" for job in lane))

    def test_plan_is_deterministic(self):
        jobs, _manifest = make_manifest.build_documents()
        p1 = [(job["job_id"], rep) for job, rep in launch.plan_jobs(jobs, 2, 20260729)]
        p2 = [(job["job_id"], rep) for job, rep in launch.plan_jobs(jobs, 2, 20260729)]
        self.assertEqual(p1, p2)
        self.assertEqual(len(p1), 152)

    def test_matching_record_is_resumable(self):
        jobs, manifest = make_manifest.build_documents()
        provenance = {
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": "a" * 64,
            "jobs_sha256": manifest["jobs_sha256"],
            "protocol_sha256": "b" * 64,
            "source_bundle_sha256": "c" * 64,
        }
        job, rep = jobs[0], 0
        rec = {
            "ok": True,
            "campaign_provenance": {
                "campaign_id": provenance["campaign_id"],
                "manifest_sha256": provenance["manifest_sha256"],
                "jobs_sha256": provenance["jobs_sha256"],
                "protocol_sha256": provenance["protocol_sha256"],
                "source_bundle_sha256": provenance["source_bundle_sha256"],
                "job_id": job["job_id"],
                "rep": rep,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            path.write_text(json.dumps(rec))
            state, loaded = launch.record_state(path, provenance, job, rep)
        self.assertEqual(state, "complete_ok")
        self.assertTrue(loaded["ok"])


if __name__ == "__main__":
    unittest.main()
