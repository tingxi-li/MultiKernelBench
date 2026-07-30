#!/usr/bin/env python3
"""CPU-only tests for fused-grid screening analysis."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_screen as screen


DSLS = ("dsl_a", "dsl_b")


class SyntheticCampaign:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw = root / "raw"
        self.raw.mkdir()
        self.manifest_path = root / "manifest.json"
        self.jobs_path = root / "jobs.json"
        self.receipt_path = root / "launch.json"
        self.adapter_path = root / "adapter.json"
        self.robust_path = root / "robust_summary.json"

        self.grid = [
            {"BM": 64 + 32 * index, "BN": 128, "BK": 32,
             "stages": 2 + index % 3, "kc": 2048}
            for index in range(5)
        ]
        self.fixed = {
            "op": "fused", "variant": "GBGS", "M": 1024, "K": 8192,
            "N": 8192, "threads": 256, "kc": 2048, "arith": "fp16",
            "cast": "precast", "wcache": "cached", "epilogue": "smem",
        }
        self.jobs = []
        for dsl in DSLS:
            for index, point in enumerate(self.grid):
                grid_id = f"g{index:02d}"
                self.jobs.append({
                    "dsl": dsl,
                    "geom": "fused",
                    "grid_id": grid_id,
                    "grid_index": index,
                    "job_id": f"{dsl}.{grid_id}",
                    "set": self._set(point),
                    "variant": "GBGS",
                })
        self._write(self.jobs_path, self.jobs)
        self.manifest = {
            "schema_version": 1,
            "campaign_id": "synthetic-fused-screen-v1",
            "dsl_order": list(DSLS),
            "fixed_factors": self.fixed,
            "grid": self.grid,
            "grid_point_count": len(self.grid),
            "job_count": len(self.jobs),
            "jobs_file": self.jobs_path.name,
            "jobs_sha256": screen.sha256_bytes(self.jobs_path.read_bytes()),
            "phase1_grid_source": "unused.json",
            "phase1_grid_source_sha256": "1" * 64,
        }
        self._write(self.manifest_path, self.manifest)

        protocol = {
            "dist": "rand", "seed": 0, "time_only": False,
            "trials": 100, "warmup_s": 2.0,
        }
        self.sources = {"impl.py": "2" * 64, "screen_only.py": "3" * 64}
        self.receipt = {
            "campaign_id": self.manifest["campaign_id"],
            "git_commit": "synthetic-commit",
            "jobs_sha256": self.manifest["jobs_sha256"],
            "launch_args": {**protocol, "order_seed": 7, "reps": 2},
            "manifest_sha256": screen.sha256_bytes(self.manifest_path.read_bytes()),
            "phase1_grid_source_sha256": self.manifest[
                "phase1_grid_source_sha256"
            ],
            "protocol_sha256": screen.sha256_bytes(screen.stable_json_bytes(protocol)),
            "source_sha256": self.sources,
            "source_bundle_sha256": screen.sha256_bytes(
                screen.stable_json_bytes(self.sources)
            ),
        }
        self._write(self.receipt_path, self.receipt)
        self._write_records()
        self._write_adapter()
        self._write_robust_summary()

    def _set(self, point: dict[str, int]) -> str:
        return ",".join((
            f"BM={point['BM']}", f"BN={point['BN']}", f"BK={point['BK']}",
            "threads=256", f"stages={point['stages']}", "kc=2048",
            "arith=fp16", "cast=precast", "x_wcache=cached",
            "x_epilogue=smem",
        ))

    @staticmethod
    def _write(path: Path, value: object) -> None:
        path.write_bytes(screen.stable_json_bytes(value))

    def _provenance(self, job: dict[str, object], rep: int) -> dict[str, object]:
        return {
            "campaign_id": self.receipt["campaign_id"],
            "git_commit": self.receipt["git_commit"],
            "grid_id": job["grid_id"],
            "grid_index": job["grid_index"],
            "job_id": job["job_id"],
            "jobs_sha256": self.receipt["jobs_sha256"],
            "manifest_sha256": self.receipt["manifest_sha256"],
            "phase1_grid_source_sha256": self.receipt[
                "phase1_grid_source_sha256"
            ],
            "protocol_sha256": self.receipt["protocol_sha256"],
            "rep": rep,
            "source_bundle_sha256": self.receipt["source_bundle_sha256"],
            "source_sha256": self.receipt["source_sha256"],
        }

    def record_path(self, job_id: str, rep: int) -> Path:
        return self.raw / f"{job_id.replace('.', '__')}__rep{rep}.json"

    def success_record(
        self, job: dict[str, object], rep: int, *, legacy_pass: bool
    ) -> dict[str, object]:
        parsed = screen.parse_set(job["set"])
        index = job["grid_index"]
        # g00 is intentionally fastest but gate-invalid; g01 is the slow old
        # incumbent, leaving g02/g03/g04 as the robust top three.
        base = {0: 0.1, 1: 9.0, 2: 1.0, 3: 2.0, 4: 3.0}[index]
        return {
            "ok": True,
            "op": "fused",
            "dsl": job["dsl"],
            "variant": "GBGS",
            "geom": "fused",
            "dist": "rand",
            "seed": 0,
            "rep": rep,
            "trials": 100,
            "warmup_s": 2.0,
            "compile_s": 1.0 + rep,
            "cfg": {
                **{name: parsed[name] for name in
                   ("BM", "BN", "BK", "threads", "stages", "kc", "arith", "cast")},
                "M": 1024, "K": 8192, "N": 8192,
                "dsl": job["dsl"], "variant": "GBGS",
                "extra": {"wcache": "cached", "epilogue": "smem"},
            },
            "timing": {"median_ms": base + rep * 0.2, "n": 100},
            "error": {
                "gate_pass": legacy_pass,
                "pct_elems_failing_gate": 1.337 if not legacy_pass else 0.0,
                "max_abs_err": 0.01 if not legacy_pass else 1e-7,
                "mean_abs_err": 1e-4,
                "budget_mean": 1e-4,
            },
            "returncode": 0,
            "campaign_provenance": self._provenance(job, rep),
        }

    def _write_records(self) -> None:
        for job in self.jobs:
            for rep in screen.EXPECTED_REPS:
                record = self.success_record(
                    job, rep, legacy_pass=job["grid_id"] != "g00"
                )
                self._write(self.record_path(job["job_id"], rep), record)

    def _write_adapter(self) -> None:
        adapter_sources = {"adapter.py": "4" * 64, "impl.py": self.sources["impl.py"]}
        self.adapter = {
            "schema_version": 1,
            "grid": {
                "manifest_sha256": self.receipt["manifest_sha256"],
                "jobs_sha256": self.receipt["jobs_sha256"],
            },
            "robust_gate": {
                "manifest_sha256": "5" * 64,
                "manifest_canonical_sha256": "6" * 64,
                "gate_spec_sha256": "7" * 64,
                "gate_spec_canonical_sha256": "8" * 64,
            },
            "source_sha256": adapter_sources,
            "source_bundle_sha256": screen.canonical_sha256(adapter_sources),
        }
        self._write(self.adapter_path, self.adapter)

    def _write_robust_summary(self) -> None:
        groups = []
        for job in self.jobs:
            if job["grid_id"] == "g00":
                continue
            job_hash = screen.canonical_sha256(job)
            for gate_id in screen.REQUIRED_GATE_IDS:
                groups.append({
                    "op": screen.ROBUST_OPERATION,
                    "gate_id": gate_id,
                    "candidate": f"fused-grid:{job['job_id']}:{job_hash[:12]}",
                    "grid_job_id": job["job_id"],
                    "grid_job_sha256": job_hash,
                    "n_records": 256,
                    "n_failed_records": 0,
                    "coverage_complete": True,
                    "success": True,
                })
        self.robust_summary = {
            "schema_version": "1.0",
            "split": "validation",
            "status": "PASS",
            "success": True,
            "grid_manifest_sha256": self.receipt["manifest_sha256"],
            "grid_jobs_sha256": self.receipt["jobs_sha256"],
            "adapter_manifest_sha256": screen.sha256_bytes(self.adapter_path.read_bytes()),
            "source_bundle_sha256": self.adapter["source_bundle_sha256"],
            "gate_spec_sha256": self.adapter["robust_gate"]["gate_spec_sha256"],
            "gate_spec_canonical_sha256": self.adapter["robust_gate"][
                "gate_spec_canonical_sha256"
            ],
            "robust_manifest_sha256": self.adapter["robust_gate"][
                "manifest_canonical_sha256"
            ],
            "groups": groups,
        }
        self._write(self.robust_path, self.robust_summary)

    def analyze(self, *, robust: bool = False) -> screen.Analysis:
        return screen.analyze_campaign(
            self.manifest_path,
            self.receipt_path,
            self.raw,
            self.robust_path if robust else None,
            self.adapter_path,
        )


class AnalyzeScreenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticCampaign(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_gate_failure_is_retained_but_never_eligible(self):
        analysis = self.fixture.analyze()
        self.assertTrue(analysis.summary["campaign_complete"])
        bad = next(cell for cell in analysis.summary["cells"]
                   if cell["job_id"] == "dsl_a.g00")
        self.assertEqual(bad["status"], "legacy_gate_failure")
        self.assertFalse(bad["screening_eligible"])
        self.assertAlmostEqual(bad["median_of_process_medians_ms"], 0.2)
        self.assertEqual(
            [row["pct_elems_failing_gate"] for row in bad["processes"]],
            [1.337, 1.337],
        )
        self.assertFalse(analysis.summary["robust_gate"]["supplied"])
        with self.assertRaisesRegex(screen.AnalysisError, "external robust"):
            screen.build_confirmation(analysis)

    def test_robust_top_three_and_incumbent_are_deterministic(self):
        first = self.fixture.analyze(robust=True)
        second = self.fixture.analyze(robust=True)
        self.assertEqual(
            screen.stable_json_bytes(first.summary),
            screen.stable_json_bytes(second.summary),
        )
        gate = first.summary["robust_gate"]
        self.assertTrue(gate["selection_ready"])
        for dsl in DSLS:
            self.assertEqual(
                gate["selected_job_ids_by_dsl"][dsl],
                [f"{dsl}.g02", f"{dsl}.g03", f"{dsl}.g04"],
            )

        document = screen.build_confirmation(first)
        self.assertEqual(len(document["jobs"]), 8)
        for dsl in DSLS:
            lane = [job for job in document["jobs"] if job["dsl"] == dsl]
            self.assertEqual(
                [job["grid_id"] for job in lane], ["g02", "g03", "g04", "g01"]
            )
            self.assertEqual(lane[-1]["selection_roles"], ["old_incumbent"])
        self.assertEqual(document, screen.build_confirmation(second))

    def test_sparse_build_failure_is_retained(self):
        job = self.fixture.jobs[2]
        rep = 1
        failed = {
            "ok": False,
            "op": "fused",
            "dsl": job["dsl"],
            "variant": "GBGS",
            "rep": rep,
            "cfg": {"extra": {"wcache": "cached", "epilogue": "smem"}},
            "error_msg": "OutOfResources: synthetic shared memory failure",
            "returncode": 1,
            "campaign_provenance": self.fixture._provenance(job, rep),
        }
        self.fixture._write(self.fixture.record_path(job["job_id"], rep), failed)
        analysis = self.fixture.analyze()
        cell = next(cell for cell in analysis.summary["cells"]
                    if cell["job_id"] == job["job_id"])
        self.assertEqual(cell["status"], "build_or_execution_failure")
        self.assertEqual(cell["processes"][1]["returncode"], 1)
        self.assertIn("OutOfResources", cell["processes"][1]["error_msg"])
        self.assertFalse(cell["screening_eligible"])

    def test_incomplete_campaign_summarizes_without_selecting(self):
        self.fixture.record_path("dsl_a.g04", 1).unlink()
        analysis = self.fixture.analyze()
        self.assertFalse(analysis.summary["campaign_complete"])
        cell = next(cell for cell in analysis.summary["cells"]
                    if cell["job_id"] == "dsl_a.g04")
        self.assertEqual(cell["status"], "incomplete")
        self.assertEqual(cell["rep_ids"], [0])
        self.assertIsNone(cell["median_of_process_medians_ms"])

    def test_duplicate_rep_and_foreign_provenance_fail_closed(self):
        original = self.fixture.record_path("dsl_a.g01", 0)
        duplicate = self.fixture.raw / "duplicate.json"
        duplicate.write_bytes(original.read_bytes())
        with self.assertRaisesRegex(screen.AnalysisError, "duplicate process"):
            self.fixture.analyze()
        duplicate.unlink()

        record = json.loads(original.read_text())
        record["campaign_provenance"]["protocol_sha256"] = "f" * 64
        self.fixture._write(original, record)
        with self.assertRaisesRegex(screen.AnalysisError, "protocol_sha256 mismatch"):
            self.fixture.analyze()

    def test_external_gate_must_cover_every_screening_candidate(self):
        robust = copy.deepcopy(self.fixture.robust_summary)
        robust["groups"] = robust["groups"][1:]
        self.fixture._write(self.fixture.robust_path, robust)
        with self.assertRaisesRegex(screen.AnalysisError, "omits 1 required"):
            self.fixture.analyze(robust=True)

    def test_external_gate_source_must_match_screened_implementation(self):
        adapter = copy.deepcopy(self.fixture.adapter)
        adapter["source_sha256"]["impl.py"] = "9" * 64
        adapter["source_bundle_sha256"] = screen.canonical_sha256(
            adapter["source_sha256"]
        )
        self.fixture._write(self.fixture.adapter_path, adapter)
        robust = copy.deepcopy(self.fixture.robust_summary)
        robust["adapter_manifest_sha256"] = screen.sha256_bytes(
            self.fixture.adapter_path.read_bytes()
        )
        robust["source_bundle_sha256"] = adapter["source_bundle_sha256"]
        self.fixture._write(self.fixture.robust_path, robust)
        with self.assertRaisesRegex(screen.AnalysisError, "shared screening/robust"):
            self.fixture.analyze(robust=True)


if __name__ == "__main__":
    unittest.main()
