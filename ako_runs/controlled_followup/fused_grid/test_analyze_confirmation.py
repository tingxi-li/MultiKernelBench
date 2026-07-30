#!/usr/bin/env python3
"""CPU-only tests for confirmation-output auditing and aggregation."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_confirmation as analysis
import analyze_screen
import common


DSLS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")


def confirmation_document() -> dict:
    jobs = []
    for dsl_index, dsl in enumerate(DSLS):
        for lane_index, grid_index in enumerate((2, 3, 4, 1)):
            roles = (
                [f"screening_rank_{lane_index + 1}"]
                if lane_index < 3
                else ["old_incumbent"]
            )
            jobs.append(
                {
                    "confirmation_id": f"{dsl}.c{len(jobs):02d}",
                    "dsl": dsl,
                    "geom": "fused",
                    "grid_id": f"g{grid_index:02d}",
                    "grid_index": grid_index,
                    "screening_job_id": f"{dsl}.g{grid_index:02d}",
                    "selection_roles": roles,
                    "set": (
                        f"BM={128 + 32 * lane_index},BN=128,BK=32,threads=256,"
                        "stages=2,kc=2048,arith=fp16,cast=precast,"
                        "x_wcache=cached,x_epilogue=smem"
                    ),
                    "variant": "GBGS",
                    "screening_median_ms": 1.0 + dsl_index * 0.2 + lane_index * 0.1,
                    "screening_robust_eligible": True,
                }
            )
    document = {
        "schema_version": 1,
        "campaign_id": "fused-gbgs-confirmation-v1",
        "top_k": 3,
        "old_incumbent_grid_id": "g01",
        "selection_rule": "synthetic",
        "provenance": {
            "screening_campaign_id": "synthetic",
            "screening_manifest_sha256": "1" * 64,
            "screening_jobs_sha256": "2" * 64,
            "screening_launch_receipt_sha256": "3" * 64,
            "screening_protocol_sha256": "4" * 64,
            "screening_records_sha256": "5" * 64,
            "robust_summary_sha256": "6" * 64,
            "robust_adapter_manifest_sha256": "7" * 64,
        },
        "jobs": jobs,
    }
    document["jobs_sha256"] = analyze_screen.canonical_sha256(jobs)
    return document


def process_map(document: dict, *, close: bool = False):
    result = {}
    bases = {
        "tilelang": 1.00,
        "triton": 1.01 if close else 1.30,
        "cuda_noptx": 1.60,
        "cuda_unlimited": 1.90,
    }
    offsets = (-0.02, -0.01, 0.0, 0.01, 0.02)
    for job in document["jobs"]:
        lane_index = next(
            index for index, role in enumerate((
                "screening_rank_1", "screening_rank_2", "screening_rank_3",
                "old_incumbent",
            )) if role in job["selection_roles"]
        )
        for rep, offset in enumerate(offsets):
            median = bases[job["dsl"]] + lane_index * 0.10 + offset
            result[(job["confirmation_id"], rep)] = {
                "rep": rep,
                "record_file": f"{job['confirmation_id']}.{rep}.json",
                "record_sha256": f"{len(result):064x}",
                "ok": True,
                "returncode": 0,
                "legacy_gate_pass": True,
                "outcome": "accepted",
                "median_ms": median,
            }
    return result


class AnalyzeConfirmationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = confirmation_document()

    def test_uses_frozen_phase1_ci_rule_exactly(self):
        processes = process_map(self.document)
        cells, _lanes, _four_way = analysis.aggregate_confirmation(
            self.document, processes
        )
        first = cells[0]
        expected = common.median_ci([0.98, 0.99, 1.0, 1.01, 1.02])
        for name, value in expected.items():
            self.assertEqual(first[name], value)

    def test_requires_all_reps_zero_through_four(self):
        processes = process_map(self.document)
        first_job = self.document["jobs"][0]
        del processes[(first_job["confirmation_id"], 4)]
        with self.assertRaisesRegex(analysis.ConfirmationAnalysisError, "missing"):
            analysis.aggregate_confirmation(self.document, processes)

    def test_failure_is_retained_and_excluded_from_winner(self):
        processes = process_map(self.document)
        first_job = self.document["jobs"][0]
        failed = processes[(first_job["confirmation_id"], 2)]
        failed.update(
            {
                "ok": False,
                "returncode": 1,
                "legacy_gate_pass": False,
                "outcome": "build_or_execution_failure",
                "error_msg": "synthetic build failure",
            }
        )
        failed.pop("median_ms")
        cells, lanes, _four_way = analysis.aggregate_confirmation(
            self.document, processes
        )
        first = cells[0]
        self.assertFalse(first["confirmation_eligible"])
        self.assertEqual(first["failure_count"], 1)
        self.assertEqual(
            lanes[0]["point_estimate_winner"]["screening_job_id"],
            self.document["jobs"][1]["screening_job_id"],
        )
        self.assertIn(first["screening_job_id"], lanes[0]["failed_job_ids"])

    def test_ci_overlap_suppresses_strict_cross_dsl_order(self):
        processes = process_map(self.document, close=True)
        _cells, lanes, four_way = analysis.aggregate_confirmation(
            self.document, processes
        )
        self.assertFalse(four_way["strict_order_resolved"])
        self.assertIsNone(four_way["strict_order"])
        self.assertIn(["tilelang", "triton"], four_way["overlapping_pairs"])
        summary = {
            "campaign_id": self.document["campaign_id"],
            "observed_process_record_count": len(processes),
            "expected_process_record_count": len(processes),
            "dsl_winners": lanes,
            "four_way": four_way,
            "all_jobs_confirmed": True,
            "cells": [],
        }
        rendered = analysis.render(summary)
        self.assertIn("No strict cross-DSL order is asserted", rendered)

    def test_nonoverlap_can_resolve_strict_order(self):
        processes = process_map(self.document)
        _cells, lanes, four_way = analysis.aggregate_confirmation(
            self.document, processes
        )
        self.assertTrue(all(lane["strict_winner_resolved"] for lane in lanes))
        self.assertTrue(four_way["strict_order_resolved"])
        self.assertEqual(four_way["strict_order"], list(DSLS))
        self.assertAlmostEqual(four_way["point_estimate_spread_x"], 1.9)

    def test_process_map_is_deterministic(self):
        first = analysis.aggregate_confirmation(
            self.document, process_map(self.document)
        )
        second = analysis.aggregate_confirmation(
            copy.deepcopy(self.document), process_map(copy.deepcopy(self.document))
        )
        self.assertEqual(first, second)

    def test_current_source_bytes_are_content_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "impl.py"
            source.write_text("frozen = True\n", encoding="utf-8")
            sources = {"impl.py": analysis.launch_confirmation.sha256_file(source)}
            receipt = {
                "source_sha256": sources,
                "source_bundle_sha256": analysis.launch_confirmation.sha256_bytes(
                    analyze_screen.stable_json_bytes(sources)
                ),
            }
            with mock.patch.object(analysis, "REPO_ROOT", root):
                analysis._verify_current_sources(receipt)
                source.write_text("frozen = False\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    analysis.ConfirmationAnalysisError, "current launch source"
                ):
                    analysis._verify_current_sources(receipt)

    def test_raw_record_must_match_receipt_job_and_rep_binding(self):
        document = self.document
        job = document["jobs"][0]
        receipt = {
            "campaign_id": document["campaign_id"],
            "confirmation_sha256": "8" * 64,
            "confirmation_jobs_sha256": document["jobs_sha256"],
            "confirmation_provenance": document["provenance"],
            "protocol": {
                "dist": "rand",
                "seed": 0,
                "time_only": False,
                "trials": 100,
                "warmup_s": 2.0,
            },
            "protocol_sha256": "9" * 64,
            "source_bundle_sha256": "a" * 64,
            "git_commit": "synthetic",
            "launch_args": {"gpu": 0, "order_seed": 20260730, "reps": 5},
        }
        campaign = SimpleNamespace(
            manifest={"fixed_factors": {"M": 1024, "K": 8192, "N": 8192}}
        )
        context = analysis.Context(
            confirmation=document,
            confirmation_sha256=receipt["confirmation_sha256"],
            launch_receipt=receipt,
            launch_receipt_sha256="b" * 64,
            campaign=campaign,
        )
        parsed = analyze_screen.parse_set(job["set"])
        record = {
            "ok": True,
            "op": "fused",
            "dsl": job["dsl"],
            "variant": job["variant"],
            "dist": "rand",
            "seed": 0,
            "rep": 0,
            "trials": 100,
            "warmup_s": 2.0,
            "returncode": 0,
            "cfg": {
                **{
                    name: parsed[name]
                    for name in (
                        "BM", "BN", "BK", "threads", "stages", "kc", "arith", "cast"
                    )
                },
                "M": 1024,
                "K": 8192,
                "N": 8192,
                "dsl": job["dsl"],
                "variant": job["variant"],
            },
            "timing": {"median_ms": 1.0},
            "error": {"gate_pass": True},
            "confirmation_provenance": analysis.launch_confirmation.record_binding(
                receipt, job, 0
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            path.write_bytes(analyze_screen.stable_json_bytes(record))
            view = analysis.validate_record(context, job, 0, path)
            self.assertEqual(view["outcome"], "accepted")
            record["confirmation_provenance"]["rep"] = 4
            path.write_bytes(analyze_screen.stable_json_bytes(record))
            with self.assertRaisesRegex(
                analysis.ConfirmationAnalysisError, "confirmation provenance"
            ):
                analysis.validate_record(context, job, 0, path)


if __name__ == "__main__":
    unittest.main()
