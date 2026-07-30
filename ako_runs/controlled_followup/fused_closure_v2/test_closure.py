#!/usr/bin/env python3
"""CPU-only integrity and statistical tests for fused closure v2."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from . import analyze, core, provenance


SCREEN_PATH = core.REPO_ROOT / (
    "ako_runs/controlled_followup/fused_grid/results/"
    "fused_gbgs_grid_rank/screen_summary.json"
)
CONFIRM_PATH = core.REPO_ROOT / (
    "ako_runs/controlled_followup/fused_grid/results/"
    "fused_gbgs_confirm_robust_v1/summary.json"
)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.campaign = core.load_campaign()

    def test_candidate_order_and_contract_arms(self):
        self.assertEqual(len(self.campaign["candidate_order"]), 9)
        self.assertEqual(
            [item["candidate_id"] for item in self.campaign["candidates"]],
            self.campaign["candidate_order"],
        )
        diagnostics = self.campaign["candidates"][:2]
        self.assertTrue(
            all(item["contract_adjudication"] == "diagnostic_nonconforming" for item in diagnostics)
        )
        self.assertTrue(all(item["structural_mismatches"] for item in diagnostics))
        self.assertTrue(
            all(
                item["contract_adjudication"] == "fused_v2_required"
                and not item["structural_mismatches"]
                for item in self.campaign["candidates"][2:]
            )
        )

    def test_strict_common_intersection_is_derived_from_screen(self):
        screen = core.read_json(SCREEN_PATH)
        dsls = sorted({cell["dsl"] for cell in screen["cells"]})
        eligible = [
            {
                cell["grid_id"]
                for cell in screen["cells"]
                if cell["dsl"] == dsl and cell["screening_eligible"]
            }
            for dsl in dsls
        ]
        intersection = sorted(set.intersection(*eligible))
        self.assertEqual(
            intersection,
            self.campaign["selection"]["strict_common_grid_ids"],
        )
        self.assertEqual(len(intersection), 9)

    def test_common_winners_are_screen_minima_inside_intersection(self):
        screen = core.read_json(SCREEN_PATH)
        common = set(self.campaign["selection"]["strict_common_grid_ids"])
        expected = {
            "tilelang": "g03",
            "triton": "g00",
            "cuda_noptx": "g04",
            "cuda_unlimited": "g02",
        }
        for dsl, grid_id in expected.items():
            cells = [
                cell
                for cell in screen["cells"]
                if cell["dsl"] == dsl
                and cell["grid_id"] in common
                and cell["screening_eligible"]
            ]
            winner = min(
                cells,
                key=lambda item: (item["median_of_process_medians_ms"], item["grid_id"]),
            )
            self.assertEqual(winner["grid_id"], grid_id)

    def test_full_frontier_winners_match_confirmation(self):
        confirmation = core.read_json(CONFIRM_PATH)
        winners = {
            item["dsl"]: item["point_estimate_winner"]["grid_id"]
            for item in confirmation["dsl_winners"]
        }
        self.assertEqual(winners["tilelang"], "g08")
        self.assertEqual(winners["triton"], "g05")
        definitions = core.candidates_by_id(self.campaign)
        self.assertEqual(definitions["tilelang_full_g08"]["grid_id"], winners["tilelang"])
        self.assertEqual(definitions["triton_full_g05"]["grid_id"], winners["triton"])

    def test_randomized_complete_block_plan(self):
        plan = core.block_plan(self.campaign)
        self.assertEqual(len(plan), 15 * 9)
        for block in range(15):
            rows = [item for item in plan if item["block"] == block]
            self.assertEqual([item["position"] for item in rows], list(range(9)))
            self.assertEqual(
                set(item["candidate_id"] for item in rows),
                set(self.campaign["candidate_order"]),
            )
        self.assertEqual(plan, core.block_plan(self.campaign))

    def test_phase2_sets_retain_fixed_factors(self):
        for definition in self.campaign["candidates"]:
            if definition["implementation"] != "phase2_custom":
                continue
            parsed = core.parse_set(definition["set"])
            self.assertEqual(parsed["arith"], "fp16")
            self.assertEqual(parsed["cast"], "precast")
            self.assertEqual(parsed["extra"], {"wcache": "cached", "epilogue": "smem"})


class StatisticsTests(unittest.TestCase):
    def test_n15_interval_is_preregistered_x4_x12(self):
        result = core.exact_median_interval(range(1, 16))
        self.assertEqual(result["interval_order_statistic_k"], 4)
        self.assertEqual(result["ci_lo"], 4)
        self.assertEqual(result["ci_hi"], 12)
        self.assertAlmostEqual(result["achieved_coverage"], 0.96484375)
        self.assertTrue(result["finite_interval_meets_target"])

    def test_n5_cannot_reach_95_percent(self):
        result = core.exact_median_interval(range(1, 6))
        self.assertFalse(result["finite_interval_meets_target"])
        self.assertAlmostEqual(result["achieved_coverage"], 0.9375)

    def test_sign_test_and_holm(self):
        sign = core.exact_sign_test([0.8] * 15)
        self.assertEqual(sign["below"], 15)
        self.assertAlmostEqual(sign["p_value_two_sided"], 2.0 / 2**15)
        self.assertEqual(core.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_paired_analysis_requires_same_contract_and_adjusted_p(self):
        campaign = core.load_campaign()
        eligibility = {
            candidate_id: not candidate_id.startswith("torch_historical")
            for candidate_id in campaign["candidate_order"]
        }
        base = {
            "torch_historical_exact": 1.30,
            "torch_historical_precast": 1.20,
            "torch_contract_fp32": 2.00,
            "tilelang_full_g08": 1.40,
            "triton_full_g05": 1.50,
            "tilelang_common_g03": 1.60,
            "triton_common_g00": 1.50,
            "cuda_noptx_common_g04": 1.90,
            "cuda_unlimited_common_g02": 1.80,
        }
        records = {}
        for block in range(15):
            drift = 1.0 + block * 0.001
            for candidate_id, value in base.items():
                records[(block, candidate_id)] = {
                    "ok": True,
                    "timing_summary": {"median_ms": value * drift},
                }
        result = analyze.analyze_records(campaign, records, eligibility)
        vendor = [
            row
            for row in result["paired_comparisons"]
            if row["family"] == "same_contract_vendor"
        ]
        self.assertTrue(all(row["faster_claim_supported"] for row in vendor))
        historical = [
            row
            for row in result["paired_comparisons"]
            if row["family"] == "historical_diagnostic"
        ]
        self.assertFalse(any(row["faster_claim_supported"] for row in historical))
        self.assertAlmostEqual(
            result["common_recipe_spread"]["spread_interval"]["median"],
            1.9 / 1.5,
        )


class ProvenanceTests(unittest.TestCase):
    def test_expected_receipt_covers_every_declared_source(self):
        receipt = provenance.expected_receipt()
        local = receipt["local_source_sha256"]
        self.assertEqual(len(local), len(provenance.LOCAL_SOURCE_NAMES))
        self.assertIn(
            "ako_runs/controlled_followup/fused_closure_v2/analyze.py", local
        )
        self.assertTrue(receipt["frozen_gate"]["accepted"])

    def test_frozen_receipt_if_present(self):
        if core.SOURCE_RECEIPT_PATH.exists():
            self.assertEqual(
                provenance.verify_receipt(),
                json.loads(core.SOURCE_RECEIPT_PATH.read_text(encoding="utf-8")),
            )


if __name__ == "__main__":
    unittest.main()
