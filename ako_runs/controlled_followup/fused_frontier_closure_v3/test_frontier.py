#!/usr/bin/env python3
"""CPU-only protocol/evidence/statistics tests for frontier closure v3."""

from __future__ import annotations

import unittest

from . import analyze, core, eligibility, provenance


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.campaign = core.load_campaign()

    def test_candidate_and_plan_coverage(self):
        self.assertEqual(self.campaign["candidate_order"], core.EXPECTED_ORDER)
        plan = core.block_plan(self.campaign)
        self.assertEqual(len(plan), 120)
        self.assertEqual(plan, core.block_plan(self.campaign))
        for block in range(15):
            rows = [row for row in plan if row["block"] == block]
            self.assertEqual([row["position"] for row in rows], list(range(8)))
            self.assertEqual(
                {row["candidate_id"] for row in rows}, set(core.EXPECTED_ORDER)
            )

    def test_gpu_and_protocol_are_frozen(self):
        self.assertEqual(self.campaign["hardware"]["physical_gpu"], 3)
        self.assertEqual(
            self.campaign["hardware"]["required_uuid"],
            "GPU-eafdd6ce-8857-40fd-f494-47a7240bf6b5",
        )
        self.assertEqual(self.campaign["performance_protocol"]["blocks"], 15)
        self.assertEqual(self.campaign["performance_protocol"]["trials"], 100)

    def test_unresolved_noptx_pair_is_retained(self):
        ids = set(self.campaign["candidate_order"])
        self.assertIn("cuda_noptx_streamed_g05", ids)
        self.assertIn("cuda_noptx_streamed_g09", ids)
        self.assertIn("UNRESOLVED", self.campaign["inference"]["noptx_selection_status"].upper())

    def test_fixed_spread_set(self):
        self.assertEqual(
            self.campaign["fixed_frontier_spread"],
            [
                "tilelang_full_g08",
                "triton_full_g05",
                "cuda_noptx_streamed_g05",
                "cuda_unlimited_streamed_g07",
            ],
        )


class EligibilityTests(unittest.TestCase):
    def test_original_evidence_yields_all_eight_eligible(self):
        receipt = eligibility.expected_receipt()
        self.assertTrue(receipt["all_candidates_original_gate_eligible"])
        self.assertEqual(list(receipt["candidates"]), core.EXPECTED_ORDER)
        for row in receipt["candidates"].values():
            self.assertEqual(row["gate_records"], 512)
            self.assertEqual(row["gate_failures"], 0)
            self.assertEqual(row["validation_seeds_per_case"], 64)
        self.assertIn("not fresh-stress", receipt["claim_limit"])

    def test_receipts_if_present(self):
        if core.ELIGIBILITY_RECEIPT_PATH.exists():
            eligibility.verify_receipt()
        if core.SOURCE_RECEIPT_PATH.exists():
            provenance.verify_receipt()


class StatisticsTests(unittest.TestCase):
    def test_exact_interval_and_sign_holm(self):
        interval = core.exact_median_interval(range(1, 16))
        self.assertEqual(interval["interval_order_statistic_k"], 4)
        self.assertEqual((interval["ci_lo"], interval["ci_hi"]), (4, 12))
        self.assertAlmostEqual(interval["achieved_coverage"], 0.96484375)
        sign = core.exact_sign_test([0.8] * 15)
        self.assertAlmostEqual(sign["p_value_two_sided"], 2 / 2**15)
        self.assertEqual(core.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_synthetic_paired_analysis(self):
        campaign = core.load_campaign()
        values = {
            "torch_contract_fp32": 2.0,
            "tilelang_full_g08": 1.1,
            "triton_full_g05": 1.0,
            "cuda_noptx_old_g04": 1.8,
            "cuda_unlimited_old_g02": 1.7,
            "cuda_noptx_streamed_g05": 1.4,
            "cuda_noptx_streamed_g09": 1.5,
            "cuda_unlimited_streamed_g07": 1.3,
        }
        records = {}
        for block in range(15):
            drift = 1 + block * 0.001
            for candidate, value in values.items():
                records[(block, candidate)] = {
                    "ok": True,
                    "timing_summary": {"median_ms": value * drift},
                }
        result = analyze.analyze_records(campaign, records)
        self.assertTrue(result["performance_complete"])
        self.assertTrue(
            all(
                row["directional_result"] != "unresolved"
                for row in result["paired_comparisons"]
            )
        )
        self.assertAlmostEqual(
            result["fixed_frontier_spread"]["spread_interval"]["median"], 1.4
        )


if __name__ == "__main__":
    unittest.main()
