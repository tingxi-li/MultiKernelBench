#!/usr/bin/env python3
"""CPU-only tests for the frozen plan and inference helpers."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.archived_current_fused_v1 import protocol
else:  # pragma: no cover
    from . import protocol


class ProtocolTests(unittest.TestCase):
    def test_source_targets_match_expected_hashes(self):
        campaign = protocol.load_campaign()
        hashes = protocol.source_hashes(campaign)
        for subject in campaign["subjects"]:
            self.assertEqual(hashes[subject["source_path"]], subject["expected_source_sha256"])

    def test_plan_is_randomized_complete_blocks(self):
        campaign = protocol.load_campaign()
        expected = {row["subject_id"] for row in campaign["subjects"]}
        plan = protocol.make_plan(campaign)
        self.assertEqual(len(plan), 60)
        orders = []
        for block in range(15):
            rows = sorted((row for row in plan if row["block"] == block), key=lambda row: row["position"])
            self.assertEqual({row["subject_id"] for row in rows}, expected)
            self.assertEqual([row["position"] for row in rows], list(range(4)))
            orders.append(tuple(row["subject_id"] for row in rows))
        self.assertGreater(len(set(orders)), 1)
        self.assertEqual(plan, protocol.make_plan(campaign))

    def test_exact_interval(self):
        result = protocol.exact_median_interval([float(value) for value in range(15)])
        self.assertEqual(result["median"], 7.0)
        self.assertEqual(result["lo"], 3.0)
        self.assertEqual(result["hi"], 11.0)
        self.assertEqual(result["achieved_coverage"], 0.96484375)

    def test_sign_test(self):
        result = protocol.exact_two_sided_sign_p([0.9] * 15)
        self.assertEqual(result["below"], 15)
        self.assertEqual(result["above"], 0)
        self.assertTrue(math.isclose(result["p_raw"], 2.0 / 2**15))
        tied = protocol.exact_two_sided_sign_p([1.0] * 15)
        self.assertEqual(tied["p_raw"], 1.0)

    def test_holm(self):
        adjusted = protocol.holm_adjust({"a": 0.01, "b": 0.04})
        self.assertEqual(adjusted, {"a": 0.02, "b": 0.04})

    def test_tags(self):
        self.assertEqual(protocol.validate_tag("main_v1"), "main_v1")
        for value in ("../x", "/abs", "bad tag", ""):
            with self.assertRaises(protocol.CampaignError):
                protocol.validate_tag(value)

    def test_receipt_is_deterministic(self):
        campaign = protocol.load_campaign()
        self.assertEqual(protocol.expected_receipt(campaign), protocol.expected_receipt(campaign))
        self.assertEqual(protocol.expected_jobs(campaign), protocol.expected_jobs(campaign))
        self.assertEqual(protocol.expected_lock(campaign), protocol.expected_lock(campaign))


if __name__ == "__main__":
    unittest.main()

