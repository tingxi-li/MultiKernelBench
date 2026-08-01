from __future__ import annotations

import math
import unittest

from ..margin_report import (
    GATE,
    RAW,
    MarginReportError,
    _record_json,
    build_report,
    validate_record,
)


THRESHOLDS = {
    "unit_gate": {
        "max_abs_err": 1.0,
        "nonfinite_count": 0.0,
    }
}


def unit_row(**updates):
    row = {
        "candidate": "candidate",
        "case_id": "case",
        "gate_id": "unit_gate",
        "seed_index": 0,
        "gate_pass": True,
        "metrics": {"max_abs_err": 0.5, "nonfinite_count": 0.0},
        "threshold_failures": [],
    }
    row.update(updates)
    return row


class MarginValidationTests(unittest.TestCase):
    def test_per_record_maximum(self) -> None:
        record = validate_record(
            unit_row(), THRESHOLDS, source_file="unit.jsonl", source_line=1
        )
        self.assertEqual(record.maximum.metric, "max_abs_err")
        self.assertEqual(record.maximum.utilization, 0.5)
        self.assertTrue(record.gate_pass)

    def test_missing_metric_fails_closed(self) -> None:
        with self.assertRaisesRegex(MarginReportError, "missing required metric"):
            validate_record(
                unit_row(metrics={"max_abs_err": 0.5}),
                THRESHOLDS,
                source_file="unit.jsonl",
                source_line=2,
            )

    def test_nonfinite_metrics_fail_closed(self) -> None:
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(MarginReportError, "must be finite"):
                    validate_record(
                        unit_row(
                            metrics={
                                "max_abs_err": invalid,
                                "nonfinite_count": 0.0,
                            }
                        ),
                        THRESHOLDS,
                        source_file="unit.jsonl",
                        source_line=3,
                    )

    def test_gate_decision_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(MarginReportError, "disagrees with metrics"):
            validate_record(
                unit_row(
                    metrics={"max_abs_err": 2.0, "nonfinite_count": 0.0}
                ),
                THRESHOLDS,
                source_file="unit.jsonl",
                source_line=4,
            )

    def test_positive_value_over_zero_threshold_is_json_safe(self) -> None:
        record = validate_record(
            unit_row(
                gate_pass=False,
                metrics={"max_abs_err": 0.5, "nonfinite_count": 1.0},
                threshold_failures=["nonfinite_count=1>0"],
            ),
            THRESHOLDS,
            source_file="unit.jsonl",
            source_line=5,
        )
        self.assertTrue(math.isinf(record.maximum.utilization))
        self.assertEqual(_record_json(record)["max_threshold_utilization"], "infinity")


class CompletedCampaignMarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report(gate_path=GATE, raw_dir=RAW)

    def test_complete_real_candidate_census_and_failure_counts(self) -> None:
        self.assertEqual(self.report["real_candidate_records"], 21_504)
        self.assertEqual(self.report["passing_records"], 14_336)
        self.assertEqual(self.report["failing_records"], 7_168)
        self.assertEqual(len(self.report["groups"]), 42)

    def test_closest_failure_is_a_record_minimum_not_a_group_maximum(self) -> None:
        closest = self.report["closest_failing_record"]
        self.assertEqual(closest["candidate"], "phase1_triton_A")
        self.assertEqual(closest["case_id"], "opposing_means")
        self.assertEqual(closest["gate_id"], "semantic_q32")
        self.assertEqual(closest["seed_index"], 465)
        self.assertEqual(closest["limiting_metric"], "max_abs_err")
        self.assertTrue(
            math.isclose(
                closest["max_threshold_utilization"],
                1.21337957962169,
                rel_tol=1e-14,
                abs_tol=0.0,
            )
        )

    def test_worst_failure(self) -> None:
        worst = self.report["worst_failing_record"]
        self.assertEqual(worst["candidate"], "phase1_triton_B")
        self.assertEqual(worst["case_id"], "legacy_u01")
        self.assertEqual(worst["gate_id"], "conformance_mixed")
        self.assertEqual(worst["seed_index"], 251)
        self.assertEqual(worst["limiting_metric"], "abs_signed_bias")
        self.assertTrue(
            math.isclose(
                worst["max_threshold_utilization"],
                9322.150545631304,
                rel_tol=1e-14,
                abs_tol=0.0,
            )
        )

    def test_group_has_failure_counts_quantiles_and_per_record_values(self) -> None:
        group = next(
            item
            for item in self.report["groups"]
            if item["candidate"] == "phase1_triton_A"
            and item["case_id"] == "opposing_means"
            and item["gate_id"] == "semantic_q32"
        )
        self.assertEqual(group["failing_records"], 512)
        self.assertEqual(group["passing_records"], 0)
        self.assertEqual(
            set(group["max_threshold_utilization_quantiles"]),
            {"p00", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p100"},
        )
        series = group["per_record_max_threshold_utilization"]
        self.assertEqual(series["axis"]["start"], 0)
        self.assertEqual(series["axis"]["stop_exclusive"], 512)
        self.assertEqual(len(series["values"]), 512)


if __name__ == "__main__":
    unittest.main()
