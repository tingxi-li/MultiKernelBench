from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from .protocol import (
    CAMPAIGN_ID,
    CAMPAIGN_PATH,
    MATERIALS_PATH,
    ProtocolError,
    canonical_sha256,
    make_timing_manifest,
    read_json,
    validate_campaign,
    validate_materials,
    validate_timing_manifest,
)


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.campaign = read_json(CAMPAIGN_PATH)
        cls.materials = read_json(MATERIALS_PATH)

    @staticmethod
    def _admission(lock_sha: str) -> dict:
        return {
            "campaign_id": CAMPAIGN_ID,
            "campaign_lock_sha256": lock_sha,
            "complete": True,
            "pairs": [{
                "pair_id": "fused_softmax_f1_f4c",
                "gate_available": True,
                "timing_eligible": True,
                "classification": "runtime_estimand",
                "artifact_identity_sha256": {
                    "high": "a" * 64,
                    "low": "b" * 64,
                },
            }],
        }

    def test_exact_fused_only_design_and_predecessor_closure(self) -> None:
        validate_campaign(self.campaign)
        resolved = validate_materials(self.materials, self.campaign)
        self.assertEqual(
            [pair["pair_id"] for pair in self.campaign["pairs"]],
            ["fused_softmax_f1_f4c"],
        )
        status = read_json(resolved["v3_admission_run_status"])
        self.assertEqual(
            status["artifact_bundle_sha256"],
            canonical_sha256(status["artifact_sha256"]),
        )

    def test_pair_drift_and_predecessor_census_tampering_fail_closed(self) -> None:
        changed = deepcopy(self.campaign)
        changed["pairs"][0]["low"]["variant"] = "F3c"
        with self.assertRaisesRegex(ProtocolError, "differs from the v3 predecessor"):
            validate_materials(self.materials, changed)

        status_path = Path(self.materials["entries"]["v3_admission_run_status"]["path"])
        original_read = read_json

        def forged_read(path):
            value = original_read(path)
            if Path(path).resolve() == status_path.resolve():
                value = deepcopy(value)
                value["artifact_sha256"].pop(next(iter(value["artifact_sha256"])))
            return value

        with (
            patch(
                "ako_runs.controlled_followup.tilelang_abstraction_v4.protocol.read_json",
                side_effect=forged_read,
            ),
            self.assertRaisesRegex(ProtocolError, "status is inconsistent"),
        ):
            validate_materials(self.materials, self.campaign)

    def test_manifest_is_exactly_120_paired_randomized_rows(self) -> None:
        lock_sha = "c" * 64
        admission = self._admission(lock_sha)
        first = make_timing_manifest(self.campaign, lock_sha, admission)
        second = make_timing_manifest(self.campaign, lock_sha, admission)
        self.assertEqual(first, second)
        self.assertEqual(len(first["rows"]), 120)
        self.assertEqual(first["eligible_pair_ids"], ["fused_softmax_f1_f4c"])
        for distribution in ("positive", "withheld_signed"):
            for block in range(15):
                rows = [
                    row for row in first["rows"]
                    if row["distribution"] == distribution and row["block"] == block
                ]
                self.assertEqual(
                    {row["role"] for row in rows},
                    {"high", "low", "sham_a", "sham_b"},
                )
                self.assertEqual(
                    [row["implementation_side"] for row in rows].count("high"),
                    3,
                )
                self.assertEqual(sorted(row["position"] for row in rows), [0, 1, 2, 3])

        tampered = deepcopy(first)
        tampered["rows"][0]["position"] = 99
        with self.assertRaisesRegex(ProtocolError, "deterministic"):
            validate_timing_manifest(tampered, self.campaign, lock_sha, admission)

    def test_ineligible_pair_projects_zero_timing_rows(self) -> None:
        lock_sha = "d" * 64
        admission = self._admission(lock_sha)
        admission["pairs"][0].update({
            "timing_eligible": False,
            "classification": "excluded_fail_closed",
            "artifact_identity_sha256": {},
        })
        manifest = make_timing_manifest(self.campaign, lock_sha, admission)
        self.assertEqual(manifest["rows"], [])
        self.assertEqual(manifest["eligible_pair_ids"], [])


if __name__ == "__main__":
    unittest.main()
