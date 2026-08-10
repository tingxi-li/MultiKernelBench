from __future__ import annotations

import copy
import unittest

from . import protocol


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = protocol.make_contract()
        cls.manifest = protocol.make_manifest(cls.contract)

    def test_materials_and_exact_census(self) -> None:
        self.assertEqual(len(self.contract["materials"]["candidates"]), 19)
        self.assertEqual(self.manifest["requested_trajectories"], 24)
        self.assertEqual(len(self.manifest["rows"]), 24)
        self.assertEqual(
            {row["gpu_slot"] for row in self.manifest["rows"]}, {0}
        )
        self.assertEqual(
            {row["gpu_uuid"] for row in self.manifest["rows"]},
            {protocol.GPU_UUIDS[0]},
        )
        self.assertEqual(
            [row["launch_sequence"] for row in self.manifest["rows"]],
            list(range(1, 25)),
        )

    def test_contract_binds_single_gpu_and_serial_policy(self) -> None:
        schedule = self.contract["execution_schedule"]
        self.assertEqual(schedule["gpu_lock_paths"], list(protocol.GPU_LOCK_PATHS))
        self.assertEqual(
            schedule["order"], "serialized_manifest_order"
        )
        self.assertEqual(
            schedule["overlap_policy"],
            "strict_previous_completion_before_next_launch",
        )
        self.assertEqual(schedule["simultaneous_trajectories"], 1)
        self.assertIn("new_successor", schedule["resume_policy"])

    def test_v1_randomization_is_preserved(self) -> None:
        from ako_runs.controlled_followup.decision_complexity_ada_v1 import (
            protocol as v1_protocol,
        )

        v1_contract = v1_protocol.make_contract()
        v1_rows = v1_protocol.make_manifest(v1_contract)["rows"]
        for before, after in zip(v1_rows, self.manifest["rows"], strict=True):
            self.assertEqual(
                (before["replicate"], before["block_position"], before["arm"]),
                (after["replicate"], after["block_position"], after["arm"]),
            )
            self.assertEqual(
                before["execution_contract"]["candidate_order"],
                after["execution_contract"]["candidate_order"],
            )
            for attempt_index in range(
                1, len(before["execution_contract"]["candidate_order"]) + 1
            ):
                self.assertEqual(
                    v1_protocol.timing_pair_order(before, attempt_index),
                    protocol.timing_pair_order(after, attempt_index),
                )

    def test_nested_treatments_and_controls(self) -> None:
        for replicate in range(4):
            block = [
                row for row in self.manifest["rows"] if row["replicate"] == replicate
            ]
            by_arm = {row["arm"]: row for row in block}
            sets = {
                arm: set(by_arm[arm]["execution_contract"]["candidate_order"])
                for arm in ("c2_open_1", "c2_open_2", "c2_open_4")
            }
            self.assertEqual([len(sets[arm]) for arm in sets], [3, 6, 19])
            self.assertLessEqual(sets["c2_open_1"], sets["c2_open_2"])
            self.assertLessEqual(sets["c2_open_2"], sets["c2_open_4"])
            sham_a = by_arm["label_sham_a"]
            sham_b = by_arm["label_sham_b"]
            self.assertEqual(
                sham_a["execution_contract_sha256"],
                sham_b["execution_contract_sha256"],
            )
            self.assertEqual(
                by_arm["sensitivity_target_hint"]["execution_contract"][
                    "candidate_order"
                ][0],
                protocol.TARGET_CELL_ID,
            )

    def test_contract_and_manifest_tampering_fail(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["materials"]["candidates"][0]["record_sha256"] = "0" * 64
        contract["materials_sha256"] = protocol.canonical_sha256(contract["materials"])
        with self.assertRaisesRegex(protocol.ProtocolError, "material projection"):
            protocol.validate_contract(contract)

        manifest = copy.deepcopy(self.manifest)
        manifest["rows"][0]["execution_contract"]["candidate_order"].reverse()
        with self.assertRaisesRegex(protocol.ProtocolError, "deterministic projection|sham execution"):
            protocol.validate_manifest(self.contract, manifest)

    def test_launch_requires_separate_successor_lock(self) -> None:
        with self.assertRaisesRegex(protocol.ProtocolError, "execution lock"):
            protocol.refuse_launch()


if __name__ == "__main__":
    unittest.main()
