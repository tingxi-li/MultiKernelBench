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
            {row["gpu_slot"] for row in self.manifest["rows"]}, set(range(4))
        )

    def test_contract_binds_lock_paths_and_wave_policy(self) -> None:
        schedule = self.contract["execution_schedule"]
        self.assertEqual(schedule["gpu_lock_paths"], list(protocol.GPU_LOCK_PATHS))
        self.assertEqual(
            schedule["within_wave_common_overlap"],
            "one_parent_release_witness_strictly_inside_all_child_lifetimes",
        )
        self.assertEqual(schedule["wave_ready_timeout_s"], 300)
        self.assertEqual(
            schedule["between_wave_separation"],
            "min_launch_ge_previous_max_completion",
        )
        self.assertIn("new_successor", schedule["resume_policy"])

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
