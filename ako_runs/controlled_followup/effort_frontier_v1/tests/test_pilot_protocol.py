from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


class PilotProtocolTest(unittest.TestCase):
    def test_noncontrolling_two_by_two_pilot_is_capped_and_budgeted(self):
        policy = json.loads((HERE / "pilot_v1.json").read_text())
        design = policy["pilot_design"]
        self.assertEqual(design["lanes"], ["cublaslt", "triton"])
        self.assertEqual(design["replicates_per_lane"], 2)
        self.assertEqual(design["checkpoint_count"], 2)
        self.assertEqual(len(policy["trajectories"]), 4)
        self.assertFalse(design["controlling"])
        self.assertFalse(policy["use_and_claim_limits"]["lane_performance_inference_permitted"])
        cap = policy["token_policy"]["per_trajectory_provider_token_admission_cap"]
        self.assertEqual(cap, 100_000)
        self.assertEqual(policy["budget"]["provider_token_admission_budget"], cap * 4)
        self.assertEqual(policy["budget"]["aggregate_active_effort_ceiling_s"], 4 * 7200)
        self.assertEqual(policy["budget"]["checkpoint_observations"], 4 * 2)

    def test_identity_and_imbalance_are_explicit_and_parent_bytes_are_unchanged(self):
        policy = json.loads((HERE / "pilot_v1.json").read_text())
        identity = policy["model_identity_policy"]
        self.assertEqual(identity["record_on_every_provider_response"], [
            "requested_alias", "response_model", "resolved_at_utc"
        ])
        self.assertFalse(identity["immutable_revision_required"])
        self.assertEqual(policy["use_and_claim_limits"]["named_limitation"],
                         "lane_by_gpu_imbalance")
        parent = policy["historical_parent"]
        for path_key, hash_key in (
            ("manifest_path", "manifest_sha256"),
            ("prereg_index_path", "prereg_index_sha256"),
        ):
            path = HERE / parent[path_key]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), parent[hash_key])


if __name__ == "__main__":
    unittest.main()
