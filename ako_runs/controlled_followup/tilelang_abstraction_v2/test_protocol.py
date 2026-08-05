"""Compact CPU-only checks for the TileLang abstraction protocol."""
from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol


def implementation(level: str) -> dict:
    prefix = "1" if level == "TL-H" else "2"
    return {
        "level": level,
        "operator": "matmul",
        "operator_family": "matmul",
        "shape": [4096, 4096, 4096],
        "gate_lock_sha256": "3" * 64,
        "input_contract_sha256": "4" * 64,
        "algorithm": "blocked-gemm",
        "dtype": "fp16-fp32acc",
        "tile": [128, 128, 32],
        "pipeline_depth": 2,
        "threads": 256,
        "instruction_family": "mma.sync+cp.async",
        "logical_work": {"flops": "2MNK", "global_bytes": "matched"},
        "dynamic_work": {"mma": 33_554_432, "global_bytes": 1_000},
        "terminal_status": "GATE_PASSED",
        "implementation_sha256": prefix * 64,
        "source_sha256": prefix * 64,
        "gate_receipt_sha256": "5" * 64,
        "ir_sha256": "6" * 64,
        "ptx_sha256": "7" * 64,
        "sass_sha256": "8" * 64,
        "dynamic_work_receipt_sha256": "9" * 64,
        "resource_receipt_sha256": "a" * 64,
    }


class ProtocolTest(unittest.TestCase):
    def test_pair_audit_is_deterministic_and_excludes_mismatch(self):
        matched = {
            "pair_id": "b",
            "implementer_id": "implementer_a",
            "implementation_order": "H_then_M",
            "isolation_receipt_sha256": "b" * 64,
            "randomization_receipt_sha256": "c" * 64,
            "high": implementation("TL-H"),
            "low": implementation("TL-M"),
        }
        excluded = deepcopy(matched)
        excluded["pair_id"] = "a"
        excluded["low"]["algorithm"] = "split-k"
        forward = protocol.audit_manifest([matched, excluded])
        reverse = protocol.audit_manifest([excluded, matched])
        self.assertEqual(forward, reverse)
        self.assertEqual([row["pair_id"] for row in forward["pairs"]], ["a", "b"])
        self.assertEqual(forward["pairs"][0]["classification"], "capability_only")
        self.assertFalse(forward["pairs"][0]["included_in_runtime_estimand"])
        self.assertEqual(forward["pairs"][0]["mismatch_fields"], ["algorithm"])
        self.assertEqual(forward["census"], {
            "total_pairs": 2, "runtime_estimand_pairs": 1,
            "capability_only_pairs": 1,
        })
        self.assertEqual(forward["claim_scope"], "local_only_unverified")
        self.assertFalse(forward["generalization_design_ready"])
        self.assertFalse(forward["material_receipts_verified"])
        failed = deepcopy(matched)
        failed["pair_id"] = "failed"
        failed["low"]["terminal_status"] = "GATE_FAILED"
        with self.assertRaisesRegex(ValueError, "GATE_PASSED"):
            protocol.audit_manifest([failed])
        unrandomized = deepcopy(matched)
        unrandomized.pop("randomization_receipt_sha256")
        with self.assertRaisesRegex(ValueError, "randomization receipt"):
            protocol.audit_manifest([unrandomized])

    def test_interval_decisions_and_epsilon_validation(self):
        decide = protocol.classify_interval
        self.assertEqual(decide(-0.20, -0.10, delta_hw=0.02, epsilon=0.05)["direction"],
                         "lower_level_faster")
        self.assertEqual(decide(0.10, 0.20, delta_hw=0.02, epsilon=0.05)["direction"],
                         "higher_level_faster")
        self.assertEqual(decide(-0.01, 0.01, delta_hw=0.02, epsilon=0.05), {
            "direction": "unresolved", "equivalence": "within_equivalence_bound",
        })
        self.assertEqual(decide(-0.06, 0.06, delta_hw=0.02, epsilon=0.05), {
            "direction": "unresolved", "equivalence": "not_demonstrated",
        })
        overlapping = decide(-0.04, -0.03, delta_hw=0.02, epsilon=0.05)
        self.assertEqual(overlapping, {
            "direction": "lower_level_faster", "equivalence": "within_equivalence_bound",
        })
        with self.assertRaisesRegex(ValueError, "at least delta_hw"):
            decide(-0.01, 0.01, delta_hw=0.02, epsilon=0.01)

    def test_a2_contract_and_launch_refusal(self):
        shared = {
            "candidate_attempt_budget": 20,
            "wall_clock_seconds": 3600,
            "hardware_binding_sha256": "b" * 64,
            "task_contract_sha256": "c" * 64,
            "gate_feedback_contract_sha256": "d" * 64,
            "searcher_lock_sha256": "1" * 64,
            "prompt_lock_sha256": "2" * 64,
            "tool_lock_sha256": "3" * 64,
            "isolation_policy_sha256": "4" * 64,
            "randomization_lock_sha256": "5" * 64,
        }
        contract = {
            "status": "design_only",
            "arms": {arm: deepcopy(shared) for arm in protocol.A2_ARMS},
            "terminal_reference": {
                "visibility": "hidden_until_search_complete",
                "gate_legal": True,
                "target_ratio": 1.05,
                "reference_sha256": "6" * 64,
                "gate_receipt_sha256": "7" * 64,
                "tuning_dataset_sha256": "8" * 64,
                "terminal_dataset_sha256": "9" * 64,
            },
        }
        protocol.validate_a2_contract(contract)
        contract["arms"]["TL-M-only"]["candidate_attempt_budget"] = 21
        with self.assertRaisesRegex(ValueError, "differ"):
            protocol.validate_a2_contract(contract)
        contract = {
            "status": "design_only",
            "arms": {arm: deepcopy(shared) for arm in protocol.A2_ARMS},
            "terminal_reference": {
                "visibility": "hidden_until_search_complete", "gate_legal": True,
                "target_ratio": 1.05, "reference_sha256": "6" * 64,
                "gate_receipt_sha256": "7" * 64, "tuning_dataset_sha256": "8" * 64,
                "terminal_dataset_sha256": "9" * 64,
            },
        }
        contract["arms"]["TL-H-only"]["candidate_attempt_budget"] = 0
        contract["arms"]["TL-M-only"]["candidate_attempt_budget"] = 0
        with self.assertRaisesRegex(ValueError, "positive integer"):
            protocol.validate_a2_contract(contract)
        with self.assertRaisesRegex(protocol.LaunchRefused, "current policy"):
            protocol.authorize_launch()
        with self.assertRaisesRegex(protocol.LaunchRefused, "no launch path"):
            protocol.authorize_launch(
                policy_authorized=True,
                material_inputs=protocol.REQUIRED_LAUNCH_INPUTS,
            )


if __name__ == "__main__":
    unittest.main()
