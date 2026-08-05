from __future__ import annotations

from copy import deepcopy
import unittest

from ako_runs.controlled_followup.decision_complexity_v1.protocol import (
    CAMPAIGN_ID,
    analyze_complete_outcomes,
    bind_outcome,
    build_manifest,
    launch,
    validate_contract,
    validate_manifest,
)


def contract() -> dict:
    def task(index: int, score: int) -> dict:
        marker = f"{index + 4:x}"
        axes = ["tile_m", "tile_n", "tile_k", "stages", "warps", "layout"]
        return {
            "task_id": f"task_{index}",
            "operator_family": f"family_{index}",
            "lane": "tilelang",
            "gate_sha256": marker * 64,
            "reference_sha256": f"{index + 7:x}" * 64,
            "tuning_dataset_sha256": f"{index + 10:x}" * 64,
            "terminal_dataset_sha256": f"{index + 13:x}" * 64,
            "visible_search_contract_sha256": f"{index + 1:x}" * 64,
            "valid_hint_sha256": f"{index + 4:x}" * 64,
            "target_candidate_sha256": f"{index + 7:x}" * 64,
            "target_gate_receipt_sha256": f"{index + 10:x}" * 64,
            "terminal_evaluator_lock_sha256": f"{index + 13:x}" * 64,
            "axis_order": axes,
            "target_values": {
                "tile_m": 128, "tile_n": 128, "tile_k": 32,
                "stages": 2, "warps": 8, "layout": "blocked",
            },
            "axis_domains": {
                "tile_m": [64, 128], "tile_n": [64, 128], "tile_k": [16, 32],
                "stages": [1, 2], "warps": [4, 8], "layout": ["linear", "blocked"],
            },
            "dependency_graph": {
                "tile_m": [], "tile_n": [], "tile_k": [], "stages": ["tile_k"],
                "warps": ["tile_m"], "layout": ["tile_n"],
            },
            "observed_complexity": {
                "score": score, "axis_count": 6, "dependency_depth": 2,
                "correctness_constraints": 3, "reachable_fraction": 0.5,
            },
        }

    return {
        "campaign_id": CAMPAIGN_ID,
        "state": "design_only_not_authorized",
        "launch_authorized": False,
        "randomization_seed_sha256": "f" * 64,
        "replicates": 2,
        "gpu_slots": 2,
        "budgets": {"max_attempts": 12, "tau_active_s": 600.0},
        "searchers": [
            {
                "searcher_id": "searcher_a",
                "immutable_revision_sha256": "1" * 64,
                "neutral_prompt_sha256": "2" * 64,
                "tool_contract_sha256": "3" * 64,
            }
        ],
        "tasks": [task(0, 2), task(1, 4), task(2, 6)],
    }


class ProtocolTest(unittest.TestCase):
    def test_manifest_census_controls_and_determinism(self) -> None:
        value = contract()
        first = build_manifest(value)
        second = build_manifest(value)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 42)
        self.assertEqual({row["arm"] for row in first}, {
            "c1_observed", "c2_open_1", "c2_open_3", "c2_open_6",
            "label_sham_a", "label_sham_b", "sensitivity_valid_hint",
        })
        shams = [row for row in first if row["arm"].startswith("label_sham_")]
        self.assertEqual(len({(row["task_id"], row["prompt_sha256"]) for row in shams}), 3)
        self.assertTrue(all(row["terminal_holdout_visibility"] == "offline_evaluator_only" for row in first))
        for task_id in {row["task_id"] for row in first}:
            for replicate in range(2):
                block = [row for row in first if row["task_id"] == task_id and row["replicate"] == replicate]
                self.assertEqual(sorted(row["position"] for row in block), list(range(7)))
                sham_slots = {row["gpu_slot"] for row in block if row["arm"].startswith("label_sham_")}
                self.assertEqual(len(sham_slots), 1)
        tampered = deepcopy(first)
        next(row for row in tampered if row["arm"] == "label_sham_a")["open_axes"] = ["tile_m"]
        with self.assertRaisesRegex(ValueError, "deterministic contract projection"):
            validate_manifest(tampered, value)

    def test_contract_rejects_leaky_or_incomplete_design(self) -> None:
        value = contract()
        value["tasks"][0]["terminal_dataset_sha256"] = value["tasks"][0]["tuning_dataset_sha256"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_contract(value)
        value = contract()
        value["tasks"][0]["axis_order"].pop()
        value["tasks"][0]["target_values"].pop("layout")
        with self.assertRaisesRegex(ValueError, "six"):
            validate_contract(value)

    def test_complete_outcomes_are_required(self) -> None:
        manifest = build_manifest(contract())
        outcomes = []
        for index, row in enumerate(manifest):
            if index % 2 == 0:
                ledger = [{
                    "attempt_index": 1, "candidate_sha256": "a" * 64,
                    "terminal_status": "GATE_PASSED", "active_s": 1.0,
                    "gate_legal": True, "candidate_latency_ms": 1.0,
                    "reference_latency_ms": 1.0,
                }]
            else:
                ledger = [{
                    "attempt_index": attempt, "candidate_sha256": "b" * 64,
                    "terminal_status": "GATE_FAILED", "active_s": 50.0,
                    "gate_legal": False, "candidate_latency_ms": None,
                    "reference_latency_ms": None,
                } for attempt in range(1, 13)]
            outcomes.append(bind_outcome(row, ledger))
        value = contract()
        result = analyze_complete_outcomes(value, manifest, outcomes, tau_attempts=12, tau_active_s=600.0)
        self.assertEqual(result["campaign_id"], CAMPAIGN_ID)
        self.assertEqual(result["status"], "design_only_noncontrolling")
        self.assertEqual(set(result["c2_primary"]["attempts"]["groups"]), {"c2_open_1", "c2_open_3", "c2_open_6"})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            analyze_complete_outcomes(value, manifest, outcomes[:-1], tau_attempts=12, tau_active_s=600.0)
        early_censor = bind_outcome(manifest[0], [{
            "attempt_index": 1, "candidate_sha256": "c" * 64,
            "terminal_status": "GATE_FAILED", "active_s": 1.0,
            "gate_legal": False, "candidate_latency_ms": None,
            "reference_latency_ms": None,
        }])
        bad = list(outcomes)
        bad[0] = early_censor
        with self.assertRaisesRegex(ValueError, "censored before"):
            analyze_complete_outcomes(value, manifest, bad, tau_attempts=12, tau_active_s=600.0)
        tampered_receipt = deepcopy(outcomes)
        tampered_receipt[0]["outcome_payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            analyze_complete_outcomes(value, manifest, tampered_receipt, tau_attempts=12, tau_active_s=600.0)
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            analyze_complete_outcomes(value, manifest[:7], outcomes[:7], tau_attempts=12, tau_active_s=600.0)
        post_event = deepcopy(outcomes)
        post_event[0] = bind_outcome(manifest[0], [
            {
                "attempt_index": attempt, "candidate_sha256": "a" * 64,
                "terminal_status": "GATE_PASSED", "active_s": 1.0,
                "gate_legal": True, "candidate_latency_ms": 1.0,
                "reference_latency_ms": 1.0,
            }
            for attempt in (1, 2)
        ])
        with self.assertRaisesRegex(ValueError, "continues after"):
            analyze_complete_outcomes(value, manifest, post_event, tau_attempts=12, tau_active_s=600.0)

    def test_launch_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "launch forbidden"):
            launch()


if __name__ == "__main__":
    unittest.main()
