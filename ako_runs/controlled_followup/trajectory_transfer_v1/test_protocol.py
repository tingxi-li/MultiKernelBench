from __future__ import annotations

import copy
import hashlib
import unittest

try:
    from . import protocol
except ImportError:  # pragma: no cover
    import protocol


def registry() -> dict:
    def sha(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def trajectory(origin: str, dsl: str, steps: int) -> dict:
        prefixes = [{
            "prefix_index": 0,
            "prefix_id": f"{origin}_p0",
            "introduced_mechanisms": [],
            "depends_on_steps": [],
            "terminal_status": "GATE_PASSED",
            "source_sha256": sha(f"{origin}-source-0"),
            "gate_receipt_sha256": sha(f"{origin}-gate-0"),
            "mechanism_audit_sha256": sha(f"{origin}-audit-0"),
            "step_artifact_sha256": None,
        }]
        for step in range(1, steps + 1):
            prefixes.append({
                "prefix_index": step,
                "prefix_id": f"{origin}_p{step}",
                "introduced_mechanisms": [f"mechanism_{step}"],
                "depends_on_steps": [1] if step == 3 else [],
                "terminal_status": "GATE_PASSED",
                "source_sha256": sha(f"{origin}-source-{step}"),
                "gate_receipt_sha256": sha(f"{origin}-gate-{step}"),
                "mechanism_audit_sha256": sha(f"{origin}-audit-{step}"),
                "step_artifact_sha256": sha(f"{origin}-step-{step}"),
                "composable": True,
            })
        return {"origin": origin, "origin_dsl": dsl, "prefixes": prefixes}

    trajectories = [
        trajectory("triton_origin", "triton", 3),
        trajectory("tilelang_origin", "tilelang", 2),
        trajectory("cuda_origin", "cuda_unlimited", 1),
    ]
    attempt_plans = {
        row["origin"]: [
            {"attempt_index": attempt, "config_sha256": sha(f"{row['origin']}-config-{attempt}")}
            for attempt in range(1, 20)
        ]
        for row in trajectories
    }
    return {
        "schema_version": 1,
        "record_type": "trajectory_transfer_donor_prefix_registry",
        "campaign_id": "trajectory_transfer_test",
        "operator": "matmul",
        "policy_status": "deferred",
        "destinations": list(protocol.DESTINATIONS),
        "transfer_modes": list(protocol.MODES),
        "translators": ["translator_b", "translator_a"],
        "order_seed": "frozen-test-seed",
        "gate_lock_sha256": sha("gate-lock"),
        "input_contract_sha256": sha("input-contract"),
        "retuned_attempts_per_cell": 19,
        "trajectories": trajectories,
        "retune_plan_sha256_by_origin": {
            origin: protocol._digest(plan) for origin, plan in attempt_plans.items()
        },
        "retune_attempt_plans_by_origin": attempt_plans,
    }


class ProtocolTest(unittest.TestCase):
    def test_t0_t1_protocol(self) -> None:
        source = registry()
        first = protocol.derive_manifest(source)
        second = protocol.derive_manifest(copy.deepcopy(source))

        # sum(L+1)=4+3+2=9; 9 prefixes x 4 destinations x 2 modes x 2 translators.
        self.assertEqual(first["census"]["donor_order_cells"], 144)
        self.assertEqual(first["census"]["valid_order_control_cells"], 64)
        self.assertEqual(first, second)
        ids = [row["cell_id"] for row in first["rows"] + first["order_control_rows"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(row["self_destination"] for row in first["rows"] if row["destination"] == row["origin_dsl"]))
        self.assertTrue(all(row["pipeline_positive_control"] == (row["self_destination"] and row["transfer_mode"] == "literal") for row in first["rows"]))

        self.assertTrue(all(row["donor_terminal_status"] == "GATE_PASSED" for row in first["rows"]))
        self.assertTrue(all(row["attempt_budget"] == (1 if row["transfer_mode"] == "literal" else 19) for row in first["rows"]))
        self.assertEqual(first["terminal_status_taxonomy"], protocol.TERMINAL_STATUS)
        ordinary_row = next(row for row in first["rows"] if not row["pipeline_positive_control"])
        for status, outcome_class in protocol.TERMINAL_STATUS.items():
            outcome = protocol.target_outcome(
                first,
                ordinary_row,
                status,
                "a" * 64,
                support_probe_receipt_sha256="b" * 64 if status == "UNSUPPORTED" else None,
                target_source_sha256="c" * 64 if status == "GATE_PASSED" else None,
            )
            self.assertEqual(outcome["terminal_status"], status)
            self.assertEqual(outcome["outcome_class"], outcome_class)
            self.assertEqual(outcome["timing_eligible"], status == "GATE_PASSED")
        positive = next(row for row in first["rows"] if row["pipeline_positive_control"])
        with self.assertRaisesRegex(protocol.ProtocolError, "byte-identical"):
            protocol.target_outcome(
                first, positive, "GATE_PASSED", "a" * 64, target_source_sha256="c" * 64
            )
        protocol.target_outcome(
            first,
            positive,
            "GATE_PASSED",
            "a" * 64,
            target_source_sha256=positive["donor_source_sha256"],
        )
        failed_positive = protocol.target_outcome(first, positive, "GATE_FAILED", "a" * 64)
        self.assertFalse(failed_positive["positive_control_passed"])
        self.assertEqual(len(first["valid_order_controls"]), 1)
        order = first["valid_order_controls"][0]
        self.assertNotEqual(order["donor_step_order"], order["control_step_order"])
        self.assertLess(order["control_step_order"].index(1), order["control_step_order"].index(3))
        control_rows = [row for row in first["order_control_rows"] if row["prefix_index"] > 0]
        self.assertTrue(all(row["applied_step_artifact_sha256"] for row in control_rows))
        self.assertTrue(all(row["constructed_source_sha256_required"] for row in control_rows))
        self.assertTrue(all(not row["pipeline_positive_control"] for row in control_rows))

        malformed = copy.deepcopy(source)
        malformed["trajectories"][0]["prefixes"][1]["introduced_mechanisms"] = ["tile", "pipeline"]
        with self.assertRaisesRegex(protocol.ProtocolError, "exactly one mechanism"):
            protocol.validate_registry(malformed)

        failed_donor = copy.deepcopy(source)
        failed_donor["trajectories"][0]["prefixes"][1]["terminal_status"] = "BUILD_FAILED"
        with self.assertRaisesRegex(protocol.ProtocolError, "donor prefixes must be gate-legal"):
            protocol.validate_registry(failed_donor)

        with self.assertRaisesRegex(
            protocol.ProtocolError,
            "execution is deferred.*source bindings are absent.*ABI is absent.*no subprocess started",
        ):
            protocol.refuse_launch(source)


if __name__ == "__main__":
    unittest.main()
