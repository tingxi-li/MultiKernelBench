"""CPU-only tests for reciprocal-v2 preregistration and inference."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze
import make_manifests
import make_retune_plans
import protocol
import validate


class ReciprocalV2Test(unittest.TestCase):
    def test_generated_documents_are_current(self):
        make_manifests.check()
        make_retune_plans.check()

    def test_retune_plans_freeze_every_attempt(self):
        plans = make_retune_plans.documents()
        self.assertEqual(len(plans), 24)
        for data in plans.values():
            plan = __import__("json").loads(data)
            self.assertEqual(len(plan["candidates"]), 19)
            self.assertTrue(all(row["failure_consumes_attempt"] for row in plan["candidates"]))

    def test_factorial_and_budgets(self):
        documents = validate.validate_static()
        self.assertEqual(len(protocol.expected_cells()), 48)
        for manifest in documents["manifests"].values():
            self.assertEqual(manifest["job_count"], 48)
            retuned = [row for row in manifest["jobs"] if row["transfer_mode"] == "retuned"]
            self.assertEqual(len(retuned), 24)
            self.assertTrue(all(row["candidate_attempt_budget"] == 19 for row in retuned))

    def test_fifteen_unique_complete_block_orders(self):
        orders = protocol.block_orders()
        expected = {protocol.cell_id(*axes) for axes in protocol.expected_cells()}
        self.assertEqual(len(orders), 15)
        self.assertEqual(len({tuple(order) for order in orders}), 15)
        self.assertTrue(all(len(order) == 48 and set(order) == expected for order in orders))
        self.assertEqual(len(protocol.block_order_sha256()), 64)

    def test_cuda_origin_is_source_grounded(self):
        card = make_manifests.cuda_card()
        self.assertEqual(card["origin"]["historical_key"],
                         "cuda_unlimited.D.128x128x32.kc2048.s3.fp16.precast")
        self.assertEqual(card["correctness_amendment"]["candidate_kc_order"],
                         list(protocol.KC_LADDER))

    def test_launch_is_blocked_before_translations(self):
        blockers = validate.dependency_blockers("audit")
        self.assertTrue(any("implementation registry" in row for row in blockers))
        self.assertTrue(any("translator isolation" in row for row in blockers))
        self.assertTrue(any("recipe-resolution" in row for row in blockers))
        self.assertTrue(any("prelaunch provenance" in row for row in blockers))
        screen = validate.dependency_blockers("screen")
        self.assertTrue(any("audit receipt" in row for row in screen))
        primary = validate.dependency_blockers("primary")
        self.assertTrue(any("selection receipt" in row for row in primary))

    def test_exact_sign_test(self):
        self.assertEqual(analyze._sign_test_p([1.0] * 15), 2 / (2**15))
        self.assertEqual(analyze._sign_test_p([0.0] * 15), 1.0)

    def test_complete_synthetic_analysis(self):
        manifest = validate.validate_static()["manifests"]["primary"]
        jobs = {row["cell_id"]: row for row in manifest["jobs"]}
        bindings = {
            cid: {
                "selection_receipt_sha256": f"selection-{index}",
                "selected_attempt": 0,
                "candidate_source_sha256": f"source-{index}",
            }
            for index, cid in enumerate(jobs)
        }
        common = {
            "primary_manifest_sha256": "manifest",
            "source_freeze_sha256": "freeze",
            "implementation_registry_sha256": "registry",
            "recipe_resolution_lock_sha256": "resolution",
            "gate_lock_sha256": "gate",
            "block_order_sha256": protocol.block_order_sha256(),
            "primary_launch_receipt_sha256": "primary-launch",
        }
        rows = []
        destination_scale = {name: index + 1 for index, name in enumerate(protocol.DESTINATIONS)}
        for block, order in enumerate(protocol.block_orders()):
            for position, cid in enumerate(order):
                job = jobs[cid]
                rows.append({
                    "campaign_id": protocol.CAMPAIGN_ID,
                    "cell_id": cid,
                    "recipe_origin": job["recipe_origin"],
                    "destination_dsl": job["destination_dsl"],
                    "transfer_mode": job["transfer_mode"],
                    "translator": job["translator"],
                    "block": block,
                    "block_position": position,
                    **common,
                    **bindings[cid],
                    "gate_pass": True,
                    "terminal_eligible": True,
                    "median_ms": float(destination_scale[job["destination_dsl"]]) + block / 1000,
                    "physical_gpu": 0,
                    "logical_device": "cuda:0",
                    "gpu_uuid": protocol.TIMING_GPU_UUID,
                    "timing_distribution": "rand_seed0_precast",
                    "process_instance_id": f"process-{block}-{position}",
                })
        grouped = analyze.validate_records(rows, bindings, common)
        result = analyze.summarize(grouped)
        self.assertEqual(result["cell_count"], 48)
        self.assertEqual(result["record_count"], 720)
        self.assertEqual(len(result["translator_bounds"]), 24)
        self.assertEqual(len(result["paired_destination_contrasts"]), 72)
        self.assertEqual(len(result["paired_origin_destination_interaction_contrasts"]), 72)
        self.assertEqual(len(result["persistent_destination_effects"]), 6)
        self.assertTrue(all(math.isclose(row["slow_over_fast_cell_median_ratio"], 1.0)
                            for row in result["translator_bounds"]))
        swapped = list(rows)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaisesRegex(ValueError, "serialized"):
            analyze.validate_records(swapped, bindings, common)


if __name__ == "__main__":
    unittest.main()
