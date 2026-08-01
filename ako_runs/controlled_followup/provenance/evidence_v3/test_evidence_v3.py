"""CPU-only contract tests for evidence-v3 selection and archive encoding."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_evidence as evidence


class EvidenceV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = evidence.selection()
        cls.names = {evidence.relative(path) for path in cls.selection.categories}
        cls.document = evidence.manifest(cls.selection)

    def test_required_documents_campaigns_and_results_are_selected(self) -> None:
        required = {evidence.relative(path) for path in evidence.REQUIRED_SELECTED}
        self.assertLessEqual(required, self.names)
        for root in evidence.PROTOCOL_ROOTS:
            prefix = evidence.relative(root) + "/"
            self.assertTrue(any(name.startswith(prefix) for name in self.names), prefix)

        matmul_raw = evidence.MATMUL / "results/raw"
        selected_matmul_raw = {
            name for name in self.names if name.startswith(evidence.relative(matmul_raw) + "/")
        }
        self.assertEqual(len(selected_matmul_raw), 6)
        self.assertIn(
            evidence.relative(evidence.SAME_SEED / "results/raw/measurements.jsonl"),
            self.names,
        )

    def test_all_plain_reachability_results_are_selected(self) -> None:
        root = evidence.FOLLOWUP / "fused_reachability_v2/results"
        expected = {
            evidence.relative(path)
            for path in root.rglob("*")
            if path.is_file() and evidence._eligible(path)
        }
        observed = {name for name in self.names if name.startswith(evidence.relative(root) + "/")}
        self.assertEqual(observed, expected)
        self.assertGreater(len(observed), 1600)

    def test_nested_pairs_are_fixed_verified_and_complete(self) -> None:
        validations = self.selection.validations["nested_evidence"]
        self.assertEqual(len(validations), len(evidence.NESTED_EVIDENCE))
        self.assertTrue(all(row["verified"] is True for row in validations))
        labels = {row["label"] for row in validations}
        self.assertIn("fused_same_seed_stress_v2", labels)
        self.assertIn("reciprocal_v2_preregistration", labels)
        self.assertIn("reciprocal_v2_production_preregistration_v2", labels)
        self.assertIn("convergence_v2_preregistration_v2", labels)
        self.assertIn("effort_frontier_v1_preregistration", labels)
        for specification in evidence.NESTED_EVIDENCE:
            self.assertIn(specification.index, self.names)
            self.assertIn(specification.bundle, self.names)

    def test_selection_has_no_forbidden_loose_files_or_interim_archives(self) -> None:
        nested = {
            name
            for specification in evidence.NESTED_EVIDENCE
            for name in (specification.index, specification.bundle)
        }
        for name in self.names:
            if name in nested:
                continue
            parts = Path(name).parts
            self.assertTrue(evidence.EXCLUDED_PARTS.isdisjoint(parts), name)
            self.assertFalse(Path(name).name.endswith(".lock"), name)
            self.assertNotIn(Path(name).suffix.lower(), evidence.EXCLUDED_SUFFIXES, name)
            self.assertIsNone(
                evidence._excluded_reason(evidence.REPO / name),
                name,
            )
        for version in range(3, 9):
            prefix = f"ako_runs/controlled_followup/provenance/evidence_v2/fused_postreview_v{version}."
            self.assertFalse(any(name.startswith(prefix) for name in self.names), prefix)

    def test_prospective_results_are_excluded_except_blocked_preflight(self) -> None:
        allowed = {
            evidence.relative(path)
            for root in evidence.PROTOCOL_ROOTS
            for path in evidence._blocked_preflight_files(root)
        }
        observed = {
            name
            for name in self.names
            if any(
                name.startswith(evidence.relative(root / directory) + "/")
                for root in evidence.PROTOCOL_ROOTS
                for directory in ("receipts", "results")
            )
            and "preflight" in Path(name).name
        }
        self.assertEqual(observed, allowed)
        for name in allowed:
            self.assertIn("preflight", Path(name).name)
            path = (evidence.REPO / name).resolve()
            self.assertIn(
                "new_campaign_blocked_preflight",
                self.selection.categories[path],
            )

    def test_campaign_states_are_receipt_bound_and_do_not_overclaim(self) -> None:
        states = self.document["campaign_states"]
        self.assertEqual(
            states["matmul_v4_instrument_v1"]["state"], "complete_receipt_bound"
        )
        self.assertEqual(states["matmul_v4_instrument_v1"]["observed_unique_records"], 66852)
        self.assertEqual(
            states["fused_same_seed_stress_v2"]["state"],
            "complete_receipt_and_archive_bound",
        )
        self.assertEqual(states["fused_same_seed_stress_v2"]["observed_unique_records"], 10240)
        self.assertEqual(states["fused_same_seed_stress_v2"]["effective_shared_seed_n"], 512)
        self.assertFalse(states["empirical_round2_program_complete"])
        self.assertEqual(
            states["convergence_v2"]["blocked_preflight_receipts_included"], 2
        )
        self.assertEqual(
            states["fused_epilogue_crossed_v1"][
                "blocked_preflight_receipts_included"
            ],
            1,
        )
        for campaign in (
            "convergence_v2",
            "fused_epilogue_crossed_v1",
            "reciprocal_v2",
            "effort_frontier_v1",
        ):
            self.assertIn("no_empirical_result", states[campaign]["state"])

    def test_release_contract_rechecks_fixed_selection_claims(self) -> None:
        entries = evidence._validate_manifest(self.document)
        evidence._validate_release_contract(self.document, entries)
        self.assertEqual(
            self.document["validations"]["plain_reachability_result_entries"],
            evidence.EXPECTED_PLAIN_REACHABILITY_RESULT_ENTRIES,
        )
        altered = copy.deepcopy(self.document)
        altered["campaign_states"]["convergence_v2"]["state"] = "complete"
        with self.assertRaises(evidence.EvidenceError):
            evidence._validate_release_contract(
                altered, evidence._validate_manifest(altered)
            )

    def test_custom_name_is_bound_into_manifest(self) -> None:
        toy = evidence.Selection()
        toy.add(HERE / "README.md", "toy")
        document = evidence.manifest(toy, bundle_id="custom-preview-v1")
        self.assertEqual(document["bundle_id"], "custom-preview-v1")

    def test_small_archive_is_deterministic_and_self_verifying(self) -> None:
        toy = evidence.Selection()
        toy.add(HERE / "README.md", "toy")
        document = evidence.manifest(toy, bundle_id="toy-v1")
        first = evidence._tar_bytes(document)
        second = evidence._tar_bytes(document)
        self.assertEqual(hashlib.sha256(first).digest(), hashlib.sha256(second).digest())
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "toy-v1.tar.gz"
            bundle.write_bytes(first)
            evidence._verify_bundle(bundle, document)

    def test_archived_blocked_preflight_is_semantically_rechecked(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            receipt = root / "launch_preflight_test.json"
            receipt.write_text(
                json.dumps({"ready": True, "action": "launch_permitted"}),
                encoding="utf-8",
            )
            toy = evidence.Selection()
            toy.add(receipt, "new_campaign_blocked_preflight")
            document = evidence.manifest(toy, bundle_id="false-preflight-v1")
            bundle = root / "false-preflight-v1.tar.gz"
            bundle.write_bytes(evidence._tar_bytes(document))
            with self.assertRaises(evidence.EvidenceError):
                evidence._verify_bundle(bundle, document)
        for receipt in (
            {"ready": True, "launch_ready": False},
            {"ready": False, "action": "launch_permitted"},
            {"ready": False, "zero_gpu_processes_started": False},
            {"ready": False, "zero_gpu_processes_started": None},
        ):
            with self.assertRaises(evidence.EvidenceError):
                evidence._validate_blocked_preflight_value(receipt, "test")

    def test_archive_paths_and_partial_trees_fail_closed(self) -> None:
        for name in ("../escape", "/absolute", "a/../b", "a\\b", "./a"):
            with self.assertRaises(evidence.EvidenceError):
                evidence._validate_member_name(name)
        self.assertIsNone(
            evidence._excluded_reason(
                evidence.FOLLOWUP / "fused_epilogue_crossed_v1/artifacts/receipt.json"
            )
        )
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            (root / "record.json.partial").write_text("{}", encoding="utf-8")
            with self.assertRaises(evidence.EvidenceError):
                evidence.Selection().add_tree(root, "temporary-test")
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            (root / "linked.md").symlink_to(HERE / "README.md")
            with self.assertRaises(evidence.EvidenceError):
                evidence.Selection().add_tree(root, "symlink-test")

    def test_normalized_tar_metadata(self) -> None:
        info = evidence.normalized_info("payload", 17)
        self.assertEqual((info.size, info.mode, info.mtime, info.uid, info.gid), (17, 0o644, 0, 0, 0))
        self.assertEqual(info.uname, "")
        self.assertEqual(info.gname, "")


if __name__ == "__main__":
    unittest.main()
