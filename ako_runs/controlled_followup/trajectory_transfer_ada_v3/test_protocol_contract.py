from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import protocol


class ContractTest(unittest.TestCase):
    def test_contract_and_deterministic_censuses(self) -> None:
        protocol.validate_contract()
        self.assertEqual(
            protocol.read_json(protocol.CAMPAIGN_PATH)["inference"],
            protocol.INFERENCE_CONTRACT,
        )
        first = protocol.make_admission_manifest()
        self.assertEqual(first, protocol.make_admission_manifest())
        self.assertEqual(first["expected_rows"], 160)
        self.assertEqual(len({row["entry_id"] for row in first["rows"]}), 160)

        fixed = [row for row in first["rows"] if row["adaptation"] == "donor_fixed"]
        tuned = [row for row in first["rows"] if row["adaptation"] == "bounded_retune"]
        self.assertEqual(len(fixed), 8)
        self.assertEqual(len(tuned), 152)
        for destination in protocol.DESTINATIONS:
            for state in protocol.MECHANISM_STATES:
                pair = [row for row in tuned if row["destination"] == destination and row["mechanism_state"] == state]
                self.assertEqual([row["attempt_index"] for row in pair], list(range(1, 20)))
                self.assertEqual([row["grid_id"] for row in pair], list(protocol.GRID_IDS))
                self.assertEqual(len({protocol.canonical_sha256(row["origin_job"]) for row in pair}), 19)
        tilelang = [row for row in first["rows"] if row["destination"] == "tilelang"]
        for adaptation in protocol.ADAPTATIONS:
            grids = (protocol.FIXED_GRID,) if adaptation == "donor_fixed" else protocol.GRID_IDS
            for grid in grids:
                pair = [
                    row for row in tilelang
                    if row["adaptation"] == adaptation and row["grid_id"] == grid
                ]
                self.assertEqual([row["mechanism_state"] for row in pair], ["off", "on"])
                self.assertLess(first["rows"].index(pair[0]), first["rows"].index(pair[1]))
        self.assertEqual(
            {row["destination"]: row["route"] for row in first["rows"]},
            protocol.ROUTE_BY_DESTINATION,
        )
        for row in first["rows"]:
            if row["route"] == "manual_reconstruction":
                self.assertEqual(
                    row["primitive_absence_receipt_sha256"],
                    protocol.canonical_sha256(row["primitive_absence_receipt"]),
                )
            else:
                self.assertIsNone(row["primitive_absence_receipt"])
                self.assertIsNone(row["primitive_absence_receipt_sha256"])

        screen = protocol.make_screen_manifest()
        self.assertEqual(screen, protocol.make_screen_manifest())
        self.assertEqual(screen["expected_records"], 304)
        self.assertEqual({row["distribution"] for row in screen["rows"]}, {"positive"})
        self.assertFalse(screen["withheld_distribution_used"])

        projection = protocol.material_projection()
        self.assertEqual(projection["campaign_id"], protocol.CAMPAIGN_ID)
        lock = protocol.make_campaign_lock()
        self.assertEqual(lock["material_projection"], projection)
        self.assertEqual(set(lock["source_sha256"]), set(protocol.SOURCE_RELATIVES))
        materials = protocol.read_json(protocol.MATERIALS_PATH)["files"]
        self.assertLessEqual(set(protocol.REQUIRED_RUNTIME_MATERIALS), set(materials))

    def test_inference_choices_are_frozen_before_results(self) -> None:
        campaign, mechanism, primitive_map, materials = protocol._base_documents()
        malformed = copy.deepcopy(campaign)
        malformed["inference"]["alpha"] = 0.10
        with mock.patch.object(
            protocol,
            "_base_documents",
            return_value=(malformed, mechanism, primitive_map, materials),
        ):
            with self.assertRaisesRegex(protocol.ProtocolError, "inference contract drift"):
                protocol.validate_contract(rehash_materials=False)
        malformed = copy.deepcopy(campaign)
        malformed["controlling"] = False
        with mock.patch.object(
            protocol,
            "_base_documents",
            return_value=(malformed, mechanism, primitive_map, materials),
        ):
            with self.assertRaisesRegex(protocol.ProtocolError, "controlling status"):
                protocol.validate_contract(rehash_materials=False)

    def test_driver_version_is_frozen(self) -> None:
        campaign, mechanism, primitive_map, materials = protocol._base_documents()
        malformed = copy.deepcopy(campaign)
        malformed["hardware"]["driver_version"] = "foreign"
        with mock.patch.object(
            protocol,
            "_base_documents",
            return_value=(malformed, mechanism, primitive_map, materials),
        ):
            with self.assertRaisesRegex(protocol.ProtocolError, "hardware identity drift"):
                protocol.validate_contract(rehash_materials=False)

    def test_direct_eligibility_requires_bound_full_gate_pass(self) -> None:
        relative = (
            "ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/"
            "crossed_v2r3/audit/records/register_fused__triton__g01.json"
        )
        record = protocol.read_json(protocol.REPO_ROOT / relative)
        protocol._validate_eligible_gate_record(record, "triton", protocol.ON_STRATEGY)
        mutations = (
            ("terminal outcome", lambda row: row.update(terminal_outcome="BUILD_FAILED")),
            ("cell identity", lambda row: row["cell"].update(lane="tilelang")),
            ("gate census", lambda row: row["gate_summary"].update(observed_records=511)),
            ("kernel census", lambda row: row["build_metadata"].update(n_kernels=1)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                malformed = copy.deepcopy(record)
                mutate(malformed)
                with self.assertRaises(protocol.ProtocolError):
                    protocol._validate_eligible_gate_record(
                        malformed, "triton", protocol.ON_STRATEGY,
                    )

    def test_material_path_census_rejects_removed_runtime_dependencies(self) -> None:
        campaign, mechanism, primitive_map, materials = protocol._base_documents()
        for relative in (
            "ako_runs/controlled_followup/fused_epilogue_crossed_v1/__init__.py",
            "ako_runs/controlled_followup/native_trajectory_replication_ada_v3/__init__.py",
            "ako_runs/phase1_matmul/variants/__init__.py",
            "ako_runs/controlled_followup/fused_epilogue_crossed_v2/candidates.py",
            "ako_runs/controlled_followup/fused_epilogue_crossed_v2/core.py",
            "ako_runs/controlled_followup/fused_epilogue_crossed_v2/support_probes.py",
            "ako_runs/phase1_matmul/common.py",
            "ako_runs/phase2_fused_sdpa/common2.py",
            "ako_runs/controlled_followup/legacy_cuda_harness_fix/checked_cuda_launch.h",
            "ako_runs/controlled_followup/trajectory_transfer_ada_v2/INCIDENT_HELD_GEMM_DRIFT_20260812.json",
        ):
            with self.subTest(relative=relative):
                malformed = copy.deepcopy(materials)
                malformed["files"].pop(relative)
                with mock.patch.object(
                    protocol,
                    "_base_documents",
                    return_value=(campaign, mechanism, primitive_map, malformed),
                ):
                    with self.assertRaisesRegex(protocol.ProtocolError, "path census"):
                        protocol.validate_contract(rehash_materials=False)

    def test_predecessor_incident_and_exact_commit_ancestry_are_bound(self) -> None:
        campaign, mechanism, primitive_map, materials = protocol._base_documents()
        protocol._validate_predecessor_incident(campaign, materials, rehash=True)
        malformed = copy.deepcopy(campaign)
        malformed["predecessor"]["result_commit"] = "0" * 40
        with mock.patch.object(
            protocol, "_base_documents",
            return_value=(malformed, mechanism, primitive_map, materials),
        ):
            with self.assertRaisesRegex(protocol.ProtocolError, "predecessor campaign binding"):
                protocol.validate_contract(rehash_materials=False)
        live_git = protocol._git_output

        def fake_git(*args: str) -> str:
            if args == ("rev-parse", f"{protocol.PREDECESSOR_RESULT_COMMIT}^{{tree}}"):
                return "0" * 40
            return live_git(*args)

        with mock.patch.object(protocol, "_git_output", side_effect=fake_git):
            with self.assertRaisesRegex(protocol.ProtocolError, "Git tree/blob binding"):
                protocol._validate_predecessor_incident(campaign, materials, rehash=False)
        with tempfile.TemporaryDirectory() as temporary:
            real = Path(temporary) / "real"
            real.mkdir()
            (real / "evidence.json").write_text("{}\n", encoding="utf-8")
            link = Path(temporary) / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(protocol.ProtocolError, "root.*symlink"):
                protocol._closure_rows([link])

    def test_tilelang_direct_route_binds_f1_softmax_gate_semantics(self) -> None:
        primitive_map = protocol.read_json(protocol.PRIMITIVE_MAP_PATH)
        review = primitive_map["destinations"]["tilelang"]["review"]
        record = protocol.read_json(protocol.REPO_ROOT / review["on_gate_record_path"])
        protocol._validate_tilelang_f1_gate_record(record, review)
        self.assertEqual(record["metadata"]["config"]["extra"]["soft_only"], "1")
        self.assertEqual(record["metadata"]["reported_artifacts"]["n_kernels"], 1)
        malformed = copy.deepcopy(record)
        malformed["metadata"]["config"]["extra"]["soft_only"] = "0"
        with self.assertRaisesRegex(protocol.ProtocolError, "coordinate drift"):
            protocol._validate_tilelang_f1_gate_record(malformed, review)

    def test_freeze_is_idempotent_but_never_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            admission_path = root / "admission_manifest.json"
            lock_path = root / "campaign_lock.json"
            admission = {"kind": "admission", "version": 1}
            lock = {"kind": "lock", "version": 1}
            patches = (
                mock.patch.object(protocol, "ADMISSION_MANIFEST_PATH", admission_path),
                mock.patch.object(protocol, "CAMPAIGN_LOCK_PATH", lock_path),
                mock.patch.object(protocol, "validate_contract"),
                mock.patch.object(protocol, "make_admission_manifest", return_value=admission),
                mock.patch.object(protocol, "make_campaign_lock", return_value=lock),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                protocol.freeze()
                protocol.freeze()
                self.assertEqual(protocol.read_json(admission_path), admission)
                self.assertEqual(protocol.read_json(lock_path), lock)
                with mock.patch.object(
                    protocol, "make_campaign_lock", return_value={**lock, "version": 2},
                ):
                    with self.assertRaisesRegex(protocol.ProtocolError, "refusing to replace"):
                        protocol.freeze()
                self.assertEqual(protocol.read_json(lock_path), lock)

            symlink = root / "symlink.json"
            symlink.symlink_to(admission_path)
            with self.assertRaisesRegex(protocol.ProtocolError, "symlink"):
                protocol.write_if_absent_or_exact_json(symlink, admission)

    def test_routes_follow_source_review_and_manual_route_binds_absence(self) -> None:
        campaign, mechanism, primitive_map, materials = protocol._base_documents()
        self.assertEqual(campaign["routes"], list(protocol.STRUCTURAL_ROUTES))
        self.assertEqual(campaign["route_by_destination"], protocol.ROUTE_BY_DESTINATION)
        absence = primitive_map["shared_receipts"]["cuda_cpp_row_reduction_primitive_absence"]
        absence_sha256 = protocol.canonical_sha256(absence)
        for destination in ("tilelang", "triton"):
            row = primitive_map["destinations"][destination]
            self.assertEqual(row["route"], "direct_primitive_mapping")
            self.assertEqual(row["direct_primitive_mapping"]["status"], "ELIGIBLE")
            self.assertEqual(row["manual_reconstruction"]["status"], "NOT_APPLICABLE")
        for destination in ("cuda_noptx", "cuda_unlimited"):
            row = primitive_map["destinations"][destination]
            self.assertEqual(row["route"], "manual_reconstruction")
            self.assertEqual(row["direct_primitive_mapping"]["status"], "PRIMITIVE_ABSENT")
            self.assertEqual(
                row["manual_reconstruction"]["primitive_absence_receipt_sha256"],
                absence_sha256,
            )

        malformed = copy.deepcopy(primitive_map)
        malformed["destinations"]["cuda_noptx"]["manual_reconstruction"][
            "primitive_absence_receipt_sha256"
        ] = "0" * 64
        with mock.patch.object(protocol, "_base_documents", return_value=(campaign, mechanism, malformed, materials)):
            with self.assertRaisesRegex(protocol.ProtocolError, "primitive-absence/recipe binding"):
                protocol.validate_contract(rehash_materials=False)

    def test_selection_never_uses_withheld_and_confirmation_reuses_configs(self) -> None:
        admission = protocol.make_admission_manifest()
        selection = {}
        for destination in protocol.DESTINATIONS:
            selection[destination] = {}
            for state in protocol.MECHANISM_STATES:
                selection[destination][state] = next(
                    row["entry_id"] for row in admission["rows"]
                    if row["destination"] == destination
                    and row["mechanism_state"] == state
                    and row["adaptation"] == "bounded_retune"
                    and row["grid_id"] == "g05"
                )
        manifest = protocol.make_confirmation_manifest(selection)
        self.assertEqual(manifest, protocol.make_confirmation_manifest(copy.deepcopy(selection)))
        self.assertEqual(manifest["expected_records"], 720)
        self.assertEqual(
            protocol.read_json(protocol.CAMPAIGN_PATH)["sequencing"],
            {
                "fixed_and_retuned_are_required_estimands": True,
                "outcome_contingent_branching": False,
                "retuned_confirmation_requires_fixed_failure": False,
            },
        )
        rows = manifest["rows"]
        for destination in protocol.DESTINATIONS:
            for state in protocol.MECHANISM_STATES:
                selected_rows = [
                    row for row in rows
                    if row["record_kind"] == "candidate"
                    and row["adaptation"] == "bounded_retune"
                    and row["destination"] == destination
                    and row["mechanism_state"] == state
                ]
                self.assertEqual({row["entry_id"] for row in selected_rows}, {selection[destination][state]})
                self.assertEqual({row["distribution"] for row in selected_rows}, set(protocol.DISTRIBUTIONS))
                self.assertEqual(len(selected_rows), 30)
        shams = [row for row in rows if row["record_kind"] == "same_artifact_label_sham"]
        self.assertEqual(len(shams), 240)
        for block in range(protocol.BLOCKS):
            for destination in protocol.DESTINATIONS:
                members = [row for row in shams if row["block"] == block and row["destination"] == destination]
                self.assertEqual({row["label"] for row in members}, set(protocol.SHAM_LABELS))
                self.assertEqual(len({row["entry_id"] for row in members}), 1)

    def test_terminal_taxonomy_is_exact(self) -> None:
        for status in protocol.TERMINAL_STATUSES:
            self.assertEqual(protocol.validate_terminal_outcome(status), status)
        with self.assertRaisesRegex(protocol.ProtocolError, "unknown terminal status"):
            protocol.validate_terminal_outcome("RELABELLED_FAILURE")


if __name__ == "__main__":
    unittest.main()
