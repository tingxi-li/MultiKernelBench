from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import analyze, artifacts, protocol, runner


HEX_A = "a" * 64
HEX_B = "b" * 64


class ExecutionTest(unittest.TestCase):
    def test_runner_source_closure_reuses_protocol_list(self) -> None:
        observed = {
            path.relative_to(protocol.HERE).as_posix()
            for path in runner._source_paths()
            if path.parent == protocol.HERE or protocol.HERE in path.parents
        }
        self.assertTrue(set(protocol.SOURCE_RELATIVES) <= observed)

    def test_provenance_rejects_tampered_campaign_lock(self) -> None:
        remote = {
            "git_commit": "1" * 40,
            "upstream": "origin/branch",
            "live_remote": "origin",
            "live_ref": "refs/heads/branch",
        }
        idle = {
            "gpu": {"index": "0", "uuid": "gpu", "name": "name", "driver_version": "driver", "compute_capability": "8.9"},
            "compute_pids": [],
            "checked_unix_ns": 2,
        }
        receipt = {
            "campaign_id": protocol.CAMPAIGN_ID,
            "campaign_lock_sha256": HEX_A,
            "created_utc": "2026-08-12T12:00:00+00:00",
            "gpu_preflight": {**idle, "checked_unix_ns": 1},
            "record_type": "trajectory_transfer_ada_v2_provenance",
            "schema_version": 1,
            "source_sha256": {"source": HEX_B},
            "toolchain": {"cache_loader_sha256": {"loader": HEX_A}},
            **remote,
        }
        with (
            mock.patch.object(protocol, "file_sha256", return_value=HEX_A),
            mock.patch.object(runner, "_source_hashes", return_value=receipt["source_sha256"]),
            mock.patch.object(runner, "_toolchain", return_value=receipt["toolchain"]),
        ):
            self.assertEqual(runner._validate_provenance(receipt, idle, remote), receipt)
            changed = copy.deepcopy(receipt)
            changed["campaign_lock_sha256"] = HEX_B
            with self.assertRaisesRegex(protocol.ProtocolError, "provenance"):
                runner._validate_provenance(changed, idle, remote)

    def test_parent_child_environment_does_not_create_admission_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory) / "admission"
            row = protocol.make_admission_manifest()["rows"][0]
            with mock.patch("subprocess.Popen") as popen:
                process = popen.return_value
                process.wait.return_value = 0
                runner._child(["time-one"], 7, entry_id=row["entry_id"], root=root)
            self.assertFalse(root.exists())
            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["TRAJECTORY_TRANSFER_ARTIFACT_MODE"], "load_only")
            self.assertNotIn(str(root), env.get("TMPDIR", ""))

    def test_terminal_census_keeps_gate_failure_but_rejects_verify_and_failure(self) -> None:
        row = protocol.make_admission_manifest()["rows"][0]
        manifest = {"rows": [row]}
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory)
            paths = artifacts.entry_paths(row["entry_id"], root)
            paths["gate"].parent.mkdir(parents=True)
            paths["gate"].write_text("{}\n", encoding="utf-8")
            protocol.atomic_json(
                paths["failure"],
                runner._failure_value(row, "AUDIT_FAILED", "audit failed", None),
            )
            statuses, passed = runner._terminal_census(manifest, root)
            self.assertEqual(statuses, {row["entry_id"]: "AUDIT_FAILED"})
            self.assertEqual(passed, [])
            protocol.atomic_json(paths["verify"], {})
            with self.assertRaisesRegex(protocol.ProtocolError, "multiple terminal"):
                runner._terminal_census(manifest, root)

    def test_failure_receipt_and_terminal_bundle_are_content_bound(self) -> None:
        row = protocol.make_admission_manifest()["rows"][0]
        manifest = {"rows": [row]}
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory)
            paths = artifacts.entry_paths(row["entry_id"], root)
            paths["failure"].parent.mkdir(parents=True)
            failure = runner._failure_value(row, "BUILD_FAILED", "failed", None)
            protocol.atomic_json(paths["failure"], failure)
            first = runner._admission_terminal_evidence(manifest, root)
            self.assertEqual(runner._validate_failure(failure, row), "BUILD_FAILED")
            changed = copy.deepcopy(failure)
            changed["admission_row_sha256"] = HEX_B
            with self.assertRaisesRegex(protocol.ProtocolError, "malformed or foreign"):
                runner._validate_failure(changed, row)
            protocol.atomic_json(paths["failure"], {**failure, "error": "changed"})
            self.assertNotEqual(first, runner._admission_terminal_evidence(manifest, root))
            (paths["root"] / "cache").mkdir()
            (paths["root"] / "cache/partial.bin").write_bytes(b"partial")
            second = runner._admission_terminal_evidence(manifest, root)
            self.assertNotEqual(first, second)
            (paths["root"] / "cache/partial.bin").write_bytes(b"changed")
            self.assertNotEqual(second, runner._admission_terminal_evidence(manifest, root))

    def test_valid_admission_entry_rejects_posthoc_overlay(self) -> None:
        row = protocol.make_admission_manifest()["rows"][0]
        manifest = {"rows": [row]}
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory)
            paths = artifacts.entry_paths(row["entry_id"], root)
            paths["cache"].mkdir(parents=True)
            paths["tmp"].mkdir()
            for path in (paths["gate"], paths["build"], paths["verify"]):
                path.write_text("{}\n", encoding="utf-8")
            (root / "launch_receipt.json").write_text("{}\n", encoding="utf-8")
            (root / "run_status.json").write_text("{}\n", encoding="utf-8")
            runner._validate_admission_census(
                manifest, root, {row["entry_id"]: "GATE_PASSED"},
            )
            (paths["root"] / "posthoc.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "entry file census"):
                runner._validate_admission_census(
                    manifest, root, {row["entry_id"]: "GATE_PASSED"},
                )

    def test_admission_rejects_symlinked_entry_root(self) -> None:
        row = protocol.make_admission_manifest()["rows"][0]
        manifest = {"rows": [row]}
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            result = Path(directory)
            root = result / "admission"
            (root / "artifacts").mkdir(parents=True)
            target = result / "foreign_entry"
            target.mkdir()
            artifacts.entry_paths(row["entry_id"], root)["root"].symlink_to(
                target, target_is_directory=True,
            )
            (root / "launch_receipt.json").write_text("{}\n", encoding="utf-8")
            (root / "run_status.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "symlink"):
                runner._validate_admission_census(
                    manifest, root, {row["entry_id"]: "GATE_PASSED"},
                )

    def test_phase_census_rejects_symlinked_expected_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            parent = Path(directory)
            stage = parent / "screen"
            stage.mkdir()
            target = parent / "record.json"
            target.write_text("{}\n", encoding="utf-8")
            linked = stage / "record.json"
            linked.symlink_to(target)
            with self.assertRaisesRegex(protocol.ProtocolError, "symlink"):
                runner._require_exact_files(stage, {linked}, "screen")

    def test_parent_unreceipted_exit_is_launch_failure(self) -> None:
        row = protocol.make_admission_manifest()["rows"][0]
        value = runner._failure_value(
            row, "LAUNCH_FAILED", "artifact admission child exited 124", None,
        )
        self.assertEqual(runner._validate_failure(value, row), "LAUNCH_FAILED")

    def test_single_entry_validator_uses_explicit_child_provenance_modes(self) -> None:
        row = protocol.make_admission_manifest()["rows"][0]
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory)
            paths = artifacts.entry_paths(row["entry_id"], root)
            paths["build"].parent.mkdir(parents=True)
            protocol.atomic_json(paths["build"], {})
            protocol.atomic_json(paths["verify"], {})
            with (
                mock.patch.object(artifacts, "validate_build_record", return_value={"provenance": {}}),
                mock.patch.object(artifacts, "validate_verify_record", return_value={"provenance": {}}),
                mock.patch.object(runner, "_validate_child_provenance") as validate,
            ):
                runner._admitted_entry(row, root)
            self.assertEqual(
                [call.kwargs["build"] for call in validate.call_args_list], [True, False],
            )

    def test_clean_child_bootstraps_exact_runtime_modules_before_cuda_verify_import(self) -> None:
        script = """
import json
from pathlib import Path
from ako_runs.controlled_followup.trajectory_transfer_ada_v2 import runner
receipt = runner._bootstrap_runtime_modules()
import common, common2, runner2
from ako_runs.controlled_followup.trajectory_transfer_ada_v2.implementations import cuda_noptx
expected = {name: str(path.resolve()) for name, path in runner.RUNTIME_MODULE_PATHS.items()}
observed = {name: str(Path(module.__file__).resolve()) for name, module in {
    'common': common, 'common2': common2, 'runner2': runner2,
}.items()}
assert observed == expected, (observed, expected)
assert Path(cuda_noptx.__file__).resolve() == (runner.HERE / 'implementations/cuda_noptx.py').resolve()
print(json.dumps(receipt, sort_keys=True))
"""
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=protocol.REPO_ROOT,
            env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), runner._runtime_module_receipt())

    def test_selection_uses_positive_screen_only_and_tie_breaks_by_config(self) -> None:
        admission = protocol.make_admission_manifest()
        entries = {
            row["entry_id"]: row for row in admission["rows"]
            if row["adaptation"] == "bounded_retune"
        }
        admitted = {
            entry_id: {"artifact_identity_sha256": HEX_A}
            for entry_id in entries
        }
        manifest = protocol.make_screen_manifest(entries)
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory)
            stage = root / "screen"
            (root / "admission").mkdir()
            protocol.atomic_json(root / "admission/launch_receipt.json", {})
            protocol.atomic_json(root / "admission/run_status.json", {})
            (stage / "raw").mkdir(parents=True)
            (stage / "position_receipts").mkdir()
            protocol.atomic_json(stage / "manifest.json", manifest)
            for row in manifest["rows"]:
                entry = entries[row["entry_id"]]
                value = 1.0 if entry["grid_id"] in {"g03", "g05"} else 2.0
                protocol.atomic_json(stage / "raw" / f"{row['record_id']}.json", {
                    "implementation_sha256": HEX_A,
                    "load_evidence": {},
                    "ok": True,
                    "primary_tail_median_ms": value,
                    "primitive_graph_sha256": entry["primitive_graph_sha256"],
                    "runtime_modules": runner._runtime_module_receipt(),
                    "coordinate_cell_id": entry["coordinate_cell_id"],
                    "implementation_id": entry["implementation_id"],
                    "structural_route": entry["route"],
                })
            for position, row in enumerate(manifest["rows"], 1):
                protocol.atomic_json(
                    stage / "position_receipts" / f"{position:04d}__{row['record_id']}.json",
                    {},
                )
            protocol.atomic_json(stage / "launch_receipt.json", {
                "admission_launch_receipt_sha256": protocol.file_sha256(root / "admission/launch_receipt.json"),
                "admission_run_status_sha256": protocol.file_sha256(root / "admission/run_status.json"),
                "campaign_id": protocol.CAMPAIGN_ID,
                "expected_records": manifest["expected_records"],
                "gpu_preflight": {},
                "manifest_sha256": protocol.canonical_sha256(manifest),
                "record_type": "trajectory_transfer_ada_v2_screen_launch",
                "schema_version": 1,
                "prelaunch_provenance_path": "provenance.json",
                "prelaunch_provenance_sha256": HEX_A,
            })
            raw_hashes = {
                row["record_id"]: protocol.file_sha256(
                    stage / "raw" / f"{row['record_id']}.json"
                ) for row in manifest["rows"]
            }
            position_hashes = {
                row["record_id"]: protocol.file_sha256(
                    stage / "position_receipts" / f"{position:04d}__{row['record_id']}.json"
                ) for position, row in enumerate(manifest["rows"], 1)
            }
            protocol.atomic_json(stage / "run_status.json", {
                "campaign_id": protocol.CAMPAIGN_ID,
                "complete": True,
                "expected_records": manifest["expected_records"],
                "launch_receipt_sha256": protocol.file_sha256(stage / "launch_receipt.json"),
                "observed_records": len(manifest["rows"]),
                "position_receipt_bundle_sha256": protocol.canonical_sha256(position_hashes),
                "raw_record_bundle_sha256": protocol.canonical_sha256(raw_hashes),
                "record_type": "trajectory_transfer_ada_v2_screen_status",
                "schema_version": 1,
            })
            with (
                mock.patch.object(runner, "result_root", return_value=root),
                mock.patch.object(protocol, "ADMISSION_MANIFEST_PATH", root / "admission_manifest.json"),
                mock.patch.object(runner, "_admitted", return_value=(admission, admitted)),
                mock.patch.object(analyze, "validate_timing_record"),
                mock.patch.object(runner, "_validate_position", return_value=1),
                mock.patch.object(protocol, "make_screen_manifest", return_value=manifest),
                mock.patch.object(artifacts, "validate_load_evidence"),
                mock.patch.object(runner, "_provenance_binding", return_value={
                    "prelaunch_provenance_path": "provenance.json",
                    "prelaunch_provenance_sha256": HEX_A,
                }),
                mock.patch.object(runner, "_validate_bound_gpu_preflight"),
            ):
                protocol.atomic_json(root / "admission_manifest.json", admission)
                selected, _hashes = runner.derive_selection("test")
            self.assertFalse(selected["withheld_distribution_used"])
            for destination in protocol.DESTINATIONS:
                for state in protocol.MECHANISM_STATES:
                    winner = entries[selected["selection"][destination][state]]
                    self.assertEqual(winner["grid_id"], "g03")

    def test_confirmation_and_final_load_rederive_selection(self) -> None:
        derived = {
            "campaign_id": protocol.CAMPAIGN_ID,
            "record_type": "trajectory_transfer_ada_v2_selection",
            "schema_version": 1,
            "selection": {},
        }
        retained = {**derived, "selection": {"tampered": {}}}
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory)
            protocol.atomic_json(root / "selection.json", retained)
            shared = (
                mock.patch.object(runner, "result_root", return_value=root),
                mock.patch.object(runner, "derive_selection", return_value=(derived, [])),
                mock.patch.object(runner, "_admitted", return_value=({}, {})),
            )
            with shared[0], shared[1], shared[2]:
                with self.assertRaisesRegex(protocol.ProtocolError, "independent screen"):
                    runner.confirm("test")
            with (
                mock.patch.object(runner, "result_root", return_value=root),
                mock.patch.object(runner, "derive_selection", return_value=(derived, [])),
                mock.patch.object(runner, "_admitted", return_value=({}, {})),
                mock.patch.object(protocol, "check_frozen"),
            ):
                with self.assertRaisesRegex(protocol.ProtocolError, "independent screen"):
                    runner.load_completed("test")

    def test_held_gemm_source_must_match_within_off_on_pair(self) -> None:
        manifest = protocol.make_admission_manifest()
        rows = [
            row for row in manifest["rows"]
            if row["destination"] == "tilelang"
            and row["adaptation"] == "donor_fixed"
        ]

        def build(row: dict, digest: str) -> dict:
            return {"source_build_metadata": {
                "adapter_dispatch_sha256": HEX_A,
                "adapter_module_sha256": HEX_A,
                "generated_source_sha256": HEX_A,
                "held_gemm_source_sha256": digest,
                "n_kernels": 2,
                "paired_config_sha256": HEX_A,
                "primitive_graph_sha256": row["primitive_graph_sha256"],
                "primitive_mapping": {
                    "primitive_map_sha256": protocol.file_sha256(protocol.PRIMITIVE_MAP_PATH),
                },
                "route": row["route"],
            }}

        entries = {row["entry_id"]: build(row, HEX_A) for row in rows}
        runner._validate_treatment_pairs(manifest, entries)
        entries[rows[1]["entry_id"]] = build(rows[1], HEX_B)
        with self.assertRaisesRegex(protocol.ProtocolError, "structural controls differ"):
            runner._validate_treatment_pairs(manifest, entries)

    def test_phase_evidence_rejects_each_tampered_trust_boundary(self) -> None:
        admission = protocol.make_admission_manifest()
        entry = next(
            row for row in admission["rows"]
            if row["adaptation"] == "bounded_retune"
        )
        row = {
            "adaptation": entry["adaptation"],
            "destination": entry["destination"],
            "distribution": "positive",
            "entry_id": entry["entry_id"],
            "mechanism_enabled": entry["mechanism_enabled"],
            "mechanism_state": entry["mechanism_state"],
            "record_id": "tts_" + "1" * 24,
            "replicate": 1,
        }
        manifest = {"expected_records": 1, "rows": [row]}
        build = {"artifact_identity_sha256": HEX_A}

        def execute(case: str | None = None) -> None:
            with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
                root = Path(directory)
                stage = root / "screen"
                (root / "admission").mkdir()
                protocol.atomic_json(root / "admission/launch_receipt.json", {})
                protocol.atomic_json(root / "admission/run_status.json", {})
                (stage / "raw").mkdir(parents=True)
                (stage / "position_receipts").mkdir()
                protocol.atomic_json(stage / "manifest.json", manifest)
                raw = stage / "raw" / f"{row['record_id']}.json"
                position_path = stage / "position_receipts" / f"0001__{row['record_id']}.json"
                record = {
                    "implementation_sha256": HEX_A,
                    "load_evidence": {"bound": True},
                    "primitive_graph_sha256": entry["primitive_graph_sha256"],
                    "runtime_modules": {},
                    "coordinate_cell_id": entry["coordinate_cell_id"],
                    "implementation_id": entry["implementation_id"],
                    "structural_route": entry["route"],
                    "t_start_unix_ns": 20,
                    "t_end_unix_ns": 30,
                }
                record_mutations = {
                    "implementation": ("implementation_sha256", HEX_B),
                    "graph": ("primitive_graph_sha256", HEX_B),
                    "coordinate": ("coordinate_cell_id", "foreign"),
                    "implementation_id": ("implementation_id", "foreign"),
                    "route": ("structural_route", "foreign"),
                    "runtime": ("runtime_modules", {"foreign": {}}),
                    "load": ("load_evidence", {"bound": False}),
                }
                if case in record_mutations:
                    key, value = record_mutations[case]
                    record[key] = value
                protocol.atomic_json(raw, record)
                receipt = {
                    "campaign_id": protocol.CAMPAIGN_ID,
                    "child_completed_unix_ns": 40,
                    "child_launched_unix_ns": 10,
                    "gpu_idle_after_child": {"compute_pids": []},
                    "position": 1,
                    "previous_child_completed_unix_ns": 0,
                    "raw_path": str(raw.relative_to(protocol.REPO_ROOT)),
                    "raw_sha256": protocol.file_sha256(raw),
                    "record_id": row["record_id"],
                    "record_type": "trajectory_transfer_ada_v2_position_receipt",
                    "returncode": 0,
                    "schema_version": 1,
                }
                if case == "position":
                    receipt["raw_sha256"] = HEX_B
                protocol.atomic_json(position_path, receipt)
                launch = {
                    "admission_launch_receipt_sha256": protocol.file_sha256(root / "admission/launch_receipt.json"),
                    "admission_run_status_sha256": protocol.file_sha256(root / "admission/run_status.json"),
                    "campaign_id": protocol.CAMPAIGN_ID,
                    "expected_records": 1,
                    "gpu_preflight": {},
                    "manifest_sha256": protocol.canonical_sha256(manifest),
                    "record_type": "trajectory_transfer_ada_v2_screen_launch",
                    "schema_version": 1,
                    "prelaunch_provenance_path": "provenance.json",
                    "prelaunch_provenance_sha256": HEX_A,
                }
                if case == "launch":
                    launch["foreign"] = True
                protocol.atomic_json(stage / "launch_receipt.json", launch)
                raw_hashes = {row["record_id"]: protocol.file_sha256(raw)}
                position_hashes = {row["record_id"]: protocol.file_sha256(position_path)}
                status = {
                    "campaign_id": protocol.CAMPAIGN_ID,
                    "complete": True,
                    "expected_records": 1,
                    "launch_receipt_sha256": protocol.file_sha256(stage / "launch_receipt.json"),
                    "observed_records": 1,
                    "position_receipt_bundle_sha256": protocol.canonical_sha256(position_hashes),
                    "raw_record_bundle_sha256": protocol.canonical_sha256(raw_hashes),
                    "record_type": "trajectory_transfer_ada_v2_screen_status",
                    "schema_version": 1,
                }
                if case == "status":
                    status["raw_record_bundle_sha256"] = HEX_B
                protocol.atomic_json(stage / "run_status.json", status)

                def validate_load(value, *_args):
                    if value != {"bound": True}:
                        raise protocol.ProtocolError("foreign load evidence")

                with (
                    mock.patch.object(runner, "result_root", return_value=root),
                    mock.patch.object(runner, "_runtime_module_receipt", return_value={}),
                    mock.patch.object(analyze, "validate_timing_record"),
                    mock.patch.object(artifacts, "validate_load_evidence", side_effect=validate_load),
                    mock.patch.object(runner, "_provenance_binding", return_value={
                        "prelaunch_provenance_path": "provenance.json",
                        "prelaunch_provenance_sha256": HEX_A,
                    }),
                    mock.patch.object(runner, "_validate_bound_gpu_preflight"),
                ):
                    runner._validate_phase_evidence(
                        "test", "screen", manifest, admission,
                        {entry["entry_id"]: build},
                    )

        execute()
        for case in (
            "launch", "implementation", "graph", "coordinate", "implementation_id", "route", "runtime",
            "load", "position", "status",
        ):
            with self.subTest(case=case), self.assertRaises(protocol.ProtocolError):
                execute(case)

    def test_analyzer_requires_same_selected_identity_across_distributions(self) -> None:
        campaign = protocol.read_json(protocol.CAMPAIGN_PATH)
        selection = {}
        admission = protocol.make_admission_manifest()
        for destination in protocol.DESTINATIONS:
            selection[destination] = {}
            for state in protocol.MECHANISM_STATES:
                selection[destination][state] = next(
                    row["entry_id"] for row in admission["rows"]
                    if row["destination"] == destination
                    and row["mechanism_state"] == state
                    and row["adaptation"] == "bounded_retune"
                    and row["grid_id"] == "g03"
                )
        manifest = protocol.make_confirmation_manifest(selection)
        records = []
        for row in manifest["rows"]:
            identity = HEX_A
            if row["record_kind"] == "candidate" and row["mechanism_state"] == "on":
                identity = HEX_B
            record = {
                "campaign_id": protocol.CAMPAIGN_ID,
                "full_median_ms": 1.0,
                "implementation_sha256": identity,
                "ok": True,
                "primary_tail_median_ms": 1.0,
                "primitive_graph_sha256": HEX_A,
                "primitive_map_sha256": protocol.file_sha256(protocol.PRIMITIVE_MAP_PATH),
                "record_type": "trajectory_transfer_ada_v2_timing_record",
                "row": row,
                "row_sha256": protocol.canonical_sha256(row),
                "schema_version": 1,
                "coordinate_cell_id": "coordinate",
                "implementation_id": "implementation",
                "structural_route": protocol.ROUTE_BY_DESTINATION[row["destination"]],
                "runtime_modules": {},
                "t_start_unix_ns": 1,
                "t_end_unix_ns": 2,
                "times_ms": [1.0] * 100,
            }
            records.append(record)
        # Sham labels must first share one artifact; candidate identities remain stable.
        result = analyze.estimate(campaign, manifest, records)
        self.assertEqual(result["timing_records"], 720)
        self.assertEqual(result["inference_contract"], protocol.INFERENCE_CONTRACT)
        self.assertFalse(result["route_comparison_claim_authorized"])
        self.assertEqual(result["route_by_destination"], protocol.ROUTE_BY_DESTINATION)
        wrong_route = copy.deepcopy(records)
        victim = next(
            record for record in wrong_route
            if record["row"]["destination"] == "cuda_noptx"
        )
        victim["structural_route"] = "direct_primitive_mapping"
        with self.assertRaisesRegex(protocol.ProtocolError, "malformed or foreign"):
            analyze.estimate(campaign, manifest, wrong_route)
        changed_campaign = copy.deepcopy(campaign)
        changed_campaign["inference"]["alpha"] = 0.10
        with self.assertRaisesRegex(protocol.ProtocolError, "inference contract"):
            analyze.estimate(changed_campaign, manifest, records)
        changed = copy.deepcopy(records)
        victims = [
            record for record in changed
            if record["row"]["record_kind"] == "candidate"
            and record["row"]["destination"] == "tilelang"
            and record["row"]["adaptation"] == "bounded_retune"
            and record["row"]["mechanism_state"] == "on"
            and record["row"]["distribution"] == "withheld_signed"
        ]
        for victim in victims:
            victim["implementation_sha256"] = "c" * 64
        with self.assertRaisesRegex(protocol.ProtocolError, "selection changed"):
            analyze.estimate(campaign, manifest, changed)


if __name__ == "__main__":
    unittest.main()
