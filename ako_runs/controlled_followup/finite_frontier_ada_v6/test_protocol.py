#!/usr/bin/env python3
"""CPU-only regressions for the terminal-only v6 successor."""
from __future__ import annotations

import ast
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from . import admit, analyze, artifacts, launch, protocol


class TerminalSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.incident = protocol.predecessor_incident()
        cls.selection = protocol.load_selection_binding()
        cls.contract = protocol.load_contract()

    def test_incident_binds_complete_v5_attempt(self) -> None:
        incident = self.incident
        self.assertEqual(incident["result_commit"], protocol.PREDECESSOR_RESULT_COMMIT)
        self.assertEqual(
            incident["sealed_census"],
            {
                "expected_artifact_admission_entries": 7,
                "expected_selection_timing_records": 240,
                "expected_terminal_timing_records": 120,
                "observed_artifact_admission_entries": 7,
                "observed_selection_timing_records": 240,
                "observed_terminal_timing_records": 0,
            },
        )
        self.assertEqual(len(protocol.predecessor_result_closure_paths(incident)), 376)
        self.assertEqual(len(protocol.predecessor_source_paths(incident)), 11)
        self.assertEqual(
            incident["artifact_selection_closure"]["sha256"],
            "d96b74686beef7c6174e2e6aa7d8a5e534a829095f6f60d457e84ebf4a58254f",
        )

    def test_selection_lock_is_independently_rederived(self) -> None:
        source_contract = protocol._predecessor_contract(self.incident)
        derived = protocol._rederive_predecessor_selection(
            self.incident, source_contract
        )
        self.assertEqual(derived, protocol.read_json(protocol.PREDECESSOR_SELECTION_PATH))
        self.assertEqual(derived["winners"], protocol.WINNERS)
        self.assertEqual(
            derived["sham"]["resolution_floor_log_ratio"],
            0.01686377744677924,
        )

    def test_selection_binding_is_exact_and_role_limited(self) -> None:
        self.assertEqual(self.selection, protocol.derive_selection_binding())
        self.assertEqual(self.selection["selection_record_count"], 240)
        self.assertEqual(len(self.selection["selection_stage_hashes"]), 242)
        self.assertEqual(
            self.selection["selection_role"], "preregistered_winner_selection_only"
        )
        self.assertTrue(self.selection["terminal_authorized"])
        self.assertEqual(self.selection["winners"], protocol.WINNERS)

    def test_terminal_plan_is_unchanged_and_exact(self) -> None:
        plan = protocol.timing_plan()
        self.assertEqual(len(plan), 120)
        self.assertEqual(
            protocol.canonical_sha256(plan),
            "fc98e60943387bc801e2c6e222c020ac53e5541fef4d5a4ca78cd4bbc20a8b23",
        )
        self.assertEqual(sum(row["record_kind"] == "candidate" for row in plan), 60)
        self.assertEqual(sum(row["record_kind"] == "sham" for row in plan), 60)
        self.assertEqual({row["stage"] for row in plan}, {"terminal_confirm"})
        for label in (*protocol.WINNERS.values(), *protocol.SHAM_LABELS):
            for distribution in protocol.DISTRIBUTIONS:
                self.assertEqual(
                    sum(
                        row["label"] == label and row["distribution"] == distribution
                        for row in plan
                    ),
                    15,
                )

    def test_selection_timing_is_not_authorized(self) -> None:
        with self.assertRaises(protocol.ProtocolError):
            protocol.timing_plan("selection_confirm")
        with self.assertRaises(protocol.ProtocolError):
            launch._stage_plan("selection_confirm")
        with self.assertRaises(protocol.ProtocolError):
            launch._timing_child_command(
                "selection_confirm", {}, Path("input"), Path("out"), Path("receipt"),
                "0" * 64,
            )

    def test_contract_preserves_terminal_estimand_and_instrument(self) -> None:
        source = protocol._predecessor_contract(self.incident)
        self.assertEqual(self.contract["question"], source["question"])
        self.assertEqual(self.contract["claim_scope"], source["claim_scope"])
        self.assertEqual(self.contract["estimand"], source["estimand"])
        for key in ("hardware", "inference", "terminal_confirm", "timing", "toolchain", "workload"):
            self.assertEqual(self.contract["manifest"][key], source["manifest"][key])
        self.assertNotIn("selection_confirm", self.contract["manifest"])
        self.assertEqual(self.contract["manifest"]["artifact_admission"]["entry_count"], 3)
        self.assertEqual(
            self.contract["manifest"]["artifact_admission"]["candidate_artifacts"], 2
        )

    def test_material_registry_is_one_exact_hash_bijection(self) -> None:
        registry = self.contract["material_registry"]
        self.assertEqual(set(registry["sha256"]), set(registry["roles"].values()))
        self.assertEqual(len(registry["roles"]), len(set(registry["roles"].values())))
        for relative, expected in registry["sha256"].items():
            self.assertEqual(protocol.file_sha256(protocol.REPO_ROOT / relative), expected)

    def test_admission_is_three_fresh_v6_artifacts(self) -> None:
        plan = artifacts.admission_plan(self.contract)
        self.assertEqual(
            [(row["cell_id"], row["role"]) for row in plan],
            [
                (protocol.WINNERS["tilelang"], "candidate"),
                (protocol.WINNERS["triton"], "candidate"),
                (protocol.SHAM_BASE_CELL, "shared_sham_base"),
            ],
        )
        for row in plan:
            root = artifacts.entry_root(row["cell_id"]).resolve()
            self.assertTrue(root.is_relative_to(protocol.RESULTS_ROOT.resolve()))
            self.assertFalse(root.is_relative_to(protocol.PREDECESSOR.resolve()))

    def test_package_qualified_module_resolution_survives_shadowing(self) -> None:
        for name, resolver, expected in (
            ("launch", protocol.local_launch_module, protocol.HERE / "launch.py"),
            ("analyze", protocol.local_analyze_module, protocol.HERE / "analyze.py"),
        ):
            previous = sys.modules.get(name)
            foreign = types.ModuleType(name)
            foreign.__file__ = f"/tmp/foreign/{name}.py"
            sys.modules[name] = foreign
            try:
                self.assertEqual(Path(resolver().__file__).resolve(), expected.resolve())
            finally:
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

    def test_local_imports_and_children_are_package_safe(self) -> None:
        local_names = {"admit", "analyze", "artifacts", "launch", "protocol"}
        for path in protocol.local_source_paths():
            if path.suffix != ".py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(
                        {alias.name for alias in node.names} & local_names,
                        f"ambiguous local import in {path.name}:{node.lineno}",
                    )
        launch_source = (protocol.HERE / "launch.py").read_text(encoding="utf-8")
        admit_source = (protocol.HERE / "admit.py").read_text(encoding="utf-8")
        self.assertNotIn('str(HERE / "launch.py")', launch_source)
        self.assertNotIn('str(HERE / "admit.py")', admit_source)

    def test_admission_child_command_uses_package_module(self) -> None:
        row = artifacts.admission_plan(self.contract)[0]
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            receipt.write_text("{}\n", encoding="utf-8")
            gpu_lock = mock.Mock()
            gpu_lock.fileno.return_value = 19
            completed = mock.Mock(returncode=0)
            with (
                mock.patch.object(artifacts, "prepare_cache_environment", return_value={}),
                mock.patch.object(admit.subprocess, "run", return_value=completed) as run,
            ):
                self.assertEqual(admit._child_command("build-one", row, receipt, gpu_lock), 0)
            command = run.call_args.args[0]
            self.assertEqual(
                command[:3],
                [
                    sys.executable,
                    "-m",
                    "ako_runs.controlled_followup.finite_frontier_ada_v6.admit",
                ],
            )

    def test_timing_child_command_uses_package_module(self) -> None:
        row = protocol.timing_plan()[0]
        command = launch._timing_child_command(
            "terminal_confirm", row, protocol.SELECTION_BINDING_PATH,
            Path("out.json"), Path("receipt.json"), "1" * 64,
        )
        self.assertEqual(
            command[:3],
            [
                sys.executable,
                "-m",
                "ako_runs.controlled_followup.finite_frontier_ada_v6.launch",
            ],
        )
        self.assertIn("terminal_confirm", command)

    def test_cli_has_no_selection_command(self) -> None:
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                launch.main(["selection-confirm"])
            with self.assertRaises(SystemExit):
                analyze.main(["select"])

    def test_negative_output_index_normalization_is_content_derived(self) -> None:
        source = {
            "source": """
@T.prim_func
def kernel(a, b, c, d):
    pass
"""
        }
        self.assertEqual(artifacts._normalized_out_idx_candidates(source, [-1]), [[3]])
        self.assertEqual(artifacts._normalized_out_idx_candidates(source, [3]), [])
        self.assertEqual(artifacts._normalized_out_idx_candidates({"source": "bad ("}, [-1]), [])

    def test_terminal_estimator_preserves_direction_and_floor_rule(self) -> None:
        self.assertEqual(
            analyze._ratio_result([0.8] * 15, [1.0] * 15, 0.01)[
                "direction_beyond_sham_floor"
            ],
            "tilelang_lower_latency",
        )
        self.assertEqual(
            analyze._ratio_result([1.2] * 15, [1.0] * 15, 0.01)[
                "direction_beyond_sham_floor"
            ],
            "triton_lower_latency",
        )
        self.assertEqual(
            analyze._ratio_result([1.0] * 15, [1.0] * 15, 0.01)[
                "direction_beyond_sham_floor"
            ],
            "unresolved",
        )

    def test_sham_labels_must_share_one_implementation(self) -> None:
        records, grouped = [], {}
        for label, implementation in (("sham_a", "a" * 64), ("sham_b", "b" * 64)):
            for distribution in protocol.DISTRIBUTIONS:
                grouped[(label, distribution)] = [(block, 1.0) for block in range(15)]
            records.append({"label": label, "implementation_sha256": implementation})
        with self.assertRaises(protocol.ProtocolError):
            analyze._sham_floor(records, grouped)

    def test_raw_census_and_process_sequence_fail_closed(self) -> None:
        plan = protocol.timing_plan()[:2]
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for row in plan:
                (raw / protocol.timing_filename(row)).write_text("{}\n", encoding="utf-8")
            protocol.validate_raw_census(raw, plan)
            (raw / "extra.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(protocol.ProtocolError):
                protocol.validate_raw_census(raw, plan)
        records = [
            {"plan_position": 0, "process_pid": 7, "t_start": 1.0, "t_end": 2.0},
            {"plan_position": 1, "process_pid": 7, "t_start": 2.0, "t_end": 3.0},
        ]
        with self.assertRaises(protocol.ProtocolError):
            launch.validate_record_sequence(records, plan)

    def test_execution_lock_binds_terminal_only_counts(self) -> None:
        lock = launch.validate_execution_lock()
        self.assertEqual(lock["artifact_admission_expected_entries"], 3)
        self.assertEqual(lock["terminal_expected_records"], 120)
        self.assertNotIn("selection_expected_records", lock)
        self.assertNotIn("selection_plan_sha256", lock)
        self.assertEqual(lock["terminal_plan_sha256"], self.selection["terminal_plan_sha256"])
        self.assertEqual(
            lock["predecessor_result_commit"], protocol.PREDECESSOR_RESULT_COMMIT
        )

    def test_artifact_readiness_closure_includes_v5_and_no_v6_selection_results(self) -> None:
        paths = launch.remote_ready_paths(self.contract, "artifact_admission")
        self.assertIn(protocol.repo_path(protocol.SELECTION_BINDING_PATH), paths)
        self.assertIn(protocol.repo_path(protocol.PREDECESSOR_INCIDENT_PATH), paths)
        self.assertEqual(
            sum("finite_frontier_ada_v5/results/" in path for path in paths), 376
        )
        self.assertFalse(any("finite_frontier_ada_v6/results/selection_confirm" in path for path in paths))


if __name__ == "__main__":
    unittest.main()
