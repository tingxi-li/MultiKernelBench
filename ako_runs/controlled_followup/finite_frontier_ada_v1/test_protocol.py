from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from . import analyze, launch, protocol
except ImportError:
    import analyze  # type: ignore
    import launch  # type: ignore
    import protocol  # type: ignore


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = protocol.load_contract()
        cls.imported = protocol.derive_imported_frontier(cls.contract)

    def test_real_imported_denominator_and_screen(self):
        self.assertEqual(len(self.imported["audit_records"]), 152)
        self.assertEqual(len(self.imported["gate_legal_cell_ids"]), 113)
        self.assertEqual(len(self.imported["screen_records"]), 226)
        self.assertEqual(
            self.imported["audit_outcome_counts"],
            {
                "UNSUPPORTED": 19,
                "BUILD_FAILED": 20,
                "LAUNCH_FAILED": 0,
                "GATE_FAILED": 0,
                "GATE_PASSED": 113,
            },
        )

    def test_mechanical_top_three_is_frozen(self):
        self.assertEqual(
            self.imported["selection_confirm_candidate_ids"],
            self.contract["manifest"]["selection_confirm"]["candidate_ids"],
        )

    def test_fresh_stage_censuses_and_shams(self):
        selection = protocol.timing_plan("selection_confirm")
        self.assertEqual(len(selection), 240)
        self.assertEqual([row["position"] for row in selection], list(range(240)))
        for block in range(15):
            rows = [row for row in selection if row["block"] == block]
            self.assertEqual(len(rows), 16)
            self.assertEqual(
                sorted(row["block_position"] for row in rows), list(range(16))
            )
            self.assertEqual(sum(row["record_kind"] == "sham" for row in rows), 4)
        winners = {
            dsl: self.contract["manifest"]["selection_confirm"]["candidate_ids"][dsl][0]
            for dsl in protocol.DSLS
        }
        terminal = protocol.timing_plan("terminal_confirm", winners)
        self.assertEqual(len(terminal), 120)
        self.assertEqual(sum(row["record_kind"] == "sham" for row in terminal), 60)

    def test_terminal_rejects_foreign_winner(self):
        with self.assertRaisesRegex(protocol.ProtocolError, "winners"):
            protocol.timing_plan(
                "terminal_confirm",
                {"tilelang": "register_fused.tilelang.g00", "triton": "register_fused.triton.g09"},
            )

    def test_contract_tamper_fails(self):
        changed = copy.deepcopy(self.contract)
        changed["manifest"]["audit"]["requested_cells"] = 151
        with self.assertRaisesRegex(protocol.ProtocolError, "audit census"):
            protocol.validate_contract(changed)

    def test_execution_bindings_cover_sources_and_import(self):
        bindings = launch.execution_bindings()
        self.assertEqual(bindings["selection_expected_records"], 240)
        self.assertEqual(bindings["terminal_expected_records"], 120)
        self.assertEqual(
            bindings["imported_frontier_sha256"],
            protocol.canonical_sha256(self.imported),
        )

    def test_ratio_direction_respects_sham_floor(self):
        faster = analyze._ratio_result([0.9] * 15, [1.0] * 15, 0.01)
        unresolved = analyze._ratio_result([0.995] * 15, [1.0] * 15, 0.01)
        self.assertEqual(faster["direction_beyond_sham_floor"], "tilelang_lower_latency")
        self.assertEqual(unresolved["direction_beyond_sham_floor"], "unresolved")

    def test_claim_is_procedure_selected_and_signed_is_generalization(self):
        self.assertIn("procedure-selected", self.contract["estimand"])
        self.assertIn("generalization", self.contract["estimand"])
        self.assertNotIn("finite-frontier winners", self.contract["estimand"])

    def test_terminal_remote_paths_seal_selection_evidence(self):
        selection = set(launch.remote_ready_paths(self.contract, "selection_confirm"))
        terminal = set(launch.remote_ready_paths(self.contract, "terminal_confirm"))
        root = "ako_runs/controlled_followup/finite_frontier_ada_v1/results"
        self.assertIn(f"{root}/imported_frontier.json", selection)
        self.assertIn(f"{root}/selection_lock.json", terminal)
        self.assertIn(f"{root}/selection_confirm/launch_receipt.json", terminal)
        self.assertIn(f"{root}/selection_confirm/run_status.json", terminal)
        self.assertEqual(len(terminal - selection), 243)

    def test_raw_census_rejects_extra_file(self):
        plan = protocol.timing_plan("terminal_confirm", {
            dsl: self.contract["manifest"]["selection_confirm"]["candidate_ids"][dsl][0]
            for dsl in protocol.DSLS
        })
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for row in plan:
                (raw / protocol.timing_filename(row)).touch()
            protocol.validate_raw_census(raw, plan)
            (raw / "foreign.json").touch()
            with self.assertRaisesRegex(protocol.ProtocolError, "extra"):
                protocol.validate_raw_census(raw, plan)

    def test_timing_child_accepts_only_the_next_missing_position(self):
        plan = protocol.timing_plan("selection_confirm")[:3]
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / protocol.timing_filename(plan[0])).touch()
            launch.require_next_plan_position(raw, plan, plan[1])
            (raw / protocol.timing_filename(plan[2])).touch()
            with self.assertRaisesRegex(protocol.ProtocolError, "next frozen"):
                launch.require_next_plan_position(raw, plan, plan[1])

    def _gpu(self):
        value = {field: "0" for field in launch.GPU_SNAPSHOT_FIELDS}
        hardware = self.contract["manifest"]["hardware"]
        value.update(
            {
                "index": "0",
                "uuid": hardware["gpu_uuid"],
                "name": hardware["gpu_name"],
                "driver_version": hardware["driver_version"],
                "compute_cap": hardware["compute_capability"],
            }
        )
        return value

    @staticmethod
    def _idle(phase, pid, checked):
        return {
            "checked_at_unix": checked,
            "compute_processes": [],
            "foreign_compute_processes": [],
            "idle_except_self": True,
            "phase": phase,
            "returncode": 0,
            "self_pid": pid,
            "stderr": "",
        }

    def _valid_record(self, position=0, start=10.0, end=20.0, pid=100):
        implementation = "a" * 64
        times = [1.0] * 100
        return {
            "schema_version": 1,
            "record_type": "finite_frontier_ada_timing_record",
            "ok": True,
            "legacy_error": {"gate_pass": True},
            "times_ms": times,
            **protocol.summarize_times(times),
            "implementation_sha256": implementation,
            "build_metadata": {
                "artifacts": {},
                "builder": "test",
                "implementation_sha256": implementation,
                "n_kernels": 2,
            },
            "compile_s": 1.0,
            "gpu_preflight": self._gpu(),
            "gpu_idle_preflight": self._idle("record_pre", pid, start - 1),
            "gpu_idle_postflight": self._idle("record_post", pid, end - 1),
            "parent_pid": 10,
            "plan_position": position,
            "process_pid": pid,
            "t_start": start,
            "t_end": end,
            "toolchain": self.contract["manifest"]["toolchain"],
            "trials": 100,
            "warmup_iterations_actual": 1,
            "warmup_s": 2.0,
        }

    def test_record_runtime_build_and_order_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            record = self._valid_record()
            path.write_text(json.dumps(record), encoding="utf-8")
            launch.validate_timing_record(path, {"plan_position": 0}, self.contract)
            record["build_metadata"]["implementation_sha256"] = "b" * 64
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "fingerprint"):
                launch.validate_timing_record(path, {"plan_position": 0}, self.contract)

            record = self._valid_record()
            record["build_metadata"]["n_kernels"] = 1
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "fingerprint"):
                launch.validate_timing_record(path, {"plan_position": 0}, self.contract)

            record = self._valid_record()
            record.update(
                {
                    "launch_created_at_utc": "1970-01-01T00:00:15+00:00",
                    "launch_stage_preflight_unix": 15.0,
                }
            )
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "process receipt"):
                launch.validate_timing_record(
                    path,
                    {
                        "plan_position": 0,
                        "launch_created_at_utc": "1970-01-01T00:00:15+00:00",
                        "launch_stage_preflight_unix": 15.0,
                    },
                    self.contract,
                )

        plan = [{"position": 0}, {"position": 1}]
        records = [
            self._valid_record(0, 10.0, 20.0, 100),
            self._valid_record(1, 19.0, 30.0, 101),
        ]
        with self.assertRaisesRegex(protocol.ProtocolError, "timestamps"):
            launch.validate_record_sequence(records, plan)

    def test_launch_receipt_validates_environment_fields(self):
        expected = {
            "campaign_id": protocol.CAMPAIGN_ID,
            "execution_lock_sha256": "a" * 64,
            "gpu_lock_id": launch.GPU_LOCK_ID,
            "input_artifact_path": "input.json",
            "input_artifact_sha256": "b" * 64,
            "plan": [],
            "plan_sha256": "c" * 64,
            "stage": "selection_confirm",
            "timing": self.contract["manifest"]["timing"],
        }
        receipt = {
            "schema_version": 1,
            "record_type": "finite_frontier_ada_launch_receipt",
            "created_at_utc": "2026-08-05T00:00:00+00:00",
            "contract": {
                **expected,
                "git_commit": "d" * 40,
                "gpu": self._gpu(),
                "toolchain": self.contract["manifest"]["toolchain"],
                "upstream_ref": "refs/heads/test",
                "upstream_remote": "origin",
            },
            "gpu_idle_preflight": self._idle("stage_pre", 100, 1.0),
        }
        with mock.patch.object(launch, "validate_recorded_git_binding"):
            launch.validate_launch_receipt(receipt, expected, self.contract)
            receipt["contract"]["toolchain"] = {"python": "wrong"}
            with self.assertRaisesRegex(protocol.ProtocolError, "toolchain"):
                launch.validate_launch_receipt(receipt, expected, self.contract)


if __name__ == "__main__":
    unittest.main()
