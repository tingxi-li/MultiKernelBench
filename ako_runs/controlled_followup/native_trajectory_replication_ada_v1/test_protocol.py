from __future__ import annotations

import copy
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import analyze, protocol, runner


class NativeReplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = protocol.make_contract()
        cls.manifest = protocol.make_manifest(cls.contract)
        cls.toolchain = runner.live_toolchain()

    def test_exact_material_and_randomized_raw_census(self) -> None:
        materials = self.contract["materials"]["selected_prefixes"]
        self.assertEqual(len(materials), 12)
        self.assertTrue(all(row["terminal_outcome"] == "GATE_PASSED" for row in materials))
        self.assertTrue(all(row["gate_evidence"]["size"] > 0 for row in materials))
        self.assertEqual(len(self.manifest["rows"]), 420)
        self.assertEqual(self.manifest["rows_per_block"], 28)
        self.assertEqual(sum(row["record_kind"] == "cell" for row in self.manifest["rows"]), 360)
        self.assertEqual(sum(row["record_kind"] == "same_config_label_sham" for row in self.manifest["rows"]), 60)

    def test_shams_have_honest_same_config_identity(self) -> None:
        self.assertFalse(self.contract["sham"]["source_byte_identity_claimed"])
        for block in range(protocol.BLOCKS):
            rows = [
                row for row in self.manifest["rows"]
                if row["block"] == block and row["record_kind"] == "same_config_label_sham"
            ]
            self.assertEqual({row["cell_id"] for row in rows}, {protocol.SHAM_CELL_ID})
            self.assertEqual({row["label"] for row in rows}, set(protocol.SHAM_LABELS))
            self.assertEqual(len({row["implementation_sha256"] for row in rows}), 1)

    def test_manifest_tampering_is_rejected(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["block_position"] = 99
        with self.assertRaisesRegex(protocol.ProtocolError, "deterministic randomized projection"):
            protocol.validate_manifest(self.contract, changed)

    def _gpu(self) -> dict:
        return {
            "compute_cap": self.contract["hardware"]["compute_capability"],
            "driver_version": "test-driver",
            "index": "0",
            "name": self.contract["hardware"]["gpu_name"],
            "uuid": self.contract["hardware"]["gpu_uuid"],
        }

    def test_gpu_projection_ignores_dynamic_telemetry(self) -> None:
        snapshot = {**self._gpu(), "memory.used": "17", "pstate": "P8", "temperature.gpu": "31"}
        with mock.patch.object(protocol.core, "gpu_snapshot", return_value=snapshot):
            self.assertEqual(runner._gpu0(self.contract, require_idle=False), self._gpu())
        self.assertEqual(
            runner.GLOBAL_GPU0_LOCK,
            Path("/tmp") / f"multikernelbench-{protocol.GPU0_UUID}-timing.lock",
        )

    def test_upstream_ref_preserves_branch_slashes(self) -> None:
        self.assertEqual(
            runner.upstream_remote_ref("origin/feature/native-replication"),
            ("origin", "refs/heads/feature/native-replication"),
        )

    def test_ready_refuses_before_any_launch(self) -> None:
        with mock.patch.object(runner.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(protocol.ProtocolError, "provenance/execution freeze"):
                runner.ready()
            popen.assert_not_called()

    def test_timing_child_requires_inherited_gpu_lock(self) -> None:
        with mock.patch.dict(runner.os.environ, {}, clear=True):
            with self.assertRaisesRegex(protocol.ProtocolError, "inherited GPU0 lock"):
                runner._validate_inherited_gpu0_lock()

    def test_execution_lock_binds_material_stage_provenance(self) -> None:
        material_lock = protocol.read_json(runner.MATERIAL_LOCK_PATH)
        receipt = {
            "campaign_id": protocol.CAMPAIGN_ID,
            "created_utc": "2026-08-05T00:00:00+00:00",
            "git_commit": "1" * 40,
            "git_upstream": "origin/main",
            "git_upstream_commit": "1" * 40,
            "gpu0": self._gpu(),
            "host": runner.platform.node(),
            "live_remote_commit": "1" * 40,
            "live_remote_name": "origin",
            "live_remote_ref": "refs/heads/main",
            "material_lock_sha256": protocol.file_sha256(runner.MATERIAL_LOCK_PATH),
            "record_type": "native_trajectory_replication_ada_v1_prelaunch_provenance",
            "remote_push_verified": True,
            "schema_version": 1,
            "source_bundle_sha256": material_lock["source_bundle_sha256"],
            "stage": "material_commit_live_upstream_verified",
            "toolchain": self.toolchain,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prelaunch_provenance.json"
            path.write_bytes(runner.stable_bytes(receipt))
            with mock.patch.object(runner, "PROVENANCE_PATH", path):
                lock = runner.make_execution_lock(
                    self.contract, self.manifest, material_lock, receipt,
                )
                malformed = {**receipt, "host": "another-host"}
                with self.assertRaisesRegex(protocol.ProtocolError, "provenance"):
                    runner.make_execution_lock(
                        self.contract, self.manifest, material_lock, malformed,
                    )
        self.assertEqual(lock["expected_raw_records"], 420)
        self.assertEqual(lock["manifest_plan_sha256"], self.manifest["plan_sha256"])

    def test_settled_tail_record_is_raw_derived_and_gpu_bound(self) -> None:
        row = self.manifest["rows"][0]
        values = [100.0] * 60 + [1.0] * 40
        record = {
            "block": row["block"],
            "block_position": row["block_position"],
            "campaign_id": protocol.CAMPAIGN_ID,
            "cell_id": row["cell_id"],
            "cell_sha256": next(
                item["cell_sha256"] for item in self.contract["materials"]["selected_prefixes"]
                if item["cell_id"] == row["cell_id"]
            ),
            "cell_sha256_expected": next(
                item["cell_sha256"] for item in self.contract["materials"]["selected_prefixes"]
                if item["cell_id"] == row["cell_id"]
            ),
            "compute_pids_preflight": [],
            "distribution": row["distribution"],
            "full_median_ms": statistics.median(values),
            "global_position": row["global_position"],
            "gpu_preflight": self._gpu(),
            "gpu_postflight": self._gpu(),
            "compute_pids_postflight": [123],
            "implementation_sha256": row["implementation_sha256"],
            "implementation_sha256_expected": row["implementation_sha256"],
            "label": row["label"],
            "live_correctness": {"gate_pass": True},
            "manifest_row_sha256": protocol.canonical_sha256(row),
            "ok": True,
            "physical_gpu": 0,
            "process_pid": 123,
            "primary_tail_median_ms": 1.0,
            "record_kind": row["record_kind"],
            "record_type": "native_trajectory_replication_ada_v1_timing_record",
            "row_id": row["row_id"],
            "schema_version": 1,
            "t_end_unix_ns": 2,
            "t_start_unix_ns": 1,
            "times_ms": values,
            "trials": 100,
            "toolchain": self.toolchain,
            "warmup_s": 2.0,
            "warmup_iterations_actual": 1,
            "compile_s": 0.0,
            "build_metadata": {"implementation_sha256": row["implementation_sha256"], "n_kernels": 2},
        }
        execution_binding = {"toolchain": self.toolchain}
        analyze.validate_timing_record(
            self.contract, row, record, execution_binding=execution_binding
        )
        changed_toolchain = copy.deepcopy(record)
        changed_toolchain["toolchain"]["triton_version"] = "changed"
        with self.assertRaisesRegex(protocol.ProtocolError, "foreign"):
            analyze.validate_timing_record(
                self.contract, row, changed_toolchain,
                execution_binding=execution_binding,
            )
        record["primary_tail_median_ms"] = 2.0
        with self.assertRaisesRegex(protocol.ProtocolError, "raw-trial-derived"):
            analyze.validate_timing_record(
                self.contract, row, record, execution_binding=execution_binding
            )
        record["primary_tail_median_ms"] = 1.0
        record["warmup_iterations_actual"] = 0
        with self.assertRaisesRegex(protocol.ProtocolError, "implementation/gate"):
            analyze.validate_timing_record(
                self.contract, row, record, execution_binding=execution_binding
            )

    def test_position_receipt_rejects_overlap(self) -> None:
        row = self.manifest["rows"][0]
        record = {
            "gpu_preflight": self._gpu(),
            "t_start_unix_ns": 2,
            "t_end_unix_ns": 3,
        }
        position = {
            "campaign_id": protocol.CAMPAIGN_ID,
            "child_completed_unix_ns": 4,
            "child_launched_unix_ns": 1,
            "compute_pids_after_child": [],
            "global_position": row["global_position"],
            "gpu_idle_after_child": self._gpu(),
            "gpu_postflight_error": None,
            "raw_path": str(runner.CONTRACT_PATH.relative_to(protocol.REPO_ROOT)),
            "raw_sha256": protocol.file_sha256(runner.CONTRACT_PATH),
            "record_type": "native_trajectory_replication_ada_v1_position_receipt",
            "returncode": 0,
            "row_id": row["row_id"],
        }
        self.assertEqual(
            analyze.validate_position_receipt(row, record, position, runner.CONTRACT_PATH, 0),
            4,
        )
        with self.assertRaisesRegex(protocol.ProtocolError, "position/timestamp"):
            analyze.validate_position_receipt(row, record, position, runner.CONTRACT_PATH, 2)

    def test_same_campaign_recurrence_requires_both_distributions(self) -> None:
        records = []
        strategy_time = {
            "tilelang": [100.0, 80.0, 64.0],
            "triton": [120.0, 100.0, 80.0],
            "cuda_noptx": [110.0, 90.0, 90.0],
            "cuda_unlimited": [105.0, 85.0, 68.0],
        }
        for row in self.manifest["rows"]:
            if row["record_kind"] == "same_config_label_sham":
                value = 75.0
            else:
                value = strategy_time[row["lane"]][protocol.STRATEGIES.index(row["strategy"])]
            records.append(
                {
                    "implementation_sha256": row["implementation_sha256"],
                    "primary_tail_median_ms": value,
                    "row_id": row["row_id"],
                }
            )
        result = analyze.estimate(self.contract, self.manifest, records)
        by_key = {
            (row["destination_lane"], row["strategy_step"]): row
            for row in result["same_campaign_recurrence_classifications"]
        }
        self.assertEqual(by_key[("triton", protocol.STEP_NAMES[0])]["classification"], "same_campaign_speedup_recurrence")
        self.assertEqual(by_key[("triton", protocol.STEP_NAMES[1])]["classification"], "same_campaign_speedup_recurrence")
        self.assertEqual(by_key[("cuda_noptx", protocol.STEP_NAMES[1])]["classification"], "unresolved_at_sham_floor")
        self.assertEqual(by_key[("tilelang", protocol.STEP_NAMES[0])]["classification"], "precommitted_same_campaign_reference")

    def test_same_direction_slowdown_is_not_called_a_gain(self) -> None:
        records = []
        slow = [80.0, 100.0, 125.0]
        for row in self.manifest["rows"]:
            value = 75.0 if row["record_kind"] == "same_config_label_sham" else slow[protocol.STRATEGIES.index(row["strategy"])]
            records.append(
                {
                    "implementation_sha256": row["implementation_sha256"],
                    "primary_tail_median_ms": value,
                    "row_id": row["row_id"],
                }
            )
        result = analyze.estimate(self.contract, self.manifest, records)
        row = next(
            item for item in result["same_campaign_recurrence_classifications"]
            if item["destination_lane"] == "triton" and item["strategy_step"] == protocol.STEP_NAMES[0]
        )
        self.assertEqual(row["classification"], "same_campaign_slowdown_recurrence")
        self.assertFalse(row["same_campaign_gain_recurrence"])


if __name__ == "__main__":
    unittest.main()
