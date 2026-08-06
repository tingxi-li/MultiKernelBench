from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

from . import analyze, protocol, runner


class ExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = protocol.make_contract()
        cls.manifest = protocol.make_manifest(cls.contract)

    def physical_gpu(self, row: dict) -> dict:
        return {
            "compute_cap": self.contract["hardware"]["compute_capability"],
            "driver_version": "test-driver",
            "index": str(row["gpu_slot"]),
            "name": self.contract["hardware"]["name"],
            "uuid": row["gpu_uuid"],
        }

    def passed_attempt(self, row: dict, *, attempt_index: int = 1) -> dict:
        candidate = row["execution_contract"]["candidate_order"][attempt_index - 1]
        material = next(item for item in self.contract["materials"]["candidates"] if item["cell_id"] == candidate)
        target = next(item for item in self.contract["materials"]["candidates"] if item["cell_id"] == protocol.TARGET_CELL_ID)
        candidate_times = [100.0] * 60 + [1.0] * 40
        target_times = [50.0] * 60 + [1.0] * 40
        return {
            "active_s": 1.0,
            "attempt_index": attempt_index,
            "authorization_sha256": "a" * 64,
            "campaign_id": protocol.CAMPAIGN_ID,
            "candidate_cell_id": candidate,
            "candidate_tail_median_ms": 1.0,
            "candidate_times_ms": candidate_times,
            "cache_isolation": protocol.attempt_cache_receipt(
                row, candidate, attempt_index
            ),
            "execution_lock_sha256": protocol.file_sha256(runner.LOCK_PATH),
            "gate_evidence_reused": {"path": material["gate_path"], "sha256": material["gate_sha256"]},
            "gpu_slot": row["gpu_slot"],
            "gpu_uuid": row["gpu_uuid"],
            "gpu_lock_paths": list(protocol.GPU_LOCK_PATHS),
            "implementation_sha256": material["implementation_sha256"],
            "live_gate": {"candidate": {"gate_pass": True}, "target": {"gate_pass": True}},
            "manifest_sha256": protocol.file_sha256(runner.MANIFEST_PATH),
            "measurement_order": protocol.timing_pair_order(row, attempt_index),
            "physical_gpu_preflight": self.physical_gpu(row),
            "parent_pid": 10,
            "ratio_to_target": 1.0,
            "record_type": "decision_complexity_ada_v1_attempt",
            "schema_version": 1,
            "child_pid": 11,
            "target_implementation_sha256": target["implementation_sha256"],
            "target_gate_evidence_reused": {
                "path": target["gate_path"], "sha256": target["gate_sha256"]
            },
            "target_tail_median_ms": 1.0,
            "target_times_ms": target_times,
            "terminal_status": "GATE_PASSED",
            "toolchain_sha256": protocol.read_json(runner.LOCK_PATH)["toolchain_sha256"],
            "trajectory_id": row["trajectory_id"],
            "warmup_iterations": {"candidate": 1, "target": 1},
        }

    def failed_attempt(self, row: dict, attempt_index: int) -> dict:
        return {
            "active_s": 1.0,
            "attempt_index": attempt_index,
            "authorization_sha256": "a" * 64,
            "cache_isolation": protocol.attempt_cache_receipt(
                row,
                row["execution_contract"]["candidate_order"][attempt_index - 1],
                attempt_index,
            ),
            "campaign_id": protocol.CAMPAIGN_ID,
            "candidate_cell_id": row["execution_contract"]["candidate_order"][attempt_index - 1],
            "child_pid": 11,
            "execution_lock_sha256": protocol.file_sha256(runner.LOCK_PATH),
            "gpu_slot": row["gpu_slot"],
            "gpu_uuid": row["gpu_uuid"],
            "gpu_lock_paths": list(protocol.GPU_LOCK_PATHS),
            "manifest_sha256": protocol.file_sha256(runner.MANIFEST_PATH),
            "physical_gpu_preflight": self.physical_gpu(row),
            "parent_pid": 10,
            "record_type": "decision_complexity_ada_v1_attempt",
            "schema_version": 1,
            "terminal_status": "BUILD_FAILED",
            "toolchain_sha256": protocol.read_json(runner.LOCK_PATH)["toolchain_sha256"],
            "trajectory_id": row["trajectory_id"],
        }

    def summary_rows(self) -> list[dict]:
        return [
            {
                "active_s": float(index + 1), "arm": row["arm"],
                "attempts_consumed": 1, "event_observed": True,
                "first_event_attempt": 1, "gpu_slot": row["gpu_slot"],
                "manifest_row_sha256": protocol.canonical_sha256(row),
                "replicate": row["replicate"], "trajectory_id": row["trajectory_id"],
            }
            for index, row in enumerate(self.manifest["rows"])
        ]

    def test_settled_tail_and_terminal_stop(self) -> None:
        row = next(item for item in self.manifest["rows"] if item["arm"] == "sensitivity_target_hint")
        attempt = self.passed_attempt(row)
        analyze.validate_attempt(self.contract, row, attempt)
        result = analyze.derive_trajectory(self.contract, row, [attempt])
        self.assertTrue(result["event_observed"])
        self.assertEqual(result["first_event_attempt"], 1)
        with self.assertRaisesRegex(protocol.ProtocolError, "continued"):
            analyze.derive_trajectory(self.contract, row, [attempt, self.passed_attempt(row, attempt_index=2)])

    def test_failures_are_charged_until_exhaustion(self) -> None:
        row = next(item for item in self.manifest["rows"] if item["arm"] == "c2_open_1")
        attempts = [self.failed_attempt(row, index) for index in (1, 2, 3)]
        result = analyze.derive_trajectory(self.contract, row, attempts)
        self.assertFalse(result["event_observed"])
        self.assertEqual(result["attempts_consumed"], 3)
        with self.assertRaisesRegex(protocol.ProtocolError, "stopped before"):
            analyze.derive_trajectory(self.contract, row, attempts[:2])

    def test_noncontrolling_survival_summary(self) -> None:
        result = analyze.summarize(self.contract, self.summary_rows())
        self.assertFalse(result["controlling"])
        self.assertTrue(result["pilot_valid"])
        self.assertTrue(result["interpretation_valid"])
        self.assertEqual(result["trajectory_count"], 24)
        self.assertEqual(set(result["arm_summaries"]), set(protocol.ARMS))
        self.assertNotIn("pairwise_logrank", result["attempt_survival"])
        self.assertEqual(len(result["paired_treatment_contrasts"]), 2)
        self.assertEqual(
            sum(
                len(contrast["outcomes"])
                for contrast in result["paired_treatment_contrasts"].values()
            ),
            4,
        )
        self.assertTrue(
            all(
                "holm_adjusted_p_value" in outcome
                for contrast in result["paired_treatment_contrasts"].values()
                for outcome in contrast["outcomes"].values()
            )
        )
        self.assertLessEqual(
            result["active_time_survival"]["effective_tau"],
            result["active_time_survival"]["common_observed_support"],
        )
        self.assertFalse(result["active_time_survival"]["rmst_extrapolated"])

    def test_preregistered_controls_mark_pilot_invalid(self) -> None:
        sham_rows = self.summary_rows()
        sham = next(
            row for row in sham_rows
            if row["replicate"] == 0 and row["arm"] == "label_sham_b"
        )
        sham.update({"attempts_consumed": 2, "first_event_attempt": 2})
        sham_result = analyze.summarize(self.contract, sham_rows)
        self.assertFalse(sham_result["pilot_valid"])
        self.assertFalse(sham_result["interpretation_valid"])
        self.assertFalse(sham_result["control_validation"]["label_sham_passed"])

        hint_rows = self.summary_rows()
        hint = next(
            row for row in hint_rows
            if row["replicate"] == 0
            and row["arm"] == "sensitivity_target_hint"
        )
        hint.update({"event_observed": False, "first_event_attempt": None})
        hint_result = analyze.summarize(self.contract, hint_rows)
        self.assertFalse(hint_result["pilot_valid"])
        self.assertFalse(hint_result["control_validation"]["target_hint_passed"])

    def test_shams_share_every_attempt_timing_order(self) -> None:
        for replicate in range(protocol.REPLICATES):
            by_arm = {
                row["arm"]: row
                for row in self.manifest["rows"]
                if row["replicate"] == replicate
            }
            for attempt_index in range(1, 7):
                self.assertEqual(
                    protocol.timing_pair_order(
                        by_arm["label_sham_a"], attempt_index
                    ),
                    protocol.timing_pair_order(
                        by_arm["label_sham_b"], attempt_index
                    ),
                )

    def test_attempt_cache_is_fresh_and_unique(self) -> None:
        row = self.manifest["rows"][0]
        candidate = row["execution_contract"]["candidate_order"][0]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_env = runner._attempt_environment(
                {"PATH": "/bin", "PHASE2_TL_CACHE": "1"},
                row, candidate, 1, row["gpu_slot"], Path(first),
            )
            second_env = runner._attempt_environment(
                {"PATH": "/bin", "PHASE2_TL_CACHE": "1"},
                row, candidate, 1, row["gpu_slot"], Path(second),
            )
            self.assertEqual(first_env["PHASE2_TL_CACHE"], "0")
            self.assertEqual(first_env["TMPDIR"], first)
            for name in protocol.CACHE_DIRECTORIES:
                self.assertNotEqual(first_env[name], second_env[name])
                self.assertEqual(Path(first_env[name]).parent, Path(first))
            with mock.patch.dict(runner.os.environ, first_env, clear=True):
                runner._validate_fresh_cache_environment(row, candidate, 1)

    def test_child_crash_is_a_charged_immutable_attempt(self) -> None:
        row = next(item for item in self.manifest["rows"] if item["arm"] == "c2_open_1")
        candidate = row["execution_contract"]["candidate_order"][0]
        authorization = {"child_pid": 11, "parent_pid": 10}
        record = runner._charged_child_failure(
            row, candidate, 1, row["gpu_slot"], self.physical_gpu(row),
            1.0, "LAUNCH_FAILED", "child exited", authorization, "a" * 64, 139,
        )
        analyze.validate_attempt(self.contract, row, record)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempt.json"
            runner.exclusive_json(path, record)
            with self.assertRaisesRegex(protocol.ProtocolError, "overwrite"):
                runner.exclusive_json(path, record)

    def test_ready_refuses_before_any_launch(self) -> None:
        with mock.patch.object(runner, "_authorized_popen") as popen:
            with self.assertRaisesRegex(protocol.ProtocolError, "provenance is missing"):
                runner.ready()
            popen.assert_not_called()

    def test_live_upstream_ref_preserves_branch_slashes(self) -> None:
        self.assertEqual(
            runner.upstream_remote_ref("origin/feature/c2-pilot"),
            ("origin", "refs/heads/feature/c2-pilot"),
        )

    def test_internal_children_require_inherited_parent_fd(self) -> None:
        with mock.patch.dict(runner.os.environ, {}, clear=True):
            with self.assertRaisesRegex(protocol.ProtocolError, "authorization FD is missing"):
                runner._consume_parent_authorization("attempt", {})

    def test_public_attempt_cannot_inject_without_parent_fd(self) -> None:
        row = self.manifest["rows"][0]
        candidate = row["execution_contract"]["candidate_order"][0]
        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary) / "results"
            runner.exclusive_json(result_root / "launch_receipt.json", {"retained": True})
            output = result_root / "trajectories" / row["trajectory_id"] / "attempts" / "attempt01.json"
            with (
                mock.patch.object(runner, "RESULTS", result_root),
                mock.patch.dict(runner.os.environ, {}, clear=True),
                mock.patch.object(runner, "_live_gpu") as live_gpu,
            ):
                with self.assertRaisesRegex(protocol.ProtocolError, "authorization FD is missing"):
                    runner.attempt(row, candidate, 1, row["gpu_slot"], output)
            live_gpu.assert_not_called()
            self.assertFalse(output.exists())

    def test_retained_launch_receipt_cannot_run_trajectory_directly(self) -> None:
        row = self.manifest["rows"][0]
        lock = protocol.read_json(runner.LOCK_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary) / "results"
            launch_path = result_root / "launch_receipt.json"
            runner.exclusive_json(
                launch_path,
                {
                    "campaign_id": protocol.CAMPAIGN_ID,
                    "execution_lock_sha256": protocol.file_sha256(runner.LOCK_PATH),
                    "manifest_sha256": protocol.file_sha256(runner.MANIFEST_PATH),
                    "record_type": "decision_complexity_ada_v1_launch_receipt",
                    "requested_trajectories": 24,
                    "schema_version": 1,
                    "toolchain_sha256": lock["toolchain_sha256"],
                },
            )
            with (
                mock.patch.object(runner, "RESULTS", result_root),
                mock.patch.dict(runner.os.environ, {}, clear=True),
                mock.patch.object(runner, "_idle_gpu_evidence") as idle,
            ):
                with self.assertRaisesRegex(protocol.ProtocolError, "authorization FD is missing"):
                    runner.trajectory(row["trajectory_id"], row["gpu_slot"], launch_path)
            idle.assert_not_called()

    def test_execute_refuses_retained_result_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary) / "results"
            result_root.mkdir()
            (result_root / "active.lock").touch()
            with mock.patch.object(runner, "RESULTS", result_root):
                runner._refuse_retained_execution_evidence()
                runner.exclusive_json(
                    result_root / "launch_receipt.json", {"retained": True}
                )
                with self.assertRaisesRegex(protocol.ProtocolError, "new successor"):
                    runner._refuse_retained_execution_evidence()

    def test_host_gpu_locks_are_uuid_global_and_sorted(self) -> None:
        from ako_runs.controlled_followup.finite_frontier_ada_v1.launch import (
            GPU_LOCK_PATH as SHARED_GPU0_LOCK_PATH,
        )

        paths = runner.gpu_lock_paths()
        expected = [
            Path("/tmp") / f"multikernelbench-{gpu_uuid}-timing.lock"
            for gpu_uuid in sorted(protocol.GPU_UUIDS)
        ]
        self.assertEqual(paths, expected)
        self.assertIn(SHARED_GPU0_LOCK_PATH, paths)

    def test_wave_intervals_require_common_overlap_and_separation(self) -> None:
        def wave(position: int, base: int) -> list[dict]:
            identifiers = [f"wave{position}-child{replicate}" for replicate in range(protocol.REPLICATES)]
            ready_receipts = {
                trajectory_id: {
                    "authorization_sha256": f"{replicate + 1:064x}",
                    "campaign_id": protocol.CAMPAIGN_ID,
                    "child_pid": 1000 + position * 10 + replicate,
                    "ready_unix_ns": base + 20 + replicate,
                    "trajectory_id": trajectory_id,
                    "wave_position": position,
                }
                for replicate, trajectory_id in enumerate(identifiers)
            }
            witness = {
                "campaign_id": protocol.CAMPAIGN_ID,
                "ready_receipts": ready_receipts,
                "release_nonce": f"{position + 1:064x}",
                "release_unix_ns": base + 30,
                "wave_position": position,
            }
            return [
                {
                    "child_launched_unix_ns": base + replicate,
                    "trajectory_ended_unix_ns": base + 40 + replicate,
                    "trajectory_id": trajectory_id,
                    "trajectory_started_unix_ns": base + 10 + replicate,
                    "wave_ready_receipt": ready_receipts[trajectory_id],
                    "wave_release_unix_ns": witness["release_unix_ns"],
                    "wave_witness": witness,
                }
                for replicate, trajectory_id in enumerate(identifiers)
            ]

        intervals = {
            position: wave(position, position * 100)
            for position in range(len(protocol.ARMS))
        }
        analyze.validate_wave_intervals(intervals)

        early_dead_child = {
            position: list(wave) for position, wave in intervals.items()
        }
        early_dead_child[0][-1] = dict(early_dead_child[0][-1])
        early_dead_child[0][-1]["trajectory_ended_unix_ns"] = 29
        with self.assertRaisesRegex(protocol.ProtocolError, "strictly inside"):
            analyze.validate_wave_intervals(early_dead_child)

        adjacent_overlap = {
            position: list(wave) for position, wave in intervals.items()
        }
        adjacent_overlap[1] = wave(1, 25)
        with self.assertRaisesRegex(protocol.ProtocolError, "adjacent.*overlap"):
            analyze.validate_wave_intervals(adjacent_overlap)

    def test_parent_wave_barrier_releases_one_shared_witness(self) -> None:
        children = []
        release_reads = []
        for replicate in range(protocol.REPLICATES):
            ready_read, ready_write = runner.os.pipe()
            release_read, release_write = runner.os.pipe()
            trajectory_id = f"barrier-child-{replicate}"
            authorization = {"authorized_unix_ns": 1, "replicate": replicate}
            process = mock.Mock(pid=2000 + replicate)
            ready = {
                "authorization_sha256": protocol.canonical_sha256(authorization),
                "campaign_id": protocol.CAMPAIGN_ID,
                "child_pid": process.pid,
                "ready_unix_ns": runner.time.time_ns(),
                "trajectory_id": trajectory_id,
                "wave_position": 0,
            }
            runner._write_canonical_pipe(ready_write, ready, "test ready")
            children.append({
                "authorization": authorization,
                "process": process,
                "ready_fd": ready_read,
                "release_fd": release_write,
                "row": {"block_position": 0, "trajectory_id": trajectory_id},
            })
            release_reads.append(release_read)
        witness = runner._release_wave(children, 0, 1)
        observed = [
            runner._read_canonical_pipe(fd, "test release") for fd in release_reads
        ]
        self.assertEqual(observed, [witness] * protocol.REPLICATES)
        self.assertEqual(set(witness["ready_receipts"]), {
            child["row"]["trajectory_id"] for child in children
        })

    def test_gpu_lock_fds_validate_as_an_inherited_chain(self) -> None:
        with runner.all_gpu_locks() as fds:
            with mock.patch.dict(
                runner.os.environ,
                {runner.GPU_LOCK_FDS_ENV: ",".join(map(str, fds))},
                clear=True,
            ):
                self.assertEqual(runner._validate_inherited_gpu_locks(), fds)

    def test_authorized_child_consumes_pipe_and_inherited_locks(self) -> None:
        source = (
            "from ako_runs.controlled_followup.decision_complexity_ada_v1 import runner;"
            "runner._consume_parent_authorization('test', {'coordinate': 7});"
            "runner._validate_inherited_gpu_locks()"
        )
        with runner.all_gpu_locks() as fds:
            process, authorization = runner._authorized_popen(
                [sys.executable, "-c", source],
                dict(runner.os.environ),
                {"coordinate": 7, "scope": "test"},
                fds,
            )
            self.assertEqual(process.wait(timeout=20), 0)
        self.assertEqual(authorization["child_pid"], process.pid)

    def test_execution_lock_binds_actual_toolchain(self) -> None:
        lock = protocol.read_json(runner.LOCK_PATH)
        self.assertEqual(lock["toolchain"], runner.toolchain_fingerprint())
        self.assertEqual(
            lock["toolchain_sha256"], protocol.canonical_sha256(lock["toolchain"])
        )

    def test_live_gpu_projects_only_static_identity(self) -> None:
        dynamic_snapshot = {
            **self.physical_gpu({"gpu_slot": 2, "gpu_uuid": self.contract["hardware"]["gpu_uuids"][2]}),
            "clocks.sm": "210",
            "memory.used": "17",
            "pstate": "P8",
            "temperature.gpu": "31",
            "utilization.gpu": "0",
        }
        with mock.patch.object(protocol.core, "gpu_snapshot", return_value=dynamic_snapshot):
            observed = runner._live_gpu(2, self.contract, require_idle=False)
        self.assertEqual(observed, self.physical_gpu({"gpu_slot": 2, "gpu_uuid": self.contract["hardware"]["gpu_uuids"][2]}))


if __name__ == "__main__":
    unittest.main()
