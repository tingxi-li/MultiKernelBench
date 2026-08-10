from __future__ import annotations

import copy
import subprocess
import tempfile
import sys
import unittest
from contextlib import nullcontext
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
            "artifacts_sha256": material["artifacts_sha256"],
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
            "record_type": "decision_complexity_ada_v2_attempt",
            "schema_version": 1,
            "child_pid": 11,
            "target_implementation_sha256": target["implementation_sha256"],
            "target_artifacts_sha256": target["artifacts_sha256"],
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
            "record_type": "decision_complexity_ada_v2_attempt",
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

    def test_primary_cuda_source_mutation_is_rejected(self) -> None:
        material = self.contract["materials"]["candidates"][0]
        metadata = protocol.read_json(
            protocol.REPO_ROOT / material["record_path"]
        )["build_metadata"]
        built = mock.Mock(metadata=metadata)
        runner._validate_built_material("candidate", built, material)
        mutated = copy.deepcopy(metadata)
        mutated["artifacts"]["cuda_source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(protocol.ProtocolError, "artifacts drift"):
            runner._validate_built_material(
                "candidate", mock.Mock(metadata=mutated), material
            )

    def test_source_closure_binds_runtime_analysis_and_v1(self) -> None:
        sources = runner.source_map()
        expected = {
            str(path.relative_to(protocol.REPO_ROOT))
            for path in runner.EXTERNAL_SOURCE_PATHS
        }
        self.assertEqual(expected, set(sources) & expected)
        self.assertEqual(
            {path: protocol.file_sha256(protocol.REPO_ROOT / path) for path in expected},
            {path: sources[path] for path in expected},
        )
        locked = protocol.read_json(runner.LOCK_PATH)["source_sha256"]
        self.assertEqual(
            {path: sources[path] for path in expected},
            {path: locked[path] for path in expected},
        )
        predecessor = runner.predecessor_evidence_binding()
        self.assertEqual(
            protocol.read_json(runner.LOCK_PATH)["predecessor_evidence"],
            predecessor,
        )
        self.assertEqual(
            predecessor["prelaunch_provenance_sha256"],
            protocol.file_sha256(runner.PREDECESSOR_PROVENANCE_PATH),
        )
        self.assertEqual(
            predecessor["result_state"],
            "sealed_prelaunch_only_no_result_evidence",
        )

    def test_script_mode_resolves_only_campaign_modules(self) -> None:
        source = (
            "import runpy,sys;"
            f"sys.path.insert(0, {str(runner.HERE)!r});"
            f"r=runpy.run_path({str(runner.HERE / 'runner.py')!r},run_name='q4_runner_probe');"
            f"a=runpy.run_path({str(runner.HERE / 'analyze.py')!r},run_name='q4_analyze_probe');"
            "from pathlib import Path;"
            f"assert Path(r['local_analyze'].__file__).resolve()==Path({str(runner.HERE / 'analyze.py')!r}).resolve();"
            f"assert Path(a['_runner_module']().__file__).resolve()==Path({str(runner.HERE / 'runner.py')!r}).resolve();"
            "assert r['protocol'].__name__=='ako_runs.controlled_followup.decision_complexity_ada_v2.protocol';"
            "assert a['protocol'].__name__=='ako_runs.controlled_followup.decision_complexity_ada_v2.protocol'"
        )
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=protocol.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)

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
        with mock.patch.object(
            runner, "all_gpu_locks", return_value=nullcontext(())
        ), mock.patch.object(runner, "_authorized_popen") as popen:
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
                    "record_type": "decision_complexity_ada_v2_launch_receipt",
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
        paths = runner.gpu_lock_paths()
        expected = [
            Path("/tmp") / f"multikernelbench-{gpu_uuid}-timing.lock"
            for gpu_uuid in sorted(protocol.GPU_UUIDS)
        ]
        self.assertEqual(paths, expected)

    def test_serial_intervals_bind_exact_previous_completion(self) -> None:
        intervals = []
        previous = None
        for row in self.manifest["rows"]:
            launched = row["launch_sequence"] * 100
            ended = launched + 50
            binding = {
                "path": f"completion/{row['trajectory_id']}.json",
                "sha256": f"{row['launch_sequence']:064x}",
                "trajectory_ended_unix_ns": ended,
                "trajectory_id": row["trajectory_id"],
            }
            intervals.append(
                {
                    "child_launched_unix_ns": launched,
                    "completion_binding": binding,
                    "launch_sequence": row["launch_sequence"],
                    "previous_completion": previous,
                    "trajectory_ended_unix_ns": ended,
                    "trajectory_id": row["trajectory_id"],
                }
            )
            previous = binding
        analyze.validate_serial_intervals(intervals)

        overlap = [dict(item) for item in intervals]
        overlap[1]["child_launched_unix_ns"] = overlap[0]["trajectory_ended_unix_ns"]
        with self.assertRaisesRegex(protocol.ProtocolError, "overlap"):
            analyze.validate_serial_intervals(overlap)

        lost_binding = [dict(item) for item in intervals]
        lost_binding[1]["previous_completion"] = None
        with self.assertRaisesRegex(protocol.ProtocolError, "predecessor"):
            analyze.validate_serial_intervals(lost_binding)

        with tempfile.TemporaryDirectory(dir=protocol.HERE) as temporary, mock.patch.object(
            runner, "RESULTS", Path(temporary) / "results"
        ):
            first, second = self.manifest["rows"][:2]
            completion = (
                runner.RESULTS / "trajectories" / first["trajectory_id"] / "completion.json"
            )
            runner.exclusive_json(completion, {"trajectory_ended_unix_ns": 123})
            self.assertEqual(
                runner.previous_completion_binding(self.manifest, second),
                {
                    "path": str(completion.relative_to(protocol.REPO_ROOT)),
                    "sha256": protocol.file_sha256(completion),
                    "trajectory_ended_unix_ns": 123,
                    "trajectory_id": first["trajectory_id"],
                },
            )

    def test_gpu_lock_fds_validate_as_an_inherited_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            protocol, "GPU_LOCK_PATHS", (str(Path(temporary) / "gpu.lock"),)
        ):
            with runner.all_gpu_locks() as fds:
                with mock.patch.dict(
                    runner.os.environ,
                    {runner.GPU_LOCK_FDS_ENV: ",".join(map(str, fds))},
                    clear=True,
                ):
                    self.assertEqual(runner._validate_inherited_gpu_locks(), fds)

    def test_authorized_child_consumes_pipe_and_inherited_locks(self) -> None:
        source = (
            "from ako_runs.controlled_followup.decision_complexity_ada_v2 import runner;"
            "runner.protocol.GPU_LOCK_PATHS=(runner.os.environ['Q4_TEST_GPU_LOCK_PATH'],);"
            "runner._consume_parent_authorization('test', {'coordinate': 7});"
            "runner._validate_inherited_gpu_locks()"
        )
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / "gpu.lock")
            env = dict(runner.os.environ)
            env["Q4_TEST_GPU_LOCK_PATH"] = lock_path
            with mock.patch.object(protocol, "GPU_LOCK_PATHS", (lock_path,)):
                with runner.all_gpu_locks() as fds:
                    process, authorization = runner._authorized_popen(
                        [sys.executable, "-c", source],
                        env,
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
            **self.physical_gpu({"gpu_slot": 0, "gpu_uuid": self.contract["hardware"]["gpu_uuids"][0]}),
            "clocks.sm": "210",
            "memory.used": "17",
            "pstate": "P8",
            "temperature.gpu": "31",
            "utilization.gpu": "0",
        }
        with mock.patch.object(protocol.core, "gpu_snapshot", return_value=dynamic_snapshot):
            observed = runner._live_gpu(0, self.contract, require_idle=False)
        self.assertEqual(observed, self.physical_gpu({"gpu_slot": 0, "gpu_uuid": self.contract["hardware"]["gpu_uuids"][0]}))


if __name__ == "__main__":
    unittest.main()
