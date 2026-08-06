from __future__ import annotations

import copy
import importlib
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import analyze, artifacts, protocol, runner


class NativeReplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract, cls.manifest = runner.prepare()
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

    def test_successor_preserves_plan_and_binds_failed_predecessor(self) -> None:
        predecessor = protocol.read_json(protocol.PREDECESSOR / "manifest.json")
        coordinates = lambda rows: [
            (row["block"], row["block_position"], row["cell_id"], row["distribution"], row["label"])
            for row in rows
        ]
        self.assertEqual(coordinates(self.manifest["rows"]), coordinates(predecessor["rows"]))
        incident = protocol.predecessor_incident()
        self.assertEqual(incident["artifact_closure"]["retained_files"], 3)
        self.assertFalse(incident["policy"]["reuse_authorized"])

    def test_shams_have_honest_same_config_identity(self) -> None:
        self.assertTrue(self.contract["sham"]["source_byte_identity_claimed"])
        for block in range(protocol.BLOCKS):
            rows = [
                row for row in self.manifest["rows"]
                if row["block"] == block and row["record_kind"] == "same_config_label_sham"
            ]
            self.assertEqual({row["cell_id"] for row in rows}, {protocol.SHAM_CELL_ID})
            self.assertEqual({row["label"] for row in rows}, set(protocol.SHAM_LABELS))
            self.assertEqual(len({row["predecessor_implementation_sha256"] for row in rows}), 1)

    def test_exact_artifact_plan_and_entry_local_atomic_temp(self) -> None:
        plan = artifacts.admission_plan(self.contract)
        self.assertEqual(len(plan), 12)
        self.assertEqual(len({row["cell_id"] for row in plan}), 12)
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            with mock.patch.object(artifacts, "ARTIFACTS_ROOT", Path(directory)):
                env = artifacts.prepare_cache_environment(plan[0]["cell_id"], "admit")
                snapshot = artifacts.atomic_temp_device_snapshot(plan[0]["cell_id"])
                hostile = {"TMPDIR": "/foreign", "TMP": "/foreign", "TEMP": "/foreign"}
                with mock.patch.dict(artifacts.os.environ, {**hostile, **env}, clear=True):
                    artifacts.validate_cache_environment(plan[0]["cell_id"], "admit")
        self.assertTrue(snapshot["same_filesystem"])
        self.assertEqual(snapshot["cache_st_dev"], snapshot["runtime_tmp_st_dev"])
        self.assertEqual(env["TMPDIR"], env["TMP"])
        self.assertEqual(env["TMPDIR"], env["TEMP"])
        self.assertTrue(env["TILELANG_TMP_DIR"].startswith(env["TMPDIR"] + "/"))

    def test_artifact_identity_excludes_diagnostic_metadata(self) -> None:
        cache = {
            "generated_sources": [{"path": "kernel.cu", "sha256": "1" * 64, "size": 1}],
            "loadable_code_objects": [{"path": "kernel.so", "sha256": "2" * 64, "size": 2}],
        }
        first = artifacts.generated_artifact_identity({**cache, "kernel_resources_error": "cache hit"})
        second = artifacts.generated_artifact_identity(
            {
                "generated_sources": [{**cache["generated_sources"][0], "path": "elsewhere.cu"}],
                "loadable_code_objects": [{**cache["loadable_code_objects"][0], "path": "elsewhere.so"}],
                "kernel_resources": {"registers": 32},
            }
        )
        self.assertEqual(first, second)

    def test_cuda_noptx_load_only_replays_admitted_resource_diagnostics(self) -> None:
        log = """ptxas info : Compiling entry function '_Z12fused_kernelv' for 'sm_89'
ptxas info : Function properties for _Z12fused_kernelv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 166 registers, 384 bytes cmem[0]
"""
        resources = protocol.core.ptxas_kernel_resources(log, "fused_kernel")
        build = {
            "cell_id": "register_fused.cuda_noptx.g01",
            "source_build_metadata": {
                "artifacts": {"ptxas_log": log, "kernel_resources": resources}
            },
        }
        replay = artifacts._admitted_cuda_noptx_side_compile(build)
        self.assertEqual(replay["ptxas_log"], log)
        self.assertEqual(replay["kernel_resources"], resources)
        changed = copy.deepcopy(build)
        changed["source_build_metadata"]["artifacts"]["kernel_resources"]["registers"] += 1
        with self.assertRaisesRegex(protocol.ProtocolError, "resource diagnostics changed"):
            artifacts._admitted_cuda_noptx_side_compile(changed)

    def test_cuda_noptx_module_alias_uses_load_only_replay(self) -> None:
        import torch.utils.cpp_extension as cpp_extension

        side_compile_module = importlib.import_module("variants.cuda_noptx_gemm")
        original_module_inline = side_compile_module.load_inline
        original_torch_inline = cpp_extension.load_inline
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            cache = Path(directory)
            build = {
                "cache": {"cache_root": protocol.repo_path(cache), "files": []},
                "cell_id": "global_intermediate.cuda_noptx.g01",
                "torch_inline_requests": [],
            }
            with artifacts.load_only_guards(build):
                self.assertIs(side_compile_module.load_inline, cpp_extension.load_inline)
                self.assertIsNot(side_compile_module.load_inline, original_module_inline)
        self.assertIs(side_compile_module.load_inline, original_module_inline)
        self.assertIs(cpp_extension.load_inline, original_torch_inline)

    def test_negative_tilelang_out_idx_loads_one_admitted_object(self) -> None:
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            code = cache / "kernel.so"
            code.write_bytes(b"admitted")
            row = {
                "path": "kernel.so",
                "sha256": protocol.file_sha256(code),
                "size": code.stat().st_size,
            }
            kernel = mock.Mock()
            kernel._tilelang_cache_path = str(cache)
            calls = []

            def one_hit(*_args, **kwargs):
                out_idx = kwargs.get("out_idx")
                calls.append(out_idx.copy() if isinstance(out_idx, list) else out_idx)
                return kernel if out_idx == [3] else None

            loaded, evidence = artifacts._load_tilelang_admitted(
                one_hit, (), {"out_idx": [-1]}, {code.resolve(): row}
            )
            self.assertIs(loaded, kernel)
            self.assertEqual(calls, [[-1], [0], [1], [2], [3]])
            self.assertEqual(evidence["resolved_out_idx"], [3])

            def two_hits(*_args, **kwargs):
                return kernel if kwargs.get("out_idx") in ([1], [3]) else None

            with self.assertRaisesRegex(protocol.ProtocolError, "found 2 admitted cache hits"):
                artifacts._load_tilelang_admitted(
                    two_hits, (), {"out_idx": [-1]}, {code.resolve(): row}
                )

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

    def test_compute_pid_query_passes_one_argv_and_parses_output(self) -> None:
        command = [
            "nvidia-smi", "--id=0", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
        for output, expected in (("", set()), ("123\n", {123})):
            with self.subTest(output=output), mock.patch.object(
                runner.subprocess, "run",
                return_value=mock.Mock(returncode=0, stdout=output),
            ) as run:
                self.assertEqual(runner._compute_pids(), expected)
                self.assertEqual(run.call_args.args, (command,))
                self.assertEqual(
                    run.call_args.kwargs,
                    {"capture_output": True, "text": True, "timeout": 20},
                )

    def test_upstream_ref_preserves_branch_slashes(self) -> None:
        self.assertEqual(
            runner.upstream_remote_ref("origin/feature/native-replication"),
            ("origin", "refs/heads/feature/native-replication"),
        )

    def test_ready_refuses_before_any_launch(self) -> None:
        with mock.patch.object(runner.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(protocol.ProtocolError, "prepare/material|provenance/execution"):
                runner.ready()
            popen.assert_not_called()

    def test_timing_child_requires_inherited_gpu_lock(self) -> None:
        with mock.patch.dict(runner.os.environ, {}, clear=True):
            with self.assertRaisesRegex(protocol.ProtocolError, "inherited GPU0 lock"):
                runner._validate_inherited_gpu0_lock()

    def test_execution_lock_binds_material_stage_provenance(self) -> None:
        material_lock = runner.make_material_lock(self.contract, self.manifest)
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
            "material_lock_sha256": "pending",
            "record_type": "native_trajectory_replication_ada_v2_prelaunch_provenance",
            "remote_push_verified": True,
            "schema_version": 1,
            "source_bundle_sha256": material_lock["source_bundle_sha256"],
            "stage": "material_commit_live_upstream_verified",
            "toolchain": self.toolchain,
        }
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            material_path = Path(directory) / "material_lock.json"
            material_path.write_bytes(runner.stable_bytes(material_lock))
            receipt["material_lock_sha256"] = protocol.file_sha256(material_path)
            path = Path(directory) / "prelaunch_provenance.json"
            path.write_bytes(runner.stable_bytes(receipt))
            with mock.patch.object(runner, "MATERIAL_LOCK_PATH", material_path), mock.patch.object(
                runner, "PROVENANCE_PATH", path
            ):
                lock = runner.make_execution_lock(
                    self.contract, self.manifest, material_lock, receipt,
                )
                malformed = {**receipt, "host": "another-host"}
                with self.assertRaisesRegex(protocol.ProtocolError, "provenance"):
                    runner.make_execution_lock(
                        self.contract, self.manifest, material_lock, malformed,
                    )
        self.assertEqual(lock["expected_raw_records"], 420)
        self.assertEqual(lock["artifact_admission_expected_entries"], 12)
        self.assertEqual(lock["manifest_plan_sha256"], self.manifest["plan_sha256"])

    def test_settled_tail_record_is_raw_derived_and_gpu_bound(self) -> None:
        row = self.manifest["rows"][0]
        values = [100.0] * 60 + [1.0] * 40
        identity = "a" * 64
        artifact_binding = {
            "artifact_admission_manifest_path": "admission/manifest.json",
            "artifact_admission_manifest_sha256": "b" * 64,
            "admitted_artifact_sha256": "c" * 64,
            "admitted_artifact_identity_sha256": identity,
            "admitted_entry_path": "admission/entry.json",
            "admitted_entry_sha256": "d" * 64,
        }
        admitted_build = {"artifact_identity_sha256": identity}
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
            "artifact_identity_sha256": identity,
            "artifact_identity_sha256_expected": identity,
            **artifact_binding,
            "label": row["label"],
            "live_correctness": {"gate_pass": True},
            "load_evidence": {},
            "manifest_row_sha256": protocol.canonical_sha256(row),
            "ok": True,
            "physical_gpu": 0,
            "process_pid": 123,
            "primary_tail_median_ms": 1.0,
            "record_kind": row["record_kind"],
            "record_type": "native_trajectory_replication_ada_v2_timing_record",
            "row_id": row["row_id"],
            "schema_version": 1,
            "t_end_unix_ns": 2,
            "t_start_unix_ns": 1,
            "times_ms": values,
            "trials": 100,
            "toolchain": self.toolchain,
            "predecessor_implementation_sha256_expected": row["predecessor_implementation_sha256"],
            "warmup_s": 2.0,
            "warmup_iterations_actual": 1,
            "compile_s": 0.0,
            "build_metadata": {
                "artifact_identity_sha256": identity,
                "n_kernels": 2,
                "predecessor_implementation_sha256_observed": "e" * 64,
            },
        }
        execution_binding = {"toolchain": self.toolchain}
        with mock.patch.object(artifacts, "validate_load_evidence"):
            analyze.validate_timing_record(
                self.contract, row, record,
                execution_binding=execution_binding,
                artifact_binding=artifact_binding, admitted_build=admitted_build,
            )
            changed_toolchain = copy.deepcopy(record)
            changed_toolchain["toolchain"]["triton_version"] = "changed"
            with self.assertRaisesRegex(protocol.ProtocolError, "foreign"):
                analyze.validate_timing_record(
                    self.contract, row, changed_toolchain,
                    execution_binding=execution_binding,
                    artifact_binding=artifact_binding, admitted_build=admitted_build,
                )
            record["primary_tail_median_ms"] = 2.0
            with self.assertRaisesRegex(protocol.ProtocolError, "raw-trial-derived"):
                analyze.validate_timing_record(
                    self.contract, row, record, execution_binding=execution_binding,
                    artifact_binding=artifact_binding, admitted_build=admitted_build,
                )
            record["primary_tail_median_ms"] = 1.0
            record["warmup_iterations_actual"] = 0
            with self.assertRaisesRegex(protocol.ProtocolError, "implementation/gate"):
                analyze.validate_timing_record(
                    self.contract, row, record, execution_binding=execution_binding,
                    artifact_binding=artifact_binding, admitted_build=admitted_build,
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
            "record_type": "native_trajectory_replication_ada_v2_position_receipt",
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
                    "artifact_identity_sha256": "a" * 64,
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
                    "artifact_identity_sha256": "a" * 64,
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
