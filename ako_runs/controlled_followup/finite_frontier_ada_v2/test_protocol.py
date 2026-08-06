from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from . import analyze, artifacts, launch, protocol
except ImportError:
    import analyze  # type: ignore
    import artifacts  # type: ignore
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
        changed = copy.deepcopy(self.contract)
        changed["manifest"]["artifact_admission"]["cache_loader_sha256"][
            "triton/runtime/cache.py"
        ] = "0" * 64
        with self.assertRaisesRegex(protocol.ProtocolError, "cache-loader"):
            protocol.validate_contract(changed)

    def test_v2_preserves_estimand_and_binds_v1_incident(self):
        v1 = json.loads(
            (
                protocol.REPO_ROOT
                / "ako_runs/controlled_followup/finite_frontier_ada_v1/contract.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(self.contract["estimand"], v1["estimand"])
        self.assertEqual(
            self.contract["manifest"]["selection_confirm"],
            v1["manifest"]["selection_confirm"],
        )
        self.assertEqual(self.contract["manifest"]["terminal_confirm"], v1["manifest"]["terminal_confirm"])
        self.assertEqual(self.contract["manifest"]["timing"], v1["manifest"]["timing"])
        self.assertEqual(
            self.contract["manifest"]["inference"]["claim_rule"],
            v1["manifest"]["inference"]["claim_rule"],
        )
        registry = self.contract["material_registry"]
        incident = registry["roles"]["predecessor_selection_incident"]
        self.assertEqual(registry["sha256"][incident], protocol.PREDECESSOR_INCIDENT_SHA256)
        self.assertEqual(len(protocol.predecessor_closure_paths()), 242)
        changed = protocol.read_json(protocol.REPO_ROOT / incident)
        changed["artifact_closure"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(protocol.ProtocolError, "closure changed"):
            protocol.predecessor_closure_paths(changed)

    def test_seven_isolated_admission_roots_and_one_shared_sham(self):
        plan = artifacts.admission_plan(self.contract)
        self.assertEqual(len(plan), 7)
        self.assertEqual(sum(row["role"] == "candidate" for row in plan), 6)
        self.assertEqual(sum(row["role"] == "shared_sham_base" for row in plan), 1)
        roots = []
        for row in plan:
            admit = artifacts.cache_environment(row["cell_id"], "admit")
            load = artifacts.cache_environment(row["cell_id"], "load_only")
            roots.append(admit["TILELANG_CACHE_DIR"])
            self.assertEqual(admit["TILELANG_CLEAR_CACHE"], "0")
            self.assertEqual(admit["TILELANG_EXECUTION_BACKEND"], "tvm_ffi")
            self.assertEqual(admit["TILELANG_TARGET"], "cuda")
            self.assertNotIn("TRITON_CACHE_MANAGER", admit)
            self.assertIn("ReadOnlyTritonCacheManager", load["TRITON_CACHE_MANAGER"])
            self.assertFalse(admit["TILELANG_TMP_DIR"].startswith(admit["TILELANG_CACHE_DIR"]))
        self.assertEqual(len(set(roots)), 7)

    def test_cache_snapshot_binds_sources_code_objects_and_tamper(self):
        cell_id = artifacts.admission_plan(self.contract)[0]["cell_id"]
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory, mock.patch.object(
            artifacts, "ARTIFACTS_ROOT", Path(directory) / "artifacts"
        ):
            cache = artifacts.entry_paths(cell_id)["cache"]
            cache.mkdir(parents=True)
            (cache / "kernel.cu").write_text("cuda", encoding="utf-8")
            (cache / "kernel.so").write_bytes(b"so")
            (cache / "softmax.cubin").write_bytes(b"cubin")
            before = artifacts.cache_snapshot(cell_id)
            self.assertEqual(len(before["generated_sources"]), 1)
            self.assertEqual(len(before["loadable_code_objects"]), 2)
            (cache / "kernel.so").write_bytes(b"changed")
            self.assertNotEqual(before["files_sha256"], artifacts.cache_snapshot(cell_id)["files_sha256"])

    def test_runtime_rejects_an_unadmitted_or_changed_code_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admitted = root / "kernel.so"
            foreign = root / "foreign.so"
            admitted.write_bytes(b"admitted")
            foreign.write_bytes(b"foreign")
            allowed = {
                admitted.resolve(): {
                    "path": "kernel.so",
                    "sha256": protocol.file_sha256(admitted),
                    "size": admitted.stat().st_size,
                }
            }
            with self.assertRaisesRegex(protocol.ProtocolError, "unadmitted artifact"):
                artifacts._verify_bound_file(foreign, allowed)
            admitted.write_bytes(b"changed")
            with self.assertRaisesRegex(protocol.ProtocolError, "unadmitted artifact"):
                artifacts._verify_bound_file(admitted, allowed)

    def test_read_only_triton_manager_rejects_miss_and_write(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"TRITON_CACHE_DIR": directory}
        ):
            missing = Path(directory) / "MISSING"
            with self.assertRaisesRegex(protocol.ProtocolError, "cache miss"):
                artifacts.ReadOnlyTritonCacheManager("MISSING")
            self.assertFalse(missing.exists())
            (Path(directory) / "PRESENT").mkdir()
            manager = artifacts.ReadOnlyTritonCacheManager("PRESENT")
            with self.assertRaisesRegex(protocol.ProtocolError, "writes are forbidden"):
                manager.put(b"x", "x")

    def test_load_evidence_census_is_fail_closed(self):
        tilelang = {
            "cell_id": "register_fused.tilelang.g09",
            "lane": "tilelang",
            "strategy": "register_fused",
        }
        cache = self._fake_cache()
        evidence = self._tilelang_load_evidence(cache)
        artifacts.validate_load_evidence(evidence, tilelang, cache)
        evidence["tilelang_cache_hits"].pop()
        with self.assertRaisesRegex(protocol.ProtocolError, "census"):
            artifacts.validate_load_evidence(evidence, tilelang, cache)
        evidence = self._tilelang_load_evidence(cache)
        evidence["tilelang_cache_hits"][1] = copy.deepcopy(
            evidence["tilelang_cache_hits"][0]
        )
        with self.assertRaisesRegex(protocol.ProtocolError, "two distinct"):
            artifacts.validate_load_evidence(evidence, tilelang, cache)
        evidence = self._tilelang_load_evidence(cache)
        evidence["tilelang_cache_hits"][0]["loadable_code_objects"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(protocol.ProtocolError, "unadmitted object"):
            artifacts.validate_load_evidence(evidence, tilelang, cache)

    def test_tilelang_frontend_cache_miss_is_rejected(self):
        import tilelang.cache as tilelang_cache
        from tilelang.cache.kernel_cache import KernelCache

        cell_id = artifacts.admission_plan(self.contract)[0]["cell_id"]
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory, mock.patch.object(
            artifacts, "ARTIFACTS_ROOT", Path(directory) / "artifacts"
        ):
            cache = artifacts.entry_paths(cell_id)["cache"]
            cache.mkdir(parents=True)
            (cache / "kernel.cu").write_text("cuda", encoding="utf-8")
            (cache / "kernel.so").write_bytes(b"so")
            (cache / "softmax.cubin").write_bytes(b"cubin")
            build = {"cache": artifacts.cache_snapshot(cell_id)}
            with mock.patch.object(tilelang_cache, "load_frontend_cached", return_value=None):
                with artifacts.load_only_guards(build):
                    with self.assertRaisesRegex(protocol.ProtocolError, "cache writes are forbidden"):
                        KernelCache._safe_write_file("unused", "wb", lambda _file: None)
                    with self.assertRaisesRegex(protocol.ProtocolError, "frontend-cache miss"):
                        tilelang_cache.load_frontend_cached({})

    def test_execution_bindings_cover_sources_and_import(self):
        bindings = launch.execution_bindings()
        self.assertEqual(bindings["artifact_admission_expected_entries"], 7)
        self.assertEqual(
            bindings["artifact_admission_plan_sha256"],
            protocol.canonical_sha256(artifacts.admission_plan(self.contract)),
        )
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

    def test_remote_readiness_requires_admission_before_timing(self):
        admission = set(launch.remote_ready_paths(self.contract, "artifact_admission"))
        root = "ako_runs/controlled_followup/finite_frontier_ada_v2/results"
        self.assertIn(f"{root}/imported_frontier.json", admission)
        self.assertIn(
            "ako_runs/controlled_followup/finite_frontier_ada_v1/results/"
            "selection_confirm/run_status.json",
            admission,
        )
        with self.assertRaisesRegex(protocol.ProtocolError, "manifest.json"):
            launch.remote_ready_paths(self.contract, "selection_confirm")

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

    @staticmethod
    def _fake_cache():
        return {
            "cache_root": (
                "ako_runs/controlled_followup/finite_frontier_ada_v2/"
                "test_fake_cache"
            ),
            "files_sha256": "c" * 64,
            "files": [
                {
                    "path": "tilelang/kernel_a/executable.so",
                    "sha256": "1" * 64,
                    "size": 1,
                },
                {
                    "path": "tilelang/kernel_b/executable.so",
                    "sha256": "2" * 64,
                    "size": 1,
                },
            ],
        }

    @staticmethod
    def _tilelang_load_evidence(cache):
        hits = []
        for row in cache["files"]:
            object_path = protocol.REPO_ROOT / cache["cache_root"] / row["path"]
            hits.append(
                {
                    "cache_path": str(object_path.parent.resolve()),
                    "loadable_code_objects": [copy.deepcopy(row)],
                }
            )
        return {
            "mode": "load_only",
            "postprocess_loads": [],
            "tilelang_cache_hits": hits,
            "triton_cache_hits": [],
            "triton_kernel_loads": [],
        }

    def _valid_record(self, position=0, start=10.0, end=20.0, pid=100):
        implementation = "a" * 64
        times = [1.0] * 100
        return {
            "schema_version": 1,
            "record_type": "finite_frontier_ada_timing_record",
            "ok": True,
            "legacy_error": {"gate_pass": True},
            "cell_id": "register_fused.tilelang.g09",
            "admitted_entry_path": "fake_entry.json",
            "admitted_entry_sha256": "d" * 64,
            "times_ms": times,
            **protocol.summarize_times(times),
            "implementation_sha256": implementation,
            "artifact_cache_files_sha256_after": "c" * 64,
            "artifact_load": self._tilelang_load_evidence(self._fake_cache()),
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
        from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core as source_core

        admitted = {
            "implementation_sha256": "a" * 64,
            "cache": self._fake_cache(),
        }
        original_read = protocol.read_json

        def read(path):
            if Path(path) == protocol.REPO_ROOT / "fake_entry.json":
                return admitted
            return original_read(Path(path))

        expected = {
            "plan_position": 0,
            "admitted_entry_path": "fake_entry.json",
            "admitted_entry_sha256": "d" * 64,
        }
        source_contract = ({}, [{
            "cell_id": "register_fused.tilelang.g09",
            "lane": "tilelang",
            "strategy": "register_fused",
        }], {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            record = self._valid_record()
            path.write_text(json.dumps(record), encoding="utf-8")
            with mock.patch.object(protocol, "read_json", side_effect=read), mock.patch.object(
                protocol, "file_sha256", return_value="d" * 64
            ), mock.patch.object(
                source_core, "load_contract", return_value=source_contract
            ):
                launch.validate_timing_record(path, expected, self.contract)
            record["build_metadata"]["implementation_sha256"] = "b" * 64
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "fingerprint"):
                launch.validate_timing_record(path, expected, self.contract)

            record = self._valid_record()
            record["build_metadata"]["n_kernels"] = 1
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "fingerprint"):
                launch.validate_timing_record(path, expected, self.contract)

            record = self._valid_record()
            record.update(
                {
                    "launch_created_at_utc": "1970-01-01T00:00:15+00:00",
                    "launch_stage_preflight_unix": 15.0,
                }
            )
            path.write_text(json.dumps(record), encoding="utf-8")
            with mock.patch.object(protocol, "read_json", side_effect=read), mock.patch.object(
                protocol, "file_sha256", return_value="d" * 64
            ), mock.patch.object(
                source_core, "load_contract", return_value=source_contract
            ), self.assertRaisesRegex(protocol.ProtocolError, "process receipt"):
                launch.validate_timing_record(
                    path,
                    {
                        **expected,
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
