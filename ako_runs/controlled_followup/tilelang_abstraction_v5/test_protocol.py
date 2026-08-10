from __future__ import annotations

import ast
import importlib
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from . import artifacts, protocol
from .protocol import (
    CAMPAIGN_ID,
    CAMPAIGN_PATH,
    MATERIALS_PATH,
    ProtocolError,
    canonical_sha256,
    make_timing_manifest,
    read_json,
    validate_campaign,
    validate_materials,
    validate_timing_manifest,
)


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.campaign = read_json(CAMPAIGN_PATH)
        cls.materials = read_json(MATERIALS_PATH)

    @staticmethod
    def _admission(lock_sha: str) -> dict:
        return {
            "campaign_id": CAMPAIGN_ID,
            "campaign_lock_sha256": lock_sha,
            "complete": True,
            "pairs": [{
                "pair_id": "fused_softmax_f1_f4c",
                "gate_available": True,
                "timing_eligible": True,
                "classification": "runtime_estimand",
                "artifact_identity_sha256": {
                    "high": "a" * 64,
                    "low": "b" * 64,
                },
            }],
        }

    def test_exact_fused_only_design_and_predecessor_closure(self) -> None:
        validate_campaign(self.campaign)
        resolved = validate_materials(self.materials, self.campaign)
        self.assertEqual(
            [pair["pair_id"] for pair in self.campaign["pairs"]],
            ["fused_softmax_f1_f4c"],
        )
        status = read_json(resolved["v4_admission_run_status"])
        self.assertEqual(
            status["artifact_bundle_sha256"],
            canonical_sha256(status["artifact_sha256"]),
        )

    def test_pair_drift_and_predecessor_census_tampering_fail_closed(self) -> None:
        changed = deepcopy(self.campaign)
        changed["pairs"][0]["low"]["variant"] = "F3c"
        with self.assertRaisesRegex(ProtocolError, "differs from the v4 predecessor"):
            validate_materials(self.materials, changed)

        incident_path = Path(self.materials["entries"]["v4_artifact_identity_incident"]["path"])
        original_read = read_json

        def forged_read(path):
            value = original_read(path)
            if Path(path).resolve() == incident_path.resolve():
                value = deepcopy(value)
                value["artifact_closure"]["file_count"] -= 1
            return value

        with (
            patch(
                "ako_runs.controlled_followup.tilelang_abstraction_v5.protocol.read_json",
                side_effect=forged_read,
            ),
            self.assertRaisesRegex(ProtocolError, "incident or result closure is inconsistent"),
        ):
            validate_materials(self.materials, self.campaign)

    def test_manifest_is_exactly_120_paired_randomized_rows(self) -> None:
        lock_sha = "c" * 64
        admission = self._admission(lock_sha)
        first = make_timing_manifest(self.campaign, lock_sha, admission)
        second = make_timing_manifest(self.campaign, lock_sha, admission)
        self.assertEqual(first, second)
        self.assertEqual(len(first["rows"]), 120)
        self.assertEqual(first["eligible_pair_ids"], ["fused_softmax_f1_f4c"])
        for distribution in ("positive", "withheld_signed"):
            for block in range(15):
                rows = [
                    row for row in first["rows"]
                    if row["distribution"] == distribution and row["block"] == block
                ]
                self.assertEqual(
                    {row["role"] for row in rows},
                    {"high", "low", "sham_a", "sham_b"},
                )
                self.assertEqual(
                    [row["implementation_side"] for row in rows].count("high"),
                    3,
                )
                self.assertEqual(sorted(row["position"] for row in rows), [0, 1, 2, 3])

        tampered = deepcopy(first)
        tampered["rows"][0]["position"] = 99
        with self.assertRaisesRegex(ProtocolError, "deterministic"):
            validate_timing_manifest(tampered, self.campaign, lock_sha, admission)

    def test_ineligible_pair_projects_zero_timing_rows(self) -> None:
        lock_sha = "d" * 64
        admission = self._admission(lock_sha)
        admission["pairs"][0].update({
            "timing_eligible": False,
            "classification": "excluded_fail_closed",
            "artifact_identity_sha256": {},
        })
        manifest = make_timing_manifest(self.campaign, lock_sha, admission)
        self.assertEqual(manifest["rows"], [])
        self.assertEqual(manifest["eligible_pair_ids"], [])

    def test_compiler_root_is_initialized_once_before_tilelang_import_sites(self) -> None:
        with patch.dict(os.environ, {"PATH": "/usr/bin:/usr/local/cuda-13.1/bin"}, clear=False):
            protocol.initialize_compiler_environment()
            self.assertEqual(os.environ["CUDA_HOME"], "/usr/local/cuda-13.1")
            self.assertEqual(os.environ["CUDA_PATH"], "/usr/local/cuda-13.1")
            self.assertEqual(os.environ["PATH"].split(os.pathsep)[0], "/usr/local/cuda-13.1/bin")
            self.assertEqual(os.environ["PATH"].split(os.pathsep).count("/usr/local/cuda-13.1/bin"), 1)
        source = Path(protocol.__file__).read_text(encoding="utf-8")
        self.assertLess(source.index("initialize_compiler_environment()"), source.index("def live_toolchain"))
        runner = Path(protocol.HERE / "campaign_runner.py").read_text(encoding="utf-8")
        for forbidden in ("profile_target2.py", "_legacy_timing_command", "export_ptx", "export_sass"):
            self.assertNotIn(forbidden, runner)
        for function in ("arm_admit", "arm_profile", "profile_target", "arm_time"):
            body = runner.split(f"def {function}(", 1)[1].split("\ndef ", 1)[0]
            self.assertLess(body.index("prepare_environment"), body.index("load_lock("))

    def test_toolchain_binds_tilelang_execution_modules_and_complete_ncu_chain(self) -> None:
        receipt = protocol.live_toolchain()
        protocol.validate_toolchain(receipt)
        expected_modules = {
            "tilelang_env": "tilelang/env.py",
            "tilelang_execution_backend": "tilelang/jit/execution_backend.py",
            "tilelang_adapter_base": "tilelang/jit/adapter/base.py",
            "tilelang_adapter_kernel_cache": "tilelang/jit/adapter/kernel_cache.py",
        }
        for name, suffix in expected_modules.items():
            binding = receipt["package_modules"][name]
            path = Path(binding["path"])
            self.assertTrue(path.as_posix().endswith(suffix))
            self.assertEqual(binding["sha256"], protocol.file_sha256(path))

        ncu = receipt["cuda_tools"]["ncu"]
        self.assertEqual(
            [row["path"] for row in ncu["executable_chain"]],
            [str(protocol.NCU_DISPATCHER), str(protocol.NCU_WRAPPER), str(protocol.NCU_FINAL_ELF)],
        )
        self.assertEqual(
            [row["format"] for row in ncu["executable_chain"]],
            ["posix_shell", "posix_shell", "elf"],
        )
        for row in ncu["executable_chain"]:
            self.assertEqual(row["sha256"], protocol.file_sha256(row["path"]))
            self.assertEqual(row["bytes"], Path(row["path"]).stat().st_size)

        changed = deepcopy(receipt)
        changed["cuda_tools"]["ncu"]["executable_chain"][-1]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ProtocolError, "NCU executable-chain"):
            protocol.validate_toolchain(changed)

        from .campaign_runner import _profile_command

        command = _profile_command(
            self.campaign["pairs"][0], "high", 0,
            protocol.HERE / "campaign_lock.json",
            protocol.HERE / "admission.json",
            protocol.HERE / "load.json",
        )
        self.assertEqual(command[0], str(protocol.NCU_FINAL_ELF))

    def test_recovery_gate_summary_imports_exact_bound_module_closure(self) -> None:
        resolved = validate_materials(self.materials, self.campaign)
        prefix = "ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1"
        expected = {
            prefix: "fused_gate_recovery_init",
            f"{prefix}.common": "fused_gate_recovery_common",
            f"{prefix}.validate": "fused_gate_recovery_validate",
            f"{prefix}.audit": "fused_gate_summary",
        }
        audit = artifacts.exact_recovery_audit()
        self.assertIs(audit, sys.modules[f"{prefix}.audit"])
        for name, material_id in expected.items():
            self.assertEqual(
                Path(sys.modules[name].__file__).resolve(),
                resolved[material_id].resolve(),
            )
        for source in (protocol.HERE / "campaign_runner.py", protocol.HERE / "analyze.py"):
            self.assertIn("artifacts.exact_recovery_audit()", source.read_text(encoding="utf-8"))

    def test_gpu_runner_exact_imports_are_frozen(self) -> None:
        tree = ast.parse((protocol.HERE / "campaign_runner.py").read_text(encoding="utf-8"))
        imported = {
            call.args[1].value
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "exact_import"
            and len(call.args) >= 2
            and isinstance(call.args[1], ast.Constant)
            and isinstance(call.args[1].value, str)
        }
        material_paths = {
            self.materials["entries"][name]["path"]
            for name in self.materials["family_bindings"]["fused_softmax"]
        }
        self.assertLessEqual(imported, set(protocol.TOOLCHAIN_LOCAL_MODULES.values()))
        self.assertLessEqual(imported, material_paths)

    def test_generic_module_shadowing_fails_closed(self) -> None:
        fake = SimpleNamespace(__file__="/tmp/foreign/common2.py")
        with patch.dict(sys.modules, {"common2": fake}):
            with self.assertRaisesRegex(ProtocolError, "generic module shadowed"):
                artifacts.exact_import("common2", "ako_runs/phase2_fused_sdpa/common2.py")
        fake_gate = SimpleNamespace(__file__="/tmp/foreign/robust_gate/__init__.py")
        with patch.dict(sys.modules, {"robust_gate": fake_gate}):
            with self.assertRaisesRegex(ProtocolError, "generic module shadowed"):
                artifacts.exact_robust_adapter()

    def test_artifact_environment_rejects_late_tilelang_or_raw_capture(self) -> None:
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory) / "arm_artifact"
            with patch.dict(sys.modules, {"tilelang": SimpleNamespace()}):
                with self.assertRaisesRegex(ProtocolError, "before importing TileLang"):
                    artifacts.prepare_environment(root, "admit", 0)
            with patch.dict(os.environ, {"TILELANG_ABSTRACTION_CAPTURE_DIR": "/tmp/forbidden"}):
                with patch.dict(sys.modules, {"tilelang": None}):
                    sys.modules.pop("tilelang", None)
                    with self.assertRaisesRegex(ProtocolError, "raw PTX/SASS capture"):
                        artifacts.prepare_environment(root, "admit", 0)

    def test_cache_snapshot_binds_two_executables_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory) / "arm_artifact"
            cache = root / "cache"
            cache.mkdir(parents=True)
            (cache / "device_kernel.cu").write_text("cuda", encoding="utf-8")
            (cache / "gemm.so").write_bytes(b"gemm")
            (cache / "softmax.so").write_bytes(b"softmax")
            before = artifacts.cache_snapshot(root)
            self.assertEqual(len(before["loadable_code_objects"]), 2)
            (cache / "softmax.so").write_bytes(b"changed")
            self.assertNotEqual(before["files_sha256"], artifacts.cache_snapshot(root)["files_sha256"])

    def test_load_only_guard_rejects_miss_compile_and_write(self) -> None:
        import tilelang.cache as tilelang_cache
        tilelang_jit = importlib.import_module("tilelang.jit")
        from tilelang.cache.kernel_cache import KernelCache

        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory) / "arm_artifact"
            cache_root = root / "cache"
            cache_root.mkdir(parents=True)
            (cache_root / "device_kernel.cu").write_text("cuda", encoding="utf-8")
            (cache_root / "gemm.so").write_bytes(b"gemm")
            (cache_root / "softmax.so").write_bytes(b"softmax")
            cache = artifacts.cache_snapshot(root)
            with patch.object(tilelang_cache, "load_frontend_cached", return_value=None):
                with artifacts.load_only_guards(cache):
                    with self.assertRaisesRegex(ProtocolError, "frontend-cache miss"):
                        tilelang_cache.load_frontend_cached({})
                    with self.assertRaisesRegex(ProtocolError, "compilation is forbidden"):
                        tilelang_jit.JITImpl.compile(None)
                    with self.assertRaisesRegex(ProtocolError, "writes are forbidden"):
                        KernelCache._safe_write_file("unused", "wb", lambda _file: None)

    def test_load_only_success_binds_two_distinct_executables(self) -> None:
        import tilelang.cache as tilelang_cache

        with tempfile.TemporaryDirectory(dir=protocol.HERE) as directory:
            root = Path(directory) / "arm_artifact"
            cache_root = root / "cache"
            gemm = cache_root / "tilelang/kernels/gemm"
            softmax = cache_root / "tilelang/kernels/softmax"
            gemm.mkdir(parents=True)
            softmax.mkdir(parents=True)
            (gemm / "device_kernel.cu").write_text("gemm source", encoding="utf-8")
            (gemm / "executable.so").write_bytes(b"gemm")
            (softmax / "device_kernel.cu").write_text("softmax source", encoding="utf-8")
            (softmax / "executable.so").write_bytes(b"softmax")
            cache = artifacts.cache_snapshot(root)
            kernels = [
                Mock(_tilelang_cache_path=str(gemm.resolve())),
                Mock(_tilelang_cache_path=str(softmax.resolve())),
            ]
            with patch.object(tilelang_cache, "load_frontend_cached", side_effect=kernels):
                with artifacts.load_only_guards(cache) as evidence:
                    tilelang_cache.load_frontend_cached({}, out_idx=[0])
                    tilelang_cache.load_frontend_cached({}, out_idx=[0])
            artifacts.validate_load_evidence(evidence, cache)
            forged = deepcopy(evidence)
            forged["tilelang_cache_hits"][1] = deepcopy(forged["tilelang_cache_hits"][0])
            with self.assertRaisesRegex(ProtocolError, "two distinct"):
                artifacts.validate_load_evidence(forged, cache)


if __name__ == "__main__":
    unittest.main()
