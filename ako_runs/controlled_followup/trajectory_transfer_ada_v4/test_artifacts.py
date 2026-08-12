#!/usr/bin/env python3
"""CPU-only checks for transfer adapters and artifact binding."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from ako_runs.controlled_followup.trajectory_transfer_ada_v4 import artifacts, protocol
from ako_runs.controlled_followup.trajectory_transfer_ada_v4 import implementations

HEX_A = "a" * 64
HEX_B = "b" * 64


def _pair_rows() -> tuple[dict, dict, dict]:
    rows = [
        row for row in protocol.make_admission_manifest()["rows"]
        if row["destination"] == "tilelang" and row["adaptation"] == "donor_fixed"
    ]
    off = next(row for row in rows if row["mechanism_state"] == "off")
    on = next(row for row in rows if row["mechanism_state"] == "on")
    return off, on, {"rows": rows}


def _write_kernel(root: Path, key: str, *, device: bytes = b"gemm-device", executable: bytes = b"gemm-object") -> None:
    kernel = root / key
    kernel.mkdir(parents=True)
    values = {
        "device_kernel.cu": device,
        "executable.so": executable,
        "host_kernel.cu": b"gemm-host",
        "params.pkl": b"params",
        "prim_func.pkl": b"prim-func",
    }
    for name, value in values.items():
        (kernel / name).write_bytes(value)


def _off_pair_fixture(root: Path) -> tuple[dict, dict, dict, dict]:
    off, on, manifest = _pair_rows()
    paths = artifacts.entry_paths(off["entry_id"], root)
    namespace = paths["cache"] / "tilelang" / "0.1-test"
    (namespace / ".staging").mkdir(parents=True)
    (namespace / "frontend").mkdir()
    (namespace / "kernels").mkdir()
    key = "c" * 64
    device = b"g04-off-device"
    _write_kernel(namespace / "kernels", key, device=device, executable=b"g04-off-object")
    (namespace / "frontend" / f"{artifacts.TILELANG_OFF_FRONTEND_KEY}.json").write_text(
        json.dumps({"kernel_key": key}, sort_keys=True), encoding="utf-8",
    )
    paths["tmp"].mkdir()
    (paths["tmp"] / "tilelang").mkdir()
    paths["build"].write_text("{}", encoding="utf-8")
    paths["verify"].write_text("{}", encoding="utf-8")
    build = {
        "source_build_metadata": {
            "held_gemm_source_binding": {
                "source_sha256": hashlib.sha256(device).hexdigest(),
            },
        },
    }
    return off, on, manifest, build


def _add_f1(root: Path, on: dict, receipt: dict) -> SimpleNamespace:
    paths = artifacts.entry_paths(on["entry_id"], root)
    namespace = paths["cache"] / "tilelang" / receipt["held_gemm_cache"]["cache_namespace"]
    f1_key = "d" * 64
    _write_kernel(namespace / "kernels", f1_key, device=b"f1-device", executable=b"f1-object")
    (namespace / "frontend" / f"{'e' * 64}.json").write_text(
        json.dumps({"kernel_key": f1_key}, sort_keys=True), encoding="utf-8",
    )
    return SimpleNamespace(metadata={
        "held_gemm_source_binding": {
            "source_sha256": receipt["held_gemm_cache"]["device_source_sha256"],
        },
        "artifacts": {
            "pair_seed": receipt,
            "seeded_gemm_cache_key": receipt["held_gemm_cache"]["kernel_key"],
            "seeded_gemm_exact_load": receipt["exact_load_contract"],
            "seeded_gemm_fallback_compile_forbidden": True,
            "tilelang_compile_mode": "admit",
            "tilelang_fresh_compile_qualnames": ["_f1.<locals>._k"],
        },
    })


def _cell(strategy: str, destination: str = "triton") -> dict:
    return {
        "cell_id": f"{strategy}.{destination}.g01",
        "grid_id": "g01",
        "lane": destination,
        "origin_job": {"dsl": destination, "grid_id": "g01", "variant": "GBGS"},
        "strategy": strategy,
    }


def _spec(state: str = "off", destination: str = "triton") -> dict:
    return next(
        row["spec"] for row in protocol.make_admission_manifest()["rows"]
        if row["destination"] == destination
        and row["adaptation"] == "donor_fixed"
        and row["mechanism_state"] == state
    )


def _config(destination: str, variant: str) -> dict:
    return {
        "dsl": destination,
        "variant": variant,
        "M": 1024,
        "N": 8192,
        "K": 8192,
        "BM": 128,
        "BN": 128,
        "BK": 32,
        "threads": 256,
        "kc": 2048,
        "stages": 3,
        "arith": "fp16",
        "cast": "precast",
        "extra": {"epilogue": "regs", "wcache": "cached"},
        "input_dtype": "torch.float16",
    }


def test_pair_dispatch_preserves_schedule_and_two_kernel_contract() -> None:
    off_spec, on_spec = _spec("off"), _spec("on")
    assert off_spec["origin_job"] == on_spec["origin_job"]
    cells = tuple(
        {**_cell(strategy), "origin_job": off_spec["origin_job"]}
        for strategy in ("register_common_postprocess", "register_fused")
    )

    def fake_build(cell):
        variant = "GBGS" if cell["strategy"] == "register_fused" else "GBG"
        return SimpleNamespace(
            compile_s=0.0,
            config=_config("triton", variant),
            metadata={"artifacts": {"cuda_source_sha256": "b" * 64}, "n_kernels": 2},
            run=lambda *_args: None,
        )

    with mock.patch.object(implementations, "_resolved_cells", return_value=cells), mock.patch(
        "ako_runs.controlled_followup.fused_epilogue_crossed_v2.candidates.build",
        side_effect=fake_build,
    ):
        off = implementations.build(off_spec, False)
        on = implementations.build(on_spec, True)
    assert off.metadata["coordinate_cell_id"].startswith("register_common_postprocess.")
    assert on.metadata["coordinate_cell_id"].startswith("register_fused.")
    assert off.metadata["implementation_id"] == off.metadata["coordinate_cell_id"]
    assert on.metadata["implementation_id"] == on.metadata["coordinate_cell_id"]
    assert off.metadata["paired_config_sha256"] == on.metadata["paired_config_sha256"]
    assert off.metadata["held_gemm_source_sha256"] == on.metadata["held_gemm_source_sha256"]
    assert off.metadata["n_kernels"] == on.metadata["n_kernels"] == 2
    assert off.metadata["primitive_graph_sha256"] != on.metadata["primitive_graph_sha256"]
    assert off.metadata["route"] == on.metadata["route"] == "direct_primitive_mapping"


def test_pair_rejects_cross_wired_mechanism_graph() -> None:
    with pytest.raises(implementations.AdapterError, match="cross-wired"):
        implementations.build(_spec("off"), True)


def test_manual_reconstruction_requires_exact_absence_receipt() -> None:
    spec = _spec(destination="cuda_noptx")
    malformed = {**spec, "primitive_absence_receipt": None}
    with pytest.raises(implementations.UnsupportedRoute, match="primitive-absence"):
        implementations.build(malformed, False, "manual_reconstruction")


def test_manual_reconstruction_reuses_existing_cuda_kernel() -> None:
    off_spec, on_spec = (
        _spec("off", "cuda_noptx"), _spec("on", "cuda_noptx")
    )
    cells = tuple(
        {
            **_cell(strategy, "cuda_noptx"),
            "origin_job": off_spec["origin_job"],
        }
        for strategy in ("register_common_postprocess", "register_fused")
    )

    def fake_build(cell):
        enabled = cell["strategy"] == "register_fused"
        source = (
            f"#define HAS_SOFTMAX {int(enabled)}\n"
            "__global__ void fused_kernel() {}\n"
            + ("/* ---- Phase-2 row softmax */\n__global__ void softmax_kernel() {}\n" if enabled else "")
            + '#include "checked_cuda_launch.h"\n'
        )
        artifacts = {"cuda_source": source}
        if not enabled:
            artifacts = {"lane_gbg": artifacts, "postprocess": {}}
        return SimpleNamespace(
            compile_s=0.0,
            config=_config("cuda_noptx", "GBGS" if enabled else "GBG"),
            metadata={"artifacts": artifacts, "n_kernels": 2},
            run=lambda *_args: None,
        )

    with mock.patch.object(implementations, "_resolved_cells", return_value=cells), mock.patch(
        "ako_runs.controlled_followup.fused_epilogue_crossed_v2.candidates.build",
        side_effect=fake_build,
    ):
        off = implementations.build(off_spec, False, "manual_reconstruction")
        on = implementations.build(on_spec, True, "manual_reconstruction")
    assert off.metadata["route"] == on.metadata["route"] == "manual_reconstruction"
    assert off.metadata["held_gemm_source_sha256"] == on.metadata["held_gemm_source_sha256"]


def test_tilelang_on_dispatches_to_existing_full_f1() -> None:
    off_spec, on_spec = _spec("off", "tilelang"), _spec("on", "tilelang")
    cells = tuple(
        {**_cell(strategy, "tilelang"), "origin_job": off_spec["origin_job"]}
        for strategy in ("register_common_postprocess", "register_fused")
    )
    gemm_source = "// held TileLang GEMM"

    def fake_built(variant: str):
        artifacts = {"cuda_source": gemm_source}
        if variant == "GBG":
            artifacts = {"lane_gbg": artifacts, "postprocess": {}}
        else:
            artifacts["cuda_source_gemm"] = gemm_source
        return SimpleNamespace(
            compile_s=0.0,
            config=_config("tilelang" if variant == "GBG" else "tilelang_abs", variant),
            metadata={"artifacts": artifacts, "n_kernels": 2},
            run=lambda *_args: None,
        )

    with mock.patch.object(implementations, "_resolved_cells", return_value=cells), mock.patch(
        "ako_runs.controlled_followup.fused_epilogue_crossed_v2.candidates.build",
        return_value=fake_built("GBG"),
    ), mock.patch.object(implementations, "_build_tilelang_f1", return_value=fake_built("F1")) as f1:
        off = implementations.build(off_spec, False)
        on = implementations.build(on_spec, True)
    f1.assert_called_once()
    assert on.metadata["coordinate_cell_id"] == "register_fused.tilelang.g01"
    assert on.metadata["implementation_id"] == "tilelang_abstraction.F1.g01"
    assert on.metadata["implementation_id"] != on.metadata["coordinate_cell_id"]
    assert "source_cell_id" not in on.metadata
    assert off.metadata["paired_config_sha256"] == on.metadata["paired_config_sha256"]
    assert off.metadata["held_gemm_source_sha256"] == on.metadata["held_gemm_source_sha256"]


def test_generated_cuda_launches_must_be_checked() -> None:
    with tempfile.TemporaryDirectory(dir=protocol.HERE) as temporary:
        root = Path(temporary)
        entry_id = "tt4_" + "d" * 24
        paths = artifacts.entry_paths(entry_id, root)
        paths["cache"].mkdir(parents=True)
        (paths["cache"] / "kernel.so").write_bytes(b"object")
        wrapper = paths["cache"] / "wrapper.cu"
        wrapper.write_text("kernel<<<1, 1>>>();\n", encoding="utf-8")
        with pytest.raises(protocol.ProtocolError, match="lacks checked launches"):
            artifacts.cache_snapshot(entry_id, root)
        wrapper.write_text(
            '#include "checked_cuda_launch.h"\nkernel<<<1, 1>>>();\n', encoding="utf-8"
        )
        snapshot = artifacts.cache_snapshot(entry_id, root)
        assert snapshot["file_count"] == 2
        identity = artifacts.generated_artifact_identity(snapshot)
        wrapper.write_text(
            '#include "checked_cuda_launch.h"\nkernel<<<2, 1>>>();\n', encoding="utf-8"
        )
        assert artifacts.generated_artifact_identity(artifacts.cache_snapshot(entry_id, root)) != identity


def test_admission_plan_is_exact_and_unique() -> None:
    manifest = protocol.make_admission_manifest()
    rows = artifacts.admission_plan(manifest)
    assert len(rows) == len({row["entry_id"] for row in rows}) == 160
    assert {row["route"] for row in rows} == {
        "direct_primitive_mapping", "manual_reconstruction",
    }


def test_dynamic_work_audit_validator_is_exact() -> None:
    names = ["gemm", "softmax"]
    valid = {
        "schema_version": 1,
        "method": "torch_profiler_cuda_activity_v1",
        "profiled_calls": 1,
        "input": {"seed": 0, "distribution": "positive"},
        "expected_cuda_device_event_count": 2,
        "observed_cuda_device_event_count": 2,
        "cuda_device_event_names": names,
        "cuda_device_event_names_sha256": protocol.canonical_sha256(names),
        "output": {"shape": [1024, 8192], "dtype": "torch.float32", "all_finite": True},
        "passed": True,
        "performance_observations": [],
    }
    assert artifacts.validate_dynamic_work_audit(valid) is valid
    with pytest.raises(protocol.ProtocolError, match="dynamic-work"):
        artifacts.validate_dynamic_work_audit({**valid, "observed_cuda_device_event_count": 3})


def test_gate_records_bind_candidate_config_and_origin_job() -> None:
    row = next(
        row for row in protocol.make_admission_manifest()["rows"]
        if row["destination"] == "triton" and row["adaptation"] == "donor_fixed"
    )
    context = SimpleNamespace(
        adapter={
            "robust_gate": {
                "shape": {"M": 1024, "N": 8192, "K": 8192},
                "gate_spec_sha256": HEX_A,
            },
            "grid": {"job_sha256": {row["origin_job"]["job_id"]: HEX_B}},
        },
        adapter_sha256=HEX_A,
        manifest_sha256=HEX_A,
        robust_manifest={"campaign_id": "robust"},
        source_bundle_sha256=HEX_B,
    )
    build = {"config": {"held": True}}
    record = {
        "adapter_manifest_sha256": HEX_A,
        "build_metadata": {"n_kernels": 2},
        "campaign_id": "robust",
        "candidate": f"trajectory-transfer:{row['entry_id']}",
        "device": "cuda:0",
        "gate_spec_sha256": HEX_A,
        "grid_job": row["origin_job"],
        "grid_job_id": row["origin_job"]["job_id"],
        "grid_job_sha256": HEX_B,
        "manifest_sha256": HEX_A,
        "phase2_config": build["config"],
        "shape": {"M": 1024, "N": 8192, "K": 8192},
        "source_bundle_sha256": HEX_B,
        "source_sha256": HEX_B,
        "split": "validation",
    }
    artifacts._validate_gate_record_bindings(context, [record], build, row)
    for field, value in (
        ("candidate", "foreign"),
        ("phase2_config", {"held": False}),
        ("grid_job", {"foreign": True}),
    ):
        changed = {**record, field: value}
        with pytest.raises(protocol.ProtocolError, match="transplanted"):
            artifacts._validate_gate_record_bindings(context, [changed], build, row)


def test_complete_gate_failure_returns_records_for_parent_retention() -> None:
    # gate_built's source contract returns the full evaluated evidence regardless
    # of pass/fail; admission_one writes it before raising GATE_FAILED.
    source = Path(artifacts.__file__).read_text(encoding="utf-8")
    body = source.split("def gate_built", 1)[1].split("def quick_gate_built", 1)[0]
    assert 'return {"records": records, "summary": summary}' in body
    assert "full_gate_pass" not in body


def test_load_only_keeps_generic_temp_outside_admitted_tree() -> None:
    root = protocol.HERE / "results" / "unit" / "admission"
    entry_id = "tt4_" + "e" * 24
    admit = artifacts.cache_environment(entry_id, "admit", root)
    load = artifacts.cache_environment(entry_id, "load_only", root)
    assert {"TMPDIR", "TMP", "TEMP"} <= set(admit)
    assert not ({"TMPDIR", "TMP", "TEMP"} & set(load))
    assert load["TILELANG_TMP_DIR"].endswith("runtime_tmp/tilelang")


def test_admission_metadata_keeps_native_diagnostics_verbatim() -> None:
    log = "ptxas info\n" * 1000
    cleaned = artifacts._clean_metadata({"artifacts": {"ptxas_log": log}})
    assert cleaned["artifacts"]["ptxas_log"] == log


def test_tilelang_pair_seed_copies_exact_cache_and_allows_only_fresh_f1() -> None:
    with tempfile.TemporaryDirectory(dir=protocol.HERE) as temporary:
        root = Path(temporary)
        off, on, manifest, off_build = _off_pair_fixture(root)
        with mock.patch.object(artifacts, "validate_build_record", return_value=off_build), \
                mock.patch.object(artifacts, "validate_verify_record", return_value={}):
            receipt = artifacts.seed_tilelang_pair(on, manifest, root)
            assert artifacts.validate_tilelang_pair_seed(on, manifest, root) == receipt
            source = root / receipt["source_paths"]["kernel_directory"]
            destination = root / receipt["destination_paths"]["kernel_directory"]
            for row in receipt["held_gemm_cache"]["files"]:
                assert (source / row["path"]).stat().st_ino != (destination / row["path"]).stat().st_ino
            frontend = destination.parents[1] / "frontend"
            assert not list(frontend.iterdir())
            built = _add_f1(root, on, receipt)
            assert artifacts.validate_tilelang_pair_seed(on, manifest, root, built) == receipt
            post = artifacts._post_build_seed_binding(on, built, receipt, root)
            assert len(post["post_build_delta_files"]) == 6
            assert post["fresh_f1_frontend_key"] == "e" * 64
            built.metadata["artifacts"]["tilelang_fresh_compile_qualnames"] = []
            with pytest.raises(protocol.ProtocolError, match="no-fallback pair-seed"):
                artifacts.validate_tilelang_pair_seed(on, manifest, root, built)


def test_tilelang_pair_seed_rejects_incomplete_kernel_even_with_extra_file() -> None:
    with tempfile.TemporaryDirectory(dir=protocol.HERE) as temporary:
        root = Path(temporary)
        off, _on, _manifest, off_build = _off_pair_fixture(root)
        namespace, _frontend, kernels = artifacts._tilelang_layout(off["entry_id"], root)
        kernel = next(kernels.iterdir())
        (kernel / "params.pkl").unlink()
        (kernel / "foreign.bin").write_bytes(b"extra")
        with pytest.raises(protocol.ProtocolError, match="exact five-file"):
            artifacts.tilelang_held_gemm_cache_binding(off, off_build, root)
        assert namespace.name == "0.1-test"


def test_g04_same_key_device_and_object_drift_is_rejected() -> None:
    incident = protocol.read_json(
        protocol.REPO_ROOT / "ako_runs/controlled_followup/trajectory_transfer_ada_v2/"
        "INCIDENT_HELD_GEMM_DRIFT_20260812.json"
    )["failure_evidence"]["g04"]
    assert incident["construction_path"]["tilelang_gemm_cache_key_equal"] is True
    assert incident["off"]["generated_gemm_cuda_sha256"] != incident["on"]["generated_gemm_cuda_sha256"]
    assert incident["off"]["generated_gemm_executable_sha256"] != incident["on"]["generated_gemm_executable_sha256"]
    with tempfile.TemporaryDirectory(dir=protocol.HERE) as temporary:
        root = Path(temporary)
        _off, on, manifest, off_build = _off_pair_fixture(root)
        with mock.patch.object(artifacts, "validate_build_record", return_value=off_build), \
                mock.patch.object(artifacts, "validate_verify_record", return_value={}):
            receipt = artifacts.seed_tilelang_pair(on, manifest, root)
            kernel = root / receipt["destination_paths"]["kernel_directory"]
            (kernel / "device_kernel.cu").write_bytes(b"g04-on-device")
            (kernel / "executable.so").write_bytes(b"g04-on-object")
            with pytest.raises(protocol.ProtocolError, match="mutated|receipt/cache"):
                artifacts.validate_tilelang_pair_seed(on, manifest, root)


def test_tilelang_pair_seed_rejects_any_held_gemm_frontend_alias() -> None:
    with tempfile.TemporaryDirectory(dir=protocol.HERE) as temporary:
        root = Path(temporary)
        _off, on, manifest, off_build = _off_pair_fixture(root)
        with mock.patch.object(artifacts, "validate_build_record", return_value=off_build), \
                mock.patch.object(artifacts, "validate_verify_record", return_value={}):
            receipt = artifacts.seed_tilelang_pair(on, manifest, root)
            kernel = receipt["held_gemm_cache"]["kernel_key"]
            destination = root / receipt["destination_paths"]["kernel_directory"]
            alias = destination.parents[1] / "frontend" / f"{'f' * 64}.json"
            alias.write_text(json.dumps({"kernel_key": kernel}), encoding="utf-8")
            with pytest.raises(protocol.ProtocolError, match="frontend alias|receipt/cache|pre-build"):
                artifacts.validate_tilelang_pair_seed(on, manifest, root)


def test_tilelang_pair_post_build_rejects_any_extra_entry() -> None:
    with tempfile.TemporaryDirectory(dir=protocol.HERE) as temporary:
        root = Path(temporary)
        _off, on, manifest, off_build = _off_pair_fixture(root)
        with mock.patch.object(artifacts, "validate_build_record", return_value=off_build), \
                mock.patch.object(artifacts, "validate_verify_record", return_value={}):
            receipt = artifacts.seed_tilelang_pair(on, manifest, root)
            built = _add_f1(root, on, receipt)
            namespace = artifacts.entry_paths(on["entry_id"], root)["cache"] / "tilelang" / "0.1-test"
            (namespace / "frontend" / f"{'f' * 64}.json").write_text(
                json.dumps({"kernel_key": receipt["held_gemm_cache"]["kernel_key"]}),
                encoding="utf-8",
            )
            with pytest.raises(protocol.ProtocolError, match="frontend census|frontend alias"):
                artifacts.validate_tilelang_pair_seed(on, manifest, root, built)
