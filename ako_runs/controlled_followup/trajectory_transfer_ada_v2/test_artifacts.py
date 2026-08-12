#!/usr/bin/env python3
"""CPU-only checks for transfer adapters and artifact binding."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from ako_runs.controlled_followup.trajectory_transfer_ada_v2 import artifacts, protocol
from ako_runs.controlled_followup.trajectory_transfer_ada_v2 import implementations

HEX_A = "a" * 64
HEX_B = "b" * 64


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
        entry_id = "tt2_" + "d" * 24
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
    entry_id = "tt2_" + "e" * 24
    admit = artifacts.cache_environment(entry_id, "admit", root)
    load = artifacts.cache_environment(entry_id, "load_only", root)
    assert {"TMPDIR", "TMP", "TEMP"} <= set(admit)
    assert not ({"TMPDIR", "TMP", "TEMP"} & set(load))
    assert load["TILELANG_TMP_DIR"].endswith("runtime_tmp/tilelang")


def test_admission_metadata_keeps_native_diagnostics_verbatim() -> None:
    log = "ptxas info\n" * 1000
    cleaned = artifacts._clean_metadata({"artifacts": {"ptxas_log": log}})
    assert cleaned["artifacts"]["ptxas_log"] == log
