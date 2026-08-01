from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = HERE.parents[2]
for path in (
    REPO_ROOT,
    REPO_ROOT / "ako_runs/phase1_matmul",
    REPO_ROOT / "ako_runs/phase2_fused_sdpa",
    HERE,
):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

import common2
from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import (
    candidates,
    core,
    cuda_noptx_register,
    cuda_unlimited,
    postprocess,
    support_probes,
    triton_smem,
)


def config(lane: str, epilogue: str, **overrides):
    extra = {"epilogue": epilogue, "wcache": "cached"}
    return common2.make_fused_config(lane, "GBGS", extra=extra, **overrides)


def test_cuda_unlimited_launch_checks_and_epilogue_conditional_smem():
    regs = cuda_unlimited.make_source(config("cuda_unlimited", "regs"))
    assert regs["epilogue_tile_shared_bytes"] == 0
    assert regs["total_dynamic_shared_bytes"] == regs["operand_shared_bytes"]
    assert '#include "checked_cuda_launch.h"' in regs["source"]
    assert 'checked_dynamic_smem((const void*)mma_gemm' in regs["source"]
    assert 'checked_kernel_launch("mma_gemm")' in regs["source"]
    assert 'checked_kernel_launch("softmax_kernel")' in regs["source"]
    assert "FSMEM_TILE_BYTES" not in regs["source"]

    smem = cuda_unlimited.make_source(config("cuda_unlimited", "smem"))
    assert smem["epilogue_tile_shared_bytes"] == 128 * (128 + 4) * 4
    assert smem["total_dynamic_shared_bytes"] == max(
        smem["operand_shared_bytes"], smem["epilogue_tile_shared_bytes"]
    )


def test_cuda_unlimited_register_recovers_grid_that_smem_rejects():
    wide = {"BM": 128, "BN": 256, "BK": 32, "threads": 256, "stages": 2, "kc": 2048}
    regs = cuda_unlimited.make_source(config("cuda_unlimited", "regs", **wide))
    assert regs["total_dynamic_shared_bytes"] <= cuda_unlimited.historical.p1._MAX_SMEM_OPTIN
    with pytest.raises(RuntimeError, match="exceeds the sm_89 cap"):
        cuda_unlimited.make_source(config("cuda_unlimited", "smem", **wide))


def test_tilelang_smem_normalization_reports_launched_allocation():
    cfg = config("tilelang", "smem")
    epilogue = cfg.BM * (cfg.BN + 4) * 4
    operand = (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * 2 * cfg.stages
    normalized = candidates._normalize_artifacts(
        {"shared_epilogue_bytes": epilogue}, cfg
    )
    assert normalized["operand_smem_bytes"] == operand
    assert normalized["epilogue_tile_bytes"] == epilogue
    assert normalized["shared_bytes"] == max(operand, epilogue)
    assert normalized["total_dynamic_shared_bytes"] == normalized["shared_bytes"]


def test_cuda_unlimited_recovers_all_eight_contaminated_register_grids():
    cells = [
        cell
        for cell in core.load_cells()
        if cell["strategy"] == "register_fused"
        and cell["lane"] == "cuda_unlimited"
        and 5 <= cell["grid_index"] <= 12
    ]
    assert len(cells) == 8
    for cell in cells:
        overrides = core.parse_set(cell["origin_job"]["set"])
        overrides.setdefault("extra", {}).update(epilogue="regs", wcache="cached")
        regs = cuda_unlimited.make_source(
            common2.make_fused_config("cuda_unlimited", "GBGS", **overrides)
        )
        assert regs["total_dynamic_shared_bytes"] <= cuda_unlimited.historical.p1._MAX_SMEM_OPTIN

        overrides["extra"]["epilogue"] = "smem"
        with pytest.raises(RuntimeError, match="exceeds the sm_89 cap"):
            cuda_unlimited.make_source(
                common2.make_fused_config("cuda_unlimited", "GBGS", **overrides)
            )


def test_no_ptx_register_probe_applies_epilogue_before_store():
    generated = cuda_noptx_register.make_source(config("cuda_noptx", "regs"))
    source = generated["source"]
    assert "wmma::load_matrix_sync(bias_fragment" in source
    assert "value += bias_fragment.x[e]" in source
    assert "float* Cs" not in source
    assert generated["epilogue_tile_shared_bytes"] == 0
    assert generated["total_dynamic_shared_bytes"] >= generated["operand_shared_bytes"]
    for forbidden in ("asm(", "asm (", "asm volatile", "__asm"):
        assert forbidden not in source
    assert '#include "checked_cuda_launch.h"' in source
    assert "#include <ATen/cuda/CUDAContext.h>" in source
    assert source.count("checked_kernel_launch(") == 2


def test_ptxas_census_selects_the_fused_kernel_not_softmax():
    log = """ptxas info : Compiling entry function '_Z14softmax_kernel' for 'sm_89'
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 48 registers, 64 bytes smem
ptxas info : Compiling entry function '_Z12fused_kernel' for 'sm_89'
176 bytes stack frame, 328 bytes spill stores, 332 bytes spill loads
ptxas info : Used 255 registers
"""
    resources = core.ptxas_kernel_resources(log, "fused_kernel")
    assert resources["registers"] == 255
    assert resources["spill_store_bytes"] == 328
    assert resources["spill_load_bytes"] == 332
    assert resources["stack_frame_bytes"] == 176


def test_common_postprocess_requires_and_records_compile_time_flags():
    with pytest.raises(TypeError):
        postprocess.make_source()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="explicit bool"):
        postprocess.make_source(has_bias=1, has_gelu=False)  # type: ignore[arg-type]
    softmax_only = postprocess.make_source(has_bias=False, has_gelu=False)
    combined = postprocess.make_source(has_bias=True, has_gelu=True)
    assert softmax_only.startswith("#define HAS_BIAS 0\n#define HAS_GELU 0")
    assert combined.startswith("#define HAS_BIAS 1\n#define HAS_GELU 1")
    assert '#include "checked_cuda_launch.h"' in softmax_only
    assert 'checked_kernel_launch("common_postprocess_kernel")' in softmax_only
    softmax_metadata = postprocess.source_metadata(has_bias=False, has_gelu=False)
    combined_metadata = postprocess.source_metadata(has_bias=True, has_gelu=True)
    assert softmax_metadata["has_bias"] is False and softmax_metadata["has_gelu"] is False
    assert softmax_metadata["cuda_source_sha256"] != combined_metadata["cuda_source_sha256"]
    assert softmax_metadata["softmax_source_sha256"] == combined_metadata["softmax_source_sha256"]


def test_candidate_dispatch_uses_v2_postprocess_flags(monkeypatch):
    import torch
    import variants

    assert Path(postprocess.__file__).resolve().parent == HERE
    calls = []

    def fake_postprocess(*, has_bias, has_gelu):
        calls.append((has_bias, has_gelu))
        return SimpleNamespace(
            artifacts={
                "cuda_source_sha256": f"cuda-{has_bias}-{has_gelu}",
                "softmax_source_sha256": "same-softmax",
            },
            compile_s=0.0,
            run=lambda output, _bias: output,
        )

    monkeypatch.setattr(postprocess, "build", fake_postprocess)
    monkeypatch.setattr(
        variants,
        "build",
        lambda _cfg: SimpleNamespace(
            artifacts={"source": "gemm"},
            compile_s=0.0,
            input_dtype=torch.float16,
            run=lambda x, _weight: x,
        ),
    )
    global_cell = next(
        cell
        for cell in core.load_cells()
        if cell["cell_id"] == "global_intermediate.tilelang.g01"
    )
    assert candidates._build_global(global_cell).metadata["n_kernels"] == 2

    lane = candidates.BuiltCell(
        "register_common_postprocess.tilelang.g01",
        lambda x, _weight, _bias: x,
        0.0,
        {"lane": "tilelang"},
        {"artifacts": {}, "build_wall_s": 0.0, "isolated_artifact_root": "/tmp/test"},
    )
    phase2_calls = []

    def fake_phase2(cell, epilogue, arm):
        phase2_calls.append((cell["lane"], epilogue, arm))
        return lane

    monkeypatch.setattr(candidates, "_build_phase2", fake_phase2)
    common_cell = {"cell_id": lane.cell_id, "lane": "tilelang"}
    assert candidates._build_register_common(common_cell).metadata["n_kernels"] == 2
    assert calls == [(True, True), (False, False)]
    assert phase2_calls == [("tilelang", "regs", "GBG")]


def test_every_supported_production_builder_consumes_strategy_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        candidates,
        "_build_phase2",
        lambda cell, epilogue, arm: calls.append((cell["lane"], "phase2", epilogue, arm)),
    )
    monkeypatch.setattr(
        candidates,
        "_build_tilelang_smem",
        lambda cell: calls.append((cell["lane"], "tilelang_smem", "smem", "GBGS")),
    )
    monkeypatch.setattr(
        candidates,
        "_build_global",
        lambda cell: calls.append((cell["lane"], "global", "global", "GBGS")),
    )
    monkeypatch.setattr(
        candidates,
        "_build_register_common",
        lambda cell: calls.append((cell["lane"], "register_common", "regs", "GBG")),
    )
    cells = [
        cell
        for cell in core.load_cells()
        if cell["grid_id"] == "g01" and cell["support_declared"] is True
    ]

    for cell in cells:
        candidates.build(cell)

    expected = []
    for cell in cells:
        if cell["strategy"] == "register_fused":
            expected.append((cell["lane"], "phase2", "regs", "GBGS"))
        elif cell["strategy"] == "smem_staged":
            builder = "tilelang_smem" if cell["lane"] == "tilelang" else "phase2"
            expected.append((cell["lane"], builder, "smem", "GBGS"))
        elif cell["strategy"] == "global_intermediate":
            expected.append((cell["lane"], "global", "global", "GBGS"))
        else:
            expected.append((cell["lane"], "register_common", "regs", "GBG"))
    assert calls == expected


def test_every_probe_builder_consumes_the_requested_epilogue():
    with pytest.raises(ValueError, match="epilogue=regs"):
        cuda_noptx_register.make_source(config("cuda_noptx", "smem"))
    with pytest.raises(ValueError, match="epilogue=smem"):
        triton_smem.build(config("triton", "regs"))
    with pytest.raises(ValueError, match="epilogue must"):
        bad = config("cuda_unlimited", "regs")
        bad.extra.pop("epilogue")
        cuda_unlimited.make_source(bad)


def test_triton_probe_retains_geometry_and_measured_api_failure():
    cfg = config("triton", "smem")
    source = triton_smem.make_source(cfg)
    assert '"BM": 128' in source and '"BN": 128' in source
    assert "tl.alloc_shared" in source
    with pytest.raises(triton_smem.ExplicitSharedMemoryUnavailable, match="Triton .*alloc_shared"):
        triton_smem.build(cfg)


def test_probe_resolution_is_receipt_derived():
    shared = [
        {
            "capability_limitation": True,
            "grid_id": grid,
            "terminal_outcome": "BUILD_FAILED",
            "failure_signature": "same",
        }
        for grid in support_probes.GRID_IDS
    ]
    assert support_probes.derive_resolution(shared)["status"] == "unsupported"
    shared[7] = {"grid_id": "g07", "terminal_outcome": "GATE_PASSED"}
    assert support_probes.derive_resolution(shared)["status"] == "supported"
    shared[7] = {
        "capability_limitation": True,
        "grid_id": "g07",
        "terminal_outcome": "BUILD_FAILED",
        "failure_signature": "other",
    }
    assert support_probes.derive_resolution(shared)["status"] == "unresolved"
    shared[7]["failure_signature"] = "same"
    shared[7]["capability_limitation"] = False
    assert support_probes.derive_resolution(shared)["status"] == "unresolved"


def test_probe_and_production_dispatch_are_one_function(monkeypatch):
    sentinel = object()

    class Fake:
        @staticmethod
        def build(cfg):
            return sentinel

    monkeypatch.setattr(support_probes, "_module", lambda _name: Fake)
    assert support_probes.build_probe_candidate("cuda_noptx_register", object()) is sentinel
    assert support_probes.build_probe_candidate("triton_smem", object()) is sentinel
