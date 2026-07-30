from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import candidate
import protocol

_ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "fused_reachability_v2_analyze", Path(__file__).with_name("analyze.py")
)
assert _ANALYZE_SPEC and _ANALYZE_SPEC.loader
analyze = importlib.util.module_from_spec(_ANALYZE_SPEC)
_ANALYZE_SPEC.loader.exec_module(analyze)


def test_job_projection_is_exact_and_streamed():
    jobs = protocol.build_jobs()
    assert len(jobs) == 16
    assert {job["lane"] for job in jobs} == set(candidate.LANES)
    assert {job["grid_id"] for job in jobs} == {f"g{i:02d}" for i in range(5, 13)}
    assert all("x_epilogue=streamed_global" in job["set"] for job in jobs)
    assert all("x_epilogue=smem" not in job["set"] for job in jobs)


def test_config_contract_for_every_job():
    for job in protocol.build_jobs():
        cfg = candidate.make_config(job)
        assert (cfg.M, cfg.K, cfg.N) == (1024, 8192, 8192)
        assert cfg.variant == "GBGS"
        assert cfg.arith == "fp16"
        assert cfg.cast == "precast"
        assert cfg.extra == {"wcache": "cached", "epilogue": "streamed_global"}


def test_campaign_is_explicitly_prospective():
    campaign = json.loads(protocol.CAMPAIGN.read_text(encoding="utf-8"))
    assert campaign["status"] == "preregistered_not_launched"
    assert "not a retroactive replacement" in campaign["claim_limit"]


def test_cuda_wrappers_check_both_launches():
    assert candidate.NOPT_WRAPPER.count("C10_CUDA_KERNEL_LAUNCH_CHECK") == 2
    assert candidate.UNLIMITED_WRAPPER.count("C10_CUDA_KERNEL_LAUNCH_CHECK") == 2
    assert "cudaFuncSetAttribute" in candidate.NOPT_WRAPPER
    assert "cudaFuncSetAttribute" in candidate.UNLIMITED_WRAPPER
    assert "BM *" not in candidate.POSTPROCESS_SOFTMAX


def test_generated_sources_define_postprocess_before_wrapper(monkeypatch):
    captured = []

    class FakeModule:
        @staticmethod
        def fused_streamed(*_args):
            raise AssertionError("source-generation test must not execute CUDA")

    def fake_load(name, source):
        captured.append((name, source))
        return FakeModule()

    monkeypatch.setattr(candidate, "_load", fake_load)
    jobs = protocol.build_jobs()
    for lane in candidate.LANES:
        job = next(row for row in jobs if row["lane"] == lane and row["grid_id"] == "g05")
        built = candidate.build(lane, candidate.make_config(job))
        assert built.artifacts["shared_bytes"] <= candidate.MAX_SMEM_OPTIN
    assert len(captured) == 2
    for _name, source in captured:
        assert source.index("void postprocess_softmax_kernel") < source.index(
            "torch::Tensor fused_streamed"
        )
        assert source.count("torch::Tensor fused_streamed") == 1
        assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source
    noptx_source = captured[0][1]
    unlimited_source = captured[1][1]
    assert "gemm_kernel<<<" in noptx_source
    assert "mma_gemm<<<" in unlimited_source
    assert "torch::Tensor gemm(torch::Tensor A" not in unlimited_source


def test_all_wide_jobs_preserve_matched_staging(monkeypatch):
    class FakeModule:
        @staticmethod
        def fused_streamed(*_args):
            raise AssertionError("source-generation test must not execute CUDA")

    monkeypatch.setattr(candidate, "_load", lambda _name, _source: FakeModule())
    by_grid = {}
    for job in protocol.build_jobs():
        built = candidate.build(job["lane"], candidate.make_config(job))
        artifact = built.artifacts
        assert artifact["configured_stages"] == artifact["realized_stages"]
        assert artifact["shared_bytes"] <= candidate.MAX_SMEM_OPTIN
        by_grid.setdefault(job["grid_id"], {})[job["lane"]] = artifact
    assert set(by_grid) == {f"g{i:02d}" for i in range(5, 13)}
    for grid_id, lanes in by_grid.items():
        assert lanes["cuda_noptx"]["shared_bytes"] == lanes["cuda_unlimited"]["shared_bytes"]
        assert lanes["cuda_noptx"]["shared_padding_halfs"] == lanes["cuda_unlimited"]["shared_padding_halfs"]


def test_safe_result_tags_and_percentile_rule():
    assert protocol.safe_result_root("screen_gpu2").parent == protocol.RESULTS.resolve()
    for unsafe in ("../escape", "/tmp/escape", ".", "..", "a/b", ""):
        try:
            protocol.safe_result_root(unsafe)
        except ValueError:
            pass
        else:  # pragma: no cover - explicit fail message
            raise AssertionError(f"unsafe tag accepted: {unsafe!r}")
    assert analyze.percentile_linear([0.0, 10.0], 0.25) == 2.5
