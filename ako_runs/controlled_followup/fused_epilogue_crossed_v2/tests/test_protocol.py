from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import (
    analyze,
    campaign_runner,
    capture_evidence,
    core,
    validate as validation,
)


def test_requested_manifest_is_deterministic_304_cell_cross():
    cells = core.load_cells()
    core.validate_cells(cells)
    assert len(cells) == 4 * 4 * 19 == 304
    assert Counter(row["strategy"] for row in cells) == {strategy: 76 for strategy in core.STRATEGIES}
    assert Counter(row["lane"] for row in cells) == {lane: 76 for lane in core.LANES}
    assert Counter(row["grid_id"] for row in cells) == {grid: 16 for grid in core.GRID_IDS}
    assert sum(row["support_declared"] is None for row in cells) == 0
    assert sum(row["support_declared"] is True for row in cells) == 285
    assert sum(row["support_declared"] is False for row in cells) == 19


def test_resolved_support_is_receipt_bound_and_probe_specific():
    cells = core.load_cells(require_resolved=True)
    unsupported = [row for row in cells if row["support_declared"] is False]
    assert {
        (row["strategy"], row["lane"], row["support_probe_key"])
        for row in unsupported
    } == {("smem_staged", "triton", "triton_smem")}
    assert all(
        row["support_declared"] is True
        for row in cells
        if row["lane"] == "cuda_noptx"
        and row["strategy"] in {"register_fused", "register_common_postprocess"}
    )


def test_confirmation_plan_randomizes_complete_cells_and_shams():
    selected = {
        "register_common_postprocess.tilelang.g01",
        "global_intermediate.tilelang.g01",
    }
    plan = core.confirmation_plan(selected)
    assert plan == core.confirmation_plan(selected)
    assert len(plan) == 15 * (len(selected) * 2 + 4)
    assert Counter(row["label"] for row in plan)["sham_a"] == 30
    assert Counter(row["label"] for row in plan)["sham_b"] == 30
    for rep in range(15):
        block = [row for row in plan if row["rep"] == rep]
        assert {(row["label"], row["distribution"]) for row in block if row["record_kind"] == "sham"} == {
            (label, distribution)
            for label in core.SHAM_LABELS
            for distribution in ("positive", "withheld_signed")
        }


def test_primary_timing_summary_uses_trials_60_through_99():
    values = [float(index + 1) for index in range(100)]
    summary = core.summarize_times(values)
    assert summary["full_median_ms"] == 50.5
    assert summary["primary_tail_median_ms"] == 80.5
    assert summary["first_decile_median_ms"] == 5.5
    assert summary["last_decile_median_ms"] == 95.5
    assert summary["first_to_last_decile_ratio"] == pytest.approx(95.5 / 5.5)


def test_gate_summary_does_not_divide_exact_zero_thresholds():
    case_ids = ["a", "b", "c", "d"]
    rows = [
        {
            "case_id": case_id,
            "gate_id": gate_id,
            "gate_pass": True,
            "metrics": {"max_abs_err": 0.5, "negative_count": 0},
            "ok": True,
            "seed_index": seed_index,
        }
        for case_id in case_ids
        for seed_index in range(64)
        for gate_id in ("conformance_mixed", "semantic_mixed")
    ]
    context = SimpleNamespace(
        adapter={"robust_gate": {"case_ids": case_ids}},
        gate_spec={
            "gates": {
                f"fused_softmax/{gate_id}": {
                    "thresholds": {
                        "max_abs_err": {"value": 1.0},
                        "negative_count": {"value": 0.0},
                    }
                }
                for gate_id in ("conformance_mixed", "semantic_mixed")
            }
        },
    )

    summary = campaign_runner._gate_summary(context, rows)

    assert summary["full_gate_pass"] is True
    assert summary["max_over_threshold_ratio"] == 0.5
    assert set(summary["zero_threshold_max_observed_by_metric"].values()) == {0.0}
    assert summary["zero_threshold_violation_records_by_metric"] == {}


def test_audit_accepts_supported_fourth_strategy_build_failures():
    record = {
        "build_metadata": None,
        "cell": {
            "cell_id": "register_common_postprocess.tilelang.g01",
            "strategy": "register_common_postprocess",
            "support_declared": True,
        },
        "terminal_outcome": "BUILD_FAILED",
    }
    analyze._validate_fourth_strategy_acceptance([record])

    record["terminal_outcome"] = "GATE_PASSED"
    record["build_metadata"] = {"n_kernels": 1}
    with pytest.raises(RuntimeError, match="full two-kernel operation"):
        analyze._validate_fourth_strategy_acceptance([record])

    record["build_metadata"]["n_kernels"] = 2
    analyze._validate_fourth_strategy_acceptance([record])


def test_sham_floor_suppresses_subfloor_effects():
    sham = {"ci_lo": 0.98, "ci_hi": 1.03}
    floor = core.resolution_floor([sham])
    assert floor == pytest.approx(max(abs(math.log(0.98)), abs(math.log(1.03))))
    assert not core.effect_is_reportable({"ci_lo": 1.001, "ci_hi": 1.02}, floor)
    assert core.effect_is_reportable({"ci_lo": 1.04, "ci_hi": 1.08}, floor)


def test_selection_is_top_two_plus_g01_per_stratum():
    cells = core.load_cells()
    medians = {}
    legal = set()
    for strategy in core.STRATEGIES:
        for lane in core.LANES:
            for grid, value in (("g00", 1.0), ("g01", 3.0), ("g02", 2.0)):
                cell_id = f"{strategy}.{lane}.{grid}"
                medians[cell_id] = [value, value]
                legal.add(cell_id)
    selected = analyze.choose_confirmation(cells, medians, legal)
    assert len(selected) == 4 * 4 * 3
    for strategy in core.STRATEGIES:
        for lane in core.LANES:
            rows = [row for row in selected if row["strategy"] == strategy and row["lane"] == lane]
            assert {row["grid_id"] for row in rows} == {"g00", "g01", "g02"}


def test_parent_results_are_content_bound():
    campaign, cells, resolution = core.make_unfrozen_contract()
    assert campaign["parent"]["controlling_result_sha256"] == "98552e48aa67411ba738e160d454e5c387e1c33cb1c543aa53a20198e4c0a70d"
    assert campaign["parent"]["tail_overlay_sha256"] == "e7296ce10585169e7dcf1f0d66c6bc59bf079d367d30491c46f42a8b3d03d1e5"
    assert len(cells) == 304
    assert resolution["status"] == "resolved"
    assert resolution["probes"]["cuda_noptx_register"]["status"] == "supported"
    assert resolution["probes"]["triton_smem"]["status"] == "unsupported"


def test_lock_aggregates_and_gate_binding_are_revalidated():
    sources = {
        relative: core.file_sha256(core.REPO_ROOT / relative)
        for relative in core.SOURCE_PATHS
    }
    dependencies = {
        relative: core.file_sha256(core.REPO_ROOT / relative)
        for relative in core.DEPENDENCY_PATHS
    }
    adapter = core.read_json(core.ROBUST_ADAPTER_PATH)
    lock = {
        "source_sha256": sources,
        "source_bundle_sha256": core.canonical_sha256(sources),
        "dependency_sha256": dependencies,
        "dependency_bundle_sha256": core.canonical_sha256(dependencies),
        "frozen_gate": {
            "adapter_manifest_sha256": core.file_sha256(core.ROBUST_ADAPTER_PATH),
            "gate_spec_sha256": adapter["robust_gate"]["gate_spec_sha256"],
            "manifest_sha256": adapter["robust_gate"]["manifest_sha256"],
        },
    }
    core.validate_frozen_hashes(lock)
    lock["dependency_bundle_sha256"] = "0" * 64
    with pytest.raises(core.ProtocolError, match="dependency_bundle_sha256"):
        core.validate_frozen_hashes(lock)


def test_remote_readiness_covers_every_frozen_dependency():
    paths = validation.remote_launch_paths("probes", {"support_evidence_sha256": {}})
    assert set(core.DEPENDENCY_PATHS) <= set(paths)


def test_timing_admission_rederives_retained_evidence(tmp_path, monkeypatch):
    results = tmp_path / "results"
    root = results / "tag"
    root.mkdir(parents=True)
    eligibility_path = root / "audit_summary.json"
    forged = {
        "campaign_id": core.CAMPAIGN_ID,
        "complete": True,
        "launch_lock_sha256": "lock",
        "record_type": "fused_crossed_v2_audit_summary",
        "source_bundle_sha256": "bundle",
        "timing_eligible_cell_ids": ["forged.cell"],
    }
    core.stable_write(eligibility_path, forged)
    monkeypatch.setattr(campaign_runner, "RESULTS_ROOT", results)
    monkeypatch.setattr(campaign_runner, "file_sha256", lambda _path: "lock")
    monkeypatch.setattr(
        campaign_runner.analysis,
        "audit_summary",
        lambda _root: {**forged, "timing_eligible_cell_ids": []},
    )
    with pytest.raises(RuntimeError, match="not re-derived"):
        campaign_runner._canonical_timing_plan(
            "screen", eligibility_path, {}, [], {"source_bundle_sha256": "bundle"}
        )


def test_evidence_index_requires_controlling_entries():
    lock_path = str(core.LOCK_PATH.relative_to(core.REPO_ROOT))
    index = {
        "campaign_id": core.CAMPAIGN_ID,
        "entries": [
            {"path": "summary.json", "sha256": "summary", "size": 1},
            {"path": lock_path, "sha256": "lock", "size": 1},
        ],
        "entry_count": 2,
        "launch_lock_sha256": "lock",
        "record_type": "fused_crossed_v2_complete_evidence_index",
        "schema_version": 2,
        "source_bundle_sha256": "bundle",
        "summary_path": "summary.json",
        "summary_sha256": "summary",
    }
    campaign = {"campaign_id": core.CAMPAIGN_ID}
    lock = {"source_bundle_sha256": "bundle"}
    assert len(capture_evidence._validated_entries(index, campaign, lock, "lock")) == 2
    index["entries"].pop()
    index["entry_count"] = 1
    with pytest.raises(RuntimeError, match="launch-lock"):
        capture_evidence._validated_entries(index, campaign, lock, "lock")
