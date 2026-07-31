from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze
import candidates
import core


def test_manifest_is_exact_balanced_228_cell_cross():
    cells = core.read_json(core.CELLS_PATH)
    core.validate_cells(cells)
    assert len(cells) == 3 * 4 * 19 == 228
    assert Counter(cell["strategy"] for cell in cells) == {strategy: 76 for strategy in core.STRATEGIES}
    assert Counter(cell["lane"] for cell in cells) == {lane: 57 for lane in core.LANES}
    assert Counter(cell["grid_id"] for cell in cells) == {grid: 12 for grid in core.GRID_IDS}


def test_every_unsupported_cell_is_explicit_and_builder_fails_before_gpu_import():
    cells = core.read_json(core.CELLS_PATH)
    unsupported = [cell for cell in cells if not cell["support_declared"]]
    assert len(unsupported) == 38
    assert all(cell["support_detail"] for cell in unsupported)
    for cell in unsupported[:2]:
        with pytest.raises(candidates.UnsupportedStrategy, match="no safe|no explicit"):
            candidates.build(cell)


def test_screen_and_confirmation_plans_are_deterministic_complete_blocks():
    cells = core.read_json(core.CELLS_PATH)
    legal = {cell["cell_id"] for cell in cells[:11]}
    first = core.screen_plan(cells, legal)
    assert first == core.screen_plan(cells, legal)
    assert len(first) == 22
    assert Counter(row["rep"] for row in first) == {0: 11, 1: 11}
    selected = set(sorted(legal)[:3])
    confirmation = core.confirmation_plan(selected)
    assert confirmation == core.confirmation_plan(selected)
    assert len(confirmation) == 3 * 15 * 2
    assert Counter(row["distribution"] for row in confirmation) == {"positive": 45, "withheld_signed": 45}
    for rep in range(15):
        block = [row for row in confirmation if row["rep"] == rep]
        assert {(row["cell_id"], row["distribution"]) for row in block} == {
            (cell_id, distribution)
            for cell_id in selected
            for distribution in ("positive", "withheld_signed")
        }


def test_selection_is_top_two_plus_g01_when_legal():
    cells = core.read_json(core.CELLS_PATH)
    medians = {}
    for strategy in core.STRATEGIES:
        for lane in core.LANES:
            medians[f"{strategy}.{lane}.g00"] = [1.0, 1.0]
            medians[f"{strategy}.{lane}.g01"] = [3.0, 3.0]
            medians[f"{strategy}.{lane}.g02"] = [2.0, 2.0]
    selected = analyze.choose_confirmation(cells, medians)
    assert len(selected) == 3 * 4 * 3
    for strategy in core.STRATEGIES:
        for lane in core.LANES:
            rows = [row for row in selected if row["strategy"] == strategy and row["lane"] == lane]
            assert {row["grid_id"] for row in rows} == {"g00", "g01", "g02"}
            assert next(row for row in rows if row["grid_id"] == "g01")["selection_reason"] == "g01_gate_legal_positive_control"


def test_gate_legal_g01_is_retained_after_screen_process_failure():
    cells = core.read_json(core.CELLS_PATH)
    legal_g01 = {
        f"{strategy}.{lane}.g01"
        for strategy in core.STRATEGIES
        for lane in core.LANES
    }
    selected = analyze.choose_confirmation(cells, {}, legal_g01)
    assert {row["cell_id"] for row in selected} == legal_g01
    assert all(row["screen_rank"] is None for row in selected)


def test_n15_interval_is_registered_x4_x12():
    interval = core.exact_median_interval(range(1, 16))
    assert interval["median"] == 8
    assert interval["ci_lo"] == 4
    assert interval["ci_hi"] == 12
    assert interval["order_k"] == 4
    assert interval["coverage"] == 0.96484375


def test_audit_analysis_fails_closed_on_missing_cell(tmp_path):
    with pytest.raises(RuntimeError, match="audit incomplete"):
        analyze.audit_summary(tmp_path)


def test_distribution_rank_statistic():
    assert core.spearman_rank([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert core.spearman_rank([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
