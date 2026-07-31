from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]

from ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1 import (
    audit,
    common,
    freeze,
)


def _rows(*, violation: bool = False):
    case_ids = ["a", "b", "c", "d"]
    rows = []
    for case_id in case_ids:
        for seed in range(64):
            for gate_id in ("conformance_mixed", "semantic_mixed"):
                value = 1 if violation and case_id == "a" and seed == 0 else 0
                rows.append(
                    {
                        "case_id": case_id,
                        "gate_id": gate_id,
                        "gate_pass": value == 0,
                        "metrics": {"max_abs_err": 0.5, "negative_count": value},
                        "ok": True,
                        "seed_index": seed,
                    }
                )
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
    return context, rows


def test_zero_thresholds_are_not_denominators():
    context, rows = _rows()
    summary = audit.fixed_gate_summary(context, rows)
    assert summary["complete"] is True
    assert summary["full_gate_pass"] is True
    assert summary["max_over_threshold_ratio"] == 0.5
    assert summary["minimum_headroom_fraction"] == 0.5
    assert set(summary["zero_threshold_max_observed_by_metric"].values()) == {0.0}
    assert summary["zero_threshold_violation_records_by_metric"] == {}


def test_zero_threshold_violations_are_separate_and_fail_gate():
    context, rows = _rows(violation=True)
    summary = audit.fixed_gate_summary(context, rows)
    assert summary["full_gate_pass"] is False
    assert summary["failed_records"] == 2
    assert set(summary["zero_threshold_violation_records_by_metric"].values()) == {1}


def test_incident_and_parent_tree_are_exact():
    incident = common.verify_incident_receipt()
    assert incident["process_exit_census"]["zero_division_error"] == 4
    assert common.verify_parent()["sealed_terminal_outcomes"] == 0


def test_recovery_lock_verifies():
    value = freeze.verify()
    assert value["reporting_change"]["frozen_gate_decisions_changed"] is False
    assert value["reporting_change"]["frozen_thresholds_changed"] is False
    assert value["source_bundle_sha256"] == common.canonical_sha256(
        value["source_sha256"]
    )


def test_binding_writers_are_exclusive():
    binding = {
        "recovery_id": common.RECOVERY_ID,
        "parent_launch_lock_sha256": common.PARENT_LAUNCH_LOCK_SHA256,
        "parent_source_bundle_sha256": common.PARENT_SOURCE_BUNDLE_SHA256,
        "recovery_lock_sha256": "1" * 64,
        "recovery_source_bundle_sha256": "2" * 64,
        "recovery_git_commit": "3" * 40,
        "result_tag": common.RESULT_TAG,
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "value.json"
        common.exclusive_json(path, common.add_binding({"value": 1}, binding))
        observed = common.read_json(path)
        common.validate_binding(observed, binding, "test")
        with pytest.raises(common.RecoveryError):
            common.exclusive_json(path, observed)


@pytest.mark.parametrize(
    "module",
    ("audit", "analyze", "launch", "run_one", "validate", "capture_evidence"),
)
def test_cli_imports(module: str):
    completed = subprocess.run(
        [sys.executable, str(HERE / f"{module}.py"), "--help"],
        cwd=common.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
