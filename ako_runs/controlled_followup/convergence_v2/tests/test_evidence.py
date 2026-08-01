from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from ako_runs.controlled_followup.convergence_v2.capture_evidence import (
    EvidenceError,
    _excluded,
    capture,
    verify_evidence,
    verify_gpu_binding_receipt,
    verify_frozen_protocol,
)
from ako_runs.controlled_followup.convergence_v2.freeze_protocol import documents


BASE = Path(__file__).resolve().parents[1]


def test_post_freeze_tool_does_not_change_historical_receipts() -> None:
    hashes = verify_frozen_protocol(BASE)
    assert "locks/protocol_freeze_receipt.json" in hashes
    for path, expected in documents(BASE).items():
        assert path.read_bytes() == expected
    receipt = json.loads((BASE / "locks/protocol_freeze_receipt.json").read_text())
    assert "capture_evidence.py" not in receipt["files"]


def test_gpu_slot_binding_is_receipt_backed_but_still_requires_runtime_recheck() -> None:
    slots = verify_gpu_binding_receipt(BASE)
    assert set(slots) == {0, 1, 2, 3}
    assert len(set(slots.values())) == 4
    receipt = json.loads((BASE / "receipts/gpu_assignment_binding_20260731.json").read_text())
    assert receipt["runtime_recheck_required"] is True
    assert receipt["state"] == "resolved_identity_runtime_recheck_pending"


def test_prereg_capture_is_deterministic_and_claims_no_execution(tmp_path: Path) -> None:
    archive_a, index_a, value_a = capture(
        base=BASE,
        mode="prereg",
        output_prefix=tmp_path / "first",
    )
    archive_b, index_b, value_b = capture(
        base=BASE,
        mode="prereg",
        output_prefix=tmp_path / "second",
    )
    assert archive_a.read_bytes() == archive_b.read_bytes()
    assert index_a.read_bytes() == index_b.read_bytes()
    assert value_a == value_b
    verified = verify_evidence(index_a)
    assert verified["ok"] is True
    assert verified["archive_sha256"] == value_a["archive_sha256"]
    with tarfile.open(archive_a, "r:gz") as bundle:
        names = bundle.getnames()
        manifest = json.load(bundle.extractfile("EVIDENCE_MANIFEST.json"))
    assert manifest["claim_state"]["capture_state"] == "preregistration_only"
    assert manifest["claim_state"]["campaign_results_complete"] is False
    assert manifest["claim_state"]["provider_calls_claimed"] is False
    assert manifest["claim_state"]["gpu_trajectories_claimed"] is False
    assert manifest["claim_state"]["performance_results_claimed"] is False
    assert manifest["claim_state"]["unresolved_launch_checks"]
    assert manifest["post_freeze_evidence_utility"]["launch_input"] is False
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert not any("active.lock" in name or ".partial" in name for name in names)


@pytest.mark.parametrize(
    "relative",
    (
        Path("results/worker.lock"),
        Path("trajectories/run.partial/events.jsonl"),
        Path("trajectories/run.tmp/events.jsonl"),
    ),
)
def test_transient_lock_and_partial_paths_are_excluded(relative: Path) -> None:
    assert _excluded(relative, support_tree=False)


def test_complete_capture_fails_closed_without_all_terminal_inputs(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError, match="complete capture requires"):
        capture(base=BASE, mode="complete", output_prefix=tmp_path / "complete")


def test_verify_fails_closed_when_bundle_bytes_change(tmp_path: Path) -> None:
    archive, index, _value = capture(base=BASE, mode="prereg", output_prefix=tmp_path / "prereg")
    payload = bytearray(archive.read_bytes())
    payload[-1] ^= 1
    archive.write_bytes(payload)
    with pytest.raises(EvidenceError, match="archive bytes differ"):
        verify_evidence(index)
