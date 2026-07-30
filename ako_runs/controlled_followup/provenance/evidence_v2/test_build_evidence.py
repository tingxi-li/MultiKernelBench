from __future__ import annotations

from pathlib import Path

from . import build_evidence as evidence


def test_current_core_selection_validates_without_matmul() -> None:
    report = evidence.selection_report(include_matmul_v4=False)
    assert report["ok"] is True
    assert report["matmul_v4_included"] is False
    assert report["selected_entries"] >= 471
    assert report["validations"]["matmul_v4_status"]["state"] in {
        "active_or_incomplete",
        "complete_candidate",
    }
    assert len(report["validations"]["nested_evidence"]) == 5


def test_forbidden_material_is_fail_closed() -> None:
    forbidden = (
        Path("x/.torch_extensions/kernel.so"),
        Path("x/__pycache__/module.pyc"),
        Path("x/.pytest_cache/state"),
        Path("x/active.lock"),
        Path("x/results.jsonl.partial"),
        Path("x/kernel.cubin"),
        Path("x/.tmp.record"),
    )
    assert all(evidence._is_forbidden(path) is not None for path in forbidden)


def test_normalized_tar_metadata() -> None:
    info = evidence.normalized_info("payload", 17)
    assert (info.size, info.mode, info.mtime, info.uid, info.gid) == (
        17,
        0o644,
        0,
        0,
        0,
    )
    assert info.uname == info.gname == ""
