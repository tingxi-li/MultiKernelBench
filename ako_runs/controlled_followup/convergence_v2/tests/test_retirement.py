from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def test_retirement_preserves_but_excludes_frozen_manifests() -> None:
    policy = json.loads((BASE / "retirement.json").read_text())
    assert policy["state"] == "retired_never_launched"
    assert policy["launch_policy"]["launch_allowed"] is False
    assert policy["launch_policy"]["allowed_manifest_paths"] == []
    assert policy["prompt_extension_policy"] == {
        "artifact_action": "preserve",
        "future_launch_action": "exclude",
        "may_be_carried_into_successor": False,
    }
    for binding in policy["historical_artifacts"].values():
        path = BASE / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]


def test_official_launcher_always_refuses() -> None:
    spec = importlib.util.spec_from_file_location("retired_convergence_launch", BASE / "launch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = io.StringIO()
    with redirect_stdout(output):
        assert module.main() == 2
    assert "no provider or GPU work started" in output.getvalue()
