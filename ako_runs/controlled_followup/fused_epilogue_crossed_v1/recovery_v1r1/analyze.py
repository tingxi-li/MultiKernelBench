#!/usr/bin/env python3
"""Run a frozen parent analysis and bind its output to v1r1."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from . import common, validate
except ImportError:  # pragma: no cover - direct-script CLI
    import common  # type: ignore
    import validate  # type: ignore


def _parent_module():
    if str(common.PARENT) not in sys.path:
        sys.path.insert(0, str(common.PARENT))
    name = "fused_crossed_parent_analyze_v1r1"
    spec = importlib.util.spec_from_file_location(name, common.PARENT / "analyze.py")
    if spec is None or spec.loader is None:
        raise common.RecoveryError("cannot load frozen parent analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _committed_binding() -> dict[str, str]:
    validate.validate_structure()
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=common.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return common.binding_for_commit(completed.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "screen", "confirmation"))
    parser.add_argument("--out", required=True)
    known, _ = parser.parse_known_args()
    output = Path(known.out).resolve()
    common.require(
        output.is_relative_to(common.RESULT_ROOT.resolve()),
        "analysis output escapes v1r1 result root",
    )
    common.require(not output.exists(), "analysis output already exists")
    binding = _committed_binding()
    common.validate_retained_tree(binding)
    parent = _parent_module()

    def stable_write(path: Path, value: dict[str, Any]) -> None:
        common.require(path.resolve() == output, "analyzer attempted another output")
        common.exclusive_json(output, common.add_binding(dict(value), binding))

    parent.stable_write = stable_write
    result = parent.main()
    common.validate_retained_tree(binding)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
