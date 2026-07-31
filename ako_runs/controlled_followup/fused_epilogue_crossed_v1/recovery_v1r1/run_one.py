#!/usr/bin/env python3
"""Bind one frozen parent timing process to the v1r1 recovery commit."""
from __future__ import annotations

import argparse
import importlib.util
import os
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
    name = "fused_crossed_parent_run_one_v1r1"
    spec = importlib.util.spec_from_file_location(name, common.PARENT / "run_one.py")
    if spec is None or spec.loader is None:
        raise common.RecoveryError("cannot load frozen parent timing process")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    known, _ = parser.parse_known_args()
    validate.validate_structure()
    commit = os.environ.get("MKB_CROSSED_RECOVERY_COMMIT", "")
    binding = common.binding_for_commit(commit)
    common.require(
        os.environ.get("MKB_CROSSED_RECOVERY_LOCK_SHA256")
        == binding["recovery_lock_sha256"],
        "timing child recovery-lock environment differs",
    )
    output = Path(known.out).resolve()
    common.require(
        output.is_relative_to(common.RESULT_ROOT.resolve()),
        "timing output escapes v1r1 result root",
    )
    common.require(not output.exists(), "timing output already exists")
    parent = _parent_module()

    def stable_write(path: Path, value: dict[str, Any]) -> None:
        resolved = path.resolve()
        common.require(resolved == output, "timing child attempted another output")
        common.exclusive_json(resolved, common.add_binding(dict(value), binding))

    parent.stable_write = stable_write
    # A direct invocation imports recovery ``common`` under the top-level name
    # that the frozen timing process later uses for the Phase-1 gate helper.
    # Temporarily clear only that alias so the parent's existing PHASE1/PHASE2
    # search order resolves its intended module.  Package-mode launches never
    # create the alias, but follow the same safe path.
    top_common = sys.modules.get("common")
    if top_common is common:
        del sys.modules["common"]
    try:
        return parent.main()
    finally:
        if top_common is common:
            sys.modules["common"] = top_common


if __name__ == "__main__":
    raise SystemExit(main())
