"""Freeze local stress sources, then preregister the two exact workloads."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ako_runs.controlled_followup.robust_gate.schema import canonical_sha256, file_sha256

from .runner import (
    FREEZE_PATH,
    HERE,
    LAUNCH_PATH,
    MANIFEST_PATH,
    REPO_ROOT,
    repo_path,
    verify_campaign,
)


LOCAL_SOURCES = (
    "__init__.py",
    "manifest.json",
    "runner.py",
    "freeze.py",
    "analyze.py",
    "README.md",
    "tests/__init__.py",
    "tests/test_stress.py",
)


def _exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def freeze() -> dict[str, Any]:
    manifest, _ = verify_campaign(require_freeze=False)
    paths = [HERE / relative for relative in LOCAL_SOURCES]
    paths.append(HERE.parent / "__init__.py")
    paths.append(repo_path(manifest["runner_adapter"]["path"]))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    sources = {_relative(path): file_sha256(path) for path in paths}
    receipt = {
        "schema_version": "1.0",
        "campaign_id": manifest["campaign_id"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "manifest_canonical_sha256": canonical_sha256(manifest),
        "original_gate": manifest["original_fused_v2"]["gate_spec"],
        "adapter_source_bundle_sha256": manifest["runner_adapter"][
            "source_bundle_sha256"
        ],
        "source_sha256": sources,
        "source_bundle_canonical_sha256": canonical_sha256(sources),
        "threshold_fitting_allowed": False,
        "threshold_mutation_authorized": False,
        "raw_results_opened": False,
    }
    _exclusive(FREEZE_PATH, receipt)
    return receipt


def launch() -> dict[str, Any]:
    manifest, _ = verify_campaign(require_freeze=True)
    freeze_receipt = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    workloads = []
    module = (
        "ako_runs.controlled_followup.robust_gate.audits."
        "fused_row_sum_stress_v1.runner"
    )
    for workload in manifest["workloads"]:
        output = _relative((HERE / workload["output"]).resolve())
        prefix = "CUDA_VISIBLE_DEVICES=1 " if workload["arm"] == "winners" else ""
        command = (
            prefix
            + "PYTHONDONTWRITEBYTECODE=1 python -m "
            + module
            + f" --arm {workload['arm']} --out {output}"
        )
        workloads.append({**workload, "output": output, "command": command})
    receipt = {
        "schema_version": "1.0",
        "campaign_id": manifest["campaign_id"],
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
        "source_bundle_canonical_sha256": freeze_receipt[
            "source_bundle_canonical_sha256"
        ],
        "workloads": workloads,
        "raw_results_opened": False,
    }
    _exclusive(LAUNCH_PATH, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--prepare-launch", action="store_true")
    args = parser.parse_args()
    receipt = freeze() if args.freeze else launch()
    path = FREEZE_PATH if args.freeze else LAUNCH_PATH
    print(f"wrote {path}; bundle={receipt['source_bundle_canonical_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
