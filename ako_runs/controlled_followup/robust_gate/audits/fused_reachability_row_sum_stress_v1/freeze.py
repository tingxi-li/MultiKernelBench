"""Freeze the audit sources and preregister its one physical-GPU workload."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    seed_plan,
    verify_campaign,
)


LOCAL_SOURCES = (
    ".gitignore",
    "__init__.py",
    "manifest.json",
    "README.md",
    "runner.py",
    "analyze.py",
    "freeze.py",
    "capture_evidence.py",
    "tests/__init__.py",
    "tests/test_stress.py",
)


def _exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite of {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def freeze() -> dict[str, Any]:
    manifest, _gate, reach_lock, seeds = verify_campaign(require_freeze=False)
    forbidden = [
        HERE / "results",
        HERE / "evidence",
        LAUNCH_PATH,
        HERE / "receipts" / "gpu_execution_receipt.json",
        HERE / "receipts" / "build_receipt.json",
        HERE / "receipts" / "collection_receipt.json",
        HERE / "receipts" / "completion_receipt.json",
    ]
    if any(path.exists() for path in forbidden):
        raise FileExistsError("freeze must precede launch/results/evidence")
    paths = [HERE / relative for relative in LOCAL_SOURCES]
    paths.append(HERE.parent / "__init__.py")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    sources = {_relative(path): file_sha256(path) for path in paths}
    receipt = {
        "schema_version": "1.0",
        "record_type": "fused_reachability_row_sum_stress_freeze_receipt",
        "campaign_id": manifest["campaign_id"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "manifest_canonical_sha256": canonical_sha256(manifest),
        "source_sha256": sources,
        "source_bundle_canonical_sha256": canonical_sha256(sources),
        "seed_plan_canonical_sha256": canonical_sha256(seeds),
        "seed_plan_count": len(seeds),
        "fresh_tensor_seed_count": len(seeds) * 3,
        "overlap_with_original_validation": 0,
        "overlap_with_prior_stress_v1": 0,
        "reachability_launch_lock_sha256": manifest["reachability_binding"]["launch_lock_sha256"],
        "reachability_source_bundle_sha256": reach_lock["source_bundle_sha256"],
        "screen_selection_sha256": manifest["reachability_binding"]["screen_selection_sha256"],
        "gate_spec_sha256": manifest["registered_gate_binding"]["gate_spec_sha256"],
        "selected_candidate_job_sha256": {
            row["job_id"]: row["job_sha256"] for row in manifest["selected_candidates"]
        },
        "physical_gpu": manifest["hardware"]["physical_gpu"],
        "required_gpu_uuid": manifest["hardware"]["required_uuid"],
        "correctness_only": True,
        "threshold_fitting_allowed": False,
        "threshold_mutation_authorized": False,
        "performance_selection_feedback_authorized": False,
        "raw_results_opened": False,
    }
    _exclusive(FREEZE_PATH, receipt)
    return receipt


def prepare_launch() -> dict[str, Any]:
    manifest, _gate, _reach_lock, seeds = verify_campaign(require_freeze=True)
    if (HERE / "results").exists() or (HERE / "evidence").exists():
        raise FileExistsError("launch preregistration must precede results/evidence")
    freeze_receipt = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    module = (
        "ako_runs.controlled_followup.robust_gate.audits."
        "fused_reachability_row_sum_stress_v1.runner"
    )
    physical = manifest["hardware"]["physical_gpu"]
    output = _relative((HERE / manifest["workload"]["output"]).resolve())
    command = (
        f"CUDA_VISIBLE_DEVICES={physical} PYTHONDONTWRITEBYTECODE=1 "
        f"python -m {module}"
    )
    receipt = {
        "schema_version": "1.0",
        "record_type": "fused_reachability_row_sum_stress_launch_receipt",
        "campaign_id": manifest["campaign_id"],
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
        "source_bundle_canonical_sha256": freeze_receipt["source_bundle_canonical_sha256"],
        "seed_plan_canonical_sha256": canonical_sha256(seeds),
        "seed_indices": [row["seed_index"] for row in seeds],
        "candidates": [row["job_id"] for row in manifest["selected_candidates"]],
        "gates": list(manifest["gates"]),
        "physical_gpu": physical,
        "required_gpu_uuid": manifest["hardware"]["required_uuid"],
        "logical_device": manifest["hardware"]["logical_device"],
        "expected_records": manifest["workload"]["expected_records"],
        "output": output,
        "command": command,
        "correctness_only": True,
        "performance_selection_feedback_authorized": False,
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
    receipt = freeze() if args.freeze else prepare_launch()
    output = FREEZE_PATH if args.freeze else LAUNCH_PATH
    print(
        f"wrote {output}; source={receipt['source_bundle_canonical_sha256']}; "
        f"seeds={receipt['seed_plan_canonical_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
