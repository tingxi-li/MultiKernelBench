"""Freeze source, seed vectors, inference, and the one authorized launch."""

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
    BUILD_PATH,
    COLLECTION_PATH,
    EXECUTION_PATH,
    FREEZE_PATH,
    HERE,
    LAUNCH_PATH,
    MANIFEST_PATH,
    POLICY_PATH,
    REPO_ROOT,
    SEED_PLAN_PATH,
    repo_path,
    seed_plan,
    verify_campaign,
)


LOCAL_SOURCES = (
    ".gitignore",
    "__init__.py",
    "README.md",
    "manifest.json",
    "inference_policy.json",
    "legacy_v1_status.json",
    "runner.py",
    "analyze.py",
    "freeze.py",
    "launch.py",
    "capture_evidence.py",
    "tests/__init__.py",
    "tests/test_same_seed_v2.py",
)


def _exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite of {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def freeze() -> dict[str, Any]:
    manifest, _gate, reach_lock, seeds = verify_campaign(require_freeze=False)
    forbidden = [HERE / "results", HERE / "evidence", FREEZE_PATH, SEED_PLAN_PATH, LAUNCH_PATH, EXECUTION_PATH, BUILD_PATH, COLLECTION_PATH]
    if any(path.exists() for path in forbidden):
        raise FileExistsError("freeze must precede receipts, results, and evidence")
    paths = [HERE / relative for relative in LOCAL_SOURCES]
    paths.append(HERE.parent / "__init__.py")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    source_sha256 = {_relative(path): file_sha256(path) for path in paths}
    _exclusive(SEED_PLAN_PATH, seeds)
    receipt = {
        "schema_version": "2.0",
        "record_type": "fused_same_seed_stress_freeze_receipt",
        "campaign_id": manifest["campaign_id"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "manifest_canonical_sha256": canonical_sha256(manifest),
        "inference_policy_sha256": file_sha256(POLICY_PATH),
        "inference_policy_canonical_sha256": canonical_sha256(json.loads(POLICY_PATH.read_text(encoding="utf-8"))),
        "source_sha256": source_sha256,
        "source_bundle_canonical_sha256": canonical_sha256(source_sha256),
        "seed_plan_path": _relative(SEED_PLAN_PATH),
        "seed_plan_sha256": file_sha256(SEED_PLAN_PATH),
        "seed_plan_canonical_sha256": canonical_sha256(seeds),
        "effective_shared_seed_n": 512,
        "tensor_seed_count": 1536,
        "legacy_replay_seed_n": 256,
        "legacy_raw_replay_verified": True,
        "known_replay_seed_vectors": {str(index): seeds[index]["tensor_seeds"] for index in (186, 197)},
        "candidate_job_sha256": {row["candidate_id"]: row["job_sha256"] for row in manifest["candidates"]},
        "old_source_bundle_sha256": manifest["old_candidate_binding"]["source_bundle_sha256"],
        "streamed_source_bundle_sha256": reach_lock["source_bundle_sha256"],
        "gate_spec_sha256": manifest["registered_gate_binding"]["gate_spec_sha256"],
        "expected_records": manifest["workload"]["expected_records"],
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
    manifest, _gate, _lock, seeds = verify_campaign(require_freeze=True)
    if (HERE / "results").exists() or (HERE / "evidence").exists() or any(path.exists() for path in (EXECUTION_PATH, BUILD_PATH, COLLECTION_PATH)):
        raise FileExistsError("launch receipt must precede collection/results/evidence")
    freeze_receipt = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    physical = manifest["hardware"]["physical_gpu"]
    module = "ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v2.runner"
    command = f"CUDA_VISIBLE_DEVICES={physical} PYTHONDONTWRITEBYTECODE=1 python -m {module}"
    receipt = {
        "schema_version": "2.0",
        "record_type": "fused_same_seed_stress_launch_receipt",
        "campaign_id": manifest["campaign_id"],
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
        "source_bundle_canonical_sha256": freeze_receipt["source_bundle_canonical_sha256"],
        "inference_policy_sha256": file_sha256(POLICY_PATH),
        "seed_plan_sha256": file_sha256(SEED_PLAN_PATH),
        "seed_plan_canonical_sha256": canonical_sha256(seeds),
        "candidate_order": manifest["candidate_order"],
        "gates": list(manifest["gates"]),
        "effective_shared_seed_n": 512,
        "physical_gpu": physical,
        "required_gpu_uuid": manifest["hardware"]["required_uuid"],
        "logical_device": manifest["hardware"]["logical_device"],
        "expected_records": manifest["workload"]["expected_records"],
        "output": _relative((HERE / manifest["workload"]["output"]).resolve()),
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
    print(json.dumps({"output": str(output), "source_bundle": receipt["source_bundle_canonical_sha256"], "expected_records": receipt["expected_records"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

