"""Create one-time source and launch receipts for the append-only audit."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from typing import Any

from ako_runs.controlled_followup.robust_gate.schema import canonical_sha256, file_sha256

from .bindings import (
    FREEZE_RECEIPT_PATH,
    LAUNCH_RECEIPT_PATH,
    REPO_ROOT,
    audit_manifest,
    audit_manifest_hashes,
    exclusive_json,
    relative,
    source_hashes,
    verify_freeze_receipt,
    verify_original_v4,
)


MODULE = (
    "ako_runs.controlled_followup.robust_gate.audits."
    "matmul_v4_instrument_v1.runner"
)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_freeze() -> dict[str, Any]:
    manifest = audit_manifest()
    gate = verify_original_v4(manifest)
    sources = source_hashes()
    namespaces = [
        manifest["splits"]["real_contact"]["namespace"],
        manifest["splits"]["synthetic"]["namespace"],
        manifest["splits"]["structural_smoke"]["namespace"],
        *[
            block["namespace"]
            for block in manifest["splits"]["replication_blocks"]
        ],
    ]
    if len(namespaces) != len(set(namespaces)):
        raise ValueError("audit namespaces are not pairwise distinct")
    receipt = {
        "schema_version": "1.0",
        "campaign_id": manifest["campaign_id"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "audit_manifest": audit_manifest_hashes(manifest),
        "original_v4": manifest["original_v4"],
        "original_gate_canonical_sha256_verified": canonical_sha256(gate),
        "source_files": sources,
        "source_bundle_canonical_sha256": canonical_sha256(sources),
        "namespaces": namespaces,
        "namespaces_pairwise_distinct": True,
        "threshold_source": "original_v4_gate_only",
        "threshold_fitting_allowed": False,
        "threshold_mutation_authorized": False,
        "raw_results_opened": False,
    }
    exclusive_json(FREEZE_RECEIPT_PATH, receipt)
    return receipt


def _expected_records(manifest: dict[str, Any], workload: dict[str, Any]) -> int:
    cases = len(manifest["cases"])
    if workload["arm"] == "replication":
        block = next(
            value
            for value in manifest["splits"]["replication_blocks"]
            if value["block_id"] == workload["block_id"]
        )
        return cases * block["seeds_per_case"] * len(manifest["gate_routes"])
    if workload["arm"] == "synthetic":
        numerical_routes = sum(
            len(control["gates"]) for control in manifest["wrong_answer_controls"]
        )
        structural_routes = len(manifest["structural_controls"]) * len(
            manifest["gate_routes"]
        )
        return (
            cases
            * manifest["splits"]["synthetic"]["seeds_per_case"]
            * numerical_routes
            + cases
            * manifest["splits"]["structural_smoke"]["seeds_per_case"]
            * structural_routes
        )
    if workload["arm"] == "real_contact":
        routes = sum(len(candidate["gates"]) for candidate in manifest["real_candidates"])
        return cases * manifest["splits"]["real_contact"]["seeds_per_case"] * routes
    raise ValueError(workload["arm"])


def create_launch() -> dict[str, Any]:
    freeze = verify_freeze_receipt()
    manifest = audit_manifest()
    workloads = []
    for item in manifest["workloads"]:
        output = relative((FREEZE_RECEIPT_PATH.parent.parent / item["output"]).resolve())
        arguments = ["--arm", item["arm"]]
        if item.get("block_id"):
            arguments += ["--block", item["block_id"]]
        arguments += ["--device", "cuda", "--out", output]
        command = (
            "CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python -m "
            + MODULE
            + " "
            + " ".join(arguments)
        )
        workloads.append(
            {
                "arm": item["arm"],
                "block_id": item.get("block_id"),
                "gpu": 1,
                "output": output,
                "expected_records": _expected_records(manifest, item),
                "command": command,
            }
        )
    receipt = {
        "schema_version": "1.0",
        "campaign_id": manifest["campaign_id"],
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "freeze_receipt_path": relative(FREEZE_RECEIPT_PATH),
        "freeze_receipt_sha256": file_sha256(FREEZE_RECEIPT_PATH),
        "source_bundle_canonical_sha256": freeze["source_bundle_canonical_sha256"],
        "cuda_visible_devices": "1",
        "workloads": workloads,
        "raw_results_opened": False,
    }
    exclusive_json(LAUNCH_RECEIPT_PATH, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--prepare-launch", action="store_true")
    args = parser.parse_args()
    receipt = create_freeze() if args.freeze else create_launch()
    path = FREEZE_RECEIPT_PATH if args.freeze else LAUNCH_RECEIPT_PATH
    print(f"wrote {path}; source_bundle={receipt['source_bundle_canonical_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
