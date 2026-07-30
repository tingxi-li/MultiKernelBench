"""Adjudicate every frozen candidate/gate without changing any threshold."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ako_runs.controlled_followup.robust_gate.schema import canonical_sha256, file_sha256, load_json

from .runner import (
    BUILD_PATH,
    COLLECTION_PATH,
    EXECUTION_PATH,
    FREEZE_PATH,
    GATES,
    HERE,
    LAUNCH_PATH,
    MANIFEST_PATH,
    _seed_map,
    seed_plan,
    threshold_failures,
    verify_campaign,
)


SUMMARY_PATH = HERE / "results" / "summary.json"
COMPLETION_PATH = HERE / "receipts" / "completion_receipt.json"


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def analyze_records(
    manifest: dict[str, Any],
    gate_spec: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    audit_source_bundle: str,
    build_receipt_sha256: str,
    seed_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    seed_rows = seed_rows or seed_plan(manifest)
    seed_by_index = {row["seed_index"]: row["tensor_seeds"] for row in seed_rows}
    candidates = {row["job_id"]: row for row in manifest["selected_candidates"]}
    expected = {
        (candidate, gate_id, seed_index)
        for candidate in candidates
        for gate_id in GATES
        for seed_index in seed_by_index
    }
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates = 0
    unexpected = []
    binding_failures = []
    decision_failures = []
    reach = manifest["reachability_binding"]
    gate_binding = manifest["registered_gate_binding"]
    manifest_sha = file_sha256(MANIFEST_PATH)
    manifest_canonical = canonical_sha256(manifest)
    seed_hash = canonical_sha256(seed_rows)

    for row in records:
        key = (row.get("candidate"), row.get("gate_id"), row.get("seed_index"))
        if key in seen:
            duplicates += 1
            continue
        seen[key] = row
        if key not in expected:
            unexpected.append(key)
            continue
        candidate = candidates[key[0]]
        required = {
            "schema_version": "1.0",
            "record_type": "fused_reachability_row_sum_stress_measurement",
            "campaign_id": manifest["campaign_id"],
            "stress_manifest_sha256": manifest_sha,
            "stress_manifest_canonical_sha256": manifest_canonical,
            "audit_source_bundle_canonical_sha256": audit_source_bundle,
            "seed_plan_canonical_sha256": seed_hash,
            "reachability_launch_lock_sha256": reach["launch_lock_sha256"],
            "reachability_source_bundle_sha256": reach["source_bundle_sha256"],
            "screen_selection_sha256": reach["screen_selection_sha256"],
            "screen_summary_sha256": reach["screen_summary_sha256"],
            "gate_spec_sha256": gate_binding["gate_spec_sha256"],
            "gate_spec_canonical_sha256": gate_binding["gate_spec_canonical_sha256"],
            "build_receipt_sha256": build_receipt_sha256,
            "candidate_job_sha256": candidate["job_sha256"],
            "lane": candidate["lane"],
            "grid_id": candidate["grid_id"],
            "screen_rank": candidate["screen_rank"],
            "case_id": manifest["case"]["id"],
            "namespace": manifest["stress_split"]["namespace"],
            "tensor_seeds": seed_by_index[key[2]],
            "shape": manifest["shape"],
            "physical_gpu": manifest["hardware"]["physical_gpu"],
            "logical_device": "cuda:0",
            "correctness_only": True,
            "performance_selection_feedback_authorized": False,
        }
        mismatches = {
            name: {"expected": value, "observed": row.get(name)}
            for name, value in required.items()
            if row.get(name) != value
        }
        if mismatches:
            binding_failures.append({"key": key, "mismatches": mismatches})

        if row.get("ok") is True:
            frozen_gate = gate_spec["gates"][f"fused_softmax/{key[1]}"]
            metrics = row.get("metrics", {})
            recomputed = threshold_failures(frozen_gate, metrics)
            row_threshold = frozen_gate["thresholds"]["row_sum_error_max"]
            raw_cutoff = row_threshold["observed_anchor_max"] * row_threshold["safety_factor"]
            raw_exceeded = (
                isinstance(metrics.get("row_sum_error_max"), (int, float))
                and math.isfinite(metrics["row_sum_error_max"])
                and metrics["row_sum_error_max"] > raw_cutoff
            )
            if (
                row.get("threshold_failures") != recomputed
                or row.get("gate_pass") is not (not recomputed)
                or row.get("registered_row_sum_threshold") != row_threshold["value"]
                or row.get("raw_safety_cutoff") != raw_cutoff
                or row.get("raw_safety_exceeded") is not raw_exceeded
            ):
                decision_failures.append(
                    {
                        "key": key,
                        "expected_threshold_failures": recomputed,
                        "expected_raw_safety_exceeded": raw_exceeded,
                    }
                )
        elif (
            row.get("ok") is not False
            or row.get("gate_pass") is not False
            or row.get("threshold_failures") != ["collection_failure"]
            or row.get("error_category") not in {"build", "setup", "reference", "execution", "metric"}
            or not isinstance(row.get("error"), str)
            or row.get("raw_safety_exceeded") is not None
        ):
            decision_failures.append({"key": key, "reason": "invalid retained collection failure"})

    missing = sorted(expected - set(seen))
    groups = []
    group_map: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in manifest["selected_candidates"]:
        for gate_id in GATES:
            rows = [
                seen[(candidate["job_id"], gate_id, index)]
                for index in seed_by_index
                if (candidate["job_id"], gate_id, index) in seen
            ]
            collection_failures = [row for row in rows if row.get("ok") is not True]
            gate_failures = [
                row for row in rows
                if row.get("ok") is True and row.get("gate_pass") is not True
            ]
            raw_exceeds = [row for row in rows if row.get("raw_safety_exceeded") is True]
            metric_counts: Counter[str] = Counter()
            for row in gate_failures:
                metric_counts.update(item.split("=", 1)[0] for item in row["threshold_failures"])
            row_sum_values = [
                row["metrics"]["row_sum_error_max"]
                for row in rows
                if row.get("ok") is True and "metrics" in row
            ]
            success = (
                len(rows) == len(seed_by_index)
                and not collection_failures
                and not gate_failures
            )
            group = {
                "candidate": candidate["job_id"],
                "candidate_job_sha256": candidate["job_sha256"],
                "lane": candidate["lane"],
                "screen_rank": candidate["screen_rank"],
                "gate_id": gate_id,
                "expected_records": len(seed_by_index),
                "observed_records": len(rows),
                "collection_failure_records": len(collection_failures),
                "registered_gate_failure_records": len(gate_failures),
                "registered_gate_failure_metrics": dict(sorted(metric_counts.items())),
                "raw_safety_exceedance_records": len(raw_exceeds),
                "row_sum_error_max_observed": max(row_sum_values) if row_sum_values else None,
                "registered_row_sum_threshold": gate_binding["registered_row_sum_threshold"],
                "raw_safety_cutoff": (
                    gate_spec["gates"][f"fused_softmax/{gate_id}"]["thresholds"]["row_sum_error_max"]["observed_anchor_max"]
                    * gate_spec["gates"][f"fused_softmax/{gate_id}"]["thresholds"]["row_sum_error_max"]["safety_factor"]
                ),
                "zero_failure_upper95": (
                    1.0 - 0.05 ** (1.0 / len(rows))
                    if rows and not collection_failures and not gate_failures
                    else None
                ),
                "success": success,
            }
            groups.append(group)
            group_map[(candidate["job_id"], gate_id)] = group

    candidate_summaries = []
    for candidate in manifest["selected_candidates"]:
        candidate_rows = [row for key, row in seen.items() if key[0] == candidate["job_id"]]
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            by_seed[row["seed_index"]].append(row)
        unique_failure_seeds = sum(
            any(row.get("ok") is True and row.get("gate_pass") is not True for row in rows)
            for rows in by_seed.values()
        )
        unique_raw_exceed_seeds = sum(
            any(row.get("raw_safety_exceeded") is True for row in rows)
            for rows in by_seed.values()
        )
        per_gate = [group_map[(candidate["job_id"], gate_id)] for gate_id in GATES]
        candidate_summaries.append(
            {
                "candidate": candidate["job_id"],
                "candidate_job_sha256": candidate["job_sha256"],
                "lane": candidate["lane"],
                "screen_rank": candidate["screen_rank"],
                "observed_unique_seeds": len(by_seed),
                "expected_unique_seeds": len(seed_by_index),
                "collection_failure_records": sum(group["collection_failure_records"] for group in per_gate),
                "registered_gate_failure_records": sum(group["registered_gate_failure_records"] for group in per_gate),
                "unique_seed_any_registered_gate_failure": unique_failure_seeds,
                "raw_safety_exceedance_records": sum(group["raw_safety_exceedance_records"] for group in per_gate),
                "unique_seed_any_raw_safety_exceedance": unique_raw_exceed_seeds,
                "row_sum_error_max_observed": max(
                    (group["row_sum_error_max_observed"] for group in per_gate if group["row_sum_error_max_observed"] is not None),
                    default=None,
                ),
                "all_registered_gates_success": all(group["success"] for group in per_gate),
                "per_gate": per_gate,
            }
        )

    evidence_complete = (
        len(seen) == len(expected)
        and not missing
        and not unexpected
        and duplicates == 0
        and not binding_failures
        and not decision_failures
    )
    return {
        "schema_version": "1.0",
        "record_type": "fused_reachability_row_sum_stress_summary",
        "campaign_id": manifest["campaign_id"],
        "correctness_only": True,
        "performance_selection_feedback_authorized": False,
        "threshold_mutation_authorized": False,
        "seed_plan_canonical_sha256": seed_hash,
        "fresh_seed_count": len(seed_by_index),
        "evidence_complete": evidence_complete,
        "all_candidate_gate_groups_success": all(group["success"] for group in groups),
        "candidate_summaries": candidate_summaries,
        "candidate_gate_groups": groups,
        "coverage": {
            "expected_records": len(expected),
            "observed_unique_records": len(seen),
            "missing_records": len(missing),
            "duplicate_records": duplicates,
            "unexpected_records": len(unexpected),
            "binding_failures": binding_failures[:100],
            "decision_failures": decision_failures[:100],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(SUMMARY_PATH))
    parser.add_argument("--receipt", default=str(COMPLETION_PATH))
    args = parser.parse_args()
    manifest, gate_spec, _reach_lock, seeds = verify_campaign(require_freeze=True)
    for required in (LAUNCH_PATH, EXECUTION_PATH, BUILD_PATH, COLLECTION_PATH):
        if not required.is_file():
            raise FileNotFoundError(f"missing production receipt: {required}")
    launch = load_json(LAUNCH_PATH)
    execution = load_json(EXECUTION_PATH)
    collection = load_json(COLLECTION_PATH)
    if (
        launch.get("freeze_receipt_sha256") != file_sha256(FREEZE_PATH)
        or execution.get("launch_receipt_sha256") != file_sha256(LAUNCH_PATH)
        or collection.get("gpu_execution_receipt_sha256") != file_sha256(EXECUTION_PATH)
        or collection.get("build_receipt_sha256") != file_sha256(BUILD_PATH)
    ):
        raise ValueError("receipt chain mismatch")
    raw_path = HERE / manifest["workload"]["output"]
    if raw_path.with_name(raw_path.name + ".partial").exists():
        raise ValueError("partial raw stream exists")
    if (
        collection.get("raw_sha256") != file_sha256(raw_path)
        or collection.get("record_count") != manifest["workload"]["expected_records"]
    ):
        raise ValueError("collection receipt/raw binding mismatch")
    freeze = load_json(FREEZE_PATH)
    records = load_jsonl(raw_path)
    summary = analyze_records(
        manifest,
        gate_spec,
        records,
        audit_source_bundle=freeze["source_bundle_canonical_sha256"],
        build_receipt_sha256=file_sha256(BUILD_PATH),
        seed_rows=seeds,
    )
    summary.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_path": str(raw_path.relative_to(HERE)),
            "raw_sha256": file_sha256(raw_path),
            "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
            "launch_receipt_sha256": file_sha256(LAUNCH_PATH),
            "gpu_execution_receipt_sha256": file_sha256(EXECUTION_PATH),
            "build_receipt_sha256": file_sha256(BUILD_PATH),
            "collection_receipt_sha256": file_sha256(COLLECTION_PATH),
            "reachability_launch_lock_sha256": manifest["reachability_binding"]["launch_lock_sha256"],
            "reachability_source_bundle_sha256": manifest["reachability_binding"]["source_bundle_sha256"],
            "screen_selection_sha256": manifest["reachability_binding"]["screen_selection_sha256"],
            "gate_spec_sha256": manifest["registered_gate_binding"]["gate_spec_sha256"],
        }
    )
    summary_path = Path(args.summary).resolve()
    _exclusive(summary_path, summary)
    completion = {
        "schema_version": "1.0",
        "record_type": "fused_reachability_row_sum_stress_completion_receipt",
        "campaign_id": manifest["campaign_id"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(summary_path.relative_to(HERE)),
        "summary_sha256": file_sha256(summary_path),
        "collection_receipt_sha256": file_sha256(COLLECTION_PATH),
        "evidence_complete": summary["evidence_complete"],
        "all_candidate_gate_groups_success": summary["all_candidate_gate_groups_success"],
        "threshold_mutation_authorized": False,
        "performance_selection_feedback_authorized": False,
    }
    _exclusive(Path(args.receipt).resolve(), completion)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["evidence_complete"] and summary["all_candidate_gate_groups_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
