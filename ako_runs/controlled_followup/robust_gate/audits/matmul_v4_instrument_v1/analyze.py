"""Check exact campaign coverage and fixed-gate outcomes without pooling blocks."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ako_runs.controlled_followup.robust_gate.schema import file_sha256

from .bindings import (
    HERE,
    FREEZE_RECEIPT_PATH,
    audit_manifest,
    audit_manifest_hashes,
    exclusive_json,
    load_json,
    raw_files_sha256,
    relative,
    threshold_failures,
    verify_freeze_receipt,
    verify_original_v4,
)
from .runner import GATE_ORDER, tensor_seeds


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: record is not an object")
                records.append(value)
    return records


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("arm"),
        record.get("block_id"),
        record.get("case_id"),
        record.get("seed_index"),
        record.get("gate_id"),
        record.get("candidate"),
    )


def expected_records(manifest: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    expected: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add(
        *,
        arm: str,
        block_id: str | None,
        namespace: str,
        case_id: str,
        seed_index: int,
        gate_id: str,
        candidate: str,
        role: str,
        outcome: str,
    ) -> None:
        descriptor = {
            "arm": arm,
            "block_id": block_id,
            "namespace": namespace,
            "case_id": case_id,
            "seed_index": seed_index,
            "gate_id": gate_id,
            "candidate": candidate,
            "role": role,
            "expected_outcome": outcome,
        }
        key = record_key(descriptor)
        if key in expected:
            raise ValueError(f"manifest creates duplicate expected record {key}")
        expected[key] = descriptor

    case_ids = [case["id"] for case in manifest["cases"]]
    for block in manifest["splits"]["replication_blocks"]:
        for case_id in case_ids:
            for index in range(block["seeds_per_case"]):
                for gate_id in GATE_ORDER:
                    add(
                        arm="replication",
                        block_id=block["block_id"],
                        namespace=block["namespace"],
                        case_id=case_id,
                        seed_index=index,
                        gate_id=gate_id,
                        candidate=manifest["gate_routes"][gate_id]["positive_control"],
                        role="positive_control",
                        outcome="pass",
                    )

    split = manifest["splits"]["synthetic"]
    for case_id in case_ids:
        for index in range(split["seeds_per_case"]):
            for control in manifest["wrong_answer_controls"]:
                outcome = "pass" if case_id in control["exact_pass_cases"] else "reject"
                role = "exact_zero_exclusion" if outcome == "pass" else "negative_control"
                for gate_id in control["gates"]:
                    add(
                        arm="synthetic",
                        block_id=None,
                        namespace=split["namespace"],
                        case_id=case_id,
                        seed_index=index,
                        gate_id=gate_id,
                        candidate=control["control_id"],
                        role=role,
                        outcome=outcome,
                    )

    split = manifest["splits"]["structural_smoke"]
    for case_id in case_ids:
        for index in range(split["seeds_per_case"]):
            for control in manifest["structural_controls"]:
                for gate_id in GATE_ORDER:
                    add(
                        arm="structural_smoke",
                        block_id=None,
                        namespace=split["namespace"],
                        case_id=case_id,
                        seed_index=index,
                        gate_id=gate_id,
                        candidate=control["control_id"],
                        role="structural_control",
                        outcome="reject",
                    )

    split = manifest["splits"]["real_contact"]
    for case_id in case_ids:
        for index in range(split["seeds_per_case"]):
            for candidate in manifest["real_candidates"]:
                for gate_id in candidate["gates"]:
                    add(
                        arm="real_contact",
                        block_id=None,
                        namespace=split["namespace"],
                        case_id=case_id,
                        seed_index=index,
                        gate_id=gate_id,
                        candidate=candidate["candidate_id"],
                        role="real_candidate",
                        outcome="pass",
                    )
    return expected


def _failure_upper95(failures: int, total: int) -> float | None:
    if total <= 0:
        return None
    if failures == 0:
        return 1.0 - 0.05 ** (1.0 / total)
    z = 1.959963984540054
    p = failures / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return min(1.0, center + half / denominator)


def _group_key(descriptor: dict[str, Any]) -> tuple[str, ...]:
    arm = descriptor["arm"]
    if arm == "replication":
        return (arm, descriptor["block_id"], descriptor["gate_id"])
    return (arm, descriptor["candidate"], descriptor["gate_id"])


def _outcome_correct(record: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    if expected["role"] == "structural_control":
        correct = (
            record.get("ok") is False
            and record.get("gate_pass") is False
            and record.get("error_category") == "metric_preflight"
        )
        return correct, "structural control must fail closed in metric preflight"
    if expected["expected_outcome"] == "reject":
        correct = record.get("ok") is True and record.get("gate_pass") is False
        return correct, "numerical negative control must collect and be rejected"
    correct = record.get("ok") is True and record.get("gate_pass") is True
    if expected["role"] == "exact_zero_exclusion":
        correct = correct and record.get("reference_exact_zero") is True
        return correct, "paired-cancellation exclusion must be exactly zero and pass"
    return correct, "positive control or real candidate must collect and pass"


def analyze_records(
    manifest: dict[str, Any],
    gate: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    source_bundle: str | None = None,
) -> dict[str, Any]:
    expected = expected_records(manifest)
    manifest_hashes = audit_manifest_hashes(manifest)
    original = manifest["original_v4"]["gate_spec"]
    observed: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates: list[tuple[Any, ...]] = []
    unexpected: list[tuple[Any, ...]] = []
    binding_failures: list[dict[str, Any]] = []

    for position, record in enumerate(records):
        key = record_key(record)
        if key in observed:
            duplicates.append(key)
            continue
        observed[key] = record
        descriptor = expected.get(key)
        if descriptor is None:
            unexpected.append(key)
            continue
        required = {
            "campaign_id": manifest["campaign_id"],
            "audit_manifest_sha256": manifest_hashes["raw_sha256"],
            "audit_manifest_canonical_sha256": manifest_hashes["canonical_sha256"],
            "original_gate_sha256": original["sha256"],
            "original_gate_canonical_sha256": original["canonical_sha256"],
            "mode": "production",
            "record_type": "matmul_v4_fixed_threshold_audit_measurement",
            "namespace": descriptor["namespace"],
            "role": descriptor["role"],
            "expected_outcome": descriptor["expected_outcome"],
            "shape": manifest["shape"],
            "tensor_seeds": tensor_seeds(
                descriptor["namespace"], descriptor["case_id"], descriptor["seed_index"]
            ),
        }
        if source_bundle is not None:
            required["source_bundle_canonical_sha256"] = source_bundle
        mismatches = {
            field: {"expected": value, "observed": record.get(field)}
            for field, value in required.items()
            if record.get(field) != value
        }
        if mismatches:
            binding_failures.append(
                {"record_position": position, "key": list(key), "mismatches": mismatches}
            )
        if record.get("ok") is True:
            frozen_gate = gate["gates"][f"matmul/{descriptor['gate_id']}"]
            recomputed = threshold_failures(frozen_gate, record.get("metrics", {}))
            if (
                record.get("threshold_failures") != recomputed
                or record.get("gate_pass") is not (not recomputed)
            ):
                binding_failures.append(
                    {
                        "record_position": position,
                        "key": list(key),
                        "mismatches": {
                            "fixed_threshold_decision": {
                                "expected_failures": recomputed,
                                "observed_failures": record.get("threshold_failures"),
                                "observed_gate_pass": record.get("gate_pass"),
                            }
                        },
                    }
                )

    missing = sorted(set(expected) - set(observed), key=str)
    outcome_failures: list[dict[str, Any]] = []
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    case_groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, descriptor in expected.items():
        group_key = _group_key(descriptor)
        group = groups.setdefault(
            group_key,
            {
                "group": list(group_key),
                "expected_records": 0,
                "observed_records": 0,
                "correct_outcomes": 0,
                "incorrect_outcomes": 0,
                "threshold_failure_metrics": defaultdict(int),
                "metric_maxima": {},
            },
        )
        group["expected_records"] += 1
        case_key = group_key + (descriptor["case_id"],)
        case_group = case_groups.setdefault(
            case_key,
            {
                "group": list(group_key),
                "case_id": descriptor["case_id"],
                "expected_records": 0,
                "observed_records": 0,
                "correct_outcomes": 0,
                "incorrect_outcomes": 0,
            },
        )
        case_group["expected_records"] += 1
        record = observed.get(key)
        if record is None:
            continue
        group["observed_records"] += 1
        case_group["observed_records"] += 1
        correct, rule = _outcome_correct(record, descriptor)
        if correct:
            group["correct_outcomes"] += 1
            case_group["correct_outcomes"] += 1
        else:
            group["incorrect_outcomes"] += 1
            case_group["incorrect_outcomes"] += 1
            outcome_failures.append(
                {
                    "key": list(key),
                    "rule": rule,
                    "ok": record.get("ok"),
                    "gate_pass": record.get("gate_pass"),
                    "error": record.get("error"),
                    "threshold_failures": record.get("threshold_failures", []),
                }
            )
        for failure in record.get("threshold_failures", []):
            metric = failure.split("=", 1)[0]
            group["threshold_failure_metrics"][metric] += 1
        if record.get("ok") is True:
            for metric, value in record.get("metrics", {}).items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    group["metric_maxima"][metric] = max(
                        value, group["metric_maxima"].get(metric, -math.inf)
                    )

    group_rows = []
    for key, group in sorted(groups.items()):
        group["threshold_failure_metrics"] = dict(group["threshold_failure_metrics"])
        failures = group["expected_records"] - group["correct_outcomes"]
        group["coverage_complete"] = group["observed_records"] == group["expected_records"]
        group["success"] = group["coverage_complete"] and failures == 0
        group["observed_incorrect_rate"] = failures / group["expected_records"]
        group["incorrect_rate_upper95"] = _failure_upper95(failures, group["expected_records"])
        gate_id = key[-1]
        threshold_values = gate["gates"][f"matmul/{gate_id}"]["thresholds"]
        group["gate_metric_headroom"] = {
            metric: {
                "observed_max": group["metric_maxima"].get(metric),
                "threshold": spec["value"],
                "max_over_threshold": (
                    group["metric_maxima"].get(metric, 0.0) / spec["value"]
                    if spec["value"] > 0
                    else (0.0 if group["metric_maxima"].get(metric, 0.0) == 0 else None)
                ),
            }
            for metric, spec in threshold_values.items()
        }
        group_rows.append(group)

    case_rows = []
    for _, group in sorted(case_groups.items()):
        group["coverage_complete"] = group["observed_records"] == group["expected_records"]
        group["success"] = (
            group["coverage_complete"]
            and group["correct_outcomes"] == group["expected_records"]
        )
        case_rows.append(group)

    def arm_success(arm: str) -> bool:
        selected = [row for row in group_rows if row["group"][0] == arm]
        return bool(selected) and all(row["success"] for row in selected)

    evidence_complete = not (missing or duplicates or unexpected or binding_failures)
    replication = arm_success("replication")
    synthetic = arm_success("synthetic") and arm_success("structural_smoke")
    real = arm_success("real_contact")
    return {
        "schema_version": "1.0",
        "campaign_id": manifest["campaign_id"],
        "audit_manifest": manifest_hashes,
        "original_gate": {
            "sha256": original["sha256"],
            "canonical_sha256": original["canonical_sha256"],
        },
        "success_semantics": (
            "Endpoints are casewise and blockwise. Real-candidate failure is retained as "
            "a candidate result and never authorizes threshold mutation."
        ),
        "expected_records": len(expected),
        "observed_unique_records": len(observed),
        "evidence_complete": evidence_complete,
        "coverage": {
            "missing_count": len(missing),
            "missing_first_100": [list(key) for key in missing[:100]],
            "duplicate_count": len(duplicates),
            "duplicates_first_100": [list(key) for key in duplicates[:100]],
            "unexpected_count": len(unexpected),
            "unexpected_first_100": [list(key) for key in unexpected[:100]],
            "binding_failure_count": len(binding_failures),
            "binding_failures_first_100": binding_failures[:100],
        },
        "endpoints": {
            "fixed_threshold_replication_success": replication,
            "synthetic_discrimination_success": synthetic,
            "real_contact_all_candidates_success": real,
            "all_preregistered_endpoints_success": evidence_complete
            and replication
            and synthetic
            and real,
        },
        "outcome_failure_count": len(outcome_failures),
        "outcome_failures_first_500": outcome_failures[:500],
        "groups": group_rows,
        "groups_by_case": case_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(HERE / "results" / "summary.json"))
    parser.add_argument(
        "--receipt", default=str(HERE / "receipts" / "completion_receipt.json")
    )
    args = parser.parse_args()

    manifest = audit_manifest()
    gate = verify_original_v4(manifest)
    freeze = verify_freeze_receipt()
    raw_paths = [HERE / workload["output"] for workload in manifest["workloads"]]
    missing_files = [str(path) for path in raw_paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"campaign raw files are missing: {missing_files}")
    records = load_jsonl(raw_paths)
    summary = analyze_records(
        manifest,
        gate,
        records,
        source_bundle=freeze["source_bundle_canonical_sha256"],
    )
    summary["generated_utc"] = datetime.now(timezone.utc).isoformat()
    summary["raw_files"] = raw_files_sha256(raw_paths)
    summary["freeze_receipt_sha256"] = file_sha256(FREEZE_RECEIPT_PATH)

    summary_path = Path(args.summary).resolve()
    receipt_path = Path(args.receipt).resolve()
    exclusive_json(summary_path, summary)
    completion = {
        "schema_version": "1.0",
        "campaign_id": manifest["campaign_id"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "summary_path": relative(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "raw_files": raw_files_sha256(raw_paths),
        "freeze_receipt_sha256": file_sha256(FREEZE_RECEIPT_PATH),
        "source_bundle_canonical_sha256": freeze["source_bundle_canonical_sha256"],
        "evidence_complete": summary["evidence_complete"],
        "endpoints": summary["endpoints"],
        "threshold_mutation_authorized": False,
    }
    exclusive_json(receipt_path, completion)
    print(
        f"summary={summary_path} evidence_complete={summary['evidence_complete']} "
        f"endpoints={summary['endpoints']}"
    )
    return 0 if summary["endpoints"]["all_preregistered_endpoints_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
