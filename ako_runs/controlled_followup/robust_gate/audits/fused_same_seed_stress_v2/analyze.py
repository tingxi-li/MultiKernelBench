"""Analyze the corrected paired audit without treating gate views as replicates."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.stats import beta, binomtest

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
    POLICY_PATH,
    SEED_PLAN_PATH,
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def exact_interval(failures: int, n: int, alpha: float = 0.05) -> list[float] | None:
    if n <= 0 or failures < 0 or failures > n:
        return None
    lower = 0.0 if failures == 0 else float(beta.ppf(alpha / 2, failures, n - failures + 1))
    upper = 1.0 if failures == n else float(beta.ppf(1 - alpha / 2, failures + 1, n - failures))
    return [lower, upper]


def quantile(values: list[float], probability: float) -> float | None:
    finite = sorted(value for value in values if isinstance(value, (int, float)) and math.isfinite(value))
    if not finite:
        return None
    position = (len(finite) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1 - fraction) + finite[upper] * fraction


def quantiles(values: list[float], probabilities: list[float]) -> dict[str, float | None]:
    return {f"q{probability:g}": quantile(values, probability) for probability in probabilities}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _seed_endpoint(rows: list[dict[str, Any]]) -> tuple[bool, bool, bool]:
    """Return collection failure, registered-gate failure, raw exceedance."""
    collection_failure = len(rows) != len(GATES) or any(row.get("ok") is not True for row in rows)
    gate_failure = collection_failure or any(row.get("gate_pass") is not True for row in rows)
    raw_exceeded = any(row.get("raw_safety_exceeded") is True for row in rows)
    return collection_failure, gate_failure, raw_exceeded


def _paired(old: dict[int, bool], new: dict[int, bool], label: str) -> dict[str, Any]:
    shared = sorted(set(old) & set(new))
    old_fail_new_pass = sum(old[index] and not new[index] for index in shared)
    old_pass_new_fail = sum(not old[index] and new[index] for index in shared)
    both_fail = sum(old[index] and new[index] for index in shared)
    both_pass = sum(not old[index] and not new[index] for index in shared)
    discordant = old_fail_new_pass + old_pass_new_fail
    p_value = 1.0 if discordant == 0 else float(
        binomtest(old_fail_new_pass, discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "endpoint": label,
        "shared_seed_n": len(shared),
        "both_pass": both_pass,
        "both_fail": both_fail,
        "old_fail_new_pass": old_fail_new_pass,
        "old_pass_new_fail": old_pass_new_fail,
        "discordant_n": discordant,
        "new_minus_old_failure_proportion": (
            (old_pass_new_fail - old_fail_new_pass) / len(shared) if shared else None
        ),
        "exact_two_sided_p": p_value,
    }


def analyze_records(
    manifest: dict[str, Any],
    gate_spec: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    source_bundle: str,
    build_receipt_sha256: str,
    seed_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    seed_rows = seed_rows or seed_plan(manifest)
    seed_by_index = {row["seed_index"]: row for row in seed_rows}
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    expected = {
        (candidate_id, gate_id, seed_index)
        for candidate_id in manifest["candidate_order"]
        for gate_id in GATES
        for seed_index in seed_by_index
    }
    seen: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicates = 0
    unexpected: list[Any] = []
    binding_failures: list[dict[str, Any]] = []
    decision_failures: list[dict[str, Any]] = []
    manifest_sha = file_sha256(MANIFEST_PATH)
    manifest_canonical = canonical_sha256(manifest)
    seed_hash = canonical_sha256(seed_rows)

    for row in records:
        key = (row.get("candidate_id"), row.get("gate_id"), row.get("seed_index"))
        if key in seen:
            duplicates += 1
            continue
        if key not in expected:
            unexpected.append(key)
            continue
        seen[key] = row
        candidate = candidates[key[0]]
        seed = seed_by_index[key[2]]
        required = {
            "schema_version": "2.0",
            "record_type": "fused_same_seed_stress_measurement",
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": manifest_sha,
            "manifest_canonical_sha256": manifest_canonical,
            "source_bundle_canonical_sha256": source_bundle,
            "seed_plan_canonical_sha256": seed_hash,
            "inference_policy_sha256": file_sha256(POLICY_PATH),
            "build_receipt_sha256": build_receipt_sha256,
            "gate_spec_sha256": manifest["registered_gate_binding"]["gate_spec_sha256"],
            "old_source_bundle_sha256": manifest["old_candidate_binding"]["source_bundle_sha256"],
            "streamed_source_bundle_sha256": manifest["streamed_candidate_binding"]["source_bundle_sha256"],
            "candidate_job_sha256": candidate["job_sha256"],
            "generation": candidate["generation"],
            "candidate_source": candidate["source"],
            "lane": candidate["lane"],
            "grid_id": candidate["grid_id"],
            "case_id": manifest["case"]["id"],
            "seed_segment": seed["segment"],
            "seed_namespace": seed["namespace"],
            "tensor_seeds": seed["tensor_seeds"],
            "shape": manifest["shape"],
            "physical_gpu": manifest["hardware"]["physical_gpu"],
            "logical_device": manifest["hardware"]["logical_device"],
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
            gate = gate_spec["gates"][f"fused_softmax/{key[1]}"]
            metrics = row.get("metrics", {})
            recomputed_failures = threshold_failures(gate, metrics)
            recomputed_ratios = {
                name: metrics[name] / rule["value"]
                for name, rule in gate["thresholds"].items()
                if name in metrics and rule["value"] > 0
            }
            row_sum = gate["thresholds"]["row_sum_error_max"]
            raw_cutoff = row_sum["observed_anchor_max"] * row_sum["safety_factor"]
            raw_exceeded = (
                isinstance(metrics.get("row_sum_error_max"), (int, float))
                and math.isfinite(metrics["row_sum_error_max"])
                and metrics["row_sum_error_max"] > raw_cutoff
            )
            if (
                row.get("threshold_failures") != recomputed_failures
                or row.get("gate_pass") is not (not recomputed_failures)
                or row.get("threshold_ratios") != recomputed_ratios
                or row.get("registered_row_sum_threshold") != row_sum["value"]
                or row.get("raw_safety_cutoff") != raw_cutoff
                or row.get("raw_safety_exceeded") is not raw_exceeded
            ):
                decision_failures.append({"key": key, "reason": "fixed decision/diagnostic mismatch"})
        elif (
            row.get("ok") is not False
            or row.get("gate_pass") is not False
            or row.get("threshold_failures") != ["collection_failure"]
            or row.get("error_category") not in {"build", "setup", "reference", "execution", "metric"}
            or not isinstance(row.get("error"), str)
            or row.get("raw_safety_exceeded") is not None
            or row.get("threshold_ratios") is not None
        ):
            decision_failures.append({"key": key, "reason": "malformed retained collection failure"})

    missing = sorted(expected - set(seen))
    probabilities = load_json(POLICY_PATH)["quantiles"]["probabilities"]
    groups: list[dict[str, Any]] = []
    for candidate_id in manifest["candidate_order"]:
        candidate = candidates[candidate_id]
        for gate_id in GATES:
            rows = [seen[(candidate_id, gate_id, index)] for index in seed_by_index if (candidate_id, gate_id, index) in seen]
            collection = [row for row in rows if row.get("ok") is not True]
            failures = [row for row in rows if row.get("gate_pass") is not True]
            raw = [row for row in rows if row.get("raw_safety_exceeded") is True]
            metric_failures: Counter[str] = Counter()
            for row in failures:
                metric_failures.update(item.split("=", 1)[0] for item in row["threshold_failures"])
            max_ratios = [max(row["threshold_ratios"].values()) for row in rows if row.get("ok") is True and row.get("threshold_ratios")]
            row_sum_threshold_ratios = [row["threshold_ratios"]["row_sum_error_max"] for row in rows if row.get("ok") is True]
            row_sum_raw_ratios = [row["metrics"]["row_sum_error_max"] / row["raw_safety_cutoff"] for row in rows if row.get("ok") is True]
            groups.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_job_sha256": candidate["job_sha256"],
                    "generation": candidate["generation"],
                    "lane": candidate["lane"],
                    "gate_id": gate_id,
                    "expected_seed_n": len(seed_by_index),
                    "observed_seed_n": len(rows),
                    "collection_failure_count": len(collection),
                    "failure_count": len(failures),
                    "failure_proportion": len(failures) / len(rows) if rows else None,
                    "failure_proportion_exact95": exact_interval(len(failures), len(rows)),
                    "failure_seed_indices": [row["seed_index"] for row in failures],
                    "failure_metrics": dict(sorted(metric_failures.items())),
                    "raw_cutoff_exceedance_count": len(raw),
                    "raw_cutoff_exceedance_seed_indices": [row["seed_index"] for row in raw],
                    "max_threshold_ratio_quantiles": quantiles(max_ratios, probabilities),
                    "row_sum_threshold_ratio_quantiles": quantiles(row_sum_threshold_ratios, probabilities),
                    "row_sum_raw_cutoff_ratio_quantiles": quantiles(row_sum_raw_ratios, probabilities),
                }
            )

    endpoint_by_candidate: dict[str, dict[int, bool]] = {}
    candidate_summaries = []
    for candidate_id in manifest["candidate_order"]:
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for key, row in seen.items():
            if key[0] == candidate_id:
                by_seed[key[2]].append(row)
        failures: dict[int, bool] = {}
        collection_count = 0
        raw_count = 0
        ratios: list[float] = []
        for index in seed_by_index:
            rows = by_seed.get(index, [])
            collection, failed, raw = _seed_endpoint(rows)
            failures[index] = failed
            collection_count += collection
            raw_count += raw
            row_ratios = [max(row["threshold_ratios"].values()) for row in rows if row.get("ok") is True and row.get("threshold_ratios")]
            if row_ratios:
                ratios.append(max(row_ratios))
        endpoint_by_candidate[candidate_id] = failures
        failure_count = sum(failures.values())
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "generation": candidates[candidate_id]["generation"],
                "lane": candidates[candidate_id]["lane"],
                "effective_seed_n": len(seed_by_index),
                "collection_failure_seed_count": collection_count,
                "any_gate_failure_seed_count": failure_count,
                "any_gate_failure_proportion": failure_count / len(seed_by_index),
                "any_gate_failure_proportion_exact95": exact_interval(failure_count, len(seed_by_index)),
                "any_gate_failure_seed_indices": [index for index, failed in failures.items() if failed],
                "any_gate_raw_cutoff_exceedance_seed_count": raw_count,
                "per_seed_max_threshold_ratio_quantiles": quantiles(ratios, probabilities),
                "zero_failures_observed": failure_count == 0,
                "robustness_claim": "not_established_by_zero_count_alone" if failure_count == 0 else "observed_registered_gate_failures",
            }
        )

    primary = []
    gate_diagnostics = []
    for old, new in manifest["paired_contrasts"]:
        row = _paired(endpoint_by_candidate[old], endpoint_by_candidate[new], "joint_all_gates")
        row.update({"old_candidate": old, "new_candidate": new})
        primary.append(row)
        for gate_id in GATES:
            old_endpoint = {
                index: seen.get((old, gate_id, index), {}).get("gate_pass") is not True
                for index in seed_by_index
            }
            new_endpoint = {
                index: seen.get((new, gate_id, index), {}).get("gate_pass") is not True
                for index in seed_by_index
            }
            diagnostic = _paired(old_endpoint, new_endpoint, gate_id)
            diagnostic.update({"old_candidate": old, "new_candidate": new, "multiplicity_adjusted": False})
            gate_diagnostics.append(diagnostic)
    adjusted = holm_adjust([row["exact_two_sided_p"] for row in primary])
    evidence_complete = (
        len(seen) == len(expected)
        and not missing
        and not unexpected
        and duplicates == 0
        and not binding_failures
        and not decision_failures
    )
    for row, adjusted_p in zip(primary, adjusted, strict=True):
        row["holm_adjusted_p"] = adjusted_p
        row["paired_fewer_failures_supported"] = bool(
            evidence_complete
            and row["shared_seed_n"] == 512
            and row["old_fail_new_pass"] > row["old_pass_new_fail"]
            and adjusted_p < 0.05
        )

    known_seed_table = []
    for index in (186, 197):
        if index not in seed_by_index:
            continue
        seed_entry = {"seed_index": index, "namespace": seed_by_index[index]["namespace"], "tensor_seeds": seed_by_index[index]["tensor_seeds"], "candidates": []}
        for candidate_id in manifest["candidate_order"]:
            gate_rows = []
            for gate_id in GATES:
                row = seen.get((candidate_id, gate_id, index))
                gate_rows.append(
                    {
                        "gate_id": gate_id,
                        "ok": row.get("ok") if row else None,
                        "gate_pass": row.get("gate_pass") if row else None,
                        "row_sum_error_max": row.get("metrics", {}).get("row_sum_error_max") if row else None,
                        "row_sum_threshold_ratio": row.get("threshold_ratios", {}).get("row_sum_error_max") if row and row.get("threshold_ratios") else None,
                        "raw_safety_exceeded": row.get("raw_safety_exceeded") if row else None,
                    }
                )
            seed_entry["candidates"].append({"candidate_id": candidate_id, "gates": gate_rows})
        known_seed_table.append(seed_entry)

    return {
        "schema_version": "2.0",
        "record_type": "fused_same_seed_stress_summary",
        "campaign_id": manifest["campaign_id"],
        "correctness_only": True,
        "threshold_mutation_authorized": False,
        "performance_selection_feedback_authorized": False,
        "effective_shared_seed_n": len(seed_by_index),
        "gate_views_per_seed_candidate": len(GATES),
        "gate_views_are_independent_replicates": False,
        "evidence_complete": evidence_complete,
        "candidate_summaries": candidate_summaries,
        "candidate_gate_groups": groups,
        "paired_primary_joint_all_gates": primary,
        "paired_gate_specific_diagnostics": gate_diagnostics,
        "known_seed_table": known_seed_table,
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
    for required in (LAUNCH_PATH, EXECUTION_PATH, BUILD_PATH, COLLECTION_PATH, SEED_PLAN_PATH):
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
        raise ValueError("partial raw stream still exists")
    if (
        collection.get("raw_sha256") != file_sha256(raw_path)
        or collection.get("record_count") != manifest["workload"]["expected_records"]
    ):
        raise ValueError("collection/raw binding mismatch")
    freeze = load_json(FREEZE_PATH)
    summary = analyze_records(
        manifest,
        gate_spec,
        load_jsonl(raw_path),
        source_bundle=freeze["source_bundle_canonical_sha256"],
        build_receipt_sha256=file_sha256(BUILD_PATH),
        seed_rows=seeds,
    )
    summary.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_path": str(raw_path.relative_to(HERE)),
            "raw_sha256": file_sha256(raw_path),
            "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
            "seed_plan_sha256": file_sha256(SEED_PLAN_PATH),
            "launch_receipt_sha256": file_sha256(LAUNCH_PATH),
            "gpu_execution_receipt_sha256": file_sha256(EXECUTION_PATH),
            "build_receipt_sha256": file_sha256(BUILD_PATH),
            "collection_receipt_sha256": file_sha256(COLLECTION_PATH),
            "inference_policy_sha256": file_sha256(POLICY_PATH),
        }
    )
    summary_path = Path(args.summary).resolve()
    _exclusive(summary_path, summary)
    completion = {
        "schema_version": "2.0",
        "record_type": "fused_same_seed_stress_completion_receipt",
        "campaign_id": manifest["campaign_id"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(summary_path.relative_to(HERE)),
        "summary_sha256": file_sha256(summary_path),
        "collection_receipt_sha256": file_sha256(COLLECTION_PATH),
        "evidence_complete": summary["evidence_complete"],
        "effective_shared_seed_n": summary["effective_shared_seed_n"],
        "threshold_mutation_authorized": False,
    }
    _exclusive(Path(args.receipt).resolve(), completion)
    print(json.dumps({"summary": str(summary_path), "evidence_complete": summary["evidence_complete"], "records": summary["coverage"]["observed_unique_records"]}, sort_keys=True))
    return 0 if summary["evidence_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
