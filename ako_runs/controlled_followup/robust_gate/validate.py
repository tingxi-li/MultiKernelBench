"""Validate locked candidate records against a frozen robust gate spec."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any

from . import SCHEMA_VERSION
from .schema import (
    SchemaError,
    canonical_sha256,
    load_json,
    load_records,
    validate_gate_spec,
    validate_manifest,
    write_json,
)
from .seeds import tensor_seeds


def _failure_upper95(failures: int, total: int) -> float | None:
    if total <= 0:
        return None
    if failures == 0:
        return 1.0 - 0.05 ** (1.0 / total)
    # Wilson two-sided 95% upper endpoint for descriptive non-zero cases.
    z = 1.959963984540054
    p = failures / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return min(1.0, center + half / denominator)


def validate_records(
    manifest: dict[str, Any],
    gate_spec: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    validate_manifest(manifest)
    validate_gate_spec(gate_spec)
    manifest_hash = canonical_sha256(manifest)
    if gate_spec["manifest_sha256"] != manifest_hash:
        raise SchemaError("gate spec was frozen from a different manifest")
    if not records:
        raise SchemaError("no validation records supplied")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str, str, int]] = set()
    for record in records:
        if record.get("split") != "validation":
            raise SchemaError("validator accepts only split='validation' records")
        if record.get("role") != "candidate":
            raise SchemaError("validator accepts only role='candidate' records")
        if record.get("manifest_sha256") != manifest_hash:
            raise SchemaError("validation record manifest hash mismatch")
        op, gate_id, candidate = (
            record.get("op"),
            record.get("gate_id"),
            record.get("candidate"),
        )
        gate_key = f"{op}/{gate_id}"
        if gate_key not in gate_spec["gates"]:
            raise SchemaError(f"no frozen threshold for {gate_key}")
        case_id, seed_index = record.get("case_id"), record.get("seed_index")
        if not isinstance(seed_index, int) or isinstance(seed_index, bool):
            raise SchemaError("seed_index must be an integer")
        expected_seeds = tensor_seeds(manifest, op, case_id, "validation", seed_index)
        if record.get("tensor_seeds") != expected_seeds:
            raise SchemaError(
                f"seed mismatch for {(op, gate_id, candidate, case_id, seed_index)}"
            )
        unique = (op, gate_id, candidate, case_id, seed_index)
        if unique in seen:
            raise SchemaError(f"duplicate validation record {unique}")
        seen.add(unique)
        groups[(op, gate_id, candidate)].append(record)

    summaries = []
    all_failures = []
    for (op, gate_id, candidate), rows in sorted(groups.items()):
        gate = gate_spec["gates"][f"{op}/{gate_id}"]
        expected = {
            (case_id, index)
            for case_id in gate["required_cases"]
            for index in range(gate["required_validation_seeds_per_case"])
        }
        observed = {(row["case_id"], row["seed_index"]) for row in rows}
        coverage_ok = allow_incomplete or observed == expected
        coverage_missing = sorted(expected - observed) if not allow_incomplete else []
        record_failures = 0
        for row in rows:
            reasons = []
            if not row.get("ok"):
                reasons.append(row.get("error", "collection failed"))
            else:
                metrics = row.get("metrics", {})
                for metric, threshold in gate["thresholds"].items():
                    value = metrics.get(metric)
                    if (
                        not isinstance(value, (int, float))
                        or not math.isfinite(value)
                    ):
                        reasons.append(f"{metric}=missing/nonfinite")
                    elif value > threshold["value"]:
                        reasons.append(
                            f"{metric}={value:.9g} > {threshold['value']:.9g}"
                        )
            if reasons:
                record_failures += 1
                all_failures.append(
                    {
                        "op": op,
                        "gate_id": gate_id,
                        "candidate": candidate,
                        "case_id": row.get("case_id"),
                        "seed_index": row.get("seed_index"),
                        "reasons": reasons,
                    }
                )
        success = coverage_ok and record_failures == 0
        summaries.append(
            {
                "op": op,
                "gate_id": gate_id,
                "candidate": candidate,
                "n_records": len(rows),
                "n_failed_records": record_failures,
                "coverage_complete": coverage_ok,
                "missing_records": len(coverage_missing),
                "success": success,
                "observed_failure_rate": record_failures / len(rows),
                "failure_rate_upper95": _failure_upper95(record_failures, len(rows)),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest_hash,
        "gate_spec_sha256": canonical_sha256(gate_spec),
        "success_rule": "all_metrics_all_cases_all_seeds_and_complete_coverage",
        "success": all(summary["success"] for summary in summaries),
        "groups": summaries,
        "failures": all_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gate-spec", required=True)
    parser.add_argument("--records", required=True, action="append")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = validate_records(
        load_json(args.manifest),
        load_json(args.gate_spec),
        [record for path in args.records for record in load_records(path)],
        allow_incomplete=args.allow_incomplete,
    )
    write_json(args.out, result)
    for group in result["groups"]:
        print(
            f"{group['op']}/{group['gate_id']} {group['candidate']}: "
            f"{'PASS' if group['success'] else 'FAIL'} "
            f"records={group['n_records']} failures={group['n_failed_records']} "
            f"coverage={group['coverage_complete']}"
        )
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
