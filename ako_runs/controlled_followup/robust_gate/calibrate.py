"""Freeze candidate-independent gate thresholds from calibration anchors."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
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


def round_up_125(value: float) -> float:
    """Round a finite non-negative number up to the 1/2/5 decimal series."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("value must be finite and non-negative")
    if value == 0:
        return 0.0
    exponent = math.floor(math.log10(value))
    unit = 10.0**exponent
    mantissa = value / unit
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if mantissa <= candidate:
            return candidate * unit
    raise AssertionError("unreachable")


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("op"),
        record.get("gate_id"),
        record.get("candidate"),
        record.get("case_id"),
        record.get("seed_index"),
    )


def calibrate_gate_spec(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    selected_ops: set[str] | None = None,
    selected_gates: set[str] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    validate_manifest(manifest)
    manifest_hash = canonical_sha256(manifest)
    if not records:
        raise SchemaError("no calibration records supplied")
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        if record.get("split") != "calibration":
            raise SchemaError("calibration accepts only split='calibration' records")
        if record.get("role") != "anchor":
            raise SchemaError("calibration accepts only role='anchor' records")
        if record.get("manifest_sha256") != manifest_hash:
            raise SchemaError("calibration record manifest hash mismatch")
        if not record.get("ok"):
            raise SchemaError(f"failed anchor record: {_record_key(record)}")
        key = _record_key(record)
        if key in seen:
            raise SchemaError(f"duplicate calibration record {key}")
        seen.add(key)
        for name, value in record.get("metrics", {}).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise SchemaError(f"record {key} has non-finite metric {name}")

    gates: dict[str, Any] = {}
    safety = float(manifest["calibration"]["safety_factor"])
    for op, op_spec in manifest["operations"].items():
        if selected_ops and op not in selected_ops:
            continue
        for gate_id, gate in op_spec["gates"].items():
            if selected_gates and gate_id not in selected_gates:
                continue
            rows = [
                row
                for row in records
                if row.get("op") == op and row.get("gate_id") == gate_id
            ]
            if not rows:
                continue
            expected_anchors = {f"anchor:{name}" for name in gate["anchors"]}
            observed_anchors = {row["candidate"] for row in rows}
            if observed_anchors != expected_anchors:
                raise SchemaError(
                    f"{op}/{gate_id}: anchors {sorted(observed_anchors)} do not match "
                    f"manifest {sorted(expected_anchors)}"
                )
            cases = [case["id"] for case in op_spec["cases"]]
            expected_indices = set(range(manifest["split_counts"]["calibration"]))
            if not allow_incomplete:
                expected = {
                    (anchor, case_id, index)
                    for anchor in expected_anchors
                    for case_id in cases
                    for index in expected_indices
                }
                observed = {
                    (row["candidate"], row["case_id"], row["seed_index"])
                    for row in rows
                }
                missing = expected - observed
                extra = observed - expected
                if missing or extra:
                    raise SchemaError(
                        f"{op}/{gate_id}: incomplete coverage "
                        f"missing={len(missing)} extra={len(extra)}"
                    )

            thresholds: dict[str, Any] = {}
            minimums = gate.get("minimum_thresholds", {})
            for metric in gate["calibrated_metrics"]:
                values = []
                for row in rows:
                    if metric not in row["metrics"]:
                        raise SchemaError(f"{op}/{gate_id}: missing metric {metric}")
                    value = float(row["metrics"][metric])
                    if value < 0 or not math.isfinite(value):
                        raise SchemaError(f"{op}/{gate_id}: invalid metric {metric}={value}")
                    values.append(value)
                observed_max = max(values)
                threshold = max(
                    round_up_125(observed_max * safety),
                    float(minimums.get(metric, 0.0)),
                )
                thresholds[metric] = {
                    "comparison": "le",
                    "value": threshold,
                    "source": "calibrated",
                    "observed_anchor_max": observed_max,
                    "safety_factor": safety,
                }
            for metric, value in gate["fixed_thresholds"].items():
                number = float(value)
                if not math.isfinite(number) or number < 0:
                    raise SchemaError(f"{op}/{gate_id}: invalid fixed threshold {metric}")
                thresholds[metric] = {
                    "comparison": "le",
                    "value": number,
                    "source": "fixed",
                }
            gates[f"{op}/{gate_id}"] = {
                "op": op,
                "gate_id": gate_id,
                "reference": gate["reference"],
                "contract": gate.get("contract", {}),
                "anchors": sorted(observed_anchors),
                "calibration_records": len(rows),
                "required_cases": cases,
                "required_validation_seeds_per_case": manifest["split_counts"]["validation"],
                "success_rule": "all_metrics_all_cases_all_seeds",
                "thresholds": thresholds,
            }
    if not gates:
        raise SchemaError("no selected gates had calibration records")
    sorted_records = sorted(records, key=_record_key)
    result = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_records_sha256": canonical_sha256(sorted_records),
        "calibration_policy": {
            "safety_factor": safety,
            "rounding": manifest["calibration"]["rounding"],
            "candidate_outputs_used": False,
        },
        "gates": gates,
    }
    validate_gate_spec(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--records", required=True, action="append")
    parser.add_argument("--op", action="append", default=[])
    parser.add_argument("--gate", action="append", default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = calibrate_gate_spec(
        load_json(args.manifest),
        [record for path in args.records for record in load_records(path)],
        selected_ops=set(args.op) or None,
        selected_gates=set(args.gate) or None,
        allow_incomplete=args.allow_incomplete,
    )
    write_json(args.out, spec)
    print(f"froze {len(spec['gates'])} gate(s) in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
