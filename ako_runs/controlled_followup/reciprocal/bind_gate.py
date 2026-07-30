#!/usr/bin/env python3
"""Bind an externally validated robust gate to this campaign.

This is an explicit post-calibration action.  It refuses moving/example hashes
and requires the SHA256 printed by the robust-gate validator, preventing mere
presence of a gate-spec file from silently unblocking the campaign.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import validate


def build_lock(
    gate_spec_path: Path,
    validated_sha256: str,
    validation_summary_path: Path,
    acceptance_receipt_path: Path,
) -> dict[str, Any]:
    if not validate._is_hash(validated_sha256, nonzero=True):
        raise validate.ValidationError("--validated-sha256 must be a nonzero SHA256")
    if gate_spec_path.resolve() != validate.GATE_SPEC:
        raise validate.ValidationError("--gate-spec must be the accepted v4 gate path")
    if validation_summary_path.resolve() != validate.VALIDATION_SUMMARY:
        raise validate.ValidationError(
            "--validation-summary must be the accepted v4 holdout-summary path"
        )
    if acceptance_receipt_path.resolve() != validate.ACCEPTANCE_RECEIPT:
        raise validate.ValidationError(
            "--acceptance-receipt must be the accepted v4 receipt path"
        )
    spec = validate.load_json(gate_spec_path)
    summary = validate.load_json(validation_summary_path)
    receipt = validate.load_json(acceptance_receipt_path)
    canonical = validate.canonical_sha256(spec)
    if canonical != validated_sha256:
        raise validate.ValidationError(
            "validated SHA256 does not match the robust validator's canonical gate hash"
        )
    file_hash = validate.sha256_file(gate_spec_path)
    if spec.get("schema_version") != "1.0":
        raise validate.ValidationError("gate spec must use schema_version='1.0'")
    for field in ("manifest_sha256", "calibration_records_sha256"):
        if not validate._is_hash(spec.get(field), nonzero=True):
            raise validate.ValidationError(f"gate spec lacks a frozen {field}")
    if not isinstance(spec.get("campaign_id"), str) or not spec["campaign_id"]:
        raise validate.ValidationError("gate spec lacks campaign_id")
    acceptance_blockers = validate.gate_acceptance_blockers(spec, summary)
    if acceptance_blockers:
        raise validate.ValidationError(
            "validation summary is not accepted: " + "; ".join(acceptance_blockers)
        )
    receipt_blockers = validate.acceptance_receipt_blockers(spec, summary, receipt)
    if receipt_blockers:
        raise validate.ValidationError(
            "acceptance receipt is invalid: " + "; ".join(receipt_blockers)
        )
    gates = spec.get("gates")
    matmul_gates = [
        gate
        for gate in gates.values()
        if isinstance(gate, dict) and gate.get("op") == "matmul"
    ] if isinstance(gates, dict) else []
    if not matmul_gates:
        raise validate.ValidationError("gate spec contains no matmul gate")
    for gate in matmul_gates:
        expected = (
            len(gate.get("anchors", []))
            * len(gate.get("required_cases", []))
            * validate.REQUIRED_GATE_CALIBRATION_SEEDS
        )
        if expected <= 0 or gate.get("calibration_records") != expected:
            raise validate.ValidationError(
                f"matmul/{gate.get('gate_id', '?')} is not a full "
                f"{validate.REQUIRED_GATE_CALIBRATION_SEEDS}-seed calibration"
            )
        if (
            gate.get("required_validation_seeds_per_case")
            != validate.REQUIRED_GATE_VALIDATION_SEEDS
        ):
            raise validate.ValidationError(
                f"matmul/{gate.get('gate_id', '?')} does not require "
                f"{validate.REQUIRED_GATE_VALIDATION_SEEDS} validation seeds"
            )
    return {
        "schema_version": 1,
        "state": "frozen",
        "operation": "matmul",
        "gate_spec": validate.make_manifests.GATE_SPEC_RELATIVE,
        "validation_summary": (
            validate.make_manifests.GATE_VALIDATION_SUMMARY_RELATIVE
        ),
        "acceptance_receipt": (
            validate.make_manifests.GATE_ACCEPTANCE_RECEIPT_RELATIVE
        ),
        "campaign_id": spec["campaign_id"],
        "manifest_sha256": spec["manifest_sha256"],
        "calibration_records_sha256": spec["calibration_records_sha256"],
        "gate_spec_sha256": canonical,
        "gate_spec_file_sha256": file_hash,
        "validation_summary_sha256": validate.sha256_file(validation_summary_path),
        "acceptance_receipt_sha256": validate.sha256_file(acceptance_receipt_path),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-spec", type=Path, default=validate.GATE_SPEC)
    parser.add_argument(
        "--validation-summary", type=Path, default=validate.VALIDATION_SUMMARY
    )
    parser.add_argument(
        "--acceptance-receipt", type=Path, default=validate.ACCEPTANCE_RECEIPT
    )
    parser.add_argument("--out", type=Path, default=validate.GATE_LOCK)
    parser.add_argument(
        "--validated-sha256",
        required=True,
        help="exact gate_spec_sha256 emitted by robust_gate/validate.py",
    )
    args = parser.parse_args()
    lock = build_lock(
        args.gate_spec.resolve(),
        args.validated_sha256,
        args.validation_summary.resolve(),
        args.acceptance_receipt.resolve(),
    )
    atomic_json(args.out.resolve(), lock)
    print(
        f"bound robust gate campaign={lock['campaign_id']} "
        f"gate_spec_sha256={lock['gate_spec_sha256']} to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
