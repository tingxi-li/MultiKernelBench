#!/usr/bin/env python3
"""Derive a validation-tagged robust summary without altering frozen results.

The historical validation aggregator omitted the top-level ``split`` field even
though every underlying record was a complete validation record.  This utility
accepts only that narrow compatibility case.  It streams and audits the JSONL
record set, verifies the PASS summary and adapter provenance, then writes a
derived summary whose sole semantic addition is ``split: validation`` and a
separate content-addressed receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_ADAPTER = HERE / "robust_adapter_manifest.json"
OPERATION = "fused_softmax"
VALIDATION_SPLIT = "validation"
HEX = set("0123456789abcdef")


class NormalizeError(ValueError):
    """The inputs do not prove the narrow missing-split compatibility case."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NormalizeError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def stable_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NormalizeError(f"value is not stable JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NormalizeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise NormalizeError(f"invalid JSON in {label}: {exc}") from exc


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NormalizeError(f"cannot read {path}: {exc}") from exc
    value = parse_json(raw, str(path))
    if not isinstance(value, dict):
        raise NormalizeError(f"{path} must contain a JSON object")
    return value, raw


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= HEX:
        raise NormalizeError(f"{label} must be a lowercase SHA256")
    return value


def require_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise NormalizeError(f"{label} must be an integer")
    return value


def same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise NormalizeError(
            f"{label} mismatch: got {actual!r}, expected {expected!r}"
        )


def relative_name(path: Path, receipt_parent: Path) -> str:
    return os.path.relpath(path.resolve(), receipt_parent.resolve())


def audit_adapter(adapter_path: Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    adapter, raw = read_json(adapter_path)
    same(adapter.get("schema_version"), 1, "adapter schema_version")
    same(adapter.get("operation"), OPERATION, "adapter operation")
    grid = adapter.get("grid")
    robust = adapter.get("robust_gate")
    sources = adapter.get("source_sha256")
    if not isinstance(grid, dict) or not isinstance(robust, dict):
        raise NormalizeError("adapter grid and robust_gate must be objects")
    if not isinstance(sources, dict) or not sources:
        raise NormalizeError("adapter source_sha256 must be a non-empty object")
    for path, digest in sources.items():
        if not isinstance(path, str) or not path:
            raise NormalizeError("adapter source path must be non-empty")
        require_sha(digest, f"adapter source {path}")
    same(
        canonical_sha256(sources),
        require_sha(adapter.get("source_bundle_sha256"), "adapter source bundle"),
        "adapter source bundle",
    )

    job_hashes = grid.get("job_sha256")
    if not isinstance(job_hashes, dict) or not job_hashes:
        raise NormalizeError("adapter grid.job_sha256 must be non-empty")
    for job_id, digest in job_hashes.items():
        if not isinstance(job_id, str) or not job_id:
            raise NormalizeError("adapter grid job ID must be non-empty")
        require_sha(digest, f"adapter job {job_id}")
    same(len(job_hashes), require_int(grid.get("job_count"), "grid job_count"),
         "adapter grid job count")

    cases = robust.get("case_ids")
    gate_keys = robust.get("gate_keys")
    splits = robust.get("split_counts")
    if (not isinstance(cases, list) or not cases or
            any(not isinstance(case, str) or not case for case in cases) or
            len(cases) != len(set(cases))):
        raise NormalizeError("adapter case_ids must be unique non-empty strings")
    if (not isinstance(gate_keys, list) or not gate_keys or
            any(not isinstance(key, str) or not key.startswith(f"{OPERATION}/")
                for key in gate_keys) or len(gate_keys) != len(set(gate_keys))):
        raise NormalizeError("adapter gate_keys are malformed or duplicated")
    if not isinstance(splits, dict):
        raise NormalizeError("adapter split_counts must be an object")
    validation_seeds = require_int(splits.get(VALIDATION_SPLIT),
                                   "validation seed count")
    if validation_seeds <= 0:
        raise NormalizeError("validation seed count must be positive")
    contract = {
        "adapter_sha256": sha256_bytes(raw),
        "campaign_id": robust.get("campaign_id"),
        "manifest_sha256": require_sha(
            robust.get("manifest_canonical_sha256"), "robust manifest canonical hash"
        ),
        "grid_manifest_sha256": require_sha(
            grid.get("manifest_sha256"), "grid manifest hash"
        ),
        "grid_jobs_sha256": require_sha(grid.get("jobs_sha256"), "grid jobs hash"),
        "source_bundle_sha256": adapter["source_bundle_sha256"],
        "gate_spec_sha256": require_sha(
            robust.get("gate_spec_sha256"), "gate spec hash"
        ),
        "gate_spec_canonical_sha256": require_sha(
            robust.get("gate_spec_canonical_sha256"), "gate spec canonical hash"
        ),
        "case_ids": tuple(cases),
        "gate_ids": tuple(key.split("/", 1)[1] for key in gate_keys),
        "validation_seeds": validation_seeds,
        "job_hashes": job_hashes,
    }
    if not isinstance(contract["campaign_id"], str) or not contract["campaign_id"]:
        raise NormalizeError("adapter robust campaign_id must be non-empty")
    return adapter, raw, contract


def audit_summary(
    summary: dict[str, Any], contract: dict[str, Any]
) -> tuple[dict[tuple[str, str], dict[str, Any]], set[str], int]:
    if "split" in summary:
        raise NormalizeError(
            "source summary already has a split field; compatibility normalization refused"
        )
    same(summary.get("schema_version"), "1.0", "summary schema_version")
    same(summary.get("campaign_id"), contract["campaign_id"], "summary campaign_id")
    same(summary.get("manifest_sha256"), contract["manifest_sha256"],
         "summary manifest hash")
    provenance = {
        "adapter_manifest_sha256": contract["adapter_sha256"],
        "source_bundle_sha256": contract["source_bundle_sha256"],
        "grid_manifest_sha256": contract["grid_manifest_sha256"],
        "grid_jobs_sha256": contract["grid_jobs_sha256"],
        "robust_manifest_sha256": contract["manifest_sha256"],
        "gate_spec_sha256": contract["gate_spec_sha256"],
        "gate_spec_canonical_sha256": contract["gate_spec_canonical_sha256"],
    }
    for name, expected in provenance.items():
        same(summary.get(name), expected, f"summary {name}")
    same(summary.get("status"), "PASS", "summary status")
    same(summary.get("success"), True, "summary success")
    same(summary.get("failures"), [], "summary failures")
    same(summary.get("absent_candidates"), [], "summary absent_candidates")

    groups = summary.get("groups")
    if not isinstance(groups, list) or not groups:
        raise NormalizeError("summary groups must be a non-empty list")
    group_map: dict[tuple[str, str], dict[str, Any]] = {}
    candidates: set[str] = set()
    per_group = len(contract["case_ids"]) * contract["validation_seeds"]
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise NormalizeError(f"summary group {index} must be an object")
        same(group.get("op"), OPERATION, f"group {index} op")
        job_id = group.get("grid_job_id")
        if job_id not in contract["job_hashes"]:
            raise NormalizeError(f"group {index} has unknown grid job {job_id!r}")
        job_hash = contract["job_hashes"][job_id]
        same(group.get("grid_job_sha256"), job_hash,
             f"group {index} grid job hash")
        candidate = f"fused-grid:{job_id}:{job_hash[:12]}"
        same(group.get("candidate"), candidate, f"group {index} candidate")
        gate_id = group.get("gate_id")
        if gate_id not in contract["gate_ids"]:
            raise NormalizeError(f"group {index} has unknown gate {gate_id!r}")
        key = (candidate, gate_id)
        if key in group_map:
            raise NormalizeError(f"duplicate summary group {key}")
        same(group.get("n_records"), per_group, f"group {index} record count")
        same(group.get("n_failed_records"), 0, f"group {index} failed count")
        same(group.get("missing_records"), 0, f"group {index} missing count")
        same(group.get("coverage_complete"), True, f"group {index} coverage")
        same(group.get("success"), True, f"group {index} success")
        failure_rate = group.get("observed_failure_rate")
        if (isinstance(failure_rate, bool) or
                not isinstance(failure_rate, (int, float)) or
                not math.isfinite(float(failure_rate)) or float(failure_rate) != 0.0):
            raise NormalizeError(f"group {index} failure rate must be finite zero")
        group_map[key] = group
        candidates.add(candidate)

    expected_group_keys = {
        (candidate, gate_id)
        for candidate in candidates
        for gate_id in contract["gate_ids"]
    }
    same(set(group_map), expected_group_keys, "summary candidate/gate Cartesian coverage")
    expected_records = len(candidates) * len(contract["gate_ids"]) * per_group
    coverage = summary.get("launch_coverage")
    if not isinstance(coverage, dict):
        raise NormalizeError("summary launch_coverage must be an object")
    same(coverage.get("complete"), True, "launch coverage complete")
    same(coverage.get("expected_records"), expected_records,
         "launch expected records")
    same(coverage.get("observed_records"), expected_records,
         "launch observed records")
    for name in ("missing_records", "unexpected_records", "duplicate_records"):
        same(coverage.get(name), 0, f"launch {name}")
    for name in ("missing_examples", "unexpected_examples"):
        same(coverage.get(name), [], f"launch {name}")
    return group_map, candidates, expected_records


def audit_records(
    records_path: Path,
    contract: dict[str, Any],
    candidates: set[str],
    expected_count: int,
) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    seen: set[tuple[str, str, str, int]] = set()
    line_count = 0
    try:
        stream = records_path.open("rb")
    except OSError as exc:
        raise NormalizeError(f"cannot read {records_path}: {exc}") from exc
    with stream:
        for line_count, raw_line in enumerate(stream, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                raise NormalizeError(f"records line {line_count} is blank")
            row = parse_json(raw_line, f"{records_path}:{line_count}")
            if not isinstance(row, dict):
                raise NormalizeError(f"records line {line_count} is not an object")
            expected_fields = {
                "schema_version": "1.0",
                "record_type": "robust_gate_measurement",
                "campaign_id": contract["campaign_id"],
                "manifest_sha256": contract["manifest_sha256"],
                "op": OPERATION,
                "split": VALIDATION_SPLIT,
                "role": "candidate",
                "adapter_manifest_sha256": contract["adapter_sha256"],
                "source_bundle_sha256": contract["source_bundle_sha256"],
                "source_sha256": contract["source_bundle_sha256"],
                "grid_manifest_sha256": contract["grid_manifest_sha256"],
                "grid_jobs_sha256": contract["grid_jobs_sha256"],
                "gate_spec_sha256": contract["gate_spec_sha256"],
                "gate_spec_canonical_sha256": contract[
                    "gate_spec_canonical_sha256"
                ],
            }
            for name, expected in expected_fields.items():
                same(row.get(name), expected, f"record {line_count} {name}")
            candidate = row.get("candidate")
            if candidate not in candidates:
                raise NormalizeError(
                    f"record {line_count} has unexpected candidate {candidate!r}"
                )
            job_id = row.get("grid_job_id")
            if job_id not in contract["job_hashes"]:
                raise NormalizeError(f"record {line_count} has unknown job {job_id!r}")
            job_hash = contract["job_hashes"][job_id]
            same(row.get("grid_job_sha256"), job_hash,
                 f"record {line_count} grid job hash")
            same(candidate, f"fused-grid:{job_id}:{job_hash[:12]}",
                 f"record {line_count} candidate binding")
            embedded_job = row.get("grid_job")
            if not isinstance(embedded_job, dict):
                raise NormalizeError(f"record {line_count} lacks embedded grid job")
            same(canonical_sha256(embedded_job), job_hash,
                 f"record {line_count} embedded grid job hash")
            gate_id = row.get("gate_id")
            case_id = row.get("case_id")
            seed_index = row.get("seed_index")
            if gate_id not in contract["gate_ids"]:
                raise NormalizeError(f"record {line_count} has unknown gate {gate_id!r}")
            if case_id not in contract["case_ids"]:
                raise NormalizeError(f"record {line_count} has unknown case {case_id!r}")
            seed_index = require_int(seed_index, f"record {line_count} seed_index")
            if not 0 <= seed_index < contract["validation_seeds"]:
                raise NormalizeError(f"record {line_count} seed_index is out of range")
            key = (candidate, gate_id, case_id, seed_index)
            if key in seen:
                raise NormalizeError(f"duplicate validation record {key}")
            seen.add(key)
            same(row.get("ok"), True, f"record {line_count} ok")
            same(row.get("gate_pass"), True, f"record {line_count} gate_pass")
            same(row.get("threshold_failures"), [],
                 f"record {line_count} threshold_failures")

    same(line_count, expected_count, "records line count")
    expected_keys = {
        (candidate, gate_id, case_id, seed_index)
        for candidate in candidates
        for gate_id in contract["gate_ids"]
        for case_id in contract["case_ids"]
        for seed_index in range(contract["validation_seeds"])
    }
    same(seen, expected_keys, "record Cartesian coverage")
    try:
        size = records_path.stat().st_size
    except OSError as exc:
        raise NormalizeError(f"cannot stat {records_path}: {exc}") from exc
    return digest.hexdigest(), line_count, size


def normalize(
    summary_path: Path,
    records_path: Path,
    adapter_path: Path,
    *,
    receipt_parent: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _adapter, adapter_raw, contract = audit_adapter(adapter_path)
    summary, summary_raw = read_json(summary_path)
    _groups, candidates, expected_count = audit_summary(summary, contract)
    records_hash, line_count, records_size = audit_records(
        records_path, contract, candidates, expected_count
    )

    derived = copy.deepcopy(summary)
    derived["split"] = VALIDATION_SPLIT
    proof = copy.deepcopy(derived)
    same(proof.pop("split"), VALIDATION_SPLIT, "derived split")
    same(proof, summary, "derived summary one-field patch")
    derived_raw = stable_bytes(derived)
    receipt = {
        "schema_version": 1,
        "receipt_type": "validation_summary_split_normalization",
        "normalization_patch": [
            {"op": "add", "path": "/split", "value": VALIDATION_SPLIT}
        ],
        "source_summary": {
            "path": relative_name(summary_path, receipt_parent),
            "sha256": sha256_bytes(summary_raw),
            "size_bytes": len(summary_raw),
        },
        "source_records": {
            "path": relative_name(records_path, receipt_parent),
            "sha256": records_hash,
            "size_bytes": records_size,
            "line_count": line_count,
        },
        "adapter_manifest": {
            "path": relative_name(adapter_path, receipt_parent),
            "sha256": sha256_bytes(adapter_raw),
        },
        "derived_summary": {
            "sha256": sha256_bytes(derived_raw),
            "size_bytes": len(derived_raw),
        },
        "verification": {
            "split": VALIDATION_SPLIT,
            "candidate_count": len(candidates),
            "gate_count": len(contract["gate_ids"]),
            "case_count": len(contract["case_ids"]),
            "validation_seeds_per_case": contract["validation_seeds"],
            "group_count": len(candidates) * len(contract["gate_ids"]),
            "expected_records": expected_count,
            "observed_records": line_count,
            "all_groups_complete_and_pass": True,
            "all_records_execution_and_gate_pass": True,
        },
    }
    return derived, receipt


def atomic_write_exact(path: Path, data: bytes) -> str:
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise NormalizeError(f"cannot read existing output {path}: {exc}") from exc
        if existing != data:
            raise NormalizeError(f"refusing to replace differing output {path}")
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return "wrote"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--receipt-out", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    records_path = args.records.resolve()
    adapter_path = args.adapter_manifest.resolve()
    out = (args.out or summary_path.with_name("summary.validation.json")).resolve()
    receipt_out = (
        args.receipt_out
        or summary_path.with_name("summary.validation.receipt.json")
    ).resolve()
    protected = {summary_path, records_path, adapter_path}
    if out in protected or receipt_out in protected or out == receipt_out:
        parser.error("derived summary and receipt must be distinct from all inputs")
    try:
        derived, receipt = normalize(
            summary_path,
            records_path,
            adapter_path,
            receipt_parent=receipt_out.parent,
        )
        if args.check_only:
            print(
                f"OK validation artifact: groups={receipt['verification']['group_count']} "
                f"records={receipt['verification']['observed_records']} "
                f"source_summary_sha256={receipt['source_summary']['sha256']}"
            )
            return 0
        derived_state = atomic_write_exact(out, stable_bytes(derived))
        receipt["derived_summary"]["path"] = relative_name(out, receipt_out.parent)
        receipt_state = atomic_write_exact(receipt_out, stable_bytes(receipt))
        print(
            f"{derived_state} {out}; {receipt_state} {receipt_out}; "
            f"records_sha256={receipt['source_records']['sha256']}"
        )
        return 0
    except NormalizeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
