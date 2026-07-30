"""Adjudicate boundary controls and each winner/gate without threshold changes."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ako_runs.controlled_followup.robust_gate.schema import file_sha256

from .runner import (
    FREEZE_PATH,
    GATES,
    HERE,
    MANIFEST_PATH,
    _seed_map,
    threshold_failures,
    verify_campaign,
)


def _exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def analyze_records(
    manifest: dict[str, Any],
    gate_spec: dict[str, Any],
    boundary: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    *,
    source_bundle: str | None = None,
) -> dict[str, Any]:
    boundary_expected = {
        (control["control_id"], gate_id): control
        for control in manifest["boundary_controls"]
        for gate_id in GATES
    }
    boundary_seen = {}
    boundary_duplicates = 0
    for row in boundary:
        key = (row.get("candidate"), row.get("gate_id"))
        if key in boundary_seen:
            boundary_duplicates += 1
        else:
            boundary_seen[key] = row
    boundary_failures = []
    for key, control in boundary_expected.items():
        row = boundary_seen.get(key)
        if row is None:
            boundary_failures.append({"key": key, "reason": "missing"})
            continue
        expected_pass = control["expected_gate_outcome"] == "pass"
        expected_raw = control["expected_raw_safety_outcome"] == "exceed"
        frozen_gate = gate_spec["gates"][f"fused_softmax/{key[1]}"]
        recomputed = threshold_failures(frozen_gate, row.get("metrics", {}))
        sole_metric = (
            control["expected_gate_outcome"] != "reject"
            or row.get("failure_metrics") == ["row_sum_error_max"]
        )
        if (
            row.get("ok") is not True
            or row.get("gate_pass") is not expected_pass
            or row.get("raw_safety_exceeded") is not expected_raw
            or not sole_metric
            or row.get("threshold_failures") != recomputed
            or row.get("gate_pass") is not (not recomputed)
            or row.get("stress_manifest_sha256") != file_sha256(MANIFEST_PATH)
            or (
                source_bundle is not None
                and row.get("source_bundle_canonical_sha256") != source_bundle
            )
        ):
            boundary_failures.append({"key": key, "reason": "outcome mismatch", "row": row})

    expected_winner = {
        (winner["job_id"], gate_id, index): winner
        for winner in manifest["winners"]
        for gate_id in GATES
        for index in range(manifest["stress_split"]["seeds"])
    }
    winner_seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates = 0
    binding_failures = []
    for row in winners:
        key = (row.get("candidate"), row.get("gate_id"), row.get("seed_index"))
        if key in winner_seen:
            duplicates += 1
            continue
        winner_seen[key] = row
        expected = expected_winner.get(key)
        if expected is None:
            binding_failures.append({"key": key, "reason": "unexpected"})
            continue
        required = {
            "case_id": manifest["case"]["id"],
            "namespace": manifest["stress_split"]["namespace"],
            "tensor_seeds": _seed_map(
                manifest["stress_split"]["namespace"],
                manifest["case"]["id"],
                key[2],
            ),
            "job_sha256": expected["job_sha256"],
            "stress_manifest_sha256": file_sha256(MANIFEST_PATH),
            "original_gate_sha256": manifest["original_fused_v2"]["gate_spec"]["sha256"],
        }
        if source_bundle is not None:
            required["source_bundle_canonical_sha256"] = source_bundle
        mismatches = {
            name: {"expected": value, "observed": row.get(name)}
            for name, value in required.items()
            if row.get(name) != value
        }
        if mismatches:
            binding_failures.append({"key": key, "mismatches": mismatches})
        if row.get("ok") is True:
            frozen_gate = gate_spec["gates"][f"fused_softmax/{key[1]}"]
            recomputed = threshold_failures(frozen_gate, row.get("metrics", {}))
            if (
                row.get("threshold_failures") != recomputed
                or row.get("gate_pass") is not (not recomputed)
            ):
                binding_failures.append(
                    {
                        "key": key,
                        "reason": "fixed-threshold decision mismatch",
                        "expected_failures": recomputed,
                    }
                )

    missing = set(expected_winner) - set(winner_seen)
    groups = []
    for winner in manifest["winners"]:
        for gate_id in GATES:
            rows = [
                winner_seen[key]
                for key in expected_winner
                if key[0] == winner["job_id"] and key[1] == gate_id and key in winner_seen
            ]
            failures = [row for row in rows if not (row.get("ok") is True and row.get("gate_pass") is True)]
            raw_exceeds = sum(row.get("raw_safety_exceeded") is True for row in rows)
            maxima = [
                row["metrics"]["row_sum_error_max"]
                for row in rows
                if row.get("ok") is True and "metrics" in row
            ]
            groups.append(
                {
                    "candidate": winner["job_id"],
                    "dsl": winner["dsl"],
                    "gate_id": gate_id,
                    "expected_records": manifest["stress_split"]["seeds"],
                    "observed_records": len(rows),
                    "failed_registered_gate_records": len(failures),
                    "raw_safety_exceedances": raw_exceeds,
                    "row_sum_error_max": max(maxima) if maxima else None,
                    "registered_threshold": manifest["registered_row_sum_threshold"],
                    "success": len(rows) == manifest["stress_split"]["seeds"] and not failures,
                    "zero_failure_upper95": (
                        1.0 - 0.05 ** (1.0 / len(rows)) if rows and not failures else None
                    ),
                }
            )
    evidence_complete = (
        len(boundary_seen) == len(boundary_expected)
        and boundary_duplicates == 0
        and not missing
        and not duplicates
        and not binding_failures
    )
    return {
        "schema_version": "1.0",
        "campaign_id": manifest["campaign_id"],
        "evidence_complete": evidence_complete,
        "boundary_success": not boundary_failures and len(boundary_seen) == len(boundary_expected),
        "boundary_failures": boundary_failures,
        "winner_groups": groups,
        "all_winner_groups_success": all(group["success"] for group in groups),
        "coverage": {
            "winner_expected": len(expected_winner),
            "winner_observed_unique": len(winner_seen),
            "winner_missing": len(missing),
            "duplicates": duplicates,
            "boundary_duplicates": boundary_duplicates,
            "binding_failures": binding_failures[:100],
        },
        "threshold_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(HERE / "results" / "summary.json"))
    parser.add_argument("--receipt", default=str(HERE / "receipts" / "completion_receipt.json"))
    args = parser.parse_args()
    manifest, gate = verify_campaign(require_freeze=True)
    boundary_path = HERE / manifest["workloads"][0]["output"]
    winner_path = HERE / manifest["workloads"][1]["output"]
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    summary = analyze_records(
        manifest,
        gate,
        _load(boundary_path),
        _load(winner_path),
        source_bundle=freeze["source_bundle_canonical_sha256"],
    )
    summary.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_files": [
                {"path": str(path.relative_to(HERE)), "sha256": file_sha256(path)}
                for path in (boundary_path, winner_path)
            ],
        }
    )
    summary_path = Path(args.summary).resolve()
    _exclusive(summary_path, summary)
    receipt = {
        "schema_version": "1.0",
        "campaign_id": manifest["campaign_id"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "summary_sha256": file_sha256(summary_path),
        "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
        "evidence_complete": summary["evidence_complete"],
        "boundary_success": summary["boundary_success"],
        "all_winner_groups_success": summary["all_winner_groups_success"],
        "threshold_mutation_authorized": False,
    }
    _exclusive(Path(args.receipt).resolve(), receipt)
    print(summary)
    return 0 if (
        summary["evidence_complete"]
        and summary["boundary_success"]
        and summary["all_winner_groups_success"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
