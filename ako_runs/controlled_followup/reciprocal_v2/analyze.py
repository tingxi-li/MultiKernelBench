#!/usr/bin/env python3
"""Validate and summarize a completed reciprocal-v2 primary campaign."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import make_manifests
import protocol
import validate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            rows.append(value)
    return rows


def _sign_test_p(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, n - positives)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def _holm(rows: list[dict[str, Any]], alpha: float = 0.05) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_raw"])
    running = 0.0
    m = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * row["p_raw"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running
        rows[index]["reject_holm_0_05"] = running <= alpha


def selection_bindings(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    blockers = validate.dependency_blockers("primary")
    if blockers:
        raise ValueError("primary dependencies are not launch-complete: " + "; ".join(blockers[:10]))
    registry = {
        row["cell_id"]: row
        for row in protocol.load_json(protocol.IMPLEMENTATION_REGISTRY)["implementations"]
    }
    bindings = {}
    for job in manifest["jobs"]:
        path = protocol.HERE / "results" / "selection" / "receipts" / f"{job['cell_id']}.json"
        selection = protocol.load_json(path)
        bindings[job["cell_id"]] = {
            "selection_receipt_sha256": protocol.file_sha256(path),
            "selected_attempt": selection["selected_attempt"],
            "candidate_source_sha256": registry[job["cell_id"]]["sha256"],
        }
    return bindings


def validate_records(
    rows: list[dict[str, Any]],
    bindings: dict[str, dict[str, Any]] | None = None,
    common_bindings: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    manifest = validate.validate_static()["manifests"]["primary"]
    jobs = {row["cell_id"]: row for row in manifest["jobs"]}
    bindings = bindings if bindings is not None else selection_bindings(manifest)
    if set(bindings) != set(jobs):
        raise ValueError("selection binding census differs from 48 primary cells")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    process_ids: set[str] = set()
    expected_sequence = [
        (cid, block, position)
        for block, order in enumerate(protocol.block_orders())
        for position, cid in enumerate(order)
    ]
    observed_sequence = []
    common = common_bindings or {
        "primary_manifest_sha256": protocol.file_sha256(make_manifests.PRIMARY_MANIFEST),
        "source_freeze_sha256": protocol.file_sha256(protocol.SOURCE_FREEZE),
        "implementation_registry_sha256": protocol.file_sha256(protocol.IMPLEMENTATION_REGISTRY),
        "recipe_resolution_lock_sha256": protocol.file_sha256(protocol.RESOLUTION_LOCK),
        "gate_lock_sha256": protocol.file_sha256(protocol.GATE_LOCK),
        "block_order_sha256": protocol.block_order_sha256(),
        "primary_launch_receipt_sha256": protocol.file_sha256(protocol.HERE / "results/primary/launch_receipt.json"),
    }
    for row in rows:
        cid = row.get("cell_id")
        block = row.get("block")
        if cid not in jobs:
            raise ValueError(f"unknown cell: {cid!r}")
        expected = jobs[cid]
        for key in ("recipe_origin", "destination_dsl", "transfer_mode", "translator"):
            if row.get(key) != expected[key]:
                raise ValueError(f"{cid}: {key} does not match manifest")
        if row.get("campaign_id") != protocol.CAMPAIGN_ID:
            raise ValueError(f"{cid}: campaign mismatch")
        for key, value in common.items():
            if row.get(key) != value:
                raise ValueError(f"{cid}: {key} binding mismatch")
        binding = bindings[cid]
        for key in ("selection_receipt_sha256", "selected_attempt", "candidate_source_sha256"):
            if row.get(key) != binding[key]:
                raise ValueError(f"{cid}: {key} does not bind frozen selection")
        if not isinstance(block, int) or isinstance(block, bool) or block not in range(15):
            raise ValueError(f"{cid}: invalid block")
        position = row.get("block_position")
        if not isinstance(position, int) or isinstance(position, bool) or position not in range(48):
            raise ValueError(f"{cid}: invalid block position")
        if protocol.block_orders()[block][position] != cid:
            raise ValueError(f"{cid}: record violates frozen randomized block order")
        observed_sequence.append((cid, block, position))
        identity = (cid, block)
        if identity in seen:
            raise ValueError(f"duplicate record {identity}")
        seen.add(identity)
        if row.get("gate_pass") is not True or row.get("terminal_eligible") is not True:
            raise ValueError(f"{cid}: timed record is not terminal-gate eligible")
        latency = row.get("median_ms")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or latency <= 0
        ):
            raise ValueError(f"{cid}: invalid latency")
        if row.get("physical_gpu") != protocol.TIMING_PHYSICAL_GPU or row.get("logical_device") != "cuda:0":
            raise ValueError(f"{cid}: confirmation was not pinned to GPU 0")
        if row.get("timing_distribution") != "rand_seed0_precast":
            raise ValueError(f"{cid}: timing distribution drift")
        if row.get("gpu_uuid") != protocol.TIMING_GPU_UUID:
            raise ValueError(f"{cid}: timing GPU UUID mismatch")
        process_id = row.get("process_instance_id")
        if not isinstance(process_id, str) or not process_id or process_id in process_ids:
            raise ValueError(f"{cid}: missing or reused process instance ID")
        process_ids.add(process_id)
        grouped[cid].append(row)
    expected_pairs = {(cid, block) for cid in jobs for block in range(15)}
    if seen != expected_pairs:
        raise ValueError(
            f"incomplete record set: missing={len(expected_pairs-seen)}, "
            f"unexpected={len(seen-expected_pairs)}"
        )
    if observed_sequence != expected_sequence:
        raise ValueError("raw stream is not serialized in frozen randomized complete-block order")
    for cid in grouped:
        grouped[cid].sort(key=lambda row: row["block"])
    return grouped


def summarize(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    manifest = protocol.load_json(make_manifests.PRIMARY_MANIFEST)
    jobs = {row["cell_id"]: row for row in manifest["jobs"]}
    cells = []
    values: dict[tuple[str, str, str, str], list[float]] = {}
    for cid, job in jobs.items():
        samples = [float(row["median_ms"]) for row in grouped[cid]]
        ordered = sorted(samples)
        key = (
            job["recipe_origin"],
            job["destination_dsl"],
            job["transfer_mode"],
            job["translator"],
        )
        values[key] = samples
        cells.append(
            {
                "cell_id": cid,
                "recipe_origin": key[0],
                "destination_dsl": key[1],
                "transfer_mode": key[2],
                "translator": key[3],
                "n": 15,
                "median_ms": statistics.median(samples),
                "median_order_statistic_interval_ms": [ordered[3], ordered[11]],
                "interval_coverage": 0.96484375,
            }
        )

    translator_bounds = []
    for origin, destination, mode in itertools.product(
        protocol.ORIGINS, protocol.DESTINATIONS, protocol.TRANSFER_MODES
    ):
        paired_ratios = [
            a / b
            for a, b in zip(
                values[(origin, destination, mode, "translator_a")],
                values[(origin, destination, mode, "translator_b")],
                strict=True,
            )
        ]
        ordered_ratios = sorted(paired_ratios)
        a = statistics.median(values[(origin, destination, mode, "translator_a")])
        b = statistics.median(values[(origin, destination, mode, "translator_b")])
        translator_bounds.append(
            {
                "recipe_origin": origin,
                "destination_dsl": destination,
                "transfer_mode": mode,
                "translator_a_ms": a,
                "translator_b_ms": b,
                "translator_a_over_b_paired_median": statistics.median(paired_ratios),
                "translator_a_over_b_order_statistic_interval": [ordered_ratios[3], ordered_ratios[11]],
                "slow_over_fast_cell_median_ratio": max(a, b) / min(a, b),
                "absolute_log_ratio": abs(math.log(a / b)),
            }
        )

    interactions = []
    for mode, translator in itertools.product(
        protocol.TRANSFER_MODES, protocol.TRANSLATORS
    ):
        table = {
            (origin, destination): math.log(
                statistics.median(values[(origin, destination, mode, translator)])
            )
            for origin in protocol.ORIGINS
            for destination in protocol.DESTINATIONS
        }
        grand = statistics.fmean(table.values())
        origin_means = {
            origin: statistics.fmean(table[(origin, dest)] for dest in protocol.DESTINATIONS)
            for origin in protocol.ORIGINS
        }
        destination_means = {
            dest: statistics.fmean(table[(origin, dest)] for origin in protocol.ORIGINS)
            for dest in protocol.DESTINATIONS
        }
        residuals = {
            f"{origin}__{dest}": table[(origin, dest)]
            - origin_means[origin]
            - destination_means[dest]
            + grand
            for origin in protocol.ORIGINS
            for dest in protocol.DESTINATIONS
        }
        interactions.append(
            {
                "transfer_mode": mode,
                "translator": translator,
                "log_interaction_residuals": residuals,
                "interaction_range_log_ms": max(residuals.values())
                - min(residuals.values()),
                "interpretation": "descriptive; nonzero values quantify recipe anchoring",
            }
        )

    interaction_contrasts = []
    for mode, translator, (origin_a, origin_b), (dest_a, dest_b) in itertools.product(
        protocol.TRANSFER_MODES,
        protocol.TRANSLATORS,
        itertools.combinations(protocol.ORIGINS, 2),
        itertools.combinations(protocol.DESTINATIONS, 2),
    ):
        per_block = [
            math.log(values[(origin_a, dest_a, mode, translator)][block])
            - math.log(values[(origin_a, dest_b, mode, translator)][block])
            - math.log(values[(origin_b, dest_a, mode, translator)][block])
            + math.log(values[(origin_b, dest_b, mode, translator)][block])
            for block in range(protocol.CONFIRM_REPS)
        ]
        interaction_contrasts.append(
            {
                "transfer_mode": mode,
                "translator": translator,
                "origin_a": origin_a,
                "origin_b": origin_b,
                "destination_a": dest_a,
                "destination_b": dest_b,
                "median_log_difference_in_differences": statistics.median(per_block),
                "median_ratio_of_destination_ratios": math.exp(statistics.median(per_block)),
                "p_raw": _sign_test_p(per_block),
            }
        )
    _holm(interaction_contrasts)

    contrasts = []
    for origin, mode, translator in itertools.product(
        protocol.ORIGINS, protocol.TRANSFER_MODES, protocol.TRANSLATORS
    ):
        for left, right in itertools.combinations(protocol.DESTINATIONS, 2):
            left_values = values[(origin, left, mode, translator)]
            right_values = values[(origin, right, mode, translator)]
            log_ratios = [math.log(a / b) for a, b in zip(left_values, right_values)]
            contrasts.append(
                {
                    "recipe_origin": origin,
                    "transfer_mode": mode,
                    "translator": translator,
                    "left": left,
                    "right": right,
                    "median_left_over_right": statistics.median(
                        a / b for a, b in zip(left_values, right_values)
                    ),
                    "p_raw": _sign_test_p(log_ratios),
                }
            )
    _holm(contrasts)
    persistent_destination_effects = []
    for left, right in itertools.combinations(protocol.DESTINATIONS, 2):
        directions = []
        for origin, mode, translator in itertools.product(
            protocol.ORIGINS, protocol.TRANSFER_MODES, protocol.TRANSLATORS
        ):
            ratio = statistics.median(values[(origin, left, mode, translator)]) / statistics.median(values[(origin, right, mode, translator)])
            directions.append(-1 if ratio < 1 else (1 if ratio > 1 else 0))
        nonzero = [direction for direction in directions if direction]
        persistent_destination_effects.append(
            {
                "left": left,
                "right": right,
                "comparisons": len(directions),
                "direction_consistent_across_all_origins_modes_translators": len(nonzero) == len(directions) and len(set(nonzero)) == 1,
                "direction": "left_faster" if nonzero and set(nonzero) == {-1} else ("right_faster" if nonzero and set(nonzero) == {1} else "not_persistent"),
            }
        )
    return {
        "schema_version": 2,
        "record_type": "reciprocal_v2_primary_analysis",
        "campaign_id": protocol.CAMPAIGN_ID,
        "complete": True,
        "cell_count": len(cells),
        "record_count": sum(len(rows) for rows in grouped.values()),
        "cells": cells,
        "translator_bounds": translator_bounds,
        "origin_destination_interactions": interactions,
        "paired_origin_destination_interaction_contrasts": interaction_contrasts,
        "paired_destination_contrasts": contrasts,
        "persistent_destination_effects": persistent_destination_effects,
        "multiple_testing": "Holm across all 72 preregistered destination contrasts",
        "interaction_multiple_testing": "Holm across all 72 preregistered pairwise origin-by-destination interaction contrasts",
        "block_order_sha256": protocol.block_order_sha256(),
        "scope": "one workload, one Ada GPU, frozen finite recipe and retune budgets",
    }


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite of {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=protocol.PRIMARY_RAW)
    parser.add_argument("--out", type=Path, default=protocol.PRIMARY_ANALYSIS)
    parser.add_argument("--receipt", type=Path, default=protocol.PRIMARY_COMPLETION)
    args = parser.parse_args()
    grouped = validate_records(_read_jsonl(args.records))
    result = summarize(grouped)
    result.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_path": protocol.repo_path(args.records.resolve()),
            "raw_sha256": protocol.file_sha256(args.records),
            "source_freeze_sha256": protocol.file_sha256(protocol.SOURCE_FREEZE),
            "primary_manifest_sha256": protocol.file_sha256(make_manifests.PRIMARY_MANIFEST),
            "implementation_registry_sha256": protocol.file_sha256(protocol.IMPLEMENTATION_REGISTRY),
            "recipe_resolution_lock_sha256": protocol.file_sha256(protocol.RESOLUTION_LOCK),
            "gate_lock_sha256": protocol.file_sha256(protocol.GATE_LOCK),
            "primary_launch_receipt_sha256": protocol.file_sha256(protocol.HERE / "results/primary/launch_receipt.json"),
        }
    )
    _exclusive_json(args.out, result)
    receipt = {
        "schema_version": 2,
        "record_type": "reciprocal_v2_primary_completion_receipt",
        "campaign_id": protocol.CAMPAIGN_ID,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": protocol.repo_path(args.records.resolve()),
        "raw_sha256": protocol.file_sha256(args.records),
        "analysis_path": protocol.repo_path(args.out.resolve()),
        "analysis_sha256": protocol.file_sha256(args.out),
        "record_count": result["record_count"],
        "cell_count": result["cell_count"],
        "complete": result["complete"],
        "source_freeze_sha256": protocol.file_sha256(protocol.SOURCE_FREEZE),
        "primary_launch_receipt_sha256": protocol.file_sha256(protocol.HERE / "results/primary/launch_receipt.json"),
    }
    _exclusive_json(args.receipt, receipt)
    print(f"validated {result['record_count']} records across {result['cell_count']} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
