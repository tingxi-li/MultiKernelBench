#!/usr/bin/env python3
"""Fail-closed v2 audit, selection, settled-tail, and sham analyses."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .core import (
        GRID_IDS, LANES, LOCK_PATH, REPO_ROOT, SHAM_BASE_CELL, SHAM_LABELS, STRATEGIES,
        TERMINAL_AUDIT_OUTCOMES, canonical_sha256, confirmation_plan,
        effect_is_reportable, exact_median_interval, file_sha256, load_contract,
        read_json, resolution_floor, screen_plan, spearman_rank, stable_write,
        summarize_times, timing_filename, validate_gpu,
    )
except ImportError:  # direct script execution
    from core import (
    GRID_IDS,
    LANES,
    LOCK_PATH,
    REPO_ROOT,
    SHAM_BASE_CELL,
    SHAM_LABELS,
    STRATEGIES,
    TERMINAL_AUDIT_OUTCOMES,
    canonical_sha256,
    confirmation_plan,
    effect_is_reportable,
    exact_median_interval,
    file_sha256,
    load_contract,
    read_json,
    resolution_floor,
    screen_plan,
    spearman_rank,
    stable_write,
    summarize_times,
    timing_filename,
    validate_gpu,
    )


def cell_filename(cell_id: str) -> str:
    return cell_id.replace(".", "__") + ".json"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read gate evidence {path}: {exc}") from exc


def _validate_audit_receipts(
    root: Path,
    campaign: dict[str, Any],
    cells: list[dict[str, Any]],
    lock: dict[str, Any],
) -> list[dict[str, str]]:
    paths = sorted((root / "audit" / "receipts").glob("shard[0-9][0-9].json"))
    if not paths:
        raise RuntimeError("audit has no shard receipts")
    shard_counts, shard_indices, commits = set(), set(), set()
    hashes = []
    for path in paths:
        receipt = read_json(path)
        contract = receipt.get("contract", {})
        expected_common = {
            "campaign_id": campaign["campaign_id"],
            "launch_lock_sha256": file_sha256(LOCK_PATH),
            "source_bundle_sha256": lock["source_bundle_sha256"],
        }
        mismatch = [key for key, value in expected_common.items() if contract.get(key) != value]
        if (
            mismatch
            or receipt.get("record_type") != "fused_crossed_v2_audit_receipt"
            or receipt.get("schema_version") != 2
        ):
            raise RuntimeError(f"invalid audit shard receipt {path}: {mismatch}")
        count, index = contract.get("shard_count"), contract.get("shard_index")
        if not isinstance(count, int) or count < 1 or not isinstance(index, int) or index not in range(count):
            raise RuntimeError(f"invalid audit shard coordinates: {path}")
        assigned = [cell["cell_id"] for cell in cells if cell["cell_index"] % count == index]
        if contract.get("assigned_cell_ids") != assigned:
            raise RuntimeError(f"audit shard assignment changed: {path}")
        validate_gpu(receipt.get("gpu", {}), campaign)
        status_path = path.with_name(path.stem + "_status.json")
        status = read_json(status_path)
        if status != {
            "complete": True,
            "expected_cells": len(assigned),
            "observed_cells": len(assigned),
            "receipt_sha256": file_sha256(path),
        }:
            raise RuntimeError(f"audit shard status is incomplete: {status_path}")
        shard_counts.add(count)
        shard_indices.add(index)
        commits.add(contract.get("git_commit"))
        hashes.extend(
            {"path": str(item.relative_to(REPO_ROOT)), "sha256": file_sha256(item)}
            for item in (path, status_path)
        )
    count = next(iter(shard_counts)) if len(shard_counts) == 1 else None
    if count is None or shard_indices != set(range(count)) or len(paths) != count or len(commits) != 1:
        raise RuntimeError("audit shard receipts are not one complete commit-bound partition")
    return hashes


def _validate_timing_run(
    root: Path,
    phase: str,
    eligibility_path: Path,
    plan: list[dict[str, Any]],
    campaign: dict[str, Any],
    lock: dict[str, Any],
) -> tuple[Path, list[dict[str, str]]]:
    receipt_path = root / phase / "launch_receipt.json"
    receipt = read_json(receipt_path)
    contract = receipt.get("contract", {})
    expected = {
        "campaign_id": campaign["campaign_id"],
        "eligibility_path": str(eligibility_path.relative_to(REPO_ROOT)),
        "eligibility_sha256": file_sha256(eligibility_path),
        "execution_order": plan,
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "phase": phase,
        "physical_gpu": 0,
        "source_bundle_sha256": lock["source_bundle_sha256"],
    }
    mismatch = [key for key, value in expected.items() if contract.get(key) != value]
    if (
        mismatch
        or receipt.get("record_type") != "fused_crossed_v2_timing_receipt"
        or receipt.get("schema_version") != 2
    ):
        raise RuntimeError(f"invalid {phase} launch receipt: {mismatch}")
    validate_gpu(receipt.get("gpu", {}), campaign)
    status_path = root / phase / "run_status.json"
    status = read_json(status_path)
    if (
        status.get("campaign_id") != campaign["campaign_id"]
        or status.get("complete") is not True
        or status.get("phase") != phase
        or status.get("expected_records") != len(plan)
        or status.get("observed_records") != len(plan)
        or status.get("launch_receipt_sha256") != file_sha256(receipt_path)
    ):
        raise RuntimeError(f"incomplete {phase} run status")
    return receipt_path, [
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
        for path in (receipt_path, status_path)
    ]


def _validate_fourth_strategy_acceptance(records: list[dict[str, Any]]) -> None:
    failures = [
        record["cell"]["cell_id"]
        for record in records
        if record["cell"]["strategy"] == "register_common_postprocess"
        and record["cell"]["support_declared"] is True
        and record["terminal_outcome"] != "BUILD_FAILED"
        and record.get("build_metadata", {}).get("n_kernels") != 2
    ]
    if failures:
        raise RuntimeError(
            "built register_common_postprocess cells did not produce the "
            f"full two-kernel operation: {failures}"
        )


def audit_summary(root: Path) -> dict[str, Any]:
    campaign, cells, lock = load_contract()
    receipt_hashes = _validate_audit_receipts(root, campaign, cells, lock)
    records = []
    for cell in cells:
        path = root / "audit" / "records" / cell_filename(cell["cell_id"])
        if not path.is_file():
            raise RuntimeError(f"audit incomplete: missing {path}")
        record = read_json(path)
        expected = {
            "campaign_id": campaign["campaign_id"],
            "cell": cell,
            "cell_sha256": canonical_sha256(cell),
            "launch_lock_sha256": file_sha256(LOCK_PATH),
            "source_bundle_sha256": lock["source_bundle_sha256"],
        }
        mismatch = [key for key, value in expected.items() if record.get(key) != value]
        outcome = record.get("terminal_outcome")
        if mismatch or outcome not in TERMINAL_AUDIT_OUTCOMES:
            raise RuntimeError(f"invalid audit record {path}: {mismatch}")
        if outcome == "UNSUPPORTED":
            if cell["support_declared"] is not False or record.get("build_attempted") is not False:
                raise RuntimeError(f"invalid measured unsupported outcome: {cell['cell_id']}")
            if record.get("support_probe_key") != cell.get("support_probe_key"):
                raise RuntimeError(f"unsupported cell lost probe binding: {cell['cell_id']}")
        elif cell["support_declared"] is not True:
            raise RuntimeError(f"unresolved/unsupported cell has executable outcome: {cell['cell_id']}")
        if outcome in {"LAUNCH_FAILED", "GATE_FAILED", "GATE_PASSED"}:
            summary = record.get("gate_summary", {})
            gate_path = REPO_ROOT / record.get("gate_jsonl_path", "")
            if not gate_path.is_file() or file_sha256(gate_path) != record.get("gate_jsonl_sha256"):
                raise RuntimeError(f"changed gate evidence: {cell['cell_id']}")
            gate_rows = _jsonl(gate_path)
            bindings = {
                "crossed_campaign_id": campaign["campaign_id"],
                "crossed_cell_id": cell["cell_id"],
                "crossed_cell_sha256": canonical_sha256(cell),
                "crossed_launch_lock_sha256": file_sha256(LOCK_PATH),
                "crossed_source_bundle_sha256": lock["source_bundle_sha256"],
            }
            if any(any(row.get(key) != value for key, value in bindings.items()) for row in gate_rows):
                raise RuntimeError(f"gate rows lost their campaign binding: {cell['cell_id']}")
            failed = sum(
                row.get("ok") is not True or row.get("gate_pass") is not True
                for row in gate_rows
            )
            coverage = {
                (row.get("case_id"), row.get("seed_index"), row.get("gate_id"))
                for row in gate_rows
            }
            expected_coverage = {
                (case, seed, gate)
                for case in campaign["frozen_gate"]["case_ids"]
                for seed in range(64)
                for gate in campaign["frozen_gate"]["gate_ids"]
            }
            complete = len(gate_rows) == 512 and coverage == expected_coverage
            if (
                summary.get("observed_records") != len(gate_rows)
                or summary.get("failed_records") != failed
                or summary.get("complete") is not complete
                or summary.get("full_gate_pass") is not (complete and failed == 0)
            ):
                raise RuntimeError(f"gate summary is not evidence-derived: {cell['cell_id']}")
            if outcome != "LAUNCH_FAILED" and not complete:
                raise RuntimeError(f"incomplete terminal gate outcome: {cell['cell_id']}")
            if (outcome == "GATE_PASSED") != (complete and failed == 0):
                raise RuntimeError(f"gate verdict mismatch: {cell['cell_id']}")
            if record.get("build_metadata", {}).get("n_kernels") != 2:
                raise RuntimeError(f"full operation did not declare exactly two kernels: {cell['cell_id']}")
        records.append(record)
    by_cell = {record["cell"]["cell_id"]: record for record in records}
    _validate_fourth_strategy_acceptance(records)
    recovered = [f"register_fused.cuda_unlimited.g{index:02d}" for index in range(5, 13)]
    if any(by_cell[cell_id]["terminal_outcome"] != "GATE_PASSED" for cell_id in recovered):
        raise RuntimeError("corrected cuda_unlimited register cells g05-g12 did not recover")
    for record in records:
        cell = record["cell"]
        oversized = cell["strategy"] == "smem_staged" and cell["grid_id"] in {
            f"g{index:02d}" for index in range(5, 13)
        }
        if oversized and cell["support_declared"] is True and (
            record["terminal_outcome"] != "BUILD_FAILED" or record.get("gate_attempted") is not False
        ):
            raise RuntimeError(f"oversized smem cell did not fail closed at setup: {cell['cell_id']}")
    legal = {record["cell"]["cell_id"] for record in records if record["terminal_outcome"] == "GATE_PASSED"}
    rows = []
    for strategy in STRATEGIES:
        for lane in LANES:
            subset = [record for record in records if record["cell"]["strategy"] == strategy and record["cell"]["lane"] == lane]
            counts = Counter(record["terminal_outcome"] for record in subset)
            rows.append(
                {
                    "counts": {name: counts.get(name, 0) for name in TERMINAL_AUDIT_OUTCOMES},
                    "gate_legal_rate_over_requested": counts.get("GATE_PASSED", 0) / 19.0,
                    "lane": lane,
                    "requested_cells": 19,
                    "strategy": strategy,
                }
            )
    common_by_strategy = {
        strategy: [grid for grid in GRID_IDS if all(f"{strategy}.{lane}.{grid}" in legal for lane in LANES)]
        for strategy in STRATEGIES
    }
    return {
        "campaign_id": campaign["campaign_id"],
        "common_feasible_grid_ids_by_strategy": common_by_strategy,
        "complete": True,
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "outcome_counts": {name: sum(record["terminal_outcome"] == name for record in records) for name in TERMINAL_AUDIT_OUTCOMES},
        "outcomes_by_strategy_lane": rows,
        "receipt_hashes": receipt_hashes,
        "record_type": "fused_crossed_v2_audit_summary",
        "requested_cells": len(cells),
        "schema_version": 2,
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "timing_eligible_cell_ids": sorted(legal),
    }


def choose_confirmation(
    cells: list[dict[str, Any]],
    process_medians: dict[str, list[float]],
    gate_legal_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    by_id = {cell["cell_id"]: cell for cell in cells}
    for strategy in STRATEGIES:
        for lane in LANES:
            candidates = []
            for grid in GRID_IDS:
                cell_id = f"{strategy}.{lane}.{grid}"
                values = process_medians.get(cell_id, [])
                if len(values) == 2:
                    candidates.append((statistics.median(values), grid, cell_id))
            ranked = sorted(candidates)
            chosen = [cell_id for _value, _grid, cell_id in ranked[:2]]
            g01 = f"{strategy}.{lane}.g01"
            g01_legal = g01 in gate_legal_ids if gate_legal_ids is not None else len(process_medians.get(g01, [])) == 2
            if g01_legal and g01 not in chosen:
                chosen.append(g01)
            for cell_id in chosen:
                rank = next((index + 1 for index, row in enumerate(ranked) if row[2] == cell_id), None)
                values = process_medians.get(cell_id, [])
                selected.append(
                    {
                        "cell_id": cell_id,
                        "grid_id": by_id[cell_id]["grid_id"],
                        "lane": lane,
                        "screen_primary_tail_median_ms": statistics.median(values) if len(values) == 2 else None,
                        "screen_rank": rank,
                        "selection_reason": "top_two" if rank is not None and rank <= 2 else "g01_gate_legal_positive_control",
                        "strategy": strategy,
                    }
                )
    return selected


def _load_timing_record(path: Path, *, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing timing record: {path}")
    record = read_json(path)
    mismatch = [key for key, value in expected.items() if record.get(key) != value]
    if mismatch or record.get("ok") is not True:
        raise RuntimeError(f"invalid timing record {path}: {mismatch or record.get('error')}")
    if record.get("legacy_error", {}).get("gate_pass") is not True:
        raise RuntimeError(f"timing record failed its legacy check: {path}")
    summary = summarize_times(record.get("times_ms", []))
    for key, value in summary.items():
        if record.get(key) != value:
            raise RuntimeError(f"timing summary mismatch {path}: {key}")
    if not record.get("implementation_sha256") or record.get("build_metadata", {}).get("n_kernels") != 2:
        raise RuntimeError(f"timing record is not a hash-bound two-kernel full operation: {path}")
    return record


def screen_selection(root: Path, audit_path: Path) -> dict[str, Any]:
    campaign, cells, lock = load_contract()
    audit_path = audit_path.resolve()
    audit = read_json(audit_path)
    if audit != audit_summary(root):
        raise RuntimeError("screen audit summary is not re-derived from the retained audit")
    if audit.get("record_type") != "fused_crossed_v2_audit_summary" or audit.get("complete") is not True:
        raise RuntimeError("screen requires a complete v2 audit summary")
    if audit.get("launch_lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("screen audit lock mismatch")
    legal = set(audit["timing_eligible_cell_ids"])
    plan = screen_plan(cells, legal)
    launch_plan = [
        {**row, "label": row["cell_id"], "record_kind": "cell"} for row in plan
    ]
    receipt_path, hashes = _validate_timing_run(
        root, "screen", audit_path, launch_plan, campaign, lock
    )
    by_id = {cell["cell_id"]: cell for cell in cells}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in plan:
        label = row["cell_id"]
        path = root / "screen" / "raw" / timing_filename(label, "positive", row["rep"])
        record = _load_timing_record(
            path,
            expected={
                "campaign_id": campaign["campaign_id"],
                "cell_id": row["cell_id"],
                "cell_sha256": canonical_sha256(by_id[row["cell_id"]]),
                "distribution": "positive",
                "eligibility_path": str(audit_path.relative_to(REPO_ROOT)),
                "eligibility_sha256": file_sha256(audit_path),
                "label": label,
                "launch_lock_sha256": file_sha256(LOCK_PATH),
                "phase": "screen",
                "physical_gpu": 0,
                "record_kind": "cell",
                "rep": row["rep"],
                "source_bundle_sha256": lock["source_bundle_sha256"],
            },
        )
        grouped[row["cell_id"]].append(float(record["primary_tail_median_ms"]))
        hashes.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)})
    selected = choose_confirmation(cells, grouped, legal)
    if SHAM_BASE_CELL not in {row["cell_id"] for row in selected}:
        raise RuntimeError("gate-legal selected g01 sham base is unavailable")
    return {
        "audit_summary_path": str(audit_path.relative_to(REPO_ROOT)),
        "audit_summary_sha256": file_sha256(audit_path),
        "campaign_id": campaign["campaign_id"],
        "complete": True,
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "record_hashes": hashes,
        "record_type": "fused_crossed_v2_confirmation_selection",
        "screen_receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
        "screen_receipt_sha256": file_sha256(receipt_path),
        "schema_version": 2,
        "selected": selected,
        "selected_cell_ids": [row["cell_id"] for row in selected],
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "timing_eligible_cell_ids": sorted(legal),
    }


def _paired_ratio(numerator: dict[str, Any], denominator: dict[str, Any], floor: float) -> dict[str, Any]:
    ratios = [a / b for a, b in zip(numerator["process_medians_ms"], denominator["process_medians_ms"])]
    interval = exact_median_interval(ratios)
    return {
        "block_ratios": ratios,
        "interval": interval,
        "reportable_above_sham_floor": effect_is_reportable(interval, floor),
    }


def _performance_contrasts(results: dict[tuple[str, str], dict[str, Any]], floor: float) -> dict[str, Any]:
    placement = []
    for lane in LANES:
        for grid in GRID_IDS:
            global_key = (f"global_intermediate.{lane}.{grid}", "positive")
            register_key = (f"register_common_postprocess.{lane}.{grid}", "positive")
            if global_key in results and register_key in results:
                placement.append(
                    {
                        "grid_id": grid,
                        "lane": lane,
                        "register_common_over_global": _paired_ratio(results[register_key], results[global_key], floor),
                    }
                )
    lane_effects = []
    for strategy in STRATEGIES:
        for grid in GRID_IDS:
            baseline = (f"{strategy}.tilelang.{grid}", "positive")
            if baseline not in results:
                continue
            for lane in LANES[1:]:
                target = (f"{strategy}.{lane}.{grid}", "positive")
                if target in results:
                    lane_effects.append(
                        {
                            "grid_id": grid,
                            "lane": lane,
                            "strategy": strategy,
                            "target_over_tilelang": _paired_ratio(results[target], results[baseline], floor),
                        }
                    )
    return {
        "lane_effects_common_selected_grids": lane_effects,
        "register_placement_effects_common_softmax": placement,
        "resolution_floor_log_ratio": floor,
    }


def confirmation_summary(root: Path, selection_path: Path) -> dict[str, Any]:
    campaign, cells, lock = load_contract()
    selection_path = selection_path.resolve()
    selection = read_json(selection_path)
    audit_path = REPO_ROOT / selection.get("audit_summary_path", "")
    if (
        not audit_path.is_file()
        or file_sha256(audit_path) != selection.get("audit_summary_sha256")
        or selection != screen_selection(root, audit_path)
    ):
        raise RuntimeError("confirmation selection is not re-derived from the retained screen")
    if selection.get("record_type") != "fused_crossed_v2_confirmation_selection" or selection.get("complete") is not True:
        raise RuntimeError("confirmation requires a complete v2 selection")
    if selection.get("launch_lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("confirmation selection lock mismatch")
    selected = set(selection["selected_cell_ids"])
    legal = set(selection.get("timing_eligible_cell_ids", []))
    if (
        selection.get("source_bundle_sha256") != lock["source_bundle_sha256"]
        or not selected <= legal
        or SHAM_BASE_CELL not in selected
    ):
        raise RuntimeError("confirmation selection is not source-bound/audit-eligible")
    plan = confirmation_plan(selected)
    receipt_path, hashes = _validate_timing_run(
        root, "confirmation", selection_path, plan, campaign, lock
    )
    by_id = {cell["cell_id"]: cell for cell in cells}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    full_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    drift_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    first_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    last_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    source_hashes: dict[str, set[str]] = defaultdict(set)
    for row in plan:
        path = root / "confirmation" / "raw" / timing_filename(row["label"], row["distribution"], row["rep"])
        record = _load_timing_record(
            path,
            expected={
                "campaign_id": campaign["campaign_id"],
                "cell_id": row["cell_id"],
                "cell_sha256": canonical_sha256(by_id[row["cell_id"]]),
                "distribution": row["distribution"],
                "eligibility_path": str(selection_path.relative_to(REPO_ROOT)),
                "eligibility_sha256": file_sha256(selection_path),
                "label": row["label"],
                "launch_lock_sha256": file_sha256(LOCK_PATH),
                "phase": "confirmation",
                "physical_gpu": 0,
                "record_kind": row["record_kind"],
                "rep": row["rep"],
                "source_bundle_sha256": lock["source_bundle_sha256"],
            },
        )
        key = (row["label"], row["distribution"])
        grouped[key].append(float(record["primary_tail_median_ms"]))
        full_grouped[key].append(float(record["full_median_ms"]))
        first_grouped[key].append(float(record["first_decile_median_ms"]))
        last_grouped[key].append(float(record["last_decile_median_ms"]))
        drift_grouped[key].append(float(record["first_to_last_decile_ratio"]))
        source_hashes[row["label"]].add(str(record.get("implementation_sha256")))
        hashes.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)})
    sham_intervals = {}
    for distribution in ("positive", "withheld_signed"):
        a = grouped[(SHAM_LABELS[0], distribution)]
        b = grouped[(SHAM_LABELS[1], distribution)]
        sham_intervals[distribution] = exact_median_interval(left / right for left, right in zip(a, b))
    if (
        source_hashes[SHAM_LABELS[0]] != source_hashes[SHAM_LABELS[1]]
        or len(source_hashes[SHAM_LABELS[0]]) != 1
        or None in source_hashes[SHAM_LABELS[0]]
    ):
        raise RuntimeError("sham labels do not bind one byte-identical implementation")
    floor = resolution_floor(sham_intervals.values())
    results: dict[tuple[str, str], dict[str, Any]] = {}
    cells_out = []
    for cell_id in sorted(selected):
        for distribution in ("positive", "withheld_signed"):
            values = grouped[(cell_id, distribution)]
            full = full_grouped[(cell_id, distribution)]
            if len(values) != 15:
                raise RuntimeError(f"incomplete confirmation block: {cell_id}/{distribution}")
            row = {
                "cell_id": cell_id,
                "distribution": distribution,
                "full_window_diagnostic_interval": exact_median_interval(full),
                "first_decile_diagnostic_interval": exact_median_interval(first_grouped[(cell_id, distribution)]),
                "first_to_last_decile_ratio_diagnostic_interval": exact_median_interval(drift_grouped[(cell_id, distribution)]),
                "interval": exact_median_interval(values),
                "last_decile_diagnostic_interval": exact_median_interval(last_grouped[(cell_id, distribution)]),
                "median_ms": statistics.median(values),
                "process_full_medians_ms": full,
                "process_medians_ms": values,
                "primary_window": [60, 100],
            }
            results[(cell_id, distribution)] = row
            cells_out.append(row)
    stability = []
    for strategy in STRATEGIES:
        for lane in LANES:
            ids = sorted(cell_id for cell_id in selected if by_id[cell_id]["strategy"] == strategy and by_id[cell_id]["lane"] == lane)
            positive = [results[(cell_id, "positive")]["median_ms"] for cell_id in ids]
            signed = [results[(cell_id, "withheld_signed")]["median_ms"] for cell_id in ids]
            stability.append(
                {
                    "cell_ids": ids,
                    "lane": lane,
                    "paired_signed_over_positive": {
                        cell_id: exact_median_interval(
                            signed_value / positive_value
                            for positive_value, signed_value in zip(
                                results[(cell_id, "positive")]["process_medians_ms"],
                                results[(cell_id, "withheld_signed")]["process_medians_ms"],
                            )
                        )
                        for cell_id in ids
                    },
                    "rank_spearman": spearman_rank(positive, signed),
                    "strategy": strategy,
                }
            )
    return {
        "campaign_id": campaign["campaign_id"],
        "cell_results": cells_out,
        "complete": True,
        "distribution_stability": stability,
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "performance_contrasts": _performance_contrasts(results, floor),
        "record_hashes": hashes,
        "record_type": "fused_crossed_v2_final_summary",
        "confirmation_receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
        "confirmation_receipt_sha256": file_sha256(receipt_path),
        "schema_version": 2,
        "selection_path": str(selection_path.relative_to(REPO_ROOT)),
        "selection_sha256": file_sha256(selection_path),
        "sham_control": {
            "base_cell": campaign["inference"]["sham_base_cell"],
            "implementation_sha256": next(iter(source_hashes[SHAM_LABELS[0]])),
            "intervals": sham_intervals,
            "labels": list(SHAM_LABELS),
            "resolution_floor_log_ratio": floor,
        },
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "scope": campaign["claim_limit"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("audit", "screen", "confirmation"):
        sub = subparsers.add_parser(phase)
        sub.add_argument("--result-root", required=True)
        sub.add_argument("--out", required=True)
        if phase == "screen":
            sub.add_argument("--audit-summary", required=True)
        elif phase == "confirmation":
            sub.add_argument("--selection", required=True)
    args = parser.parse_args()
    root = Path(args.result_root).resolve()
    if args.phase == "audit":
        value = audit_summary(root)
    elif args.phase == "screen":
        value = screen_selection(root, Path(args.audit_summary).resolve())
    else:
        value = confirmation_summary(root, Path(args.selection).resolve())
    stable_write(Path(args.out), value)
    print(f"{args.phase}: complete={value['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
