#!/usr/bin/env python3
"""Read-only audit and aggregation of fused-grid confirmation records.

The analyzer accepts only records bound to one exact confirmation launch
receipt.  It requires repetitions 0--4 for every selected job, retains build
and correctness failures as first-class outcomes, and applies Phase 1's frozen
``common.median_ci`` rule to the five independent process medians.

No input is ever modified.  By default the human-readable result is printed to
stdout; ``--summary-out`` optionally writes a separate, deterministic JSON
summary.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PHASE1 = REPO_ROOT / "ako_runs/phase1_matmul"
DEFAULT_CONFIRMATION = HERE / "jobs/confirm_robust.json"
EXPECTED_REPS = tuple(range(5))
REQUIRED_ARTIFACTS = {
    "confirmation",
    "robust_adapter_manifest",
    "robust_summary",
    "screening_launch_receipt",
    "screening_manifest",
}

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PHASE1))
import analyze_screen  # noqa: E402
import launch_confirmation  # noqa: E402
import common  # noqa: E402


class ConfirmationAnalysisError(ValueError):
    """An input is missing, malformed, stale, or not content-bound."""


@dataclass(frozen=True)
class Context:
    confirmation: dict[str, Any]
    confirmation_sha256: str
    launch_receipt: dict[str, Any]
    launch_receipt_sha256: str
    campaign: analyze_screen.Campaign


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ConfirmationAnalysisError(
            f"{label} mismatch: got {actual!r}, expected {expected!r}"
        )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfirmationAnalysisError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfirmationAnalysisError(f"{label} must be an integer")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfirmationAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ConfirmationAnalysisError(f"{label} must be finite and positive")
    return result


def _strict_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        value, raw = analyze_screen.read_json(path)
    except analyze_screen.AnalysisError as exc:
        raise ConfirmationAnalysisError(str(exc)) from exc
    document = _object(value, label)
    try:
        stable = analyze_screen.stable_json_bytes(document)
    except analyze_screen.AnalysisError as exc:
        raise ConfirmationAnalysisError(str(exc)) from exc
    _same(raw, stable, f"{label} stable serialization")
    return document, raw


def _repo_artifact(path_text: Any, label: str) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise ConfirmationAnalysisError(f"{label} path must be non-empty")
    path = Path(path_text)
    if not path.is_absolute():
        raise ConfirmationAnalysisError(f"{label} path must be absolute")
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ConfirmationAnalysisError(f"{label} path escapes the repository") from exc
    return path


def _verify_current_sources(receipt: dict[str, Any]) -> None:
    sources = _object(receipt.get("source_sha256"), "launch source_sha256")
    if not sources:
        raise ConfirmationAnalysisError("launch source_sha256 cannot be empty")
    for relative, expected in sources.items():
        if not isinstance(relative, str) or not relative:
            raise ConfirmationAnalysisError("launch source path must be non-empty")
        path = (REPO_ROOT / relative).resolve()
        try:
            path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ConfirmationAnalysisError(
                f"launch source escapes the repository: {relative!r}"
            ) from exc
        _same(
            launch_confirmation.sha256_file(path),
            expected,
            f"current launch source {relative}",
        )
    _same(
        launch_confirmation.sha256_bytes(analyze_screen.stable_json_bytes(sources)),
        receipt.get("source_bundle_sha256"),
        "launch source bundle hash",
    )


def validate_launch_receipt(
    receipt: dict[str, Any],
    confirmation: dict[str, Any],
    confirmation_path: Path,
    confirmation_sha256: str,
) -> analyze_screen.Campaign:
    """Validate the launch receipt and every artifact/source hash it names."""
    _same(receipt.get("schema_version"), 1, "launch receipt schema_version")
    _same(
        receipt.get("record_type"),
        "fused_grid_confirmation_launch",
        "launch receipt record_type",
    )
    _same(receipt.get("campaign_id"), confirmation["campaign_id"], "campaign_id")
    _same(
        receipt.get("confirmation_sha256"),
        confirmation_sha256,
        "launch receipt confirmation_sha256",
    )
    _same(
        receipt.get("confirmation_jobs_sha256"),
        confirmation["jobs_sha256"],
        "launch receipt confirmation_jobs_sha256",
    )
    _same(
        receipt.get("confirmation_provenance"),
        confirmation["provenance"],
        "launch receipt confirmation provenance",
    )

    protocol = _object(receipt.get("protocol"), "launch protocol")
    _same(
        set(protocol),
        {"dist", "seed", "time_only", "trials", "warmup_s"},
        "confirmation timing protocol fields",
    )
    if protocol["dist"] not in ("rand", "randn"):
        raise ConfirmationAnalysisError("confirmation dist must be rand or randn")
    _integer(protocol["seed"], "confirmation seed")
    _same(protocol["time_only"], False, "confirmation time_only")
    if _integer(protocol["trials"], "confirmation trials") <= 0:
        raise ConfirmationAnalysisError("confirmation trials must be positive")
    warmup_s = protocol["warmup_s"]
    if (
        isinstance(warmup_s, bool)
        or not isinstance(warmup_s, (int, float))
        or not math.isfinite(float(warmup_s))
        or warmup_s < 0
    ):
        raise ConfirmationAnalysisError(
            "confirmation warmup_s must be finite and non-negative"
        )
    protocol_hash = launch_confirmation.sha256_bytes(
        analyze_screen.stable_json_bytes(protocol)
    )
    _same(protocol_hash, receipt.get("protocol_sha256"), "launch protocol hash")
    _same(
        protocol_hash,
        confirmation["provenance"]["screening_protocol_sha256"],
        "confirmation/screening protocol hash",
    )

    launch_args = _object(receipt.get("launch_args"), "launch_args")
    _same(launch_args.get("reps"), len(EXPECTED_REPS), "launch repetitions")
    _integer(launch_args.get("gpu"), "launch gpu")
    _integer(launch_args.get("order_seed"), "launch order_seed")
    if _integer(launch_args.get("timeout"), "launch timeout") <= 0:
        raise ConfirmationAnalysisError("launch timeout must be positive")

    artifact_paths = _object(receipt.get("artifact_path"), "artifact_path")
    artifact_hashes = _object(receipt.get("artifact_sha256"), "artifact_sha256")
    _same(set(artifact_paths), REQUIRED_ARTIFACTS, "launch artifact path names")
    _same(set(artifact_hashes), REQUIRED_ARTIFACTS, "launch artifact hash names")
    paths = {
        name: _repo_artifact(artifact_paths[name], f"artifact {name}")
        for name in REQUIRED_ARTIFACTS
    }
    _same(paths["confirmation"], confirmation_path.resolve(), "confirmation path")
    for name, path in paths.items():
        _same(
            launch_confirmation.sha256_file(path),
            artifact_hashes[name],
            f"artifact {name} hash",
        )
    provenance = confirmation["provenance"]
    expected_artifact_hashes = {
        "confirmation": confirmation_sha256,
        "robust_adapter_manifest": provenance["robust_adapter_manifest_sha256"],
        "robust_summary": provenance["robust_summary_sha256"],
        "screening_launch_receipt": provenance[
            "screening_launch_receipt_sha256"
        ],
        "screening_manifest": provenance["screening_manifest_sha256"],
    }
    _same(artifact_hashes, expected_artifact_hashes, "bound artifact hashes")

    try:
        campaign = analyze_screen.load_campaign(paths["screening_manifest"])
    except analyze_screen.AnalysisError as exc:
        raise ConfirmationAnalysisError(str(exc)) from exc
    _same(
        campaign.manifest_sha256,
        provenance["screening_manifest_sha256"],
        "screening manifest hash",
    )
    _same(
        campaign.manifest["jobs_sha256"],
        provenance["screening_jobs_sha256"],
        "screening jobs hash",
    )
    source_grid = REPO_ROOT / campaign.manifest["phase1_grid_source"]
    _same(
        launch_confirmation.sha256_file(source_grid),
        campaign.manifest["phase1_grid_source_sha256"],
        "current Phase-1 grid source hash",
    )

    dsl_order = campaign.manifest["dsl_order"]
    observed_dsl_order = list(dict.fromkeys(job["dsl"] for job in confirmation["jobs"]))
    _same(observed_dsl_order, dsl_order, "confirmation DSL order")
    _same(len(dsl_order), 4, "four-way DSL count")
    for index, job in enumerate(confirmation["jobs"]):
        screening_id = job["screening_job_id"]
        if screening_id not in campaign.jobs_by_id:
            raise ConfirmationAnalysisError(
                f"confirmation job {index} has unknown screening job {screening_id!r}"
            )
        source = campaign.jobs_by_id[screening_id]
        expected = {
            "dsl": source["dsl"],
            "geom": source["geom"],
            "grid_id": source["grid_id"],
            "grid_index": source["grid_index"],
            "set": source["set"],
            "variant": source["variant"],
        }
        for name, wanted in expected.items():
            _same(job.get(name), wanted, f"confirmation job {index} {name}")

    _verify_current_sources(receipt)
    return campaign


def load_context(confirmation_path: Path, launch_receipt_path: Path) -> Context:
    confirmation, confirmation_raw = _strict_json(confirmation_path, "confirmation")
    try:
        launch_confirmation.validate_confirmation_document(confirmation)
    except (launch_confirmation.ConfirmationError, analyze_screen.AnalysisError) as exc:
        raise ConfirmationAnalysisError(str(exc)) from exc
    receipt, receipt_raw = _strict_json(launch_receipt_path, "launch receipt")
    confirmation_hash = launch_confirmation.sha256_bytes(confirmation_raw)
    try:
        campaign = validate_launch_receipt(
            receipt, confirmation, confirmation_path, confirmation_hash
        )
    except launch_confirmation.ConfirmationError as exc:
        raise ConfirmationAnalysisError(str(exc)) from exc
    return Context(
        confirmation=confirmation,
        confirmation_sha256=confirmation_hash,
        launch_receipt=receipt,
        launch_receipt_sha256=launch_confirmation.sha256_bytes(receipt_raw),
        campaign=campaign,
    )


def _expected_cfg(context: Context, job: dict[str, Any]) -> dict[str, Any]:
    parsed = analyze_screen.parse_set(job["set"])
    fixed = context.campaign.manifest["fixed_factors"]
    expected = {
        name: parsed[name]
        for name in ("BM", "BN", "BK", "threads", "stages", "kc", "arith", "cast")
    }
    expected.update({name: fixed[name] for name in ("M", "K", "N")})
    expected.update({"dsl": job["dsl"], "variant": job["variant"]})
    return expected


def validate_record(
    context: Context,
    job: dict[str, Any],
    rep: int,
    path: Path,
) -> dict[str, Any]:
    record, raw = _strict_json(path, f"raw record {path.name}")
    expected_binding = launch_confirmation.record_binding(
        context.launch_receipt, job, rep
    )
    _same(
        record.get("confirmation_provenance"),
        expected_binding,
        f"{path.name} confirmation provenance",
    )
    for name, wanted in {
        "op": "fused",
        "dsl": job["dsl"],
        "variant": job["variant"],
        "rep": rep,
    }.items():
        _same(record.get(name), wanted, f"{path.name} {name}")
    if not isinstance(record.get("ok"), bool):
        raise ConfirmationAnalysisError(f"{path.name} ok must be boolean")
    returncode = _integer(record.get("returncode"), f"{path.name} returncode")

    expected_cfg = _expected_cfg(context, job)
    cfg_value = record.get("cfg")
    cfg = _object(cfg_value, f"{path.name} cfg") if cfg_value is not None else {}
    if record["ok"]:
        for name, wanted in expected_cfg.items():
            _same(cfg.get(name), wanted, f"{path.name} cfg {name}")
        for name, wanted in context.launch_receipt["protocol"].items():
            if name == "time_only":
                continue
            _same(record.get(name), wanted, f"{path.name} {name}")
    else:
        for name, wanted in expected_cfg.items():
            if name in cfg:
                _same(cfg[name], wanted, f"{path.name} cfg {name}")

    view: dict[str, Any] = {
        "rep": rep,
        "record_file": path.name,
        "record_sha256": launch_confirmation.sha256_bytes(raw),
        "ok": record["ok"],
        "returncode": returncode,
    }
    if record["ok"]:
        timing = _object(record.get("timing"), f"{path.name} timing")
        view["median_ms"] = _positive(
            timing.get("median_ms"), f"{path.name} timing.median_ms"
        )
        error = record.get("error")
        if not isinstance(error, dict) or not isinstance(error.get("gate_pass"), bool):
            view["legacy_gate_pass"] = False
            view["outcome"] = "legacy_gate_missing"
        else:
            view["legacy_gate_pass"] = error["gate_pass"]
            view["outcome"] = "accepted" if error["gate_pass"] else "legacy_gate_failure"
            for name in (
                "pct_elems_failing_gate",
                "max_abs_err",
                "mean_abs_err",
                "budget_mean",
            ):
                value = error.get(name)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    view[name] = float(value)
        if returncode != 0:
            view["outcome"] = "build_or_execution_failure"
            view["legacy_gate_pass"] = False
    else:
        view["legacy_gate_pass"] = False
        view["outcome"] = "build_or_execution_failure"
        message = record.get("error_msg")
        view["error_msg"] = message if isinstance(message, str) and message else "unspecified"
    if "compile_s" in record:
        value = record["compile_s"]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)) and value >= 0:
                view["compile_s"] = float(value)
    return view


def load_records(context: Context, raw_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not raw_dir.is_dir():
        raise ConfirmationAnalysisError(f"raw directory does not exist: {raw_dir}")
    expected_paths: dict[str, tuple[dict[str, Any], int]] = {}
    for job in context.confirmation["jobs"]:
        for rep in EXPECTED_REPS:
            name = launch_confirmation.output_path(job, rep, Path(".")).name
            if name in expected_paths:
                raise ConfirmationAnalysisError(f"duplicate expected raw filename {name}")
            expected_paths[name] = (job, rep)
    observed_paths = {path.name: path for path in raw_dir.glob("*.json")}
    missing = sorted(set(expected_paths) - set(observed_paths))
    extra = sorted(set(observed_paths) - set(expected_paths))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} records; first={missing[0]}")
        if extra:
            parts.append(f"unexpected {len(extra)} records; first={extra[0]}")
        raise ConfirmationAnalysisError("raw record set mismatch: " + "; ".join(parts))

    result: dict[tuple[str, int], dict[str, Any]] = {}
    for name, (job, rep) in expected_paths.items():
        key = (job["confirmation_id"], rep)
        result[key] = validate_record(context, job, rep, observed_paths[name])
    return result


def intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return not (
        left["ci95_hi_ms"] < right["ci95_lo_ms"]
        or right["ci95_hi_ms"] < left["ci95_lo_ms"]
    )


def aggregate_confirmation(
    confirmation: dict[str, Any],
    processes: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Aggregate a complete validated record map into cells and lane winners."""
    expected_keys = {
        (job["confirmation_id"], rep)
        for job in confirmation["jobs"]
        for rep in EXPECTED_REPS
    }
    missing = sorted(expected_keys - set(processes))
    extra = sorted(set(processes) - expected_keys)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} job/reps; first={missing[0]}")
        if extra:
            detail.append(f"unexpected {len(extra)} job/reps; first={extra[0]}")
        raise ConfirmationAnalysisError("process map is incomplete: " + "; ".join(detail))

    cells: list[dict[str, Any]] = []
    for job in confirmation["jobs"]:
        rows = [processes[(job["confirmation_id"], rep)] for rep in EXPECTED_REPS]
        outcomes = [row["outcome"] for row in rows]
        accepted = outcomes.count("accepted")
        cell: dict[str, Any] = {
            "confirmation_id": job["confirmation_id"],
            "screening_job_id": job["screening_job_id"],
            "dsl": job["dsl"],
            "grid_id": job["grid_id"],
            "grid_index": job["grid_index"],
            "selection_roles": job["selection_roles"],
            "set": job["set"],
            "screening_median_ms": job["screening_median_ms"],
            "rep_ids": [row["rep"] for row in rows],
            "processes": rows,
            "accepted_process_count": accepted,
            "failure_count": len(rows) - accepted,
            "outcome_counts": {
                name: outcomes.count(name)
                for name in (
                    "accepted",
                    "legacy_gate_failure",
                    "legacy_gate_missing",
                    "build_or_execution_failure",
                )
            },
            "confirmation_eligible": accepted == len(EXPECTED_REPS),
            "status": "confirmed" if accepted == len(EXPECTED_REPS) else "failed",
        }
        if cell["confirmation_eligible"]:
            medians = [row["median_ms"] for row in rows]
            stats = common.median_ci(medians)
            cell.update(stats)
            cell["selection_bias_pct"] = 100.0 * (
                cell["median_of_medians_ms"] / cell["screening_median_ms"] - 1.0
            )
        cells.append(cell)

    dsl_order = list(dict.fromkeys(job["dsl"] for job in confirmation["jobs"]))
    lanes: list[dict[str, Any]] = []
    for dsl in dsl_order:
        lane_cells = [cell for cell in cells if cell["dsl"] == dsl]
        eligible = [cell for cell in lane_cells if cell["confirmation_eligible"]]
        eligible.sort(
            key=lambda cell: (
                cell["median_of_medians_ms"],
                cell["grid_index"],
                cell["confirmation_id"],
            )
        )
        winner = eligible[0] if eligible else None
        failed_ids = [
            cell["screening_job_id"] for cell in lane_cells if not cell["confirmation_eligible"]
        ]
        lane: dict[str, Any] = {
            "dsl": dsl,
            "job_count": len(lane_cells),
            "confirmed_job_count": len(eligible),
            "failed_job_ids": failed_ids,
            "point_estimate_winner": None,
            "strict_winner_resolved": False,
            "unresolved_with": [],
        }
        if winner is not None:
            overlaps = [
                other["screening_job_id"]
                for other in eligible[1:]
                if intervals_overlap(winner, other)
            ]
            screen_rank_one = next(
                (
                    cell["screening_job_id"]
                    for cell in lane_cells
                    if "screening_rank_1" in cell["selection_roles"]
                ),
                None,
            )
            lane["point_estimate_winner"] = {
                "confirmation_id": winner["confirmation_id"],
                "screening_job_id": winner["screening_job_id"],
                "grid_id": winner["grid_id"],
                "median_of_medians_ms": winner["median_of_medians_ms"],
                "ci95_lo_ms": winner["ci95_lo_ms"],
                "ci95_hi_ms": winner["ci95_hi_ms"],
                "selection_roles": winner["selection_roles"],
                "screening_rank_one_held": winner["screening_job_id"] == screen_rank_one,
            }
            lane["unresolved_with"] = overlaps
            lane["strict_winner_resolved"] = not failed_ids and not overlaps
        lanes.append(lane)

    winners = [lane["point_estimate_winner"] for lane in lanes]
    four_way: dict[str, Any] = {
        "available": all(winner is not None for winner in winners) and len(winners) == 4,
        "point_estimate_order": [],
        "strict_order": None,
        "strict_order_resolved": False,
        "overlapping_pairs": [],
        "point_estimate_spread_x": None,
    }
    if four_way["available"]:
        ordered = sorted(
            zip((lane["dsl"] for lane in lanes), winners),
            key=lambda item: (item[1]["median_of_medians_ms"], item[0]),
        )
        four_way["point_estimate_order"] = [dsl for dsl, _winner in ordered]
        fastest_dsl, fastest = ordered[0]
        slowest_dsl, slowest = ordered[-1]
        four_way["fastest_point_estimate_dsl"] = fastest_dsl
        four_way["slowest_point_estimate_dsl"] = slowest_dsl
        four_way["point_estimate_spread_x"] = (
            slowest["median_of_medians_ms"] / fastest["median_of_medians_ms"]
        )
        overlap_pairs = []
        for (left_dsl, left), (right_dsl, right) in itertools.combinations(ordered, 2):
            if intervals_overlap(left, right):
                overlap_pairs.append([left_dsl, right_dsl])
        four_way["overlapping_pairs"] = overlap_pairs
        all_lane_winners_resolved = all(lane["strict_winner_resolved"] for lane in lanes)
        adjacent_separated = all(
            left["ci95_hi_ms"] < right["ci95_lo_ms"]
            for (_left_dsl, left), (_right_dsl, right) in zip(ordered, ordered[1:])
        )
        four_way["strict_order_resolved"] = (
            all_lane_winners_resolved and adjacent_separated
        )
        if four_way["strict_order_resolved"]:
            four_way["strict_order"] = four_way["point_estimate_order"]
            four_way["inference"] = "strict order resolved by non-overlapping 95% CIs"
        else:
            four_way["inference"] = (
                "point estimates only; at least one within-DSL winner or cross-DSL "
                "ordering is unresolved"
            )
    return cells, lanes, four_way


def build_summary(context: Context, raw_dir: Path) -> dict[str, Any]:
    processes = load_records(context, raw_dir)
    cells, lanes, four_way = aggregate_confirmation(context.confirmation, processes)
    record_bundle = [
        {
            "confirmation_id": job["confirmation_id"],
            "rep": rep,
            "record_sha256": processes[(job["confirmation_id"], rep)]["record_sha256"],
        }
        for job in context.confirmation["jobs"]
        for rep in EXPECTED_REPS
    ]
    return {
        "schema_version": 1,
        "campaign_id": context.confirmation["campaign_id"],
        "confirmation_sha256": context.confirmation_sha256,
        "confirmation_jobs_sha256": context.confirmation["jobs_sha256"],
        "launch_receipt_sha256": context.launch_receipt_sha256,
        "source_bundle_sha256": context.launch_receipt["source_bundle_sha256"],
        "screening_records_sha256": context.confirmation["provenance"][
            "screening_records_sha256"
        ],
        "robust_summary_sha256": context.confirmation["provenance"][
            "robust_summary_sha256"
        ],
        "expected_reps": list(EXPECTED_REPS),
        "expected_process_record_count": len(context.confirmation["jobs"])
        * len(EXPECTED_REPS),
        "observed_process_record_count": len(processes),
        "records_sha256": analyze_screen.canonical_sha256(record_bundle),
        "ci_rule": {
            "name": "phase1_common.median_ci",
            "point_estimator": "median_of_process_medians",
            "interval": "t-based 95% CI on mean of process medians",
            "source": "ako_runs/phase1_matmul/common.py",
            "source_sha256": launch_confirmation.sha256_file(PHASE1 / "common.py"),
        },
        "all_jobs_confirmed": all(cell["confirmation_eligible"] for cell in cells),
        "cells": cells,
        "dsl_winners": lanes,
        "four_way": four_way,
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        f"{summary['campaign_id']}: {summary['observed_process_record_count']}/"
        f"{summary['expected_process_record_count']} bound process records",
        "CI rule: median of five process medians; Phase-1 t-based 95% CI on "
        "the mean of those process medians.",
        "",
        "| DSL | point-estimate winner | confirmed ms | 95% CI ms | screening rank | inference |",
        "|---|---|---:|---:|---|---|",
    ]
    for lane in summary["dsl_winners"]:
        winner = lane["point_estimate_winner"]
        if winner is None:
            lines.append(
                f"| {lane['dsl']} | none | — | — | — | no fully confirmed candidate |"
            )
            continue
        if lane["strict_winner_resolved"]:
            inference = "resolved against all confirmed candidates"
        else:
            reasons = []
            if lane["unresolved_with"]:
                reasons.append("CI overlaps " + ", ".join(lane["unresolved_with"]))
            if lane["failed_job_ids"]:
                reasons.append("failed candidates: " + ", ".join(lane["failed_job_ids"]))
            inference = "point estimate only; " + "; ".join(reasons)
        rank = "held" if winner["screening_rank_one_held"] else "flipped"
        lines.append(
            f"| {lane['dsl']} | `{winner['screening_job_id']}` | "
            f"{winner['median_of_medians_ms']:.4f} | "
            f"[{winner['ci95_lo_ms']:.4f}, {winner['ci95_hi_ms']:.4f}] | "
            f"{rank} | {inference} |"
        )
    four_way = summary["four_way"]
    lines.append("")
    if four_way["available"]:
        lines.append(
            f"Four-way point-estimate spread: {four_way['point_estimate_spread_x']:.3f}x "
            f"({four_way['fastest_point_estimate_dsl']} fastest point estimate; "
            f"{four_way['slowest_point_estimate_dsl']} slowest)."
        )
        if four_way["strict_order_resolved"]:
            lines.append(
                "Strict cross-DSL order resolved by the reported intervals: "
                + " < ".join(four_way["strict_order"])
                + "."
            )
        else:
            overlaps = ", ".join(
                f"{left}/{right}" for left, right in four_way["overlapping_pairs"]
            ) or "within-DSL winner uncertainty"
            lines.append(
                "No strict cross-DSL order is asserted; unresolved comparisons: "
                + overlaps
                + "."
            )
    else:
        lines.append("Four-way spread unavailable because at least one DSL has no confirmed job.")
    if not summary["all_jobs_confirmed"]:
        failed = [cell["screening_job_id"] for cell in summary["cells"] if not cell[
            "confirmation_eligible"
        ]]
        lines.append("Retained failed confirmation jobs: " + ", ".join(failed) + ".")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--json", action="store_true", help="print full JSON instead of table")
    parser.add_argument("--require-all-pass", action="store_true")
    args = parser.parse_args()
    try:
        context = load_context(args.confirmation.resolve(), args.launch_receipt.resolve())
        summary = build_summary(context, args.raw_dir.resolve())
        if args.summary_out:
            analyze_screen.atomic_write(
                args.summary_out.resolve(), analyze_screen.stable_json_bytes(summary)
            )
        if args.json:
            print(analyze_screen.stable_json_bytes(summary).decode("utf-8"), end="")
        else:
            print(render(summary))
        return 1 if args.require_all_pass and not summary["all_jobs_confirmed"] else 0
    except (
        ConfirmationAnalysisError,
        analyze_screen.AnalysisError,
        launch_confirmation.ConfirmationError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
