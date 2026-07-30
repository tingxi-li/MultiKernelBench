#!/usr/bin/env python3
"""Fail-closed paired analysis for the frozen fused closure-v2 campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.fused_closure_v2 import core, provenance
else:  # pragma: no cover
    from . import core, provenance


TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _load_stable(path: Path, record_type: str) -> dict[str, Any]:
    value = core.read_json(path)
    if value.get("record_type") != record_type:
        raise core.ClosureError(f"unexpected record type in {path}")
    if path.read_bytes() != core.stable_json_bytes(value):
        raise core.ClosureError(f"artifact is not stable JSON: {path}")
    return value


def _raw_relative(item: dict[str, Any]) -> Path:
    return Path("raw") / f"block{item['block']:02d}" / f"{item['candidate_id']}.json"


def _validate_launch(
    campaign: dict[str, Any], source_receipt: dict[str, Any], path: Path, tag: str
) -> tuple[dict[str, Any], str]:
    launch = _load_stable(path, "fused_closure_v2_performance_launch")
    if launch.get("tag") != tag or launch.get("plan") != core.block_plan(campaign):
        raise core.ClosureError("launch tag or randomized block plan differs")
    binding = launch.get("binding", {})
    expected = {
        "campaign_canonical_sha256": core.canonical_sha256(campaign),
        "source_receipt_sha256": core.sha256_file(core.SOURCE_RECEIPT_PATH),
        "performance_protocol_sha256": core.protocol_sha256(campaign),
        "candidate_sha256": source_receipt["candidate_sha256"],
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise core.ClosureError(f"launch {key} binding differs")
    return launch, core.sha256_file(path)


def _validate_gate(launch: dict[str, Any]) -> dict[str, Any]:
    path = core.REPO_ROOT / launch["gate_summary_path"]
    summary = _load_stable(path, "fused_closure_v2_gate_summary")
    if core.sha256_file(path) != launch["binding"]["gate_summary_sha256"]:
        raise core.ClosureError("gate summary hash differs from performance launch")
    if not summary.get("coverage_complete") or not summary.get(
        "performance_launch_allowed"
    ):
        raise core.ClosureError("gate summary is incomplete")
    eligibility = {
        item["candidate_id"]: item["same_contract_eligible"]
        for item in summary["adjudications"]
    }
    if eligibility != launch["binding"]["gate_eligibility"]:
        raise core.ClosureError("gate eligibility differs from launch receipt")
    return summary


def _validate_record(
    record: dict[str, Any],
    *,
    path: Path,
    item: dict[str, Any],
    campaign: dict[str, Any],
    launch: dict[str, Any],
    launch_sha256: str,
) -> None:
    if record.get("record_type") != "fused_closure_v2_performance_measurement":
        raise core.ClosureError(f"foreign raw record: {path}")
    definition = core.candidates_by_id(campaign)[item["candidate_id"]]
    binding = launch["binding"]
    exact = {
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": binding["campaign_canonical_sha256"],
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "performance_protocol_sha256": binding["performance_protocol_sha256"],
        "gate_summary_sha256": binding["gate_summary_sha256"],
        "performance_launch_receipt_sha256": launch_sha256,
        "candidate_id": item["candidate_id"],
        "candidate_definition": definition,
        "candidate_definition_sha256": binding["candidate_sha256"][
            item["candidate_id"]
        ],
        "block": item["block"],
        "position": item["position"],
        "protocol": campaign["performance_protocol"],
    }
    for key, value in exact.items():
        if record.get(key) != value:
            raise core.ClosureError(f"raw record {key} differs: {path}")
    if path.read_bytes() != core.stable_json_bytes(record):
        raise core.ClosureError(f"raw record is not stable JSON: {path}")
    if record.get("ok"):
        trials = record.get("trial_times_ms")
        if not isinstance(trials, list) or len(trials) != campaign[
            "performance_protocol"
        ]["trials"]:
            raise core.ClosureError(f"raw record trial count differs: {path}")
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in trials
        ):
            raise core.ClosureError(f"raw record has invalid trial time: {path}")
        median = record.get("timing_summary", {}).get("median_ms")
        if not isinstance(median, (int, float)) or not math.isfinite(median):
            raise core.ClosureError(f"raw record has invalid process median: {path}")


def analyze_records(
    campaign: dict[str, Any],
    records: dict[tuple[int, str], dict[str, Any]],
    eligibility: dict[str, bool],
) -> dict[str, Any]:
    blocks = campaign["performance_protocol"]["blocks"]
    cells = []
    cell_times: dict[str, dict[int, float]] = {}
    for candidate_id in campaign["candidate_order"]:
        by_block = {
            block: records[(block, candidate_id)]["timing_summary"]["median_ms"]
            for block in range(blocks)
            if records[(block, candidate_id)].get("ok")
        }
        cell_times[candidate_id] = by_block
        cell: dict[str, Any] = {
            "candidate_id": candidate_id,
            "same_contract_eligible": eligibility[candidate_id],
            "n_successful_processes": len(by_block),
            "process_medians_ms_by_block": {
                str(block): value for block, value in sorted(by_block.items())
            },
            "complete": len(by_block) == blocks,
        }
        if cell["complete"]:
            interval = core.exact_median_interval(by_block.values())
            cell.update(
                {
                    "median_of_process_medians_ms": interval["median"],
                    "median_interval_ms": interval,
                }
            )
        cells.append(cell)

    comparisons = []
    for family, pairs in campaign["preregistered_families"].items():
        family_rows = []
        raw_p_values = []
        for numerator, denominator in pairs:
            common_blocks = sorted(
                set(cell_times[numerator]) & set(cell_times[denominator])
            )
            ratios = [
                cell_times[numerator][block] / cell_times[denominator][block]
                for block in common_blocks
            ]
            complete = len(ratios) == blocks
            row: dict[str, Any] = {
                "family": family,
                "numerator": numerator,
                "denominator": denominator,
                "ratio_definition": "numerator_ms / denominator_ms",
                "n_paired_blocks": len(ratios),
                "ratios_by_block": {
                    str(block): ratio for block, ratio in zip(common_blocks, ratios)
                },
                "complete": complete,
                "both_same_contract_eligible": bool(
                    eligibility[numerator] and eligibility[denominator]
                ),
            }
            if complete:
                row["ratio_interval"] = core.exact_median_interval(ratios)
                row["sign_test"] = core.exact_sign_test(ratios)
                raw_p_values.append(row["sign_test"]["p_value_two_sided"])
            else:
                raw_p_values.append(1.0)
            family_rows.append(row)
        adjusted = core.holm_adjust(raw_p_values)
        for row, value in zip(family_rows, adjusted):
            row["holm_adjusted_p_value"] = value
            confirmatory_scope = row["family"] == "same_contract_vendor"
            row["confirmatory_same_contract_scope"] = confirmatory_scope
            row["faster_claim_supported"] = bool(
                confirmatory_scope
                and row["complete"]
                and row["both_same_contract_eligible"]
                and row["ratio_interval"]["ci_hi"] < 1.0
                and value < 0.05
            )
            comparisons.append(row)

    common_ids = [
        "tilelang_common_g03",
        "triton_common_g00",
        "cuda_noptx_common_g04",
        "cuda_unlimited_common_g02",
    ]
    common_spreads = {}
    for block in range(blocks):
        values = [cell_times[candidate_id].get(block) for candidate_id in common_ids]
        if all(isinstance(value, (int, float)) for value in values):
            common_spreads[str(block)] = max(values) / min(values)
    common: dict[str, Any] = {
        "candidate_ids": common_ids,
        "all_same_contract_eligible": all(eligibility[item] for item in common_ids),
        "spread_definition": "within-block max(process median) / min(process median)",
        "spread_by_block": common_spreads,
        "n_complete_blocks": len(common_spreads),
        "complete": len(common_spreads) == blocks,
        "scope": "frozen strict-common recipes; not latent DSL optima",
    }
    if common["complete"]:
        common["spread_interval"] = core.exact_median_interval(
            common_spreads.values()
        )
    failed = [
        {
            "block": block,
            "candidate_id": candidate_id,
            "error": record.get("error", "unknown failure"),
        }
        for (block, candidate_id), record in sorted(records.items())
        if not record.get("ok")
    ]
    return {
        "performance_complete": not failed and len(records) == blocks * len(
            campaign["candidate_order"]
        ),
        "failed_measurements": failed,
        "cells": cells,
        "paired_comparisons": comparisons,
        "common_recipe_spread": common,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fused closure v2 results",
        "",
        f"Status: **{summary['status']}**. Raw measurements: "
        f"{summary['observed_records']}/{summary['expected_records']}.",
        "",
        "## Candidate process medians",
        "",
        "| candidate | eligible | median ms | exact median interval ms | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in summary["analysis"]["cells"]:
        if cell["complete"]:
            interval = cell["median_interval_ms"]
            median = f"{cell['median_of_process_medians_ms']:.6f}"
            ci = f"[{interval['ci_lo']:.6f}, {interval['ci_hi']:.6f}]"
        else:
            median, ci = "NA", "NA"
        lines.append(
            f"| {cell['candidate_id']} | {cell['same_contract_eligible']} | "
            f"{median} | {ci} | {cell['n_successful_processes']} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered paired contrasts",
            "",
            "Ratios are numerator time divided by denominator time; values below one "
            "favor the numerator.",
            "",
            "| family | numerator / denominator | median ratio | exact interval | Holm p | claim |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["analysis"]["paired_comparisons"]:
        if row["complete"]:
            interval = row["ratio_interval"]
            median = f"{interval['median']:.6f}"
            ci = f"[{interval['ci_lo']:.6f}, {interval['ci_hi']:.6f}]"
        else:
            median, ci = "NA", "NA"
        lines.append(
            f"| {row['family']} | {row['numerator']} / {row['denominator']} | "
            f"{median} | {ci} | {row['holm_adjusted_p_value']:.6g} | "
            f"{row['faster_claim_supported']} |"
        )
    common = summary["analysis"]["common_recipe_spread"]
    lines.extend(["", "## Strict-common frozen-recipe spread", ""])
    if common["complete"]:
        interval = common["spread_interval"]
        lines.append(
            f"Median within-block slowest/fastest spread: {interval['median']:.6f}x "
            f"(exact interval [{interval['ci_lo']:.6f}, {interval['ci_hi']:.6f}]x)."
        )
    else:
        lines.append("The common-recipe spread is unavailable because a block failed.")
    lines.extend(
        [
            "",
            "Historical-torch contrasts are diagnostic because the historical arms "
            "do not satisfy the fused-v2 structural arithmetic contract. The common "
            "spread concerns four frozen recipes, not universal DSL optima.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="performance_v1")
    args = parser.parse_args()
    if not TAG_PATTERN.fullmatch(args.tag):
        parser.error("--tag contains unsupported characters")
    campaign = core.load_campaign()
    source_receipt = provenance.verify_receipt()
    result_root = core.RESULTS_ROOT / args.tag
    launch_path = result_root / "launch_receipt.json"
    launch, launch_sha = _validate_launch(
        campaign, source_receipt, launch_path, args.tag
    )
    gate = _validate_gate(launch)
    plan = launch["plan"]
    expected_paths = {result_root / _raw_relative(item): item for item in plan}
    observed_paths = set((result_root / "raw").rglob("*.json"))
    if observed_paths != set(expected_paths):
        raise core.ClosureError(
            "raw record path set differs from preregistered launch plan"
        )
    records: dict[tuple[int, str], dict[str, Any]] = {}
    raw_hashes = {}
    for path, item in expected_paths.items():
        record = core.read_json(path)
        _validate_record(
            record,
            path=path,
            item=item,
            campaign=campaign,
            launch=launch,
            launch_sha256=launch_sha,
        )
        key = (item["block"], item["candidate_id"])
        if key in records:
            raise core.ClosureError(f"duplicate raw record key {key!r}")
        records[key] = record
        raw_hashes[str(path.relative_to(core.REPO_ROOT))] = core.sha256_file(path)
    eligibility = {
        item["candidate_id"]: item["same_contract_eligible"]
        for item in gate["adjudications"]
    }
    analysis = analyze_records(campaign, records, eligibility)
    summary = {
        "schema_version": 1,
        "record_type": "fused_closure_v2_performance_analysis",
        "campaign_id": campaign["campaign_id"],
        "tag": args.tag,
        "status": "COMPLETE" if analysis["performance_complete"] else "FAILED_CLOSED",
        "source_receipt_sha256": core.sha256_file(core.SOURCE_RECEIPT_PATH),
        "launch_receipt_sha256": launch_sha,
        "gate_summary_sha256": launch["binding"]["gate_summary_sha256"],
        "expected_records": len(plan),
        "observed_records": len(records),
        "raw_record_sha256": raw_hashes,
        "analysis": analysis,
    }
    summary_path = result_root / "analysis_summary.json"
    report_path = result_root / "ANALYSIS.md"
    if summary_path.exists() and core.read_json(summary_path) != summary:
        raise core.ClosureError("existing analysis summary differs; refusing overwrite")
    core.atomic_json(summary_path, summary)
    report = _markdown(summary)
    if report_path.exists() and report_path.read_text(encoding="utf-8") != report:
        raise core.ClosureError("existing analysis report differs; refusing overwrite")
    _atomic_text(report_path, report)
    print(f"wrote {summary_path} and {report_path}; status={summary['status']}")
    return 0 if analysis["performance_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
