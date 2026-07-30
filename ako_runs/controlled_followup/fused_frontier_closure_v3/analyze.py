#!/usr/bin/env python3
"""Fail-closed paired analysis for fused frontier closure v3."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.fused_frontier_closure_v3 import (
        core,
        eligibility,
        provenance,
    )
else:  # pragma: no cover
    from . import core, eligibility, provenance


TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _raw_relative(item: dict[str, Any]) -> Path:
    return Path("raw") / f"block{item['block']:02d}" / f"{item['candidate_id']}.json"


def _load_launch(
    campaign: dict[str, Any], source: dict[str, Any], path: Path, tag: str
) -> tuple[dict[str, Any], str]:
    launch = core.read_json(path)
    if (
        launch.get("record_type") != "fused_frontier_closure_v3_launch"
        or launch.get("tag") != tag
        or launch.get("plan") != core.block_plan(campaign)
        or path.read_bytes() != core.stable_json_bytes(launch)
    ):
        raise core.ClosureError("launch receipt differs from frozen plan")
    binding = launch["binding"]
    expected = {
        "campaign_canonical_sha256": core.canonical_sha256(campaign),
        "source_receipt_sha256": core.sha256_file(core.SOURCE_RECEIPT_PATH),
        "eligibility_receipt_sha256": core.sha256_file(
            core.ELIGIBILITY_RECEIPT_PATH
        ),
        "performance_protocol_sha256": core.protocol_sha256(campaign),
        "candidate_sha256": source["candidate_sha256"],
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise core.ClosureError(f"launch {key} binding differs")
    return launch, core.sha256_file(path)


def _validate_raw(
    record: dict[str, Any],
    path: Path,
    item: dict[str, Any],
    campaign: dict[str, Any],
    launch: dict[str, Any],
    launch_sha: str,
) -> None:
    definition = core.candidates_by_id(campaign)[item["candidate_id"]]
    binding = launch["binding"]
    expected = {
        "record_type": "fused_frontier_closure_v3_measurement",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": binding["campaign_canonical_sha256"],
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "eligibility_receipt_sha256": binding["eligibility_receipt_sha256"],
        "performance_protocol_sha256": binding["performance_protocol_sha256"],
        "launch_receipt_sha256": launch_sha,
        "candidate_id": item["candidate_id"],
        "candidate_definition": definition,
        "candidate_definition_sha256": binding["candidate_sha256"][
            item["candidate_id"]
        ],
        "block": item["block"],
        "position": item["position"],
        "protocol": campaign["performance_protocol"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise core.ClosureError(f"raw {key} differs: {path}")
    if path.read_bytes() != core.stable_json_bytes(record):
        raise core.ClosureError(f"raw record is not stable JSON: {path}")
    if record.get("ok"):
        times = record.get("trial_times_ms")
        if (
            not isinstance(times, list)
            or len(times) != 100
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in times
            )
        ):
            raise core.ClosureError(f"invalid trial list: {path}")
        median = record.get("timing_summary", {}).get("median_ms")
        if not isinstance(median, (int, float)) or not math.isfinite(median):
            raise core.ClosureError(f"invalid process median: {path}")


def analyze_records(
    campaign: dict[str, Any], records: dict[tuple[int, str], dict[str, Any]]
) -> dict[str, Any]:
    blocks = campaign["performance_protocol"]["blocks"]
    times: dict[str, dict[int, float]] = {}
    cells = []
    for candidate_id in campaign["candidate_order"]:
        by_block = {
            block: records[(block, candidate_id)]["timing_summary"]["median_ms"]
            for block in range(blocks)
            if records[(block, candidate_id)].get("ok")
        }
        times[candidate_id] = by_block
        cell: dict[str, Any] = {
            "candidate_id": candidate_id,
            "original_gate_eligible": True,
            "eligibility_scope": campaign["eligibility"]["scope"],
            "n_processes": len(by_block),
            "complete": len(by_block) == blocks,
            "process_medians_ms_by_block": {
                str(block): value for block, value in sorted(by_block.items())
            },
        }
        if cell["complete"]:
            interval = core.exact_median_interval(by_block.values())
            cell["median_of_process_medians_ms"] = interval["median"]
            cell["median_interval_ms"] = interval
        cells.append(cell)
    comparisons = []
    for family, pairs in campaign["preregistered_families"].items():
        family_rows, p_values = [], []
        for numerator, denominator in pairs:
            paired_blocks = sorted(set(times[numerator]) & set(times[denominator]))
            ratios = [
                times[numerator][block] / times[denominator][block]
                for block in paired_blocks
            ]
            row: dict[str, Any] = {
                "family": family,
                "numerator": numerator,
                "denominator": denominator,
                "ratio_definition": "numerator_ms / denominator_ms",
                "n_paired_blocks": len(ratios),
                "complete": len(ratios) == blocks,
                "both_original_gate_eligible": True,
                "ratios_by_block": {
                    str(block): ratio for block, ratio in zip(paired_blocks, ratios)
                },
            }
            if row["complete"]:
                row["ratio_interval"] = core.exact_median_interval(ratios)
                row["sign_test"] = core.exact_sign_test(ratios)
                p_values.append(row["sign_test"]["p_value_two_sided"])
            else:
                p_values.append(1.0)
            family_rows.append(row)
        for row, adjusted in zip(family_rows, core.holm_adjust(p_values)):
            row["holm_adjusted_p_value"] = adjusted
            direction = "unresolved"
            if row["complete"] and adjusted < 0.05:
                if row["ratio_interval"]["ci_hi"] < 1:
                    direction = "numerator_faster"
                elif row["ratio_interval"]["ci_lo"] > 1:
                    direction = "numerator_slower"
            row["directional_result"] = direction
            comparisons.append(row)
    spread_ids = campaign["fixed_frontier_spread"]
    spreads = {}
    for block in range(blocks):
        values = [times[candidate_id].get(block) for candidate_id in spread_ids]
        if all(isinstance(value, (int, float)) for value in values):
            spreads[str(block)] = max(values) / min(values)
    spread: dict[str, Any] = {
        "candidate_ids": spread_ids,
        "scope": "fixed recipes only; no universal DSL optimum or noptx winner claim",
        "spread_definition": "within-block slowest process median / fastest process median",
        "spread_by_block": spreads,
        "n_blocks": len(spreads),
        "complete": len(spreads) == blocks,
    }
    if spread["complete"]:
        spread["spread_interval"] = core.exact_median_interval(spreads.values())
    failures = [
        {
            "block": block,
            "candidate_id": candidate,
            "error": record.get("error", "unknown failure"),
        }
        for (block, candidate), record in sorted(records.items())
        if not record.get("ok")
    ]
    return {
        "performance_complete": not failures
        and len(records) == blocks * len(campaign["candidate_order"]),
        "failed_measurements": failures,
        "cells": cells,
        "paired_comparisons": comparisons,
        "fixed_frontier_spread": spread,
        "eligibility_scope": campaign["eligibility"]["scope"],
        "eligibility_claim_limit": campaign["eligibility"]["claim_limit"],
        "claim_limit": campaign["claim_limit"],
    }


def _markdown(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    lines = [
        "# Fused frontier closure v3 results",
        "",
        f"Status: **{summary['status']}**. Raw measurements: "
        f"{summary['observed_records']}/{summary['expected_records']}.",
        "",
        "## Candidate medians",
        "",
        "| candidate | median ms | exact interval ms | n |",
        "|---|---:|---:|---:|",
    ]
    for cell in analysis["cells"]:
        if cell["complete"]:
            interval = cell["median_interval_ms"]
            point = f"{interval['median']:.6f}"
            ci = f"[{interval['ci_lo']:.6f}, {interval['ci_hi']:.6f}]"
        else:
            point = ci = "NA"
        lines.append(f"| {cell['candidate_id']} | {point} | {ci} | {cell['n_processes']} |")
    lines.extend(
        [
            "",
            "## Preregistered paired ratios",
            "",
            "Ratios are numerator time divided by denominator time.",
            "",
            "| family | numerator / denominator | median | exact interval | Holm p | result |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in analysis["paired_comparisons"]:
        if row["complete"]:
            interval = row["ratio_interval"]
            point = f"{interval['median']:.6f}"
            ci = f"[{interval['ci_lo']:.6f}, {interval['ci_hi']:.6f}]"
        else:
            point = ci = "NA"
        lines.append(
            f"| {row['family']} | {row['numerator']} / {row['denominator']} | "
            f"{point} | {ci} | {row['holm_adjusted_p_value']:.6g} | "
            f"{row['directional_result']} |"
        )
    spread = analysis["fixed_frontier_spread"]
    lines.extend(["", "## Fixed-frontier spread", ""])
    if spread["complete"]:
        interval = spread["spread_interval"]
        lines.append(
            f"Median within-block spread: {interval['median']:.6f}x "
            f"(exact interval [{interval['ci_lo']:.6f}, {interval['ci_hi']:.6f}]x)."
        )
    else:
        lines.append("Spread unavailable because at least one fixed-set block failed.")
    lines.extend(
        [
            "",
            "Eligibility is imported from the original frozen 4-case × 64-seed "
            "mixed validation gates. It is not fresh-stress eligibility.",
            "",
            "The no-PTX g05/g09 winner remains unresolved by design. All conclusions "
            "concern these fixed recipes on one GPU, not universal DSL optima.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="performance_v1")
    args = parser.parse_args()
    if not TAG.fullmatch(args.tag):
        parser.error("unsafe result tag")
    campaign = core.load_campaign()
    source = provenance.verify_receipt()
    eligibility.verify_receipt()
    root = core.RESULTS_ROOT / args.tag
    launch, launch_sha = _load_launch(
        campaign, source, root / "launch_receipt.json", args.tag
    )
    plan = launch["plan"]
    expected = {root / _raw_relative(item): item for item in plan}
    observed = set((root / "raw").rglob("*.json"))
    if observed != set(expected):
        raise core.ClosureError("raw path set differs from plan")
    records, hashes = {}, {}
    for path, item in expected.items():
        record = core.read_json(path)
        _validate_raw(record, path, item, campaign, launch, launch_sha)
        key = (item["block"], item["candidate_id"])
        if key in records:
            raise core.ClosureError(f"duplicate record {key}")
        records[key] = record
        hashes[str(path.relative_to(core.REPO_ROOT))] = core.sha256_file(path)
    analysis = analyze_records(campaign, records)
    summary = {
        "schema_version": 1,
        "record_type": "fused_frontier_closure_v3_analysis",
        "campaign_id": campaign["campaign_id"],
        "tag": args.tag,
        "status": "COMPLETE" if analysis["performance_complete"] else "FAILED_CLOSED",
        "source_receipt_sha256": core.sha256_file(core.SOURCE_RECEIPT_PATH),
        "eligibility_receipt_sha256": core.sha256_file(
            core.ELIGIBILITY_RECEIPT_PATH
        ),
        "launch_receipt_sha256": launch_sha,
        "expected_records": len(plan),
        "observed_records": len(records),
        "raw_record_sha256": hashes,
        "analysis": analysis,
    }
    summary_path, report_path = root / "analysis_summary.json", root / "ANALYSIS.md"
    if summary_path.exists() and core.read_json(summary_path) != summary:
        raise core.ClosureError("existing analysis summary differs")
    core.atomic_json(summary_path, summary)
    report = _markdown(summary)
    if report_path.exists() and report_path.read_text(encoding="utf-8") != report:
        raise core.ClosureError("existing analysis report differs")
    _atomic_text(report_path, report)
    print(f"wrote {summary_path} and {report_path}; status={summary['status']}")
    return 0 if analysis["performance_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
