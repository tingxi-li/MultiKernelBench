#!/usr/bin/env python3
"""Validate and analyze one complete archived-current fused result namespace."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.archived_current_fused_v1 import protocol
else:  # pragma: no cover
    from . import protocol


def contained(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise protocol.CampaignError(f"path is outside {root}: {resolved}") from exc
    return resolved


def load_stable(path: Path) -> dict[str, Any]:
    value = protocol.read_json(path)
    if path.read_bytes() != protocol.stable_json_bytes(value):
        raise protocol.CampaignError(f"artifact is not stable JSON: {path}")
    return value


def validate_records(result: Path):
    campaign, receipt, jobs, lock = protocol.verify_lock()
    launch_path = result / "launch_receipt.json"
    completion_path = result / "completion_receipt.json"
    launch = load_stable(launch_path)
    completion = load_stable(completion_path)
    if launch.get("record_type") != "archived_current_fused_v1_launch":
        raise protocol.CampaignError("invalid launch receipt type")
    if launch.get("campaign_canonical_sha256") != protocol.canonical_sha256(campaign):
        raise protocol.CampaignError("launch campaign binding differs")
    launch_sha256 = protocol.sha256_file(launch_path)
    if completion.get("launch_receipt_file_sha256") != launch_sha256:
        raise protocol.CampaignError("completion launch binding differs")
    if completion.get("success") is not True or completion.get("failures") != 0:
        raise protocol.CampaignError("campaign did not complete without failures")
    expected_paths = {
        (result / protocol.raw_relative(job)).resolve(): job for job in jobs["plan"]
    }
    observed_paths = {path.resolve() for path in (result / "raw").glob("block*/*.json")}
    if observed_paths != set(expected_paths):
        missing = sorted(str(path) for path in set(expected_paths) - observed_paths)
        extra = sorted(str(path) for path in observed_paths - set(expected_paths))
        raise protocol.CampaignError(f"raw census differs; missing={missing} extra={extra}")
    outcome_map = {
        row["job_id"]: row for row in completion.get("outcomes", [])
    }
    if set(outcome_map) != {job["job_id"] for job in jobs["plan"]}:
        raise protocol.CampaignError("completion outcome census differs")
    subject_defs = protocol.subject_map(campaign)
    records: list[dict[str, Any]] = []
    for path, job in expected_paths.items():
        record = load_stable(path)
        if record.get("record_type") != "archived_current_fused_v1_measurement":
            raise protocol.CampaignError(f"invalid raw type: {path}")
        if record.get("job") != job or record.get("subject") != subject_defs[job["subject_id"]]:
            raise protocol.CampaignError(f"raw job/subject binding differs: {path}")
        if record.get("campaign_canonical_sha256") != protocol.canonical_sha256(campaign):
            raise protocol.CampaignError(f"raw campaign binding differs: {path}")
        if record.get("launch_receipt_file_sha256") != launch_sha256:
            raise protocol.CampaignError(f"raw launch binding differs: {path}")
        outcome = outcome_map[job["job_id"]]
        if outcome.get("raw_sha256") != protocol.sha256_file(path):
            raise protocol.CampaignError(f"completion raw hash differs: {path}")
        if not record.get("ok") or record.get("correctness_diagnostic", {}).get("pass") is not True:
            raise protocol.CampaignError(f"failed or unqualified record: {path}")
        if record.get("loaded_source_sha256") != subject_defs[job["subject_id"]]["expected_source_sha256"]:
            raise protocol.CampaignError(f"loaded source hash differs: {path}")
        times = record.get("trial_times_ms")
        if not isinstance(times, list) or len(times) != campaign["performance_protocol"]["trials"]:
            raise protocol.CampaignError(f"raw trial census differs: {path}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in times):
            raise protocol.CampaignError(f"raw timing values invalid: {path}")
        median = statistics.median(times)
        if abs(median - record["timing_summary"]["median_ms"]) > 1e-12:
            raise protocol.CampaignError(f"raw median does not recompute: {path}")
        records.append(record)
    return campaign, receipt, jobs, lock, launch, completion, records


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Archived versus current fused rebenchmark — results",
        "",
        f"Campaign: `{summary['campaign_id']}`. All {summary['record_census']['accepted']} preregistered records passed the bound per-process correctness diagnostic.",
        "",
        "Each number below is the median of 15 independent process medians; brackets are the exact [x4,x12] order-statistic interval (96.484% achieved coverage).",
        "",
        "| Subject | Median ms | Exact interval ms |",
        "|---|---:|---:|",
    ]
    for subject in summary["subject_order_by_median"]:
        row = summary["subjects"][subject]
        ci = row["exact_median_interval"]
        lines.append(f"| {subject} | {ci['median']:.6f} | [{ci['lo']:.6f}, {ci['hi']:.6f}] |")
    lines.extend(
        [
            "",
            "The preregistered paired estimand is `current / archived` within each randomized block; values below 1 favor the current file.",
            "",
            "| Contrast | Median ratio | Exact interval | Holm p | Decision |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for contrast_id in [row["contrast_id"] for row in summary["preregistered_contrasts"]]:
        row = summary["contrasts"][contrast_id]
        ci = row["exact_median_interval"]
        lines.append(
            f"| {contrast_id} | {ci['median']:.6f} | [{ci['lo']:.6f}, {ci['hi']:.6f}] | "
            f"{row['sign_test']['p_holm']:.6g} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation is deliberately artifact-level: archived and current files differ in more than code generation, so these data resolve contemporaneous file performance, not an intrinsic DSL effect. The performance-input diagnostic is not a full fused-v2 4×64 gate acceptance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = contained(args.result, protocol.HERE / "results")
    out_dir = contained(args.out_dir, protocol.HERE / "analysis")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise protocol.CampaignError(f"refusing to overwrite analysis: {out_dir}")
    campaign, receipt, jobs, lock, launch, completion, records = validate_records(result)
    by_subject: dict[str, list[dict[str, Any]]] = {
        row["subject_id"]: [] for row in campaign["subjects"]
    }
    by_block: dict[int, dict[str, dict[str, Any]]] = {
        block: {} for block in range(campaign["performance_protocol"]["blocks"])
    }
    for record in records:
        subject_id = record["job"]["subject_id"]
        by_subject[subject_id].append(record)
        by_block[record["job"]["block"]][subject_id] = record
    subjects: dict[str, Any] = {}
    for subject_id, rows in by_subject.items():
        rows.sort(key=lambda row: row["job"]["block"])
        medians = [row["timing_summary"]["median_ms"] for row in rows]
        subjects[subject_id] = {
            "n_processes": len(rows),
            "process_medians_by_block_ms": medians,
            "exact_median_interval": protocol.exact_median_interval(medians),
            "max_correctness_error": max(row["correctness_diagnostic"]["max_abs_err"] for row in rows),
            "max_row_sum_error": max(row["correctness_diagnostic"]["row_sum_error_max"] for row in rows),
            "source_sha256": rows[0]["loaded_source_sha256"],
        }
    contrasts: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for definition in campaign["preregistered_contrasts"]:
        contrast_id = definition["contrast_id"]
        ratios = []
        blocks = []
        for block in range(campaign["performance_protocol"]["blocks"]):
            current = by_block[block][definition["current"]]["timing_summary"]["median_ms"]
            archived = by_block[block][definition["archived"]]["timing_summary"]["median_ms"]
            ratio = current / archived
            ratios.append(ratio)
            blocks.append(
                {
                    "block": block,
                    "current_ms": current,
                    "archived_ms": archived,
                    "current_over_archived": ratio,
                }
            )
        interval = protocol.exact_median_interval(ratios)
        sign = protocol.exact_two_sided_sign_p(ratios)
        raw_p[contrast_id] = sign["p_raw"]
        contrasts[contrast_id] = {
            **definition,
            "n_paired_blocks": len(ratios),
            "paired_blocks": blocks,
            "exact_median_interval": interval,
            "median_current_delta_pct": 100.0 * (interval["median"] - 1.0),
            "sign_test": sign,
        }
    adjusted = protocol.holm_adjust(raw_p)
    for contrast_id, row in contrasts.items():
        row["sign_test"]["p_holm"] = adjusted[contrast_id]
        interval = row["exact_median_interval"]
        if interval["hi"] < 1.0 and adjusted[contrast_id] < 0.05:
            row["decision"] = "current_faster"
        elif interval["lo"] > 1.0 and adjusted[contrast_id] < 0.05:
            row["decision"] = "archived_faster"
        else:
            row["decision"] = "unresolved"
    summary = {
        "schema_version": 1,
        "record_type": "archived_current_fused_v1_analysis",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": protocol.canonical_sha256(campaign),
        "source_receipt_file_sha256": protocol.sha256_file(protocol.SOURCE_RECEIPT_PATH),
        "jobs_file_sha256": protocol.sha256_file(protocol.JOBS_PATH),
        "launch_lock_file_sha256": protocol.sha256_file(protocol.LOCK_PATH),
        "launch_receipt_file_sha256": protocol.sha256_file(result / "launch_receipt.json"),
        "completion_receipt_file_sha256": protocol.sha256_file(result / "completion_receipt.json"),
        "inference": campaign["inference"],
        "record_census": {
            "expected": jobs["expected_records"],
            "observed": len(records),
            "accepted": sum(bool(row["ok"] and row["correctness_diagnostic"]["pass"]) for row in records),
        },
        "subjects": subjects,
        "subject_order_by_median": sorted(
            subjects,
            key=lambda subject: (subjects[subject]["exact_median_interval"]["median"], subject),
        ),
        "preregistered_contrasts": campaign["preregistered_contrasts"],
        "contrasts": contrasts,
        "qualification": campaign["scope"]["exclusions"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol.atomic_json(out_dir / "summary.json", summary)
    (out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(protocol.stable_json_bytes(summary).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

