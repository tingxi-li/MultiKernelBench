#!/usr/bin/env python3
"""Audit fused-grid screening records and prepare gated confirmation jobs.

This program never imports the GPU launcher and never mutates its result tree.
It treats ``ok`` as execution/build success only: a screening candidate is
eligible only when both process records also contain an explicit successful
legacy gate.  Robust eligibility comes exclusively from a separately supplied,
content-addressed robust-gate summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "manifest.json"
DEFAULT_ADAPTER_MANIFEST = HERE / "robust_adapter_manifest.json"
EXPECTED_REPS = (0, 1)
TOP_K = 3
OLD_INCUMBENT_GRID_ID = "g01"
ROBUST_OPERATION = "fused_softmax"
REQUIRED_GATE_IDS = ("semantic_mixed", "conformance_mixed")
HEX64 = set("0123456789abcdef")


class AnalysisError(ValueError):
    """An input is malformed, stale, ambiguous, or incompletely bound."""


@dataclass(frozen=True)
class Campaign:
    manifest: dict[str, Any]
    manifest_sha256: str
    jobs: tuple[dict[str, Any], ...]
    jobs_by_id: dict[str, dict[str, Any]]
    jobs_path: Path


@dataclass(frozen=True)
class Analysis:
    campaign: Campaign
    receipt: dict[str, Any]
    receipt_sha256: str
    summary: dict[str, Any]


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
        raise AnalysisError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def stable_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"value is not stable JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path: Path) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AnalysisError(f"invalid JSON in {path}: {exc}") from exc
    return value, raw


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must be an object")
    return value


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= HEX64:
        raise AnalysisError(f"{label} must be a lowercase SHA256")
    return value


def _int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AnalysisError(f"{label} must be an integer")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise AnalysisError(f"{label} must be {qualifier}")
    return result


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AnalysisError(f"{label} mismatch: got {actual!r}, expected {expected!r}")


def parse_set(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise AnalysisError("job set must be a non-empty string")
    result: dict[str, Any] = {}
    for item in value.split(","):
        if item.count("=") != 1:
            raise AnalysisError(f"invalid set component {item!r}")
        key, raw = (part.strip() for part in item.split("=", 1))
        if not key or key in result:
            raise AnalysisError(f"empty or duplicate set key {key!r}")
        try:
            result[key] = int(raw)
        except ValueError:
            result[key] = raw
    return result


def load_campaign(manifest_path: Path) -> Campaign:
    manifest_value, manifest_raw = read_json(manifest_path)
    manifest = _object(manifest_value, "manifest")
    _same(manifest.get("schema_version"), 1, "manifest schema_version")
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise AnalysisError("manifest campaign_id must be non-empty")

    jobs_file = manifest.get("jobs_file")
    if not isinstance(jobs_file, str) or not jobs_file:
        raise AnalysisError("manifest jobs_file must be non-empty")
    jobs_path = (manifest_path.parent / jobs_file).resolve()
    try:
        jobs_path.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise AnalysisError("manifest jobs_file escapes the manifest directory") from exc
    jobs_value, jobs_raw = read_json(jobs_path)
    if not isinstance(jobs_value, list):
        raise AnalysisError("jobs file must contain a list")
    _same(sha256_bytes(jobs_raw), _hex64(manifest.get("jobs_sha256"), "jobs_sha256"),
          "jobs file hash")
    _same(len(jobs_value), _int(manifest.get("job_count"), "job_count"),
          "job count")

    dsls = manifest.get("dsl_order")
    if (not isinstance(dsls, list) or not dsls or
            any(not isinstance(dsl, str) or not dsl for dsl in dsls) or
            len(dsls) != len(set(dsls))):
        raise AnalysisError("dsl_order must contain unique non-empty strings")
    grid_count = _int(manifest.get("grid_point_count"), "grid_point_count")
    if grid_count <= 1:
        raise AnalysisError("grid must contain the g01 incumbent")
    if len(jobs_value) != len(dsls) * grid_count:
        raise AnalysisError("job_count is not dsl_count x grid_point_count")

    manifest_grid = manifest.get("grid")
    if not isinstance(manifest_grid, list) or len(manifest_grid) != grid_count:
        raise AnalysisError("manifest grid length mismatch")
    fixed = _object(manifest.get("fixed_factors"), "fixed_factors")
    expected_axes = {"BM", "BN", "BK", "stages", "kc"}
    by_dsl: dict[str, list[dict[str, Any]]] = {dsl: [] for dsl in dsls}
    jobs_by_id: dict[str, dict[str, Any]] = {}
    required = {"dsl", "geom", "grid_id", "grid_index", "job_id", "set", "variant"}
    for position, raw_job in enumerate(jobs_value):
        job = _object(raw_job, f"job {position}")
        if not required <= set(job):
            raise AnalysisError(f"job {position} lacks {sorted(required - set(job))}")
        dsl = job["dsl"]
        if dsl not in by_dsl:
            raise AnalysisError(f"job {position} has unknown DSL {dsl!r}")
        expected_dsl = dsls[position // grid_count]
        expected_index = position % grid_count
        _same(dsl, expected_dsl, f"job {position} DSL-major order")
        _same(job["grid_index"], expected_index, f"job {position} grid_index")
        _same(job["grid_id"], f"g{expected_index:02d}", f"job {position} grid_id")
        _same(job["job_id"], f"{dsl}.{job['grid_id']}", f"job {position} job_id")
        _same(job["variant"], fixed.get("variant"), f"job {position} variant")
        _same(job["geom"], "fused", f"job {position} geom")
        if job["job_id"] in jobs_by_id:
            raise AnalysisError(f"duplicate job_id {job['job_id']!r}")

        parsed = parse_set(job["set"])
        axes = {name: parsed.get(name) for name in expected_axes}
        _same(axes, manifest_grid[expected_index], f"job {position} grid axes")
        expected_fixed = {
            "threads": fixed.get("threads"),
            "kc": fixed.get("kc"),
            "arith": fixed.get("arith"),
            "cast": fixed.get("cast"),
            "x_wcache": fixed.get("wcache"),
            "x_epilogue": fixed.get("epilogue"),
        }
        for name, expected in expected_fixed.items():
            _same(parsed.get(name), expected, f"job {position} set {name}")
        by_dsl[dsl].append(job)
        jobs_by_id[job["job_id"]] = job

    reference = [job["set"] for job in by_dsl[dsls[0]]]
    for dsl in dsls[1:]:
        _same([job["set"] for job in by_dsl[dsl]], reference, f"{dsl} shared grid")
    return Campaign(
        manifest=manifest,
        manifest_sha256=sha256_bytes(manifest_raw),
        jobs=tuple(jobs_value),
        jobs_by_id=jobs_by_id,
        jobs_path=jobs_path,
    )


def load_receipt(path: Path, campaign: Campaign) -> tuple[dict[str, Any], str]:
    value, raw = read_json(path)
    receipt = _object(value, "launch receipt")
    manifest = campaign.manifest
    expected = {
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": campaign.manifest_sha256,
        "jobs_sha256": manifest["jobs_sha256"],
        "phase1_grid_source_sha256": manifest["phase1_grid_source_sha256"],
    }
    for name, wanted in expected.items():
        _same(receipt.get(name), wanted, f"launch receipt {name}")
    for name in ("manifest_sha256", "jobs_sha256", "phase1_grid_source_sha256",
                 "protocol_sha256", "source_bundle_sha256"):
        _hex64(receipt.get(name), f"launch receipt {name}")

    args = _object(receipt.get("launch_args"), "launch_args")
    _same(args.get("reps"), 2, "launch reps")
    _same(args.get("time_only"), False, "launch time_only")
    if args.get("dist") not in ("rand", "randn"):
        raise AnalysisError("launch dist must be rand or randn")
    _int(args.get("seed"), "launch seed")
    if _int(args.get("trials"), "launch trials") <= 0:
        raise AnalysisError("launch trials must be positive")
    warmup_s = _finite(args.get("warmup_s"), "launch warmup_s")
    if warmup_s < 0:
        raise AnalysisError("launch warmup_s must be non-negative")
    protocol = {name: args[name] for name in
                ("dist", "seed", "time_only", "trials", "warmup_s")}
    _same(sha256_bytes(stable_json_bytes(protocol)), receipt["protocol_sha256"],
          "launch protocol hash")

    sources = _object(receipt.get("source_sha256"), "launch source_sha256")
    if not sources:
        raise AnalysisError("launch source_sha256 cannot be empty")
    for name, digest in sources.items():
        if not isinstance(name, str) or not name:
            raise AnalysisError("launch source path must be non-empty")
        _hex64(digest, f"launch source {name}")
    _same(sha256_bytes(stable_json_bytes(sources)), receipt["source_bundle_sha256"],
          "launch source bundle hash")
    return receipt, sha256_bytes(raw)


def _record_expected_cfg(campaign: Campaign, job: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_set(job["set"])
    fixed = campaign.manifest["fixed_factors"]
    expected = {
        name: parsed[name]
        for name in ("BM", "BN", "BK", "threads", "stages", "kc", "arith", "cast")
    }
    expected.update({name: fixed[name] for name in ("M", "K", "N")})
    expected.update({"dsl": job["dsl"], "variant": job["variant"]})
    return expected


def _validate_record(
    rec: dict[str, Any], path: Path, campaign: Campaign, receipt: dict[str, Any]
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    provenance = _object(rec.get("campaign_provenance"),
                         f"{path.name} campaign_provenance")
    job_id = provenance.get("job_id")
    if job_id not in campaign.jobs_by_id:
        raise AnalysisError(f"{path.name}: unknown job_id {job_id!r}")
    job = campaign.jobs_by_id[job_id]
    rep = _int(provenance.get("rep"), f"{path.name} provenance rep")
    if rep not in EXPECTED_REPS:
        raise AnalysisError(f"{path.name}: rep {rep} is outside {EXPECTED_REPS}")

    expected_provenance = {
        "campaign_id": receipt["campaign_id"],
        "manifest_sha256": receipt["manifest_sha256"],
        "jobs_sha256": receipt["jobs_sha256"],
        "phase1_grid_source_sha256": receipt["phase1_grid_source_sha256"],
        "protocol_sha256": receipt["protocol_sha256"],
        "source_bundle_sha256": receipt["source_bundle_sha256"],
        "source_sha256": receipt["source_sha256"],
        "git_commit": receipt.get("git_commit"),
        "grid_id": job["grid_id"],
        "grid_index": job["grid_index"],
        "job_id": job_id,
        "rep": rep,
    }
    for name, wanted in expected_provenance.items():
        _same(provenance.get(name), wanted, f"{path.name} provenance {name}")

    _same(rec.get("rep"), rep, f"{path.name} top-level rep")
    optional_top = {
        "op": campaign.manifest["fixed_factors"].get("op", "fused"),
        "dsl": job["dsl"],
        "variant": job["variant"],
        "geom": job["geom"],
        "dist": receipt["launch_args"]["dist"],
        "seed": receipt["launch_args"]["seed"],
        "trials": receipt["launch_args"]["trials"],
        "warmup_s": receipt["launch_args"]["warmup_s"],
    }
    for name, wanted in optional_top.items():
        if name in rec:
            _same(rec[name], wanted, f"{path.name} {name}")

    if not isinstance(rec.get("ok"), bool):
        raise AnalysisError(f"{path.name}: ok must be boolean")
    cfg_value = rec.get("cfg")
    cfg = _object(cfg_value, f"{path.name} cfg") if cfg_value is not None else {}
    expected_cfg = _record_expected_cfg(campaign, job)
    if rec["ok"]:
        for name, wanted in expected_cfg.items():
            _same(cfg.get(name), wanted, f"{path.name} cfg {name}")
    else:
        # Driver-level failures can contain only cfg.extra.  Validate every
        # identity-bearing field that survived, but retain the sparse record.
        for name, wanted in expected_cfg.items():
            if name in cfg:
                _same(cfg[name], wanted, f"{path.name} cfg {name}")
    extra = cfg.get("extra")
    if extra is not None:
        extra = _object(extra, f"{path.name} cfg.extra")
        parsed = parse_set(job["set"])
        for name, set_name in (("wcache", "x_wcache"), ("epilogue", "x_epilogue")):
            if name in extra:
                _same(extra[name], parsed[set_name], f"{path.name} cfg.extra {name}")

    process: dict[str, Any] = {
        "rep": rep,
        "record_file": path.name,
        "ok": rec["ok"],
    }
    if rec["ok"]:
        timing = _object(rec.get("timing"), f"{path.name} timing")
        process["median_ms"] = _finite(
            timing.get("median_ms"), f"{path.name} timing.median_ms", positive=True
        )
        error = rec.get("error")
        if not isinstance(error, dict) or not isinstance(error.get("gate_pass"), bool):
            process["legacy_gate_pass"] = False
            process["outcome"] = "legacy_gate_missing"
        else:
            process["legacy_gate_pass"] = error["gate_pass"]
            process["outcome"] = "ok" if error["gate_pass"] else "legacy_gate_failure"
            for name in ("pct_elems_failing_gate", "max_abs_err", "mean_abs_err",
                         "budget_mean"):
                if name in error and isinstance(error[name], (int, float)) and not isinstance(
                    error[name], bool
                ) and math.isfinite(float(error[name])):
                    process[name] = float(error[name])
        if "compile_s" in rec:
            process["compile_s"] = _finite(rec["compile_s"], f"{path.name} compile_s")
    else:
        process["legacy_gate_pass"] = False
        process["outcome"] = "build_or_execution_failure"
        message = rec.get("error_msg")
        if not isinstance(message, str) or not message:
            message = "unspecified failure"
        process["error_msg"] = message
        if "returncode" in rec:
            process["returncode"] = rec["returncode"]
    return job, rep, process


def _cell_status(processes: list[dict[str, Any]]) -> str:
    reps = [row["rep"] for row in processes]
    if reps != list(EXPECTED_REPS):
        return "incomplete"
    if any(not row["ok"] for row in processes):
        return "build_or_execution_failure"
    if any(row["outcome"] == "legacy_gate_missing" for row in processes):
        return "legacy_gate_missing"
    if any(not row["legacy_gate_pass"] for row in processes):
        return "legacy_gate_failure"
    return "eligible"


def _summarize_records(
    raw_dir: Path, campaign: Campaign, receipt: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    if not raw_dir.is_dir():
        raise AnalysisError(f"raw directory does not exist: {raw_dir}")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    hashes: dict[tuple[str, int], str] = {}
    paths = sorted(raw_dir.glob("*.json"), key=lambda path: path.name)
    for path in paths:
        value, raw = read_json(path)
        rec = _object(value, f"raw record {path.name}")
        job, rep, process = _validate_record(rec, path, campaign, receipt)
        key = (job["job_id"], rep)
        if key in indexed:
            raise AnalysisError(f"duplicate process record for {key}")
        digest = sha256_bytes(raw)
        process["record_sha256"] = digest
        indexed[key] = process
        hashes[key] = digest

    cells: list[dict[str, Any]] = []
    for job in campaign.jobs:
        processes = [indexed[(job["job_id"], rep)] for rep in EXPECTED_REPS
                     if (job["job_id"], rep) in indexed]
        status = _cell_status(processes)
        medians = [row["median_ms"] for row in processes if "median_ms" in row]
        cell = {
            "dsl": job["dsl"],
            "grid_id": job["grid_id"],
            "grid_index": job["grid_index"],
            "job_id": job["job_id"],
            "set": job["set"],
            "rep_count": len(processes),
            "rep_ids": [row["rep"] for row in processes],
            "processes": processes,
            "process_medians_ms": medians,
            "median_of_process_medians_ms": (
                float(statistics.median(medians)) if len(medians) == 2 else None
            ),
            "status": status,
            "screening_eligible": status == "eligible",
        }
        cells.append(cell)
    bundle = [
        {"job_id": job["job_id"], "rep": rep,
         "record_sha256": hashes[(job["job_id"], rep)]}
        for job in campaign.jobs for rep in EXPECTED_REPS
        if (job["job_id"], rep) in hashes
    ]
    return cells, canonical_sha256(bundle)


def _dsl_summaries(campaign: Campaign, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for dsl in campaign.manifest["dsl_order"]:
        lane = [cell for cell in cells if cell["dsl"] == dsl]
        counts = {name: sum(cell["status"] == name for cell in lane) for name in (
            "eligible", "incomplete", "build_or_execution_failure",
            "legacy_gate_missing", "legacy_gate_failure",
        )}
        result.append({
            "dsl": dsl,
            "job_count": len(lane),
            "complete_job_count": sum(cell["rep_ids"] == list(EXPECTED_REPS) for cell in lane),
            "screening_eligible_count": counts["eligible"],
            "status_counts": counts,
        })
    return result


def _job_hash(job: dict[str, Any]) -> str:
    return canonical_sha256(job)


def _candidate_name(job: dict[str, Any]) -> str:
    digest = _job_hash(job)
    return f"fused-grid:{job['job_id']}:{digest[:12]}"


def _adapter_value(adapter: dict[str, Any], *path: str) -> Any:
    value: Any = adapter
    for name in path:
        if not isinstance(value, dict) or name not in value:
            raise AnalysisError(f"adapter manifest lacks {'.'.join(path)}")
        value = value[name]
    return value


def apply_robust_summary(
    analysis: Analysis, robust_summary_path: Path, adapter_manifest_path: Path
) -> None:
    """Validate an external robust summary and annotate ``analysis.summary``."""
    summary_value, summary_raw = read_json(robust_summary_path)
    robust = _object(summary_value, "robust summary")
    adapter_value, adapter_raw = read_json(adapter_manifest_path)
    adapter = _object(adapter_value, "robust adapter manifest")
    campaign = analysis.campaign

    expected_bindings = {
        "grid_manifest_sha256": campaign.manifest_sha256,
        "grid_jobs_sha256": campaign.manifest["jobs_sha256"],
        "adapter_manifest_sha256": sha256_bytes(adapter_raw),
        "source_bundle_sha256": _adapter_value(adapter, "source_bundle_sha256"),
        "gate_spec_sha256": _adapter_value(adapter, "robust_gate", "gate_spec_sha256"),
        "gate_spec_canonical_sha256": _adapter_value(
            adapter, "robust_gate", "gate_spec_canonical_sha256"
        ),
        "robust_manifest_sha256": _adapter_value(
            adapter, "robust_gate", "manifest_canonical_sha256"
        ),
    }
    for name, wanted in expected_bindings.items():
        _same(robust.get(name), wanted, f"robust summary {name}")
        _hex64(robust.get(name), f"robust summary {name}")
    _same(robust.get("split"), "validation", "robust summary split")
    _same(_adapter_value(adapter, "grid", "manifest_sha256"),
          campaign.manifest_sha256, "adapter grid manifest hash")
    _same(_adapter_value(adapter, "grid", "jobs_sha256"),
          campaign.manifest["jobs_sha256"], "adapter grid jobs hash")

    # The adapter has additional execution sources, so its bundle hash cannot
    # equal the screening bundle.  Every shared implementation file must,
    # however, be byte-identical.
    adapter_sources = _object(adapter.get("source_sha256"), "adapter source_sha256")
    for name, digest in adapter_sources.items():
        if not isinstance(name, str) or not name:
            raise AnalysisError("adapter source path must be non-empty")
        _hex64(digest, f"adapter source {name}")
    _same(canonical_sha256(adapter_sources), adapter["source_bundle_sha256"],
          "adapter source bundle hash")
    screening_sources = analysis.receipt["source_sha256"]
    shared = sorted(set(adapter_sources) & set(screening_sources))
    if not shared:
        raise AnalysisError("screening and robust adapter have no shared source paths")
    for name in shared:
        _same(adapter_sources[name], screening_sources[name],
              f"shared screening/robust source {name}")

    groups = robust.get("groups")
    if not isinstance(groups, list):
        raise AnalysisError("robust summary groups must be a list")
    group_map: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_group in enumerate(groups):
        group = _object(raw_group, f"robust group {index}")
        job_id = group.get("grid_job_id")
        if job_id not in campaign.jobs_by_id:
            raise AnalysisError(f"robust group {index} has unknown grid_job_id {job_id!r}")
        job = campaign.jobs_by_id[job_id]
        _same(group.get("grid_job_sha256"), _job_hash(job),
              f"robust group {index} grid_job_sha256")
        _same(group.get("candidate"), _candidate_name(job),
              f"robust group {index} candidate")
        _same(group.get("op"), ROBUST_OPERATION, f"robust group {index} op")
        gate_id = group.get("gate_id")
        if gate_id not in REQUIRED_GATE_IDS:
            raise AnalysisError(f"robust group {index} has unexpected gate {gate_id!r}")
        key = (job_id, gate_id)
        if key in group_map:
            raise AnalysisError(f"duplicate robust group {key}")
        if not isinstance(group.get("success"), bool):
            raise AnalysisError(f"robust group {index} success must be boolean")
        if not isinstance(group.get("coverage_complete"), bool):
            raise AnalysisError(
                f"robust group {index} coverage_complete must be boolean"
            )
        group_map[key] = group

    cells = analysis.summary["cells"]
    screening_ids = {cell["job_id"] for cell in cells if cell["screening_eligible"]}
    missing = sorted(
        (job_id, gate_id) for job_id in screening_ids for gate_id in REQUIRED_GATE_IDS
        if (job_id, gate_id) not in group_map
    )
    if missing:
        raise AnalysisError(
            f"robust summary omits {len(missing)} required screening candidate/gate groups; "
            f"first={missing[0]}"
        )

    for cell in cells:
        group_views = []
        for gate_id in REQUIRED_GATE_IDS:
            group = group_map.get((cell["job_id"], gate_id))
            if group is not None:
                group_views.append({
                    "gate_id": gate_id,
                    "success": group["success"],
                    "coverage_complete": group["coverage_complete"],
                    "n_records": group.get("n_records"),
                    "n_failed_records": group.get("n_failed_records"),
                })
        robust_eligible = (
            cell["screening_eligible"]
            and len(group_views) == len(REQUIRED_GATE_IDS)
            and all(view["success"] and view["coverage_complete"] for view in group_views)
        )
        cell["robust_gate"] = {
            "groups": group_views,
            "robust_eligible": robust_eligible,
        }

    selected: dict[str, list[str]] = {}
    blockers: list[str] = []
    for dsl in campaign.manifest["dsl_order"]:
        eligible = [cell for cell in cells if cell["dsl"] == dsl and
                    cell["robust_gate"]["robust_eligible"]]
        eligible.sort(key=lambda cell: (
            cell["median_of_process_medians_ms"], cell["grid_index"], cell["job_id"]
        ))
        selected[dsl] = [cell["job_id"] for cell in eligible[:TOP_K]]
        if len(eligible) < TOP_K:
            blockers.append(f"{dsl}: only {len(eligible)} robust-eligible candidates")

    complete = analysis.summary["campaign_complete"]
    if not complete:
        blockers.insert(0, "screening campaign is incomplete")
    analysis.summary["robust_gate"] = {
        "supplied": True,
        "summary_sha256": sha256_bytes(summary_raw),
        "adapter_manifest_sha256": sha256_bytes(adapter_raw),
        "required_gate_ids": list(REQUIRED_GATE_IDS),
        "selection_ready": not blockers,
        "selection_blockers": blockers,
        "top_k": TOP_K,
        "selected_job_ids_by_dsl": selected,
    }


def analyze_campaign(
    manifest_path: Path,
    launch_receipt_path: Path,
    raw_dir: Path,
    robust_summary_path: Path | None = None,
    adapter_manifest_path: Path = DEFAULT_ADAPTER_MANIFEST,
) -> Analysis:
    campaign = load_campaign(manifest_path)
    receipt, receipt_hash = load_receipt(launch_receipt_path, campaign)
    cells, record_bundle_hash = _summarize_records(raw_dir, campaign, receipt)
    campaign_complete = all(cell["rep_ids"] == list(EXPECTED_REPS) for cell in cells)
    summary = {
        "schema_version": 1,
        "campaign_id": campaign.manifest["campaign_id"],
        "manifest_sha256": campaign.manifest_sha256,
        "jobs_sha256": campaign.manifest["jobs_sha256"],
        "launch_receipt_sha256": receipt_hash,
        "protocol_sha256": receipt["protocol_sha256"],
        "screening_source_bundle_sha256": receipt["source_bundle_sha256"],
        "screening_records_sha256": record_bundle_hash,
        "expected_reps": list(EXPECTED_REPS),
        "expected_process_record_count": len(campaign.jobs) * len(EXPECTED_REPS),
        "observed_process_record_count": sum(cell["rep_count"] for cell in cells),
        "campaign_complete": campaign_complete,
        "dsl_summaries": _dsl_summaries(campaign, cells),
        "cells": cells,
        "robust_gate": {
            "supplied": False,
            "selection_ready": False,
            "selection_blockers": ["external robust summary not supplied"],
        },
    }
    analysis = Analysis(campaign, receipt, receipt_hash, summary)
    if robust_summary_path is not None:
        apply_robust_summary(analysis, robust_summary_path, adapter_manifest_path)
    return analysis


def build_confirmation(analysis: Analysis) -> dict[str, Any]:
    gate = analysis.summary["robust_gate"]
    if not gate.get("supplied"):
        raise AnalysisError("confirmation requires an external robust summary")
    if not gate.get("selection_ready"):
        raise AnalysisError(
            "confirmation is not ready: " + "; ".join(gate["selection_blockers"])
        )
    cells = {cell["job_id"]: cell for cell in analysis.summary["cells"]}
    campaign = analysis.campaign
    jobs: list[dict[str, Any]] = []
    for dsl in campaign.manifest["dsl_order"]:
        ranked_ids = gate["selected_job_ids_by_dsl"][dsl]
        incumbent_id = f"{dsl}.{OLD_INCUMBENT_GRID_ID}"
        ordered_ids = list(ranked_ids)
        if incumbent_id not in ordered_ids:
            ordered_ids.append(incumbent_id)
        for job_id in ordered_ids:
            source_job = campaign.jobs_by_id[job_id]
            cell = cells[job_id]
            roles = []
            if job_id in ranked_ids:
                roles.append(f"screening_rank_{ranked_ids.index(job_id) + 1}")
            if job_id == incumbent_id:
                roles.append("old_incumbent")
            jobs.append({
                "confirmation_id": f"{dsl}.c{len(jobs):02d}",
                "dsl": source_job["dsl"],
                "geom": source_job["geom"],
                "grid_id": source_job["grid_id"],
                "grid_index": source_job["grid_index"],
                "screening_job_id": job_id,
                "selection_roles": roles,
                "set": source_job["set"],
                "variant": source_job["variant"],
                "screening_median_ms": cell["median_of_process_medians_ms"],
                "screening_robust_eligible": cell["robust_gate"]["robust_eligible"],
            })

    provenance = {
        "screening_campaign_id": analysis.summary["campaign_id"],
        "screening_manifest_sha256": analysis.summary["manifest_sha256"],
        "screening_jobs_sha256": analysis.summary["jobs_sha256"],
        "screening_launch_receipt_sha256": analysis.summary["launch_receipt_sha256"],
        "screening_protocol_sha256": analysis.summary["protocol_sha256"],
        "screening_records_sha256": analysis.summary["screening_records_sha256"],
        "robust_summary_sha256": gate["summary_sha256"],
        "robust_adapter_manifest_sha256": gate["adapter_manifest_sha256"],
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": "fused-gbgs-confirmation-v1",
        "top_k": TOP_K,
        "old_incumbent_grid_id": OLD_INCUMBENT_GRID_ID,
        "selection_rule": (
            "per DSL: robust-eligible candidates ordered by median of two process "
            "medians, then grid index and job ID; append g01 incumbent if absent"
        ),
        "provenance": provenance,
        "jobs": jobs,
    }
    document["jobs_sha256"] = canonical_sha256(jobs)
    return document


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _print_status(summary: dict[str, Any]) -> None:
    print(
        f"{summary['campaign_id']}: {summary['observed_process_record_count']}/"
        f"{summary['expected_process_record_count']} process records; "
        f"complete={summary['campaign_complete']}"
    )
    for lane in summary["dsl_summaries"]:
        counts = lane["status_counts"]
        print(
            f"  {lane['dsl']}: eligible={counts['eligible']} "
            f"build/exec-fail={counts['build_or_execution_failure']} "
            f"legacy-gate-fail={counts['legacy_gate_failure']} "
            f"legacy-gate-missing={counts['legacy_gate_missing']} "
            f"incomplete={counts['incomplete']}"
        )
    gate = summary["robust_gate"]
    if gate["supplied"]:
        print(f"  robust selection ready={gate['selection_ready']}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--robust-summary", type=Path)
    parser.add_argument(
        "--robust-adapter-manifest", type=Path, default=DEFAULT_ADAPTER_MANIFEST
    )
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--confirmation-out", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.confirmation_out is not None and args.robust_summary is None:
            raise AnalysisError("--confirmation-out requires --robust-summary")
        analysis = analyze_campaign(
            args.manifest.resolve(),
            args.launch_receipt.resolve(),
            args.raw_dir.resolve(),
            args.robust_summary.resolve() if args.robust_summary else None,
            args.robust_adapter_manifest.resolve(),
        )
        if args.require_complete and not analysis.summary["campaign_complete"]:
            raise AnalysisError("screening campaign is incomplete")
        if args.summary_out is not None:
            atomic_write(args.summary_out, stable_json_bytes(analysis.summary))
        if args.confirmation_out is not None:
            atomic_write(args.confirmation_out, stable_json_bytes(build_confirmation(analysis)))
        _print_status(analysis.summary)
        return 0
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
