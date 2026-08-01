#!/usr/bin/env python3
"""Run the resolved v2 audit and fresh-process timing stages."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .candidates import build
    from .core import (
        CAMPAIGN_ID, HERE, LOCK_PATH, REPO_ROOT, RESULTS_ROOT,
        SUPPORT_RESOLUTION_PATH, canonical_sha256, confirmation_plan, file_sha256,
        gpu_snapshot, load_contract, nvcc_fingerprint, read_json, result_root,
        screen_plan, stable_write, summarize_times, timing_filename, validate_gpu,
    )
    from .validate import validate_launch_ready
    from . import analyze as analysis
except ImportError:  # direct script execution
    from candidates import build
    from core import (
    CAMPAIGN_ID,
    HERE,
    LOCK_PATH,
    REPO_ROOT,
    RESULTS_ROOT,
    SUPPORT_RESOLUTION_PATH,
    canonical_sha256,
    confirmation_plan,
    file_sha256,
    gpu_snapshot,
    load_contract,
    nvcc_fingerprint,
    read_json,
    result_root,
    screen_plan,
    stable_write,
    summarize_times,
    timing_filename,
    validate_gpu,
    )
    from validate import validate_launch_ready
    import analyze as analysis


FUSED_GRID = REPO_ROOT / "ako_runs/controlled_followup/fused_grid"
if str(FUSED_GRID) not in sys.path:
    sys.path.insert(0, str(FUSED_GRID))
import robust_adapter as adapter  # noqa: E402


def compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        import hashlib
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, str) and len(value) > 4096:
        import hashlib
        encoded = value.encode("utf-8")
        return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _cell_filename(cell_id: str) -> str:
    return cell_id.replace(".", "__") + ".json"


def _gate_summary(context, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = 4 * 64 * 2
    if len(rows) != expected:
        raise RuntimeError(f"gate row count {len(rows)} != {expected}")
    ratios: dict[str, float] = {}
    for row in rows:
        if row.get("ok") is not True:
            continue
        gate = context.gate_spec["gates"][f"fused_softmax/{row['gate_id']}"]
        for metric, threshold in gate["thresholds"].items():
            value = row.get("metrics", {}).get(metric)
            if isinstance(value, (int, float)):
                key = f"{row['gate_id']}/{metric}"
                ratios[key] = max(ratios.get(key, float("-inf")), float(value) / float(threshold["value"]))
    failed = [row for row in rows if row.get("ok") is not True or row.get("gate_pass") is not True]
    coverage = {(row.get("case_id"), row.get("seed_index"), row.get("gate_id")) for row in rows}
    expected_coverage = {
        (case, seed, gate)
        for case in context.adapter["robust_gate"]["case_ids"]
        for seed in range(64)
        for gate in adapter.GATE_IDS
    }
    complete = coverage == expected_coverage and len(rows) == expected
    return {
        "complete": complete,
        "expected_records": expected,
        "failed_records": len(failed),
        "full_gate_pass": complete and not failed,
        "max_over_threshold_ratio": max(ratios.values()) if ratios else None,
        "max_over_threshold_ratio_by_metric": ratios,
        "minimum_headroom_fraction": 1.0 - max(ratios.values()) if ratios else None,
        "observed_records": len(rows),
    }


def _candidate_plan(context, cell: dict[str, Any], built):
    metadata = compact(built.metadata)
    metadata.update(
        {
            "crossed_campaign_id": CAMPAIGN_ID,
            "crossed_cell_id": cell["cell_id"],
            "crossed_cell_sha256": canonical_sha256(cell),
            "crossed_strategy": cell["strategy"],
            "reported_compile_s": built.compile_s,
        }
    )

    def execute(inputs, prepared):
        if "x_fp16" not in prepared:
            prepared["x_fp16"] = inputs["x"].half().contiguous()
        return built.run(prepared["x_fp16"], inputs["weight"], inputs["bias"])

    return adapter.CandidatePlan(
        candidate=f"fused-crossed-v2:{cell['cell_id']}:{canonical_sha256(cell)[:12]}",
        job=cell["origin_job"],
        job_sha256=cell["origin_job_sha256"],
        config=built.config,
        build_metadata=metadata,
        execute=execute,
    )


def _canonical_timing_plan(
    phase: str,
    eligibility_path: Path,
    campaign: dict[str, Any],
    cells: list[dict[str, Any]],
    lock: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligibility_path = eligibility_path.resolve()
    try:
        eligibility_path.relative_to(RESULTS_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("eligibility artifact escapes the v2 result root") from exc
    eligibility = read_json(eligibility_path)
    common = {
        "campaign_id": CAMPAIGN_ID,
        "complete": True,
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "source_bundle_sha256": lock["source_bundle_sha256"],
    }
    mismatch = [key for key, value in common.items() if eligibility.get(key) != value]
    if mismatch:
        raise RuntimeError(f"foreign eligibility artifact: {mismatch}")
    root = eligibility_path.parent
    by_id = {cell["cell_id"]: cell for cell in cells}
    if phase == "screen":
        if eligibility.get("record_type") != "fused_crossed_v2_audit_summary":
            raise RuntimeError("screen requires a v2 audit summary")
        if eligibility != analysis.audit_summary(root):
            raise RuntimeError("screen eligibility is not re-derived from retained audit evidence")
        legal_list = eligibility.get("timing_eligible_cell_ids")
        if not isinstance(legal_list, list) or len(legal_list) != len(set(legal_list)):
            raise RuntimeError("audit eligibility IDs are malformed")
        legal = set(legal_list)
        if not legal <= set(by_id) or any(by_id[cell_id]["support_declared"] is not True for cell_id in legal):
            raise RuntimeError("audit eligibility contains an unknown or unsupported cell")
        plan = [
            {**row, "label": row["cell_id"], "record_kind": "cell"}
            for row in screen_plan(cells, legal)
        ]
    else:
        if eligibility.get("record_type") != "fused_crossed_v2_confirmation_selection":
            raise RuntimeError("confirmation requires a v2 selection")
        audit_path = REPO_ROOT / eligibility.get("audit_summary_path", "")
        if not audit_path.is_file() or eligibility != analysis.screen_selection(root, audit_path):
            raise RuntimeError("confirmation eligibility is not re-derived from retained screen evidence")
        selected_list = eligibility.get("selected_cell_ids")
        legal_list = eligibility.get("timing_eligible_cell_ids")
        if (
            not isinstance(selected_list, list)
            or not isinstance(legal_list, list)
            or len(selected_list) != len(set(selected_list))
            or len(legal_list) != len(set(legal_list))
        ):
            raise RuntimeError("confirmation selection IDs are malformed")
        selected, legal = set(selected_list), set(legal_list)
        if not selected <= legal or not legal <= set(by_id):
            raise RuntimeError("confirmation selection is not audit-eligible")
        if {row.get("cell_id") for row in eligibility.get("selected", [])} != selected:
            raise RuntimeError("confirmation selection rows disagree with selected IDs")
        plan = confirmation_plan(selected)
    return eligibility, plan


def _validate_retained_timing(
    path: Path,
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    record = read_json(path)
    mismatch = [key for key, value in expected.items() if record.get(key) != value]
    if mismatch or record.get("ok") is not True:
        raise RuntimeError(f"foreign or failed retained timing record {path}: {mismatch}")
    if record.get("legacy_error", {}).get("gate_pass") is not True:
        raise RuntimeError(f"retained timing record failed its legacy check: {path}")
    for key, value in summarize_times(record.get("times_ms", [])).items():
        if record.get(key) != value:
            raise RuntimeError(f"retained timing summary changed: {path}: {key}")
    if not isinstance(record.get("implementation_sha256"), str):
        raise RuntimeError(f"retained timing record lacks an implementation hash: {path}")
    return record

def _audit_cell(context, root: Path, cell: dict[str, Any], lock: dict, gpu: int) -> dict[str, Any]:
    record_path = root / "audit" / "records" / _cell_filename(cell["cell_id"])
    gate_path = root / "audit" / "gate" / (_cell_filename(cell["cell_id"]).removesuffix(".json") + ".jsonl")
    if record_path.exists():
        return read_json(record_path)
    base = {
        "campaign_id": CAMPAIGN_ID,
        "cell": cell,
        "cell_sha256": canonical_sha256(cell),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "physical_gpu": gpu,
        "schema_version": 2,
        "source_bundle_sha256": lock["source_bundle_sha256"],
    }
    if cell["support_declared"] is False:
        resolution = read_json(SUPPORT_RESOLUTION_PATH)
        probe = resolution["probes"][cell["support_probe_key"]]
        record = {
            **base,
            "build_attempted": False,
            "gate_attempted": False,
            "support_probe_index_path": probe["result_index_path"],
            "support_probe_index_sha256": probe["result_index_sha256"],
            "support_probe_key": cell["support_probe_key"],
            "terminal_outcome": "UNSUPPORTED",
            "terminal_reason": cell["support_detail"],
        }
        stable_write(record_path, record)
        return record
    started = time.perf_counter()
    try:
        built = build(cell)
    except Exception as exc:
        record = {
            **base,
            "build_attempted": True,
            "build_wall_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "gate_attempted": False,
            "terminal_outcome": "BUILD_FAILED",
            "traceback": traceback.format_exc(),
        }
        stable_write(record_path, record)
        return record
    plan = _candidate_plan(context, cell, built)
    rows = []
    live_inputs = None
    try:
        for case_id in context.adapter["robust_gate"]["case_ids"]:
            for seed_index in range(64):
                prior = live_inputs
                evaluated, live_inputs = adapter.evaluate_case_seed(
                    context,
                    [plan],
                    case_id=case_id,
                    split="validation",
                    seed_index=seed_index,
                    device="cuda:0",
                )
                if prior is not None:
                    del prior
                for row in evaluated[plan.candidate]:
                    row.update(
                        {
                            "crossed_campaign_id": CAMPAIGN_ID,
                            "crossed_cell_id": cell["cell_id"],
                            "crossed_cell_sha256": canonical_sha256(cell),
                            "crossed_launch_lock_sha256": file_sha256(LOCK_PATH),
                            "crossed_source_bundle_sha256": lock["source_bundle_sha256"],
                        }
                    )
                    rows.append(row)
    except Exception as exc:
        _write_jsonl(gate_path, rows)
        record = {
            **base,
            "build_attempted": True,
            "build_metadata": compact(built.metadata),
            "build_wall_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "gate_attempted": True,
            "gate_jsonl_path": str(gate_path.relative_to(REPO_ROOT)),
            "gate_jsonl_sha256": file_sha256(gate_path),
            "gate_summary": {
                "complete": False,
                "expected_records": 512,
                "failed_records": sum(
                    row.get("ok") is not True or row.get("gate_pass") is not True
                    for row in rows
                ),
                "full_gate_pass": False,
                "observed_records": len(rows),
            },
            "reported_compile_s": built.compile_s,
            "terminal_outcome": "LAUNCH_FAILED",
            "traceback": traceback.format_exc(),
        }
        stable_write(record_path, record)
        return record
    summary = _gate_summary(context, rows)
    _write_jsonl(gate_path, rows)
    execution_error = any(row.get("ok") is not True for row in rows)
    record = {
        **base,
        "build_attempted": True,
        "build_metadata": compact(built.metadata),
        "build_wall_s": time.perf_counter() - started,
        "gate_attempted": True,
        "gate_jsonl_path": str(gate_path.relative_to(REPO_ROOT)),
        "gate_jsonl_sha256": file_sha256(gate_path),
        "gate_summary": summary,
        "reported_compile_s": built.compile_s,
        "terminal_outcome": "LAUNCH_FAILED" if execution_error else "GATE_PASSED" if summary["full_gate_pass"] else "GATE_FAILED",
    }
    stable_write(record_path, record)
    return record


def run_audit(args) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13.1"
    os.environ["PATH"] = "/usr/local/cuda-13.1/bin:" + os.environ.get("PATH", "")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(HERE / ".torch_ext" / f"gpu{args.gpu}")
    os.environ.setdefault("MAX_JOBS", "4")
    ready = validate_launch_ready("campaign", args.gpu, allow_busy=args.allow_busy)
    campaign, cells, lock = load_contract()
    if args.shard_count < 1 or args.shard_index not in range(args.shard_count):
        raise ValueError("invalid shard")
    assigned = [cell for cell in cells if cell["cell_index"] % args.shard_count == args.shard_index]
    root = result_root(args.tag)
    receipt = root / "audit" / "receipts" / f"shard{args.shard_index:02d}.json"
    contract = {
        "assigned_cell_ids": [cell["cell_id"] for cell in assigned],
        "campaign_id": campaign["campaign_id"],
        "git_commit": ready["git_commit"],
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "physical_gpu": args.gpu,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "source_bundle_sha256": lock["source_bundle_sha256"],
    }
    if receipt.exists() and read_json(receipt).get("contract") != contract:
        raise RuntimeError("existing shard receipt has another contract")
    if not receipt.exists():
        stable_write(
            receipt,
            {
                "contract": contract,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "gpu": ready["gpu"],
                "host": platform.node(),
                "nvcc": nvcc_fingerprint(),
                "record_type": "fused_crossed_v2_audit_receipt",
                "schema_version": 2,
            },
        )
    context = adapter.load_repository()
    for position, cell in enumerate(assigned, 1):
        record = _audit_cell(context, root, cell, lock, args.gpu)
        print(f"[{position}/{len(assigned)}] {cell['cell_id']} -> {record['terminal_outcome']}", flush=True)
    stable_write(
        receipt.with_name(receipt.stem + "_status.json"),
        {
            "complete": True,
            "expected_cells": len(assigned),
            "observed_cells": len(assigned),
            "receipt_sha256": file_sha256(receipt),
        },
    )
    return 0


def _time_one(args) -> int:
    campaign, cells, lock = load_contract()
    row = json.loads(args.row_json)
    eligibility_path = Path(args.eligibility).resolve()
    _eligibility, plan = _canonical_timing_plan(
        args.phase, eligibility_path, campaign, cells, lock
    )
    if row not in plan:
        raise RuntimeError("timing row is not in the canonical eligibility-bound plan")
    by_id = {cell["cell_id"]: cell for cell in cells}
    cell = by_id[row["cell_id"]]
    label = row["label"]
    if args.physical_gpu != campaign["hardware"]["timing_gpu"]:
        raise RuntimeError("timing is frozen to physical GPU 0")
    snapshot = gpu_snapshot(args.physical_gpu)
    validate_gpu(snapshot, campaign)
    output = Path(args.out).resolve()
    if output.parent != eligibility_path.parent / args.phase / "raw":
        raise RuntimeError("timing output is outside the eligibility-bound raw directory")
    expected_name = timing_filename(label, row["distribution"], row["rep"])
    if output.name != expected_name or output.exists():
        raise RuntimeError("unsafe or existing timing output")
    record = {
        "campaign_id": CAMPAIGN_ID,
        "cell_id": cell["cell_id"],
        "cell_sha256": canonical_sha256(cell),
        "distribution": row["distribution"],
        "eligibility_path": str(eligibility_path.relative_to(REPO_ROOT)),
        "eligibility_sha256": file_sha256(eligibility_path),
        "label": label,
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "phase": args.phase,
        "physical_gpu": args.physical_gpu,
        "record_kind": row.get("record_kind", "cell"),
        "rep": row["rep"],
        "schema_version": 2,
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "trials": 100,
        "warmup_s": 2.0,
        "gpu_preflight": snapshot,
        "t_start": time.time(),
    }
    try:
        import common
        import common2
        import runner2
        import torch

        seed, dist = (0, "rand") if row["distribution"] == "positive" else (2026073101, "randn")
        built = build(cell)
        x, weight, bias = common2.fused_inputs(seed=seed, dist=dist)
        with torch.no_grad():
            reference = common2.fused_reference(x, weight, bias, arm="GBGS", dtype=torch.float32)
        x16 = x.half().contiguous()
        with torch.no_grad():
            observed = built.run(x16, weight, bias)
            torch.cuda.synchronize()
        record["legacy_error"] = common.gate_stats(reference, observed.float())
        del reference, observed
        torch.cuda.empty_cache()
        times, warmup_iterations = runner2.time_kernel3(
            built.run,
            x16,
            weight,
            bias,
            num_trials=100,
            warmup_s=2.0,
            flush_l2=True,
        )
        numeric = [float(value) for value in times]
        record.update(
            {
                "build_metadata": compact(built.metadata),
                "compile_s": built.compile_s,
                "implementation_sha256": built.metadata["implementation_sha256"],
                "ok": True,
                "times_ms": numeric,
                "warmup_iterations_actual": warmup_iterations,
                **summarize_times(numeric),
            }
        )
    except Exception as exc:
        record.update({"error": f"{type(exc).__name__}: {exc}", "ok": False, "traceback": traceback.format_exc()})
    record["t_end"] = time.time()
    stable_write(output, record)
    return 0 if record["ok"] else 1


def _run_timing(args) -> int:
    campaign, cells, lock = load_contract()
    if args.gpu != campaign["hardware"]["timing_gpu"]:
        raise RuntimeError("timing must use physical GPU 0")
    ready = validate_launch_ready("campaign", args.gpu, allow_busy=args.allow_busy)
    eligibility_path = Path(args.eligibility).resolve()
    try:
        eligibility_path.relative_to(result_root(args.tag).resolve())
    except ValueError as exc:
        raise RuntimeError("eligibility artifact does not belong to this result tag") from exc
    eligibility, plan = _canonical_timing_plan(
        args.phase, eligibility_path, campaign, cells, lock
    )
    root = result_root(args.tag) / args.phase
    root.mkdir(parents=True, exist_ok=True)
    active = (root / "active.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"another {args.phase} launcher is active") from None
    receipt = root / "launch_receipt.json"
    contract = {
        "campaign_id": CAMPAIGN_ID,
        "eligibility_path": str(eligibility_path.relative_to(REPO_ROOT)),
        "eligibility_sha256": file_sha256(eligibility_path),
        "execution_order": plan,
        "git_commit": ready["git_commit"],
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "phase": args.phase,
        "physical_gpu": args.gpu,
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "tag": args.tag,
    }
    if receipt.exists() and read_json(receipt).get("contract") != contract:
        raise RuntimeError("existing timing receipt has another contract")
    if not receipt.exists():
        stable_write(receipt, {"contract": contract, "created_utc": datetime.now(timezone.utc).isoformat(), "gpu": ready["gpu"], "record_type": "fused_crossed_v2_timing_receipt", "schema_version": 2})
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "CUDA_HOME": "/usr/local/cuda-13.1",
            "TORCH_EXTENSIONS_DIR": str(HERE / ".torch_ext" / "gpu0"),
        }
    )
    env["PATH"] = "/usr/local/cuda-13.1/bin:" + env.get("PATH", "")
    env.setdefault("MAX_JOBS", "4")
    attempts = root / "attempts.jsonl"
    for position, row in enumerate(plan, 1):
        output = raw / timing_filename(row["label"], row["distribution"], row["rep"])
        expected = {
            "campaign_id": CAMPAIGN_ID,
            "cell_id": row["cell_id"],
            "cell_sha256": canonical_sha256(
                next(cell for cell in cells if cell["cell_id"] == row["cell_id"])
            ),
            "distribution": row["distribution"],
            "eligibility_path": str(eligibility_path.relative_to(REPO_ROOT)),
            "eligibility_sha256": file_sha256(eligibility_path),
            "label": row["label"],
            "launch_lock_sha256": file_sha256(LOCK_PATH),
            "phase": args.phase,
            "physical_gpu": args.gpu,
            "record_kind": row["record_kind"],
            "rep": row["rep"],
            "source_bundle_sha256": lock["source_bundle_sha256"],
        }
        if output.exists():
            _validate_retained_timing(output, expected=expected)
            continue
        command = [
            sys.executable,
            str(HERE / "campaign_runner.py"),
            "time-one",
            "--phase",
            args.phase,
            "--eligibility",
            str(eligibility_path),
            "--row-json",
            json.dumps(row, sort_keys=True, separators=(",", ":")),
            "--physical-gpu",
            str(args.gpu),
            "--out",
            str(output),
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
        with attempts.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"label": row["label"], "record_exists": output.exists(), "rep": row["rep"], "returncode": completed.returncode, "utc": datetime.now(timezone.utc).isoformat()}, sort_keys=True) + "\n")
        print(f"[{position}/{len(plan)}] {row['label']} {row['distribution']} -> {completed.returncode}", flush=True)
        if completed.returncode != 0:
            raise RuntimeError(f"timing child failed: {row['label']}")
    observed = sum(
        _validate_retained_timing(
            raw / timing_filename(row["label"], row["distribution"], row["rep"]),
            expected={
                "campaign_id": CAMPAIGN_ID,
                "cell_id": row["cell_id"],
                "cell_sha256": canonical_sha256(
                    next(cell for cell in cells if cell["cell_id"] == row["cell_id"])
                ),
                "distribution": row["distribution"],
                "eligibility_path": str(eligibility_path.relative_to(REPO_ROOT)),
                "eligibility_sha256": file_sha256(eligibility_path),
                "label": row["label"],
                "launch_lock_sha256": file_sha256(LOCK_PATH),
                "phase": args.phase,
                "physical_gpu": args.gpu,
                "record_kind": row["record_kind"],
                "rep": row["rep"],
                "source_bundle_sha256": lock["source_bundle_sha256"],
            },
        )
        is not None
        for row in plan
    )
    stable_write(
        root / "run_status.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "complete": True,
            "expected_records": len(plan),
            "observed_records": observed,
            "phase": args.phase,
            "gpu_after": gpu_snapshot(args.gpu),
            "launch_receipt_sha256": file_sha256(receipt),
        },
    )
    active.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--tag", required=True)
    audit.add_argument("--gpu", type=int, required=True)
    audit.add_argument("--shard-index", type=int, required=True)
    audit.add_argument("--shard-count", type=int, default=4)
    audit.add_argument("--allow-busy", action="store_true")
    for phase in ("screen", "confirmation"):
        timing = subparsers.add_parser(phase)
        timing.add_argument("--tag", required=True)
        timing.add_argument("--gpu", type=int, default=0)
        timing.add_argument("--eligibility", required=True)
        timing.add_argument("--allow-busy", action="store_true")
        timing.set_defaults(phase=phase)
    one = subparsers.add_parser("time-one")
    one.add_argument("--phase", choices=("screen", "confirmation"), required=True)
    one.add_argument("--eligibility", required=True)
    one.add_argument("--row-json", required=True)
    one.add_argument("--physical-gpu", type=int, required=True)
    one.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "audit":
        return run_audit(args)
    if args.command == "time-one":
        return _time_one(args)
    return _run_timing(args)


if __name__ == "__main__":
    raise SystemExit(main())
