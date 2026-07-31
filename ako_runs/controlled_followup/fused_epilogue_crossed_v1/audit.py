#!/usr/bin/env python3
"""Build/reachability audit and complete fused-v2 gate for one GPU shard."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from candidates import UnsupportedStrategy, build
from core import (
    HERE,
    LOCK_PATH,
    REPO_ROOT,
    canonical_sha256,
    cell_filename,
    file_sha256,
    read_json,
    result_root,
    nvcc_fingerprint,
    stable_write,
)
from validate import validate_launch_ready


FUSED_GRID = REPO_ROOT / "ako_runs/controlled_followup/fused_grid"
if str(FUSED_GRID) not in sys.path:
    sys.path.insert(0, str(FUSED_GRID))
import robust_adapter as adapter  # noqa: E402


def compact_artifact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): compact_artifact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact_artifact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        import hashlib
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, str) and len(value) > 4096:
        import hashlib
        return {"bytes": len(value.encode("utf-8")), "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def make_plan(context, cell, built):
    metadata = compact_artifact(built.metadata)
    metadata.update(
        {
            "crossed_campaign_id": "fused-epilogue-crossed-v1",
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
        candidate=f"fused-crossed:{cell['cell_id']}:{canonical_sha256(cell)[:12]}",
        job=cell["origin_job"],
        job_sha256=cell["origin_job_sha256"],
        config=built.config,
        build_metadata=metadata,
        execute=execute,
    )


def gate_summary(context, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = 4 * 64 * 2
    if len(rows) != expected:
        raise RuntimeError(f"gate row count {len(rows)} != {expected}")
    max_ratios: dict[str, float] = {}
    for row in rows:
        if row.get("ok") is not True:
            continue
        gate = context.gate_spec["gates"][f"fused_softmax/{row['gate_id']}"]
        for metric, threshold in gate["thresholds"].items():
            value = row.get("metrics", {}).get(metric)
            if isinstance(value, (int, float)):
                key = f"{row['gate_id']}/{metric}"
                max_ratios[key] = max(max_ratios.get(key, float("-inf")), float(value) / float(threshold["value"]))
    failed = [row for row in rows if row.get("ok") is not True or row.get("gate_pass") is not True]
    coverage = {
        (row.get("case_id"), row.get("seed_index"), row.get("gate_id")) for row in rows
    }
    expected_coverage = {
        (case_id, seed_index, gate_id)
        for case_id in context.adapter["robust_gate"]["case_ids"]
        for seed_index in range(64)
        for gate_id in adapter.GATE_IDS
    }
    return {
        "complete": coverage == expected_coverage and len(rows) == expected,
        "expected_records": expected,
        "failed_records": len(failed),
        "full_gate_pass": coverage == expected_coverage and len(rows) == expected and not failed,
        "max_over_threshold_ratio": max(max_ratios.values()) if max_ratios else None,
        "max_over_threshold_ratio_by_metric": max_ratios,
        "minimum_headroom_fraction": 1.0 - max(max_ratios.values()) if max_ratios else None,
        "observed_records": len(rows),
    }


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def audit_cell(context, root: Path, cell: dict[str, Any], lock: dict[str, Any], gpu: int) -> dict[str, Any]:
    record_path = root / "audit" / "records" / cell_filename(cell["cell_id"])
    gate_path = root / "audit" / "gate" / (cell_filename(cell["cell_id"]).removesuffix(".json") + ".jsonl")
    if record_path.exists():
        record = read_json(record_path)
        expected = {
            "cell": cell,
            "cell_sha256": canonical_sha256(cell),
            "launch_lock_sha256": file_sha256(LOCK_PATH),
            "physical_gpu": gpu,
            "source_bundle_sha256": lock["source_bundle_sha256"],
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"foreign retained audit record: {record_path}")
        if record.get("terminal_outcome") == "GATE_PASSED" or record.get("terminal_outcome") == "GATE_FAILED":
            if not gate_path.is_file() or record.get("gate_jsonl_sha256") != file_sha256(gate_path):
                raise RuntimeError(f"retained gate evidence missing or changed: {gate_path}")
        return record
    base = {
        "campaign_id": "fused-epilogue-crossed-v1",
        "cell": cell,
        "cell_sha256": canonical_sha256(cell),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "physical_gpu": gpu,
        "schema_version": 1,
        "source_bundle_sha256": lock["source_bundle_sha256"],
    }
    if not cell["support_declared"]:
        record = {
            **base,
            "build_attempted": False,
            "gate_attempted": False,
            "terminal_outcome": "UNSUPPORTED",
            "terminal_reason": cell["support_detail"],
        }
        stable_write(record_path, record)
        return record
    started = time.perf_counter()
    try:
        built = build(cell)
    except UnsupportedStrategy as exc:
        raise RuntimeError("supported manifest cell raised UnsupportedStrategy") from exc
    except Exception as exc:  # build failure is retained experimental evidence
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
    plan = make_plan(context, cell, built)
    rows = []
    live_inputs = None
    for case_id in context.adapter["robust_gate"]["case_ids"]:
        for seed_index in range(64):
            prior_inputs = live_inputs
            evaluated, live_inputs = adapter.evaluate_case_seed(
                context, [plan], case_id=case_id, split="validation",
                seed_index=seed_index, device="cuda:0",
            )
            if prior_inputs is not None:
                del prior_inputs
            seed_rows = evaluated[plan.candidate]
            for row in seed_rows:
                row.update(
                    {
                        "crossed_campaign_id": "fused-epilogue-crossed-v1",
                        "crossed_cell_id": cell["cell_id"],
                        "crossed_cell_sha256": canonical_sha256(cell),
                        "crossed_launch_lock_sha256": file_sha256(LOCK_PATH),
                        "crossed_source_bundle_sha256": lock["source_bundle_sha256"],
                    }
                )
            rows.extend(seed_rows)
    summary = gate_summary(context, rows)
    write_jsonl_atomic(gate_path, rows)
    any_execution_error = any(row.get("ok") is not True for row in rows)
    record = {
        **base,
        "build_attempted": True,
        "build_metadata": compact_artifact(built.metadata),
        "build_wall_s": time.perf_counter() - started,
        "gate_attempted": True,
        "gate_jsonl_path": str(gate_path.relative_to(REPO_ROOT)),
        "gate_jsonl_sha256": file_sha256(gate_path),
        "gate_summary": summary,
        "reported_compile_s": built.compile_s,
        "terminal_outcome": (
            "LAUNCH_FAILED" if any_execution_error else
            "GATE_PASSED" if summary["full_gate_pass"] else "GATE_FAILED"
        ),
    }
    stable_write(record_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or args.shard_index not in range(args.shard_count):
        raise ValueError("invalid shard")
    # Each audit process exposes exactly its assigned physical card as logical
    # cuda:0. Set this before any builder can initialize the CUDA runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13.1"
    os.environ["PATH"] = "/usr/local/cuda-13.1/bin:" + os.environ.get("PATH", "")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(HERE / ".torch_ext" / f"gpu{args.gpu}")
    os.environ.setdefault("MAX_JOBS", "4")
    ready = validate_launch_ready(args.gpu, allow_busy=args.allow_busy)
    campaign, cells, lock = __import__("core").load_contract()
    assigned = [cell for cell in cells if cell["cell_index"] % args.shard_count == args.shard_index]
    root = result_root(args.tag)
    receipt_path = root / "audit" / "receipts" / f"shard{args.shard_index:02d}.json"
    contract = {
        "assigned_cell_ids": [cell["cell_id"] for cell in assigned],
        "campaign_id": campaign["campaign_id"],
        "git_commit": ready["git_commit"],
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "physical_gpu": args.gpu,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "tag": args.tag,
    }
    if receipt_path.exists():
        if read_json(receipt_path).get("contract") != contract:
            raise RuntimeError("existing shard receipt has another contract")
    else:
        stable_write(receipt_path, {"contract": contract, "created_utc": datetime.now(timezone.utc).isoformat(), "gpu": ready["gpu"], "host": platform.node(), "nvcc": nvcc_fingerprint(), "python": sys.version, "record_type": "fused_crossed_audit_receipt", "schema_version": 1})
    context = adapter.load_repository()
    outcomes = []
    for position, cell in enumerate(assigned, 1):
        print(f"[{position}/{len(assigned)}] {cell['cell_id']}", flush=True)
        outcomes.append(audit_cell(context=context, root=root, cell=cell, lock=lock, gpu=args.gpu))
    stable_write(
        root / "audit" / "receipts" / f"shard{args.shard_index:02d}_status.json",
        {
            "campaign_id": campaign["campaign_id"],
            "complete": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "expected_cells": len(assigned),
            "gpu_after": __import__("core").gpu_snapshot(args.gpu),
            "launch_lock_sha256": file_sha256(LOCK_PATH),
            "observed_cells": len(outcomes),
            "outcome_counts": {
                name: sum(row["terminal_outcome"] == name for row in outcomes)
                for name in ("UNSUPPORTED", "BUILD_FAILED", "LAUNCH_FAILED", "GATE_FAILED", "GATE_PASSED")
            },
            "receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
            "receipt_sha256": file_sha256(receipt_path),
            "record_type": "fused_crossed_audit_shard_status",
            "schema_version": 1,
        },
    )
    print(f"audit shard complete: {len(assigned)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
