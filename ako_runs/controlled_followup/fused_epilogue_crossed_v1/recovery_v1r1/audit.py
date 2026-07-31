#!/usr/bin/env python3
"""Run one append-only v1r1 audit shard with the reporting-only fix."""
from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from pathlib import Path
from typing import Any

try:
    from . import common, validate
except ImportError:  # pragma: no cover - direct-script CLI
    import common  # type: ignore
    import validate  # type: ignore


def fixed_gate_summary(context, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize margins without treating exact-zero constraints as ratios.

    Gate decisions remain the frozen adapter's decisions. Strictly positive
    thresholds receive ordinary value/threshold ratios. Exact-zero count
    constraints are reported separately as observed maxima and violations.
    """
    expected = 4 * 64 * 2
    if len(rows) != expected:
        raise RuntimeError(f"gate row count {len(rows)} != {expected}")
    max_ratios: dict[str, float] = {}
    zero_maxima: dict[str, float] = {}
    zero_violations: dict[str, int] = {}
    for row in rows:
        if row.get("ok") is not True:
            continue
        gate = context.gate_spec["gates"][f"fused_softmax/{row['gate_id']}"]
        for metric, threshold in gate["thresholds"].items():
            value = row.get("metrics", {}).get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            numeric = float(value)
            limit = float(threshold["value"])
            if not math.isfinite(numeric) or not math.isfinite(limit) or limit < 0:
                raise RuntimeError(f"invalid diagnostic metric/threshold: {metric}")
            key = f"{row['gate_id']}/{metric}"
            if limit == 0.0:
                zero_maxima[key] = max(zero_maxima.get(key, 0.0), numeric)
                if numeric > 0.0:
                    zero_violations[key] = zero_violations.get(key, 0) + 1
            else:
                max_ratios[key] = max(max_ratios.get(key, float("-inf")), numeric / limit)
    failed = [
        row
        for row in rows
        if row.get("ok") is not True or row.get("gate_pass") is not True
    ]
    coverage = {
        (row.get("case_id"), row.get("seed_index"), row.get("gate_id"))
        for row in rows
    }
    expected_coverage = {
        (case_id, seed_index, gate_id)
        for case_id in context.adapter["robust_gate"]["case_ids"]
        for seed_index in range(64)
        for gate_id in ("conformance_mixed", "semantic_mixed")
    }
    maximum = max(max_ratios.values()) if max_ratios else None
    return {
        "complete": coverage == expected_coverage and len(rows) == expected,
        "expected_records": expected,
        "failed_records": len(failed),
        "full_gate_pass": coverage == expected_coverage and len(rows) == expected and not failed,
        "max_over_threshold_ratio": maximum,
        "max_over_threshold_ratio_by_metric": max_ratios,
        "minimum_headroom_fraction": 1.0 - maximum if maximum is not None else None,
        "observed_records": len(rows),
        "zero_threshold_max_observed_by_metric": zero_maxima,
        "zero_threshold_violation_records_by_metric": zero_violations,
    }


def _parent_audit_module():
    if str(common.PARENT) not in sys.path:
        sys.path.insert(0, str(common.PARENT))
    return importlib.import_module("audit")


def _bound_writers(binding: dict[str, str]):
    def stable_write(path: Path, value: dict[str, Any]) -> None:
        bound = common.add_binding(dict(value), binding)
        if path.exists():
            observed = common.read_json(path)
            common.require(observed == bound, f"refusing changed retained JSON: {path}")
            return
        common.exclusive_json(path, bound)

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        bound = [common.add_binding(dict(row), binding) for row in rows]
        if path.exists():
            observed = common.read_jsonl(path)
            common.require(observed == bound, f"refusing changed retained JSONL: {path}")
            return
        common.exclusive_jsonl(path, bound)

    return stable_write, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    common.require(args.tag == common.RESULT_TAG, "recovery result tag differs")
    common.require(args.shard_count == 4, "recovery requires four shards")
    common.require(args.gpu == args.shard_index, "GPU/shard binding differs")
    # Bind the physical card before any validation import can transitively load
    # a CUDA-aware module.  The frozen parent repeats these assignments before
    # it imports a candidate builder.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13.1"
    os.environ["PATH"] = "/usr/local/cuda-13.1/bin:" + os.environ.get("PATH", "")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(
        common.PARENT / ".torch_ext" / f"gpu{args.gpu}"
    )
    os.environ.setdefault("MAX_JOBS", "4")
    ready = validate.validate_launch_ready(args.gpu, allow_busy=args.allow_busy)
    binding = common.binding_for_commit(ready["recovery_git_commit"])
    validate.ensure_remote_receipt(ready)
    common.validate_retained_tree(binding)
    parent = _parent_audit_module()
    stable_write, write_jsonl = _bound_writers(binding)
    parent.gate_summary = fixed_gate_summary
    parent.stable_write = stable_write
    parent.write_jsonl_atomic = write_jsonl
    # In direct-script mode the parent's absolute ``validate`` import can
    # resolve to this recovery module.  Normalize the callable explicitly and
    # retain the parent's return schema for its frozen receipt writer.
    parent.validate_launch_ready = lambda gpu, allow_busy=False: (
        validate.validate_launch_ready(gpu, allow_busy=allow_busy)["parent"]
    )
    result = parent.main()
    common.validate_retained_tree(binding)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
