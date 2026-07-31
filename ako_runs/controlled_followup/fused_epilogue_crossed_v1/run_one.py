#!/usr/bin/env python3
"""Build, check, and time one frozen crossed-epilogue cell in a fresh process."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

from candidates import build
from core import (
    LOCK_PATH,
    REPO_ROOT,
    canonical_sha256,
    file_sha256,
    gpu_snapshot,
    load_contract,
    read_json,
    stable_write,
    timing_filename,
    validate_gpu,
)


PHASE1 = REPO_ROOT / "ako_runs/phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs/phase2_fused_sdpa"
for path in (str(PHASE2), str(PHASE1)):
    if path not in sys.path:
        sys.path.insert(0, path)


def compact_artifact(value):
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
        encoded = value.encode("utf-8")
        return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "confirmation"), required=True)
    parser.add_argument("--cell-json", required=True)
    parser.add_argument("--distribution", choices=("positive", "withheld_signed"), required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    campaign, cells, lock = load_contract()
    cell = json.loads(args.cell_json)
    frozen = {row["cell_id"]: row for row in cells}
    if cell.get("cell_id") not in frozen or cell != frozen[cell["cell_id"]]:
        raise RuntimeError("cell differs from frozen manifest")
    if args.physical_gpu != campaign["hardware"]["timing_gpu"]:
        raise RuntimeError("all timing is frozen to physical GPU 0")
    eligibility_path = Path(args.eligibility).resolve()
    eligibility = read_json(eligibility_path)
    if eligibility.get("campaign_id") != campaign["campaign_id"] or eligibility.get("launch_lock_sha256") != file_sha256(LOCK_PATH):
        raise RuntimeError("foreign eligibility artifact")
    legal_ids = set(eligibility.get("timing_eligible_cell_ids", []))
    if cell["cell_id"] not in legal_ids:
        raise RuntimeError("cell is not complete-fused-v2-gate eligible")
    if args.phase == "screen":
        if args.distribution != "positive" or args.rep not in range(2):
            raise RuntimeError("screen phase contract mismatch")
        seed, dist = 0, "rand"
    else:
        selected_ids = set(eligibility.get("selected_cell_ids", []))
        if cell["cell_id"] not in selected_ids:
            raise RuntimeError("cell is not in frozen confirmation selection")
        if args.rep not in range(15):
            raise RuntimeError("confirmation rep must be 0--14")
        mapping = {
            "positive": (0, "rand"),
            "withheld_signed": (2026073101, "randn"),
        }
        seed, dist = mapping[args.distribution]
    output = Path(args.out).resolve()
    expected_name = timing_filename(cell["cell_id"], args.distribution, args.rep)
    if output.name != expected_name or output.exists():
        raise RuntimeError("unsafe, mismatched, or existing timing output")
    snapshot = gpu_snapshot(args.physical_gpu)
    validate_gpu(snapshot, campaign)
    record = {
        "campaign_id": campaign["campaign_id"],
        "cell": cell,
        "cell_sha256": canonical_sha256(cell),
        "distribution": args.distribution,
        "eligibility_path": str(eligibility_path.relative_to(REPO_ROOT)),
        "eligibility_sha256": file_sha256(eligibility_path),
        "launch_lock_sha256": file_sha256(LOCK_PATH),
        "logical_device": "cuda:0",
        "phase": args.phase,
        "physical_gpu": args.physical_gpu,
        "pid": os.getpid(),
        "record_type": "fused_crossed_timing_process",
        "rep": args.rep,
        "schema_version": 1,
        "seed": seed,
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "timing_dist": dist,
        "trials": campaign["performance"]["trials"],
        "warmup_s": campaign["performance"]["warmup_s"],
        "gpu_preflight": snapshot,
        "host": platform.node(),
        "t_start": time.time(),
    }
    try:
        import torch
        import common
        import common2
        import runner2

        built = build(cell)
        x, weight, bias = common2.fused_inputs(seed=seed, dist=dist)
        with torch.no_grad():
            reference = common2.fused_reference(x, weight, bias, arm="GBGS", dtype=torch.float32)
        x16 = x.half().contiguous()
        with torch.no_grad():
            output_tensor = built.run(x16, weight, bias)
            torch.cuda.synchronize()
        record["legacy_error"] = common.gate_stats(reference, output_tensor.float())
        record["output_dtype"] = str(output_tensor.dtype)
        record["output_shape"] = list(output_tensor.shape)
        del reference, output_tensor
        torch.cuda.empty_cache()
        times, warmup_iterations = runner2.time_kernel3(
            built.run, x16, weight, bias,
            num_trials=campaign["performance"]["trials"],
            warmup_s=campaign["performance"]["warmup_s"],
            flush_l2=campaign["performance"]["flush_l2"],
        )
        numeric = [float(value) for value in times]
        if len(numeric) != 100 or any(not math.isfinite(value) or value <= 0 for value in numeric):
            raise RuntimeError("timing returned invalid samples")
        record.update(
            {
                "build_metadata": compact_artifact(built.metadata),
                "compile_s": built.compile_s,
                "max_ms": max(numeric),
                "mean_ms": statistics.fmean(numeric),
                "median_ms": statistics.median(numeric),
                "min_ms": min(numeric),
                "ok": True,
                "times_ms": numeric,
                "warmup_iterations_actual": warmup_iterations,
            }
        )
    except Exception as exc:  # retained timing-process failure
        record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "ok": False,
                "traceback": traceback.format_exc(),
            }
        )
    record["t_end"] = time.time()
    stable_write(output, record)
    print("###JSON###")
    print(json.dumps(record, default=str))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
