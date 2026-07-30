#!/usr/bin/env python3
"""Build, validate, and time one fused-reachability-v2 candidate."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PHASE1 = REPO_ROOT / "ako_runs" / "phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs" / "phase2_fused_sdpa"
for path in (str(REPO_ROOT), str(PHASE2), str(PHASE1), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
if "/usr/local/cuda-13.1/bin" not in os.environ.get("PATH", "").split(":"):
    os.environ["PATH"] = "/usr/local/cuda-13.1/bin:" + os.environ.get("PATH", "")

import torch  # noqa: E402
import common  # noqa: E402
import common2  # noqa: E402
import runner2  # noqa: E402
from candidate import build, make_config  # noqa: E402
from protocol import (  # noqa: E402
    JOBS,
    LOCK,
    RESULTS,
    canonical_sha256,
    file_sha256,
    read_json,
    record_filename,
    gpu_snapshot,
    validate_gpu_snapshot,
    stable_write,
    verify_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--phase", choices=("screen", "confirmation"), required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist", choices=("rand", "randn"), default="rand")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = verify_lock()
    job = json.loads(args.job_json)
    frozen_jobs = {row["job_id"]: row for row in read_json(JOBS)}
    if job.get("job_id") not in frozen_jobs or job != frozen_jobs[job["job_id"]]:
        raise RuntimeError("job does not exactly match the frozen job map")
    if lock["job_sha256"][job["job_id"]] != canonical_sha256(job):
        raise RuntimeError("job canonical hash mismatch")
    expected_reps = lock["launch_policy"][args.phase]["reps"]
    if args.rep not in range(expected_reps):
        raise ValueError(f"rep must be in [0,{expected_reps})")
    if (args.trials, args.warmup_s, args.seed, args.dist) != (100, 2.0, 0, "rand"):
        raise ValueError("timing protocol differs from the frozen campaign")
    output_path = Path(args.out).resolve()
    output_path.relative_to(RESULTS.resolve())
    if output_path.name != record_filename(job["job_id"], args.rep):
        raise ValueError("output filename does not match frozen job/rep")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    record = {
        "schema_version": 1,
        "record_type": "fused_reachability_v2_process",
        "campaign_id": "fused-reachability-streamed-epilogue-v2",
        "phase": args.phase,
        "job": job,
        "job_sha256": lock["job_sha256"][job["job_id"]],
        "rep": args.rep,
        "seed": args.seed,
        "dist": args.dist,
        "trials": args.trials,
        "warmup_s": args.warmup_s,
        "physical_gpu": args.physical_gpu,
        "logical_device": "cuda:0",
        "launch_lock_sha256": file_sha256(LOCK),
        "source_bundle_sha256": lock["source_bundle_sha256"],
        "pid": os.getpid(),
        "host": platform.node(),
        "t_start": time.time(),
        "gpu_preflight": gpu_snapshot(args.physical_gpu),
    }
    validate_gpu_snapshot(record["gpu_preflight"])
    try:
        cfg = make_config(job)
        record["config"] = cfg.to_dict()
        built = build(job["lane"], cfg)
        record["compile_s"] = built.compile_s
        record["artifacts"] = built.artifacts

        x, W, bias = common2.fused_inputs(seed=args.seed, dist=args.dist)
        with torch.no_grad():
            reference = common2.fused_reference(x, W, bias, arm="GBGS")
        x16 = x.half().contiguous()
        with torch.no_grad():
            output = built.run(x16, W, bias)
            torch.cuda.synchronize()
        record["output_dtype"] = str(output.dtype)
        record["legacy_error"] = common.gate_stats(reference, output.float())
        del reference, output
        torch.cuda.empty_cache()

        times, warm_iters = runner2.time_kernel3(
            built.run,
            x16,
            W,
            bias,
            num_trials=args.trials,
            warmup_s=args.warmup_s,
            flush_l2=True,
        )
        record["warmup_iters_actual"] = warm_iters
        record["times_ms"] = [float(value) for value in times]
        record["median_ms"] = statistics.median(record["times_ms"])
        record["mean_ms"] = statistics.fmean(record["times_ms"])
        record["min_ms"] = min(record["times_ms"])
        record["max_ms"] = max(record["times_ms"])
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - retained experimental outcome
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    try:
        record["env"] = common.env_fingerprint()
        record["env_scope"] = (
            "logical CUDA namespace; gpu_preflight and launch receipt are the "
            "authoritative physical-device records"
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        record["env_error"] = f"{type(exc).__name__}: {exc}"
    record["t_end"] = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stable_write(output_path, record)
    print("###JSON###")
    print(json.dumps(record, default=str))
    return 0 if record.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
