#!/usr/bin/env python3
"""Load and measure one exact frozen artifact in one fresh process."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.archived_current_fused_v1 import protocol
else:  # pragma: no cover
    from . import protocol


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def command(argv: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv, check=False, capture_output=True, text=True, timeout=timeout
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    point = probability * (len(ordered) - 1)
    low, high = math.floor(point), math.ceil(point)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
        "p05_ms": quantile(values, 0.05),
        "p25_ms": quantile(values, 0.25),
        "p75_ms": quantile(values, 0.75),
        "p95_ms": quantile(values, 0.95),
    }


def load_exact_model(torch, nn, subject: dict[str, Any]):
    target = (protocol.REPO_ROOT / subject["source_path"]).resolve()
    observed_hash = protocol.sha256_file(target)
    if observed_hash != subject["expected_source_sha256"]:
        raise protocol.CampaignError(
            f"target hash changed before import: {target} {observed_hash}"
        )
    module_name = "archived_current_fused_" + subject["subject_id"]
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise protocol.CampaignError(f"cannot construct exact loader for {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = Path(module.__file__).resolve()
    if loaded != target:
        raise protocol.CampaignError(f"loader resolved {loaded}, expected {target}")
    cls = getattr(module, "Model", None)
    if not isinstance(cls, type) or not issubclass(cls, nn.Module):
        raise protocol.CampaignError(f"{target} has no faithful nn.Module Model")
    model = cls(8192, 8192).to("cuda:0").eval()
    linears = [item for item in model.modules() if isinstance(item, nn.Linear)]
    if len(linears) != 1:
        raise protocol.CampaignError(
            f"expected exactly one nn.Linear in {subject['subject_id']}, found {len(linears)}"
        )
    linear = linears[0]
    if tuple(linear.weight.shape) != (8192, 8192) or tuple(linear.bias.shape) != (8192,):
        raise protocol.CampaignError("loaded Model has an unexpected linear shape")
    return module, model, linear, target, observed_hash


def make_inputs(torch, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bound = 1.0 / math.sqrt(8192)
    x = torch.rand((1024, 8192), generator=generator, dtype=torch.float32)
    weight = (torch.rand((8192, 8192), generator=generator) * 2.0 - 1.0) * bound
    bias = (torch.rand((8192,), generator=generator) * 2.0 - 1.0) * bound
    return x.to("cuda:0"), weight.to("cuda:0"), bias.to("cuda:0")


def exact_reference(torch, x, weight, bias):
    value = x @ weight.transpose(0, 1)
    value = value + bias
    value = 0.5 * value * (1.0 + torch.erf(value / math.sqrt(2.0)))
    return torch.softmax(value, dim=1)


def correctness_diagnostic(torch, output, reference, specification: dict[str, Any]):
    if not isinstance(output, torch.Tensor):
        raise protocol.CampaignError("Model.forward did not return a tensor")
    if tuple(output.shape) != (1024, 8192):
        raise protocol.CampaignError(f"unexpected output shape: {tuple(output.shape)}")
    if output.dtype != torch.float32:
        raise protocol.CampaignError(f"unexpected output dtype: {output.dtype}")
    finite = bool(torch.isfinite(output).all().item())
    if not finite:
        raise protocol.CampaignError("output contains NaN or Inf")
    difference = (output - reference).abs()
    budget = specification["historical_atol"] + specification["historical_rtol"] * reference.abs()
    failures = int((difference > budget).sum().item())
    negative_count = int((output < 0).sum().item())
    row_sum_error_max = float((output.double().sum(dim=1) - 1.0).abs().max().item())
    diagnostic = {
        "reference": specification["reference"],
        "finite": finite,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "historical_atol": specification["historical_atol"],
        "historical_rtol": specification["historical_rtol"],
        "historical_failure_count": failures,
        "max_abs_err": float(difference.max().item()),
        "mean_abs_err": float(difference.mean().item()),
        "negative_count": negative_count,
        "row_sum_error_max": row_sum_error_max,
        "row_sum_threshold": 5e-7,
        "qualification": specification["qualification"],
    }
    diagnostic["pass"] = (
        failures == 0 and negative_count == 0 and row_sum_error_max <= 5e-7
    )
    if not diagnostic["pass"]:
        raise protocol.CampaignError(f"correctness diagnostic failed: {diagnostic}")
    return diagnostic


def time_model(torch, model, x, performance: dict[str, Any]):
    warmup_started = time.perf_counter()
    warmup_iterations = 0
    with torch.no_grad():
        while True:
            model(x)
            warmup_iterations += 1
            if warmup_iterations % 4 == 0:
                torch.cuda.synchronize()
                if time.perf_counter() - warmup_started >= performance["warmup_s"]:
                    break
        torch.cuda.synchronize()
        flusher = None
        if performance["flush_l2"]:
            if performance["l2_flush_bytes"] % 8:
                raise protocol.CampaignError("l2_flush_bytes must be divisible by 8")
            flusher = torch.empty(
                performance["l2_flush_bytes"] // 8,
                dtype=torch.int64,
                device="cuda:0",
            )
        values: list[float] = []
        total = performance["trials"] + performance["discard_first"]
        for trial in range(total):
            torch.cuda.synchronize()
            if flusher is not None:
                flusher.fill_(42)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            elapsed = float(begin.elapsed_time(end))
            if trial >= performance["discard_first"]:
                values.append(elapsed)
        del flusher
    if len(values) != performance["trials"]:
        raise protocol.CampaignError("timing trial census differs from protocol")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise protocol.CampaignError("timing contains a nonpositive/nonfinite value")
    return values, warmup_iterations


def load_launch(path: Path, lock: dict[str, Any]) -> dict[str, Any]:
    launch = protocol.read_json(path)
    if launch.get("record_type") != "archived_current_fused_v1_launch":
        raise protocol.CampaignError("not an archived-current fused launch receipt")
    if launch.get("launch_lock_file_sha256") != protocol.sha256_file(protocol.LOCK_PATH):
        raise protocol.CampaignError("launch receipt lock binding differs")
    if launch.get("launch_lock_canonical_sha256") != protocol.canonical_sha256(lock):
        raise protocol.CampaignError("launch receipt canonical lock binding differs")
    if path.read_bytes() != protocol.stable_json_bytes(launch):
        raise protocol.CampaignError("launch receipt is not stable JSON")
    return launch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise protocol.CampaignError(f"refusing to overwrite raw record: {args.out}")
    campaign, receipt, jobs, lock = protocol.verify_lock()
    launch = load_launch(args.launch_receipt.resolve(), lock)
    matching = [row for row in jobs["plan"] if row["job_id"] == args.job_id]
    if len(matching) != 1:
        raise protocol.CampaignError(f"job is absent or duplicated: {args.job_id}")
    job = matching[0]
    expected_output = (
        protocol.REPO_ROOT / launch["result_root"] / protocol.raw_relative(job)
    ).resolve()
    if args.out.resolve() != expected_output:
        raise protocol.CampaignError("raw output path differs from launch plan")
    subject = protocol.subject_map(campaign)[job["subject_id"]]
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "archived_current_fused_v1_measurement",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": protocol.canonical_sha256(campaign),
        "source_receipt_file_sha256": protocol.sha256_file(protocol.SOURCE_RECEIPT_PATH),
        "launch_lock_file_sha256": protocol.sha256_file(protocol.LOCK_PATH),
        "jobs_file_sha256": protocol.sha256_file(protocol.JOBS_PATH),
        "launch_receipt_file_sha256": protocol.sha256_file(args.launch_receipt.resolve()),
        "job": job,
        "subject": subject,
        "process_id": os.getpid(),
        "process_started_utc": utc_now(),
        "performance_protocol": campaign["performance_protocol"],
        "ok": False,
    }
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise protocol.CampaignError("child requires CUDA_VISIBLE_DEVICES=0")
        import torch
        import torch.nn as nn

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise protocol.CampaignError("child must see exactly one CUDA device")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        expected_cc = campaign["performance_protocol"]["expected_compute_capability"]
        if [props.major, props.minor] != expected_cc:
            raise protocol.CampaignError(
                f"compute capability {[props.major, props.minor]} differs from {expected_cc}"
            )
        module, model, linear, target, target_hash = load_exact_model(torch, nn, subject)
        x, weight, bias = make_inputs(torch, campaign["performance_protocol"]["data_seed"])
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
            output = model(x)
            torch.cuda.synchronize()
            reference = exact_reference(torch, x, weight, bias)
            torch.cuda.synchronize()
            diagnostic = correctness_diagnostic(
                torch, output, reference, campaign["correctness_diagnostic"]
            )
            del output, reference, weight, bias
            torch.cuda.empty_cache()
            values, warmup_iterations = time_model(
                torch, model, x, campaign["performance_protocol"]
            )
        versions = {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tilelang": getattr(module, "tilelang", None).__version__
            if getattr(module, "tilelang", None) is not None
            else None,
            "triton": getattr(module, "triton", None).__version__
            if getattr(module, "triton", None) is not None
            else None,
        }
        record.update(
            {
                "ok": True,
                "loaded_source_path": str(target.relative_to(protocol.REPO_ROOT)),
                "loaded_source_sha256": target_hash,
                "loaded_module_file": str(Path(module.__file__).resolve().relative_to(protocol.REPO_ROOT)),
                "correctness_diagnostic": diagnostic,
                "warmup_iterations": warmup_iterations,
                "trial_times_ms": values,
                "timing_summary": summarize(values),
                "environment": {
                    **versions,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "logical_device": 0,
                    "device_name": props.name,
                    "compute_capability": [props.major, props.minor],
                    "multi_processor_count": props.multi_processor_count,
                    "total_memory_bytes": props.total_memory,
                    "nvidia_smi_identity": command(
                        [
                            "nvidia-smi",
                            "-i",
                            "0",
                            "--query-gpu=uuid,name,driver_version,persistence_mode,temperature.gpu,clocks.sm,clocks.max.sm,power.draw",
                            "--format=csv,noheader,nounits",
                        ]
                    ),
                },
                "process_completed_utc": utc_now(),
            }
        )
    except Exception as exc:  # noqa: BLE001 - failure is evidence
        record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "process_completed_utc": utc_now(),
            }
        )
    protocol.atomic_json(args.out.resolve(), record)
    print(
        f"{job['job_id']} ok={record['ok']} out={args.out}",
        flush=True,
    )
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

