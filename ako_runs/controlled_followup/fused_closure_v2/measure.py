#!/usr/bin/env python3
"""Measure one frozen closure candidate in one fresh serialized process."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.fused_closure_v2 import candidates, core, provenance
else:  # pragma: no cover
    from . import candidates, core, provenance


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _time_candidate(
    torch,
    run: Callable[[Any, Any, Any, Any], Any],
    operands: tuple[Any, Any, Any, Any],
    *,
    trials: int,
    warmup_s: float,
    flush_l2: bool,
) -> tuple[list[float], int]:
    started, iterations = time.perf_counter(), 0
    while True:
        run(*operands)
        iterations += 1
        if iterations % 4 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() - started >= warmup_s:
                break
    torch.cuda.synchronize()
    flusher = (
        torch.empty(int(128e6 // 4), dtype=torch.float32, device="cuda:0")
        if flush_l2
        else None
    )
    times = []
    for _ in range(trials):
        if flusher is not None:
            flusher.zero_()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        run(*operands)
        end.record()
        torch.cuda.synchronize()
        times.append(float(begin.elapsed_time(end)))
    del flusher
    return times, iterations


def _load_launch(path: Path) -> dict[str, Any]:
    launch = core.read_json(path)
    if launch.get("record_type") != "fused_closure_v2_performance_launch":
        raise core.ClosureError("not a closure-v2 performance launch receipt")
    if path.read_bytes() != core.stable_json_bytes(launch):
        raise core.ClosureError("performance launch receipt is not stable JSON")
    return launch


def _logical_environment(torch) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "logical_device": 0,
        "device_name": props.name,
        "compute_capability": [props.major, props.minor],
        "multi_processor_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--block", required=True, type=int)
    parser.add_argument("--position", required=True, type=int)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise core.ClosureError(f"refusing to overwrite existing raw record: {args.out}")
    source_receipt = provenance.verify_receipt()
    campaign = core.load_campaign()
    launch = _load_launch(args.launch_receipt.resolve())
    binding = launch.get("binding", {})
    if binding.get("campaign_canonical_sha256") != core.canonical_sha256(campaign):
        raise core.ClosureError("launch receipt campaign binding differs")
    if binding.get("source_receipt_sha256") != core.sha256_file(
        core.SOURCE_RECEIPT_PATH
    ):
        raise core.ClosureError("launch receipt source binding differs")
    if binding.get("performance_protocol_sha256") != core.protocol_sha256(campaign):
        raise core.ClosureError("launch receipt protocol binding differs")
    if binding.get("candidate_sha256") != source_receipt["candidate_sha256"]:
        raise core.ClosureError("launch receipt candidate bindings differ")
    planned = {
        (item["candidate_id"], item["block"], item["position"])
        for item in launch.get("plan", [])
    }
    key = (args.candidate, args.block, args.position)
    if key not in planned:
        raise core.ClosureError(f"measurement {key!r} is absent from launch plan")
    result_root = (core.REPO_ROOT / launch["result_root"]).resolve()
    try:
        args.out.resolve().relative_to((result_root / "raw").resolve())
    except ValueError as exc:
        raise core.ClosureError("raw output path is outside launch raw directory") from exc
    definitions = core.candidates_by_id(campaign)
    if args.candidate not in definitions:
        raise core.ClosureError(f"unknown candidate {args.candidate!r}")
    definition = definitions[args.candidate]
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "fused_closure_v2_performance_measurement",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": binding["campaign_canonical_sha256"],
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "performance_protocol_sha256": binding["performance_protocol_sha256"],
        "gate_summary_sha256": binding["gate_summary_sha256"],
        "performance_launch_receipt_sha256": core.sha256_file(
            args.launch_receipt.resolve()
        ),
        "candidate_id": args.candidate,
        "candidate_definition": definition,
        "candidate_definition_sha256": binding["candidate_sha256"][args.candidate],
        "block": args.block,
        "position": args.position,
        "process_id": os.getpid(),
        "process_started_utc": _utc_now(),
        "protocol": campaign["performance_protocol"],
        "ok": False,
    }
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise core.ClosureError("measurement requires CUDA_VISIBLE_DEVICES=0")
        built = candidates.build_candidate(
            definition, expected_candidate_id=args.candidate
        )
        # candidates._imports has now loaded torch/common2 under the frozen source.
        import torch

        phase2 = core.REPO_ROOT / "ako_runs/phase2_fused_sdpa"
        phase1 = core.REPO_ROOT / "ako_runs/phase1_matmul"
        for directory in (phase2, phase1):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        import common
        import common2

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise core.ClosureError("measurement must see exactly one CUDA GPU")
        protocol = campaign["performance_protocol"]
        x32, weight, bias = common2.fused_inputs(
            seed=protocol["seed"], dist=protocol["dist"], device="cuda:0"
        )
        x16 = x32.half().contiguous()
        operands = (x32, x16, weight, bias)
        with torch.no_grad():
            # This untimed call both validates the output contract and populates
            # the frozen address-keyed fp16 weight cache with the real weight.
            output = built.run(*operands)
            torch.cuda.synchronize()
            expected_shape = (common2.F_M, common2.F_N)
            if tuple(output.shape) != expected_shape:
                raise core.ClosureError(
                    f"output shape {tuple(output.shape)} differs from {expected_shape}"
                )
            if output.dtype != torch.float32:
                raise core.ClosureError(f"output dtype {output.dtype} is not fp32")
            reference = common2.fused_reference(
                x32, weight, bias, arm="GBGS", dtype=torch.float32
            )
            torch.cuda.synchronize()
            legacy_gate = common.gate_stats(reference, output)
            del reference, output
            times, warmup_iterations = _time_candidate(
                torch,
                built.run,
                operands,
                trials=protocol["trials"],
                warmup_s=protocol["warmup_s"],
                flush_l2=protocol["flush_l2"],
            )
        if len(times) != protocol["trials"] or any(
            not math.isfinite(value) or value <= 0 for value in times
        ):
            raise core.ClosureError("timing returned missing/nonpositive/nonfinite trials")
        record.update(
            {
                "ok": True,
                "build_metadata": built.build_metadata,
                "environment": _logical_environment(torch),
                "output_dtype": "torch.float32",
                "output_shape": list(expected_shape),
                "legacy_gate_diagnostic": legacy_gate,
                "warmup_iterations": warmup_iterations,
                "trial_times_ms": times,
                "timing_summary": common.summarize(times),
                "process_completed_utc": _utc_now(),
            }
        )
    except Exception as exc:  # noqa: BLE001 - failure is retained as raw evidence
        record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "process_completed_utc": _utc_now(),
            }
        )
    core.atomic_json(args.out.resolve(), record)
    print(
        f"{args.candidate} block={args.block} position={args.position} "
        f"ok={record['ok']} out={args.out}",
        flush=True,
    )
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
