#!/usr/bin/env python3
"""Measure one v3 candidate in one fresh process."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.fused_frontier_closure_v3 import (
        candidates,
        core,
        provenance,
    )
else:  # pragma: no cover
    from . import candidates, core, provenance


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_launch(path: Path) -> dict[str, Any]:
    value = core.read_json(path)
    if value.get("record_type") != "fused_frontier_closure_v3_launch":
        raise core.ClosureError("foreign launch receipt")
    if path.read_bytes() != core.stable_json_bytes(value):
        raise core.ClosureError("launch receipt is not stable JSON")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--block", required=True, type=int)
    parser.add_argument("--position", required=True, type=int)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise core.ClosureError(f"refusing overwrite: {args.out}")
    source_receipt = provenance.verify_receipt()
    campaign = core.load_campaign()
    launch_path = args.launch_receipt.resolve()
    launch = _load_launch(launch_path)
    binding = launch["binding"]
    expected_binding = {
        "campaign_canonical_sha256": core.canonical_sha256(campaign),
        "source_receipt_sha256": core.sha256_file(core.SOURCE_RECEIPT_PATH),
        "eligibility_receipt_sha256": core.sha256_file(
            core.ELIGIBILITY_RECEIPT_PATH
        ),
        "performance_protocol_sha256": core.protocol_sha256(campaign),
        "candidate_sha256": source_receipt["candidate_sha256"],
    }
    for key, value in expected_binding.items():
        if binding.get(key) != value:
            raise core.ClosureError(f"launch {key} binding differs")
    key = (args.candidate, args.block, args.position)
    planned = {
        (row["candidate_id"], row["block"], row["position"])
        for row in launch["plan"]
    }
    if key not in planned:
        raise core.ClosureError(f"measurement absent from launch plan: {key}")
    result_root = (core.REPO_ROOT / launch["result_root"]).resolve()
    args.out.resolve().relative_to((result_root / "raw").resolve())
    definitions = core.candidates_by_id(campaign)
    definition = definitions.get(args.candidate)
    if definition is None:
        raise core.ClosureError(f"unknown candidate {args.candidate}")
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "fused_frontier_closure_v3_measurement",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": binding["campaign_canonical_sha256"],
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "eligibility_receipt_sha256": binding["eligibility_receipt_sha256"],
        "performance_protocol_sha256": binding["performance_protocol_sha256"],
        "launch_receipt_sha256": core.sha256_file(launch_path),
        "candidate_id": args.candidate,
        "candidate_definition": definition,
        "candidate_definition_sha256": binding["candidate_sha256"][args.candidate],
        "block": args.block,
        "position": args.position,
        "protocol": campaign["performance_protocol"],
        "pid": os.getpid(),
        "started_utc": _utc_now(),
        "ok": False,
    }
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
            raise core.ClosureError("measurement requires CUDA_VISIBLE_DEVICES=3")
        built = candidates.build_candidate(
            definition, expected_candidate_id=args.candidate
        )
        import torch

        for directory in (
            core.REPO_ROOT / "ako_runs/phase2_fused_sdpa",
            core.REPO_ROOT / "ako_runs/phase1_matmul",
        ):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        import common
        import common2
        from ako_runs.controlled_followup.fused_closure_v2.measure import (
            _time_candidate,
        )

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise core.ClosureError("measurement must see exactly one logical GPU")
        props = torch.cuda.get_device_properties(0)
        hardware = campaign["hardware"]
        if props.name != hardware["required_name"] or f"{props.major}.{props.minor}" != (
            hardware["required_compute_capability"]
        ):
            raise core.ClosureError("logical GPU model/capability differs")
        protocol = campaign["performance_protocol"]
        x32, weight, bias = common2.fused_inputs(
            seed=protocol["seed"], dist=protocol["dist"], device="cuda:0"
        )
        x16 = x32.half().contiguous()
        operands = (x32, x16, weight, bias)
        with torch.no_grad():
            output = built.run(*operands)
            torch.cuda.synchronize()
            expected_shape = (common2.F_M, common2.F_N)
            if tuple(output.shape) != expected_shape or output.dtype != torch.float32:
                raise core.ClosureError(
                    f"output contract differs: shape={tuple(output.shape)} dtype={output.dtype}"
                )
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
        if len(times) != 100 or any(
            not math.isfinite(value) or value <= 0 for value in times
        ):
            raise core.ClosureError("invalid timing samples")
        record.update(
            {
                "ok": True,
                "build_metadata": built.build_metadata,
                "environment": {
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "logical_device": 0,
                    "device_name": props.name,
                    "compute_capability": [props.major, props.minor],
                    "total_memory_bytes": props.total_memory,
                    "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                },
                "output_dtype": "torch.float32",
                "output_shape": list(expected_shape),
                "legacy_gate_diagnostic": legacy_gate,
                "warmup_iterations": warmup_iterations,
                "trial_times_ms": times,
                "timing_summary": common.summarize(times),
                "completed_utc": _utc_now(),
            }
        )
    except Exception as exc:  # noqa: BLE001 - retained evidence
        record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "completed_utc": _utc_now(),
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
