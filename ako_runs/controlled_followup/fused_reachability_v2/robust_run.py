#!/usr/bin/env python3
"""Adjudicate frozen v2 candidates under the unchanged fused-v2 gates."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PHASE1 = REPO_ROOT / "ako_runs" / "phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs" / "phase2_fused_sdpa"
FUSED_GRID = REPO_ROOT / "ako_runs" / "controlled_followup" / "fused_grid"
for path in (str(REPO_ROOT), str(PHASE2), str(PHASE1), str(FUSED_GRID), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from candidate import build, make_config  # noqa: E402
from protocol import (  # noqa: E402
    ADAPTER,
    JOBS,
    LOCK,
    canonical_sha256,
    file_sha256,
    read_json,
    safe_result_root,
    acquire_active_lock,
    gpu_snapshot,
    nvcc_fingerprint,
    validate_gpu_snapshot,
    stable_write,
    verify_lock,
)
import robust_adapter as adapter  # noqa: E402
from robust_gate.schema import write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("cuda_noptx", "cuda_unlimited"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--allow-busy", action="store_true")
    return parser.parse_args()


def ensure_idle(gpu: int, allow_busy: bool) -> None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    occupants = [line for line in completed.stdout.splitlines() if line.strip()]
    if (completed.returncode != 0 or occupants) and not allow_busy:
        raise RuntimeError(f"GPU {gpu} preflight failed or is busy: {occupants}")


def make_context(lock: dict, all_jobs: list[dict]) -> adapter.RepositoryContext:
    original = adapter.load_repository()
    prospective = read_json(ADAPTER)
    frozen = lock["frozen_gate"]
    robust_info = prospective["robust_gate"]
    if (
        robust_info["manifest_sha256"] != frozen["manifest_sha256"]
        or robust_info["gate_spec_sha256"] != frozen["gate_spec_sha256"]
        or original.adapter["robust_gate"] != robust_info
        or original.manifest_sha256 != robust_info["manifest_canonical_sha256"]
        or canonical_sha256(original.gate_spec) != robust_info["gate_spec_canonical_sha256"]
    ):
        raise RuntimeError("loaded robust context differs from the frozen v2 gate binding")
    return adapter.RepositoryContext(
        adapter_path=ADAPTER,
        adapter=prospective,
        adapter_sha256=file_sha256(ADAPTER),
        grid_manifest=read_json(HERE / "campaign.json"),
        jobs=tuple(all_jobs),
        robust_manifest=original.robust_manifest,
        gate_spec=original.gate_spec,
    )


def make_plan(context, job: dict) -> adapter.CandidatePlan:
    started = time.perf_counter()
    try:
        config = make_config(job)
        built = build(job["lane"], config)

        def execute(inputs, prepared, *, _built=built):
            if "x_fp16" not in prepared:
                prepared["x_fp16"] = inputs["x"].half().contiguous()
            return _built.run(prepared["x_fp16"], inputs["weight"], inputs["bias"])

        return adapter.CandidatePlan(
            candidate=adapter.candidate_name(context, job),
            job=job,
            job_sha256=context.adapter["grid"]["job_sha256"][job["job_id"]],
            config=config.to_dict(),
            build_metadata={
                "adapter_build_wall_s": time.perf_counter() - started,
                "reported_compile_s": built.compile_s,
                "n_kernels": built.n_kernels,
                "artifacts": built.artifacts,
                "prospective_campaign": context.adapter["campaign_id"],
            },
            execute=execute,
        )
    except Exception as exc:  # noqa: BLE001 - retained build evidence
        return adapter.CandidatePlan(
            candidate=adapter.candidate_name(context, job),
            job=job,
            job_sha256=context.adapter["grid"]["job_sha256"][job["job_id"]],
            config=None,
            build_metadata={"adapter_build_wall_s": time.perf_counter() - started},
            build_error=f"BuildError: {type(exc).__name__}: {exc}",
            build_traceback=traceback.format_exc(),
        )


def bundle_path(root: Path, job_id: str, case_id: str, seed_index: int) -> Path:
    return root / "raw" / job_id / case_id / f"seed{seed_index:03d}.json"


def validate_existing(path: Path, plan, context, case_id: str, seed_index: int):
    value = read_json(path)
    rows = value.get("records")
    if not isinstance(rows, list) or len(rows) != 2:
        raise RuntimeError(f"invalid retained bundle: {path}")
    expected = {
        "candidate": plan.candidate,
        "case_id": case_id,
        "seed_index": seed_index,
        "source_bundle_sha256": context.source_bundle_sha256,
        "adapter_manifest_sha256": context.adapter_sha256,
    }
    for row in rows:
        if any(row.get(key) != wanted for key, wanted in expected.items()):
            raise RuntimeError(f"foreign retained bundle: {path}")
    return rows


def main() -> int:
    args = parse_args()
    lock = verify_lock()
    ensure_idle(args.gpu, args.allow_busy)
    gpu_before = gpu_snapshot(args.gpu)
    validate_gpu_snapshot(gpu_before)
    all_jobs = read_json(JOBS)
    selection_path = Path(args.selection)
    selection = read_json(selection_path)
    if (
        selection.get("record_type") != "fused_reachability_v2_screen_selection"
        or selection.get("campaign_id") != lock["campaign_id"]
        or selection.get("launch_lock_sha256") != file_sha256(LOCK)
        or selection.get("source_bundle_sha256") != lock["source_bundle_sha256"]
    ):
        raise RuntimeError("screen selection is not bound to this frozen campaign")
    selected_rows = selection.get("selected")
    if not isinstance(selected_rows, list) or len(selected_rows) != 6:
        raise RuntimeError("screen selection must contain exactly six rows")
    if len({row.get("job_id") for row in selected_rows}) != 6:
        raise RuntimeError("screen selection contains duplicate jobs")
    for lane in ("cuda_noptx", "cuda_unlimited"):
        ranks = sorted(
            row.get("screen_rank") for row in selected_rows if row.get("lane") == lane
        )
        if ranks != [1, 2, 3]:
            raise RuntimeError(f"screen selection ranks invalid for {lane}: {ranks}")
    screen_summary_path = Path(selection.get("screen_summary_path", ""))
    if (
        not screen_summary_path.is_file()
        or selection.get("screen_summary_sha256") != file_sha256(screen_summary_path)
    ):
        raise RuntimeError("screen selection does not bind its analysis summary")
    screen_summary = read_json(screen_summary_path)
    if (
        screen_summary.get("campaign_id") != lock["campaign_id"]
        or screen_summary.get("phase") != "screen"
        or screen_summary.get("record_bundle_sha256")
        != selection.get("screen_bundle_sha256")
    ):
        raise RuntimeError("screen summary/selection binding mismatch")
    recomputed = []
    for lane in ("cuda_noptx", "cuda_unlimited"):
        eligible = sorted(
            (
                row for row in screen_summary.get("cells", [])
                if row.get("lane") == lane and row.get("eligible") is True
            ),
            key=lambda row: (row["median_of_process_medians_ms"], row["job_id"]),
        )[:3]
        recomputed.extend(
            (lane, rank, row["job_id"], row["median_of_process_medians_ms"])
            for rank, row in enumerate(eligible, 1)
        )
    observed = sorted(
        (row["lane"], row["screen_rank"], row["job_id"], row["screen_median_ms"])
        for row in selected_rows
    )
    if sorted(recomputed) != observed:
        raise RuntimeError("screen selection does not reproduce the frozen top-three rule")
    selected_ids = {
        row["job_id"] for row in selected_rows if row["lane"] == args.lane
    }
    jobs = [job for job in all_jobs if job["job_id"] in selected_ids]
    if len(jobs) != 3:
        raise RuntimeError(f"expected frozen top three for {args.lane}, got {len(jobs)}")
    for job in jobs:
        selected = next(row for row in selection["selected"] if row["job_id"] == job["job_id"])
        if selected["job_sha256"] != lock["job_sha256"][job["job_id"]]:
            raise RuntimeError(f"selection job hash mismatch: {job['job_id']}")

    root = safe_result_root(args.tag)
    active_lock = acquire_active_lock(root)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(HERE / ".torch_ext" / f"gpu{args.gpu}")
    os.environ.setdefault("MAX_JOBS", "4")
    context = make_context(lock, all_jobs)
    plans = [make_plan(context, job) for job in jobs]
    cases = context.cases
    indices = tuple(range(64))
    receipt = {
        "schema_version": 1,
        "record_type": "fused_reachability_v2_robust_receipt",
        "campaign_id": lock["campaign_id"],
        "lane": args.lane,
        "physical_gpu": args.gpu,
        "logical_device": "cuda:0",
        "selection_path": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "selected_jobs": [job["job_id"] for job in jobs],
        "case_ids": list(cases),
        "seed_indices": list(indices),
        "launch_lock_sha256": file_sha256(LOCK),
        "source_bundle_sha256": context.source_bundle_sha256,
        "gate_spec_sha256": context.adapter["robust_gate"]["gate_spec_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "gpu_before": gpu_before,
        "nvcc": nvcc_fingerprint(),
    }
    receipt_path = root / "launch_receipt.json"
    if receipt_path.exists():
        old = read_json(receipt_path)
        stable_keys = (
            "schema_version", "record_type", "campaign_id", "lane",
            "physical_gpu", "logical_device", "selection_path",
            "selection_sha256", "selected_jobs", "case_ids", "seed_indices",
            "launch_lock_sha256", "source_bundle_sha256", "gate_spec_sha256",
        )
        if any(old.get(key) != receipt.get(key) for key in stable_keys):
            raise RuntimeError("existing robust tag has a different contract")
    else:
        stable_write(receipt_path, receipt)

    records = []
    live_inputs = None
    total = len(cases) * len(indices)
    position = 0
    for case_id in cases:
        for seed_index in indices:
            position += 1
            paths = {
                plan.candidate: bundle_path(root, plan.job_id, case_id, seed_index)
                for plan in plans
            }
            if all(path.exists() for path in paths.values()):
                for plan in plans:
                    records.extend(
                        validate_existing(
                            paths[plan.candidate], plan, context, case_id, seed_index
                        )
                    )
                print(f"[resume {position}/{total}] {case_id} seed={seed_index}", flush=True)
                continue
            if any(path.exists() for path in paths.values()):
                raise RuntimeError(
                    f"partial seed bundle for {case_id}/{seed_index}; inspect before retry"
                )
            # Keep the prior input mapping live until the new seed has been
            # allocated. common2's cached weight is keyed by data_ptr; deleting
            # early lets the CUDA allocator reuse that pointer and would serve
            # the previous seed's fp16 weight for the new case.
            prior_inputs = live_inputs
            evaluated, next_inputs = adapter.evaluate_case_seed(
                context,
                plans,
                case_id=case_id,
                split="validation",
                seed_index=seed_index,
                device="cuda:0",
            )
            live_inputs = next_inputs
            if prior_inputs is not None:
                del prior_inputs
            for plan in plans:
                rows = evaluated[plan.candidate]
                path = paths[plan.candidate]
                stable_write(
                    path,
                    {
                        "schema_version": 1,
                        "record_type": "fused_reachability_v2_robust_seed_bundle",
                        "records": rows,
                    },
                )
                records.extend(rows)
            print(f"[run {position}/{total}] {case_id} seed={seed_index}", flush=True)
    if live_inputs is not None:
        del live_inputs

    write_jsonl(root / "records.jsonl", records)
    candidates = [plan.candidate for plan in plans]
    summary = adapter.summarize_validation(
        context,
        records,
        candidates=candidates,
        case_ids=cases,
        indices=indices,
    )
    summary.update(
        {
            "prospective_campaign_id": lock["campaign_id"],
            "lane": args.lane,
            "selection_sha256": file_sha256(selection_path),
            "selected_jobs": [job["job_id"] for job in jobs],
            "complete_frozen_validation_split": True,
        }
    )
    stable_write(root / "summary.json", summary)
    stable_write(root / "run_status.json", {
        "campaign_id": lock["campaign_id"],
        "lane": args.lane,
        "status": summary["status"],
        "record_count": len(records),
        "gpu_after": gpu_snapshot(args.gpu),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"robust status={summary['status']} records={len(records)}")
    active_lock.close()
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
