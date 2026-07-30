#!/usr/bin/env python3
"""Run selected fused-grid GBGS jobs through both frozen mixed gates.

Examples (none are launched by this file itself):

  # CPU-only contract smoke
  python robust_launch.py --cpu-test

  # One tuning seed across all 76 grid jobs
  python robust_launch.py --all-jobs --split tuning --max-seeds 1 \
      --gpu 0 --tag tuning_smoke_all76

  # Complete tuning split for an explicitly screened subset
  python robust_launch.py --job tilelang.g01 --job triton.g01 \
      --split tuning --gpu 0 --tag tuning_locked_subset
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RESULTS_ROOT = HERE / "results/robust"
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--job",
        action="append",
        default=[],
        help="existing grid job ID; repeat or use comma-separated IDs",
    )
    selection.add_argument("--all-jobs", action="store_true")
    parser.add_argument("--split", choices=("tuning", "validation"), default="tuning")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--cases", default="", help="comma-separated case IDs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--adapter-manifest", default=str(HERE / "robust_adapter_manifest.json"))
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="rerun retained build/runtime/threshold failures",
    )
    return parser.parse_args()


def comma_values(values: list[str]) -> list[str]:
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def git_value(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"


def preflight(gpu: int, *, allow_busy: bool) -> None:
    problems = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
                "-i",
                str(gpu),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            problems.append(proc.stderr.strip() or "nvidia-smi query failed")
        occupants = [line for line in proc.stdout.splitlines() if line.strip()]
        if occupants:
            problems.append(f"GPU {gpu} already has compute processes: {occupants}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not query GPU {gpu}: {type(exc).__name__}: {exc}")
    if problems:
        print("[preflight] " + "\n[preflight] ".join(problems), flush=True)
        if not allow_busy:
            raise RuntimeError("GPU preflight failed; use --allow-busy only deliberately")
    else:
        print(f"[preflight] GPU {gpu} has no reported compute process", flush=True)


def run_contract(context, jobs, split, cases, indices, gpu) -> dict[str, Any]:
    return {
        "adapter_campaign_id": context.adapter["campaign_id"],
        "adapter_manifest_path": str(context.adapter_path.relative_to(REPO_ROOT)),
        "adapter_manifest_sha256": context.adapter_sha256,
        "source_bundle_sha256": context.source_bundle_sha256,
        "robust_manifest_sha256": context.manifest_sha256,
        "gate_spec_sha256": context.adapter["robust_gate"]["gate_spec_sha256"],
        "grid_manifest_sha256": context.adapter["grid"]["manifest_sha256"],
        "grid_jobs_sha256": context.adapter["grid"]["jobs_sha256"],
        "selected_jobs": [
            {
                "job_id": job["job_id"],
                "job_sha256": context.adapter["grid"]["job_sha256"][job["job_id"]],
            }
            for job in jobs
        ],
        "split": split,
        "case_ids": list(cases),
        "seed_indices": list(indices),
        "physical_gpu": gpu,
        "logical_device": "cuda:0",
    }


def initialize_run(path: Path, contract: dict[str, Any], context, *, force: bool) -> None:
    from robust_gate.schema import write_json

    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("run_contract") == contract and not force:
            print(f"[resume] compatible run contract: {path}", flush=True)
            return
        if not force:
            raise RuntimeError(
                f"{path} belongs to a different run; choose a new --tag or pass --force"
            )
    runtime = {
        "schema_version": 1,
        "record_type": "fused_grid_robust_run",
        "run_contract": contract,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
        "host": platform.node(),
        "python": sys.version,
        "python_executable": sys.executable,
        "source_sha256": context.adapter["source_sha256"],
    }
    write_json(path, runtime)


def bundle_path(root: Path, job_id: str, split: str, case_id: str, index: int) -> Path:
    return root / "raw" / job_id / split / case_id / f"seed{index:03d}.json"


def bundle_binding(context, job, split, case_id, index) -> dict[str, Any]:
    from robust_gate.seeds import tensor_seeds

    job_hash = context.adapter["grid"]["job_sha256"][job["job_id"]]
    return {
        "adapter_manifest_sha256": context.adapter_sha256,
        "source_bundle_sha256": context.source_bundle_sha256,
        "robust_manifest_sha256": context.manifest_sha256,
        "gate_spec_sha256": context.adapter["robust_gate"]["gate_spec_sha256"],
        "grid_manifest_sha256": context.adapter["grid"]["manifest_sha256"],
        "grid_jobs_sha256": context.adapter["grid"]["jobs_sha256"],
        "grid_job_id": job["job_id"],
        "grid_job_sha256": job_hash,
        "candidate": f"fused-grid:{job['job_id']}:{job_hash[:12]}",
        "split": split,
        "case_id": case_id,
        "seed_index": index,
        "tensor_seeds": tensor_seeds(
            context.robust_manifest, "fused_softmax", case_id, split, index
        ),
        "gate_ids": ["semantic_mixed", "conformance_mixed"],
    }


def read_bundle_state(path: Path, binding: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    if not path.exists():
        return "pending", None
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid", None
    if bundle.get("binding") != binding:
        return "foreign", bundle
    records = bundle.get("records")
    if not isinstance(records, list) or len(records) != 2:
        return "invalid", bundle
    if [row.get("gate_id") for row in records] != binding["gate_ids"]:
        return "invalid", bundle
    for row in records:
        expected = {
            "candidate": binding["candidate"],
            "grid_job_id": binding["grid_job_id"],
            "grid_job_sha256": binding["grid_job_sha256"],
            "split": binding["split"],
            "case_id": binding["case_id"],
            "seed_index": binding["seed_index"],
            "tensor_seeds": binding["tensor_seeds"],
            "adapter_manifest_sha256": binding["adapter_manifest_sha256"],
            "source_bundle_sha256": binding["source_bundle_sha256"],
            "manifest_sha256": binding["robust_manifest_sha256"],
            "gate_spec_sha256": binding["gate_spec_sha256"],
            "grid_manifest_sha256": binding["grid_manifest_sha256"],
        }
        if any(row.get(key) != value for key, value in expected.items()):
            return "invalid", bundle
    failed = any(
        not row.get("ok", False) or not row.get("gate_pass", False)
        for row in records
    )
    return ("failed" if failed else "complete"), bundle


def write_bundle(path: Path, binding: dict[str, Any], records: list[dict[str, Any]]) -> None:
    from robust_gate.schema import write_json

    write_json(
        path,
        {
            "schema_version": 1,
            "record_type": "fused_grid_robust_seed_bundle",
            "binding": binding,
            "records": records,
        },
    )


def inspect_tasks(context, jobs, split, cases, indices, result_root, args):
    tasks = {}
    counts = {"pending": 0, "complete": 0, "failed": 0, "invalid": 0, "foreign": 0}
    for job in jobs:
        for case_id in cases:
            for index in indices:
                path = bundle_path(result_root, job["job_id"], split, case_id, index)
                binding = bundle_binding(context, job, split, case_id, index)
                state, _bundle = read_bundle_state(path, binding)
                if args.force or (state == "failed" and args.retry_failures):
                    state = "pending"
                if state in ("invalid", "foreign") and not args.force:
                    raise RuntimeError(
                        f"{state} bundle at {path}; use a new --tag or inspect before --force"
                    )
                if state in ("invalid", "foreign"):
                    state = "pending"
                tasks[(job["job_id"], case_id, index)] = (state, path, binding)
                counts[state] += 1
    return tasks, counts


def aggregate(context, jobs, split, cases, indices, result_root, tasks) -> tuple[list, dict]:
    from robust_gate.schema import write_json, write_jsonl
    import robust_adapter as adapter

    records = []
    for job in jobs:
        for case_id in cases:
            for index in indices:
                state, path, binding = tasks[(job["job_id"], case_id, index)]
                observed, bundle = read_bundle_state(path, binding)
                if observed not in ("complete", "failed") or bundle is None:
                    raise RuntimeError(f"campaign ended without a valid bundle: {path}")
                records.extend(bundle["records"])
    write_jsonl(result_root / "records.jsonl", records)
    candidates = [adapter.candidate_name(context, job) for job in jobs]
    if split == "tuning":
        summary = adapter.summarize_tuning(
            context,
            records,
            candidates=candidates,
            case_ids=cases,
            indices=indices,
        )
    else:
        summary = adapter.summarize_validation(
            context,
            records,
            candidates=candidates,
            case_ids=cases,
            indices=indices,
        )
    write_json(result_root / "summary.json", summary)
    return records, summary


def main() -> int:
    args = parse_args()
    # CUDA visibility must be fixed before robust_adapter imports torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    if str(HERE.parent) not in sys.path:
        sys.path.insert(0, str(HERE.parent))

    import robust_adapter as adapter

    context = adapter.load_repository(args.adapter_manifest)
    if args.list_jobs:
        for job in context.jobs:
            print(f"{job['job_id']:<22} {job['dsl']:<16} {job['set']}")
        return 0

    if args.cpu_test:
        records = adapter.cpu_smoke_records(context)
        collection_failures = sum(not row.get("ok", False) for row in records)
        threshold_failures = sum(not row.get("gate_pass", False) for row in records)
        print(
            f"CPU smoke: records={len(records)} collection_failures={collection_failures} "
            f"threshold_failures={threshold_failures}"
        )
        for row in records:
            print(
                f"  {row['case_id']:<24} {row['gate_id']:<18} "
                f"{'PASS' if row.get('gate_pass') else 'FAIL'}"
            )
        return 0 if collection_failures == 0 else 2

    requested = comma_values(args.job)
    jobs = adapter.select_jobs(context, requested, all_jobs=args.all_jobs)
    cases = tuple(
        value.strip() for value in args.cases.split(",") if value.strip()
    ) or context.cases
    if len(cases) != len(set(cases)):
        raise ValueError("case selection contains duplicates")
    unknown_cases = sorted(set(cases) - set(context.cases))
    if unknown_cases:
        raise ValueError(f"unknown case(s): {', '.join(unknown_cases)}")
    indices = adapter.seed_indices(context, args.split, args.seed_start, args.max_seeds)

    candidate_runs = len(jobs) * len(cases) * len(indices)
    records_expected = candidate_runs * 2
    weight_gib = len(jobs) * 8192 * 8192 * 2 / 2**30
    print(
        f"plan: jobs={len(jobs)} cases={len(cases)} seeds={len(indices)} "
        f"candidate_runs={candidate_runs} gate_records={records_expected} split={args.split}"
    )
    print(
        f"reference reuse: 2 references per case/seed; selected cached-weight ceiling "
        f"approximately {weight_gib:.2f} GiB"
    )
    if args.split == "validation" and (
        set(cases) != set(context.cases) or set(indices) != set(range(64))
    ):
        print("validation selection is incomplete and will fail closed (never PASS)")
    if args.dry_run:
        for job in jobs:
            print(f"  {job['job_id']}: {job['set']}")
        return 0
    if not args.tag or not TAG_RE.fullmatch(args.tag):
        raise ValueError("execution requires --tag matching [A-Za-z0-9][A-Za-z0-9_.-]*")

    preflight(args.gpu, allow_busy=args.allow_busy)
    result_root = RESULTS_ROOT / args.tag
    contract = run_contract(context, jobs, args.split, cases, indices, args.gpu)
    initialize_run(result_root / "run.json", contract, context, force=args.force)
    tasks, counts = inspect_tasks(
        context, jobs, args.split, cases, indices, result_root, args
    )
    print(f"resume states: {counts}", flush=True)
    pending_job_ids = {
        job_id for (job_id, _case, _index), (state, _path, _binding) in tasks.items()
        if state == "pending"
    }
    if pending_job_ids:
        preflight_jobs = [job for job in jobs if job["job_id"] in pending_job_ids]

        def progress(plan, done, total):
            state = "FAIL" if plan.build_error else "OK"
            print(f"[build {done}/{total}] {plan.job_id}: {state}", flush=True)

        plans = adapter.build_phase2_candidates(context, preflight_jobs, progress)
        plans_by_job = {plan.job_id: plan for plan in plans}
        # Phase-2's cached-weight path keys on W.data_ptr().  Keep the source W
        # that each candidate last consumed alive until that same candidate
        # successfully consumes a replacement.  A single global previous-W
        # guard is insufficient on resume: sparse pending candidates can skip
        # seeds, allowing an older per-candidate address to be recycled.
        retained_weights: dict[str, list[Any]] = {}
        for case_id in cases:
            for index in indices:
                active_jobs = [
                    job
                    for job in jobs
                    if tasks[(job["job_id"], case_id, index)][0] == "pending"
                ]
                if not active_jobs:
                    continue
                active_plans = [plans_by_job[job["job_id"]] for job in active_jobs]
                rows_by_candidate, next_inputs = adapter.evaluate_case_seed(
                    context,
                    active_plans,
                    case_id=case_id,
                    split=args.split,
                    seed_index=index,
                    device="cuda:0",
                )
                if next_inputs is not None:
                    retained_ptrs = {
                        weight.data_ptr()
                        for weights in retained_weights.values()
                        for weight in weights
                    }
                    if next_inputs["weight"].data_ptr() in retained_ptrs:
                        raise RuntimeError(
                            "weight pointer reused while a candidate's prior source was live"
                        )
                for job, plan in zip(active_jobs, active_plans):
                    state, path, binding = tasks[(job["job_id"], case_id, index)]
                    assert state == "pending"
                    rows = rows_by_candidate[plan.candidate]
                    if next_inputs is not None and not plan.build_error:
                        candidate_error = any(
                            str(row.get("error", "")).startswith("CandidateError:")
                            for row in rows
                        )
                        if candidate_error:
                            # The exception may have occurred immediately before
                            # or after wf(W) refreshed its private cache.  Retain
                            # both possible source tensors and fail closed.
                            retained_weights.setdefault(job["job_id"], []).append(
                                next_inputs["weight"]
                            )
                        else:
                            retained_weights[job["job_id"]] = [next_inputs["weight"]]
                    write_bundle(path, binding, rows)
                    status = "PASS" if all(row.get("gate_pass") for row in rows) else "FAIL"
                    print(
                        f"[{case_id} seed={index:03d}] {job['job_id']}: {status}",
                        flush=True,
                    )

    # Re-inspect so aggregation never trusts in-memory success alone.
    tasks, _counts = inspect_tasks(
        context, jobs, args.split, cases, indices, result_root, args=argparse.Namespace(
            force=False, retry_failures=False
        )
    )
    records, summary = aggregate(
        context, jobs, args.split, cases, indices, result_root, tasks
    )
    print(
        f"wrote {len(records)} rows to {result_root / 'records.jsonl'}; "
        f"summary={summary.get('status')} selected_success="
        f"{summary.get('selected_success', summary.get('success'))}"
    )
    if args.split == "tuning":
        return 0 if summary["selected_success"] else 2
    return 0 if summary["success"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; completed atomic seed bundles are resumable", file=sys.stderr)
        raise SystemExit(130)
