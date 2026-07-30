#!/usr/bin/env python3
"""Validate, list, dry-run, or resume the controlled fused-GBGS grid.

Execution delegates each fresh-process measurement to Phase 2's
``driver2.run_job``/``runner2.py`` interface.  This wrapper adds an isolated
result directory, content-addressed provenance, deterministic plan ordering,
and safe resume checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PHASE1 = REPO_ROOT / "ako_runs/phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs/phase2_fused_sdpa"
DEFAULT_MANIFEST = HERE / "manifest.json"
RESULTS_ROOT = HERE / "results"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PHASE2))
import make_manifest  # noqa: E402
import driver2  # noqa: E402


SOURCE_PATHS = (
    HERE / "make_manifest.py",
    HERE / "launch.py",
    PHASE1 / "common.py",
    PHASE1 / "jobs/native_tuned.json",
    PHASE2 / "common2.py",
    PHASE2 / "driver2.py",
    PHASE2 / "runner2.py",
    PHASE2 / "variants2/SPEC2.md",
    PHASE2 / "variants2/fused_tilelang.py",
    PHASE2 / "variants2/fused_triton.py",
    PHASE2 / "variants2/fused_cuda_noptx.py",
    PHASE2 / "variants2/fused_cuda_unlimited.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(stable_json_bytes(value))
    os.replace(tmp, path)


def git_value(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
    )
    return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"


def load_and_validate(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # The default files must match the deterministic generator byte-for-byte.
    if manifest_path.resolve() == DEFAULT_MANIFEST.resolve():
        make_manifest.check_files()

    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    jobs_path = manifest_path.parent / manifest["jobs_file"]
    jobs_raw = jobs_path.read_bytes()
    jobs = json.loads(jobs_raw)

    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    if sha256_bytes(jobs_raw) != manifest.get("jobs_sha256"):
        raise ValueError("jobs file hash does not match manifest")
    if sha256_file(REPO_ROOT / manifest["phase1_grid_source"]) != manifest.get(
        "phase1_grid_source_sha256"
    ):
        raise ValueError("Phase-1 source grid changed after manifest generation")
    if len(jobs) != manifest.get("job_count"):
        raise ValueError("job count does not match manifest")

    expected_dsls = tuple(manifest["dsl_order"])
    seen_ids: set[str] = set()
    points_by_dsl: dict[str, list[str]] = {dsl: [] for dsl in expected_dsls}
    for job in jobs:
        required = {"dsl", "geom", "grid_id", "grid_index", "job_id", "set", "variant"}
        if not required.issubset(job):
            raise ValueError(f"job lacks required fields: {job!r}")
        if job["job_id"] in seen_ids:
            raise ValueError(f"duplicate job_id {job['job_id']}")
        seen_ids.add(job["job_id"])
        if job["dsl"] not in points_by_dsl:
            raise ValueError(f"unexpected DSL {job['dsl']}")
        if job["variant"] != "GBGS" or job["geom"] != "fused":
            raise ValueError(f"job left the fused GBGS contract: {job!r}")
        points_by_dsl[job["dsl"]].append(job["set"])

    reference = points_by_dsl[expected_dsls[0]]
    if len(reference) != manifest["grid_point_count"]:
        raise ValueError("wrong number of grid points in first DSL")
    for dsl in expected_dsls[1:]:
        if points_by_dsl[dsl] != reference:
            raise ValueError(f"job grid/order differs for {dsl}")
    return manifest, jobs


def runner_command(job: dict[str, Any], rep: int, args: argparse.Namespace, out: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(PHASE2 / "runner2.py"),
        "--op",
        "fused",
        "--dsl",
        job["dsl"],
        "--variant",
        job["variant"],
        "--dist",
        args.dist,
        "--seed",
        str(args.seed),
        "--rep",
        str(rep),
        "--trials",
        str(args.trials),
        "--warmup-s",
        str(args.warmup_s),
        "--out",
        str(out),
        "--set",
        job["set"],
    ]
    if args.time_only:
        cmd.append("--time-only")
    return cmd


def output_path(job: dict[str, Any], rep: int, args: argparse.Namespace, rawdir: Path) -> Path:
    return rawdir / f"{driver2.job_name(job, args, rep)}.json"


def plan_jobs(jobs: list[dict[str, Any]], reps: int, order_seed: int):
    plan = [(job, rep) for job in jobs for rep in range(reps)]
    random.Random(order_seed).shuffle(plan)
    return plan


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in SOURCE_PATHS
    }


def runtime_provenance(
    manifest_path: Path, manifest: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    protocol = {
        "dist": args.dist,
        "seed": args.seed,
        "time_only": args.time_only,
        "trials": args.trials,
        "warmup_s": args.warmup_s,
    }
    sources = source_hashes()
    return {
        "campaign_id": manifest["campaign_id"],
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
        "host": platform.node(),
        "jobs_sha256": manifest["jobs_sha256"],
        "launch_args": {
            **protocol,
            "order_seed": args.order_seed,
            "reps": args.reps,
        },
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "phase1_grid_source_sha256": manifest["phase1_grid_source_sha256"],
        "protocol_sha256": sha256_bytes(stable_json_bytes(protocol)),
        "python": sys.version,
        "python_executable": sys.executable,
        "source_bundle_sha256": sha256_bytes(stable_json_bytes(sources)),
        "source_sha256": sources,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }


def record_state(
    path: Path, provenance: dict[str, Any], job: dict[str, Any], rep: int
) -> tuple[str, dict[str, Any] | None]:
    if not path.exists():
        return "pending", None
    try:
        rec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "invalid", None
    got = rec.get("campaign_provenance", {})
    expected = (
        provenance["campaign_id"],
        provenance["manifest_sha256"],
        provenance["jobs_sha256"],
        provenance["protocol_sha256"],
        provenance["source_bundle_sha256"],
        job["job_id"],
        rep,
    )
    actual = (
        got.get("campaign_id"),
        got.get("manifest_sha256"),
        got.get("jobs_sha256"),
        got.get("protocol_sha256"),
        got.get("source_bundle_sha256"),
        got.get("job_id"),
        got.get("rep"),
    )
    if actual != expected:
        return "foreign", rec
    return ("complete_ok" if rec.get("ok") else "complete_failed"), rec


def record_provenance(
    rec: dict[str, Any], provenance: dict[str, Any], job: dict[str, Any], rep: int
) -> dict[str, Any]:
    rec["campaign_provenance"] = {
        "campaign_id": provenance["campaign_id"],
        "git_commit": provenance["git_commit"],
        "grid_id": job["grid_id"],
        "grid_index": job["grid_index"],
        "job_id": job["job_id"],
        "jobs_sha256": provenance["jobs_sha256"],
        "manifest_sha256": provenance["manifest_sha256"],
        "phase1_grid_source_sha256": provenance["phase1_grid_source_sha256"],
        "protocol_sha256": provenance["protocol_sha256"],
        "rep": rep,
        "source_bundle_sha256": provenance["source_bundle_sha256"],
        "source_sha256": provenance["source_sha256"],
    }
    return rec


def print_summary(counts: dict[str, int], total: int) -> None:
    detail = ", ".join(f"{key}={counts.get(key, 0)}" for key in (
        "pending", "complete_ok", "complete_failed", "foreign", "invalid"
    ))
    print(f"plan: {total} process records ({detail})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--dist", choices=("rand", "randn"), default="rand")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--order-seed", type=int, default=20260729)
    parser.add_argument("--tag", default="fused_gbgs_grid_rank")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--time-only", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace every planned record")
    parser.add_argument(
        "--retry-failed", action="store_true", help="replace matching failed records"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list", action="store_true", help="list deterministic plan/status")
    parser.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="operate on only the first N entries after deterministic shuffle",
    )
    args = parser.parse_args()

    if args.reps < 1 or args.trials < 1 or args.warmup_s < 0 or args.limit < 0:
        parser.error("reps/trials must be positive; warmup-s/limit must be non-negative")
    manifest_path = args.manifest.resolve()
    manifest, jobs = load_and_validate(manifest_path)
    provenance = runtime_provenance(manifest_path, manifest, args)
    plan = plan_jobs(jobs, args.reps, args.order_seed)
    if args.limit:
        plan = plan[: args.limit]

    outdir = RESULTS_ROOT / args.tag
    rawdir = outdir / "raw"
    entries = []
    counts: dict[str, int] = {}
    for job, rep in plan:
        out = output_path(job, rep, args, rawdir)
        state, rec = record_state(out, provenance, job, rep)
        counts[state] = counts.get(state, 0) + 1
        entries.append((job, rep, out, state, rec))

    print(
        f"validated {manifest['campaign_id']}: {manifest['grid_point_count']} points "
        f"x {len(manifest['dsl_order'])} DSLs = {manifest['job_count']} jobs; "
        f"order_seed={args.order_seed}"
    )
    print_summary(counts, len(entries))
    if args.validate_only and not (args.list or args.dry_run):
        return 0

    if args.list or args.dry_run:
        for index, (job, rep, out, state, _rec) in enumerate(entries, 1):
            print(
                f"[{index:03d}] {state:<15s} {job['job_id']} rep{rep} -> "
                f"{out.relative_to(HERE)}"
            )
            if args.dry_run and (
                args.force
                or state == "pending"
                or (args.retry_failed and state == "complete_failed")
            ):
                print("      " + shlex.join(runner_command(job, rep, args, out)))
        return 0

    unsafe = counts.get("foreign", 0) + counts.get("invalid", 0)
    if unsafe and not args.force:
        print(
            f"ABORT: {unsafe} output record(s) lack matching provenance; "
            "use a new --tag or inspect them before --force",
            file=sys.stderr,
        )
        return 4

    runnable = []
    for entry in entries:
        _job, _rep, _out, state, _rec = entry
        if args.force or state == "pending" or (
            args.retry_failed and state == "complete_failed"
        ):
            runnable.append(entry)
    if not runnable:
        print("nothing to run; all matching records are already complete")
        return 0

    driver2.preflight(args.gpu, strict=not args.allow_busy)
    rawdir.mkdir(parents=True, exist_ok=True)
    session_path = outdir / (
        "launch_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{os.getpid()}.json"
    )
    atomic_json(session_path, provenance)
    print(f"running {len(runnable)} records; session provenance -> {session_path}")

    nok = nfail = 0
    for index, (job, rep, out, state, _old) in enumerate(runnable, 1):
        run_args = SimpleNamespace(
            op="fused",
            dist=args.dist,
            seed=args.seed,
            trials=args.trials,
            warmup_s=args.warmup_s,
            timeout=args.timeout,
            force=(args.force or state != "pending"),
            time_only=args.time_only,
        )
        print(
            f"[{index:03d}/{len(runnable)}] {job['job_id']} rep{rep} "
            f"grid={job['set']}",
            flush=True,
        )
        rec = driver2.run_job(job, rep, args.gpu, run_args, str(rawdir))
        rec = record_provenance(rec, provenance, job, rep)
        atomic_json(out, rec)
        if rec.get("ok"):
            nok += 1
            median = rec.get("timing", {}).get("median_ms")
            print(f"      ok median={median:.4f} ms" if median else "      ok")
        else:
            nfail += 1
            print(f"      FAILED: {str(rec.get('error_msg', '?'))[:200]}")
        atomic_json(
            outdir / "status.json",
            {
                "campaign_id": manifest["campaign_id"],
                "complete_failed_this_session": nfail,
                "complete_ok_this_session": nok,
                "jobs_sha256": manifest["jobs_sha256"],
                "last_job_id": job["job_id"],
                "last_rep": rep,
                "manifest_sha256": provenance["manifest_sha256"],
                "planned_this_session": len(runnable),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    print(f"done: {nok} ok, {nfail} failed -> {rawdir}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
