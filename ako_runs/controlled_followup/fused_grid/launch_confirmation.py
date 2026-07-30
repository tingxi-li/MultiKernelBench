#!/usr/bin/env python3
"""Validate and run the robust-selected fused-grid confirmation campaign.

Every (candidate, repetition) pair is executed by a new ``runner2.py``
process.  The confirmation selection is reconstructed from the screening raw
records and robust-gate summary before any GPU work starts; a hand-edited,
stale, or incompletely bound selection therefore fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIRMATION = HERE / "jobs/confirm_robust.json"
DEFAULT_SCREENING_MANIFEST = HERE / "manifest.json"
DEFAULT_ADAPTER_MANIFEST = HERE / "robust_adapter_manifest.json"
RESULTS_ROOT = HERE / "results"
EXPECTED_REPS = 5
DEFAULT_ORDER_SEED = 20260730
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
HEX64 = set("0123456789abcdef")

sys.path.insert(0, str(HERE))
import analyze_screen  # noqa: E402
import launch as screen_launch  # noqa: E402


class ConfirmationError(ValueError):
    """The confirmation selection or one of its bindings is invalid."""


@dataclass(frozen=True)
class BoundConfirmation:
    document: dict[str, Any]
    raw_sha256: str
    analysis: analyze_screen.Analysis


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ConfirmationError(f"cannot hash {path}: {exc}") from exc


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ConfirmationError(
            f"{label} mismatch: got {actual!r}, expected {expected!r}"
        )


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= HEX64:
        raise ConfirmationError(f"{label} must be a lowercase SHA256")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfirmationError(f"{label} must be an object")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfirmationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ConfirmationError(f"{label} must be finite and positive")
    return result


def validate_confirmation_document(document: Any) -> dict[str, Any]:
    """Validate the standalone confirmation schema before artifact binding."""
    doc = _object(document, "confirmation")
    _same(doc.get("schema_version"), 1, "confirmation schema_version")
    _same(
        doc.get("campaign_id"),
        "fused-gbgs-confirmation-v1",
        "confirmation campaign_id",
    )
    _same(doc.get("top_k"), analyze_screen.TOP_K, "confirmation top_k")
    _same(
        doc.get("old_incumbent_grid_id"),
        analyze_screen.OLD_INCUMBENT_GRID_ID,
        "confirmation old_incumbent_grid_id",
    )

    provenance = _object(doc.get("provenance"), "confirmation provenance")
    required_provenance = {
        "screening_campaign_id",
        "screening_manifest_sha256",
        "screening_jobs_sha256",
        "screening_launch_receipt_sha256",
        "screening_protocol_sha256",
        "screening_records_sha256",
        "robust_summary_sha256",
        "robust_adapter_manifest_sha256",
    }
    missing = required_provenance - set(provenance)
    if missing:
        raise ConfirmationError(
            f"confirmation provenance lacks {sorted(missing)}"
        )
    for name in required_provenance - {"screening_campaign_id"}:
        _hex64(provenance[name], f"confirmation provenance {name}")
    if not isinstance(provenance["screening_campaign_id"], str) or not provenance[
        "screening_campaign_id"
    ]:
        raise ConfirmationError("screening_campaign_id must be non-empty")

    jobs = doc.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ConfirmationError("confirmation jobs must be a non-empty list")
    _same(
        _hex64(doc.get("jobs_sha256"), "confirmation jobs_sha256"),
        analyze_screen.canonical_sha256(jobs),
        "confirmation canonical jobs hash",
    )

    required_job = {
        "confirmation_id",
        "dsl",
        "geom",
        "grid_id",
        "grid_index",
        "screening_job_id",
        "selection_roles",
        "set",
        "variant",
        "screening_median_ms",
        "screening_robust_eligible",
    }
    confirmation_ids: set[str] = set()
    screening_ids: set[str] = set()
    lanes: dict[str, list[dict[str, Any]]] = {}
    for index, value in enumerate(jobs):
        job = _object(value, f"confirmation job {index}")
        missing = required_job - set(job)
        if missing:
            raise ConfirmationError(
                f"confirmation job {index} lacks {sorted(missing)}"
            )
        confirmation_id = job["confirmation_id"]
        if not isinstance(confirmation_id, str) or not ID_RE.fullmatch(confirmation_id):
            raise ConfirmationError(
                f"confirmation job {index} has unsafe confirmation_id"
            )
        if confirmation_id in confirmation_ids:
            raise ConfirmationError(f"duplicate confirmation_id {confirmation_id!r}")
        confirmation_ids.add(confirmation_id)

        dsl = job["dsl"]
        grid_id = job["grid_id"]
        grid_index = job["grid_index"]
        if not isinstance(dsl, str) or not dsl:
            raise ConfirmationError(f"confirmation job {index} DSL is invalid")
        if not isinstance(grid_index, int) or isinstance(grid_index, bool) or grid_index < 0:
            raise ConfirmationError(f"confirmation job {index} grid_index is invalid")
        _same(grid_id, f"g{grid_index:02d}", f"confirmation job {index} grid_id")
        _same(
            job["screening_job_id"],
            f"{dsl}.{grid_id}",
            f"confirmation job {index} screening_job_id",
        )
        if job["screening_job_id"] in screening_ids:
            raise ConfirmationError(
                f"duplicate screening_job_id {job['screening_job_id']!r}"
            )
        screening_ids.add(job["screening_job_id"])
        _same(job["geom"], "fused", f"confirmation job {index} geom")
        _same(job["variant"], "GBGS", f"confirmation job {index} variant")
        if not isinstance(job["set"], str) or not job["set"]:
            raise ConfirmationError(f"confirmation job {index} set is invalid")
        analyze_screen.parse_set(job["set"])
        _finite_positive(
            job["screening_median_ms"],
            f"confirmation job {index} screening_median_ms",
        )
        # Confirmation timings are only meaningful for candidates that passed
        # both frozen robust gates.  This also makes an invalid old incumbent a
        # visible launch blocker rather than a silently quoted baseline.
        _same(
            job["screening_robust_eligible"],
            True,
            f"confirmation job {index} screening_robust_eligible",
        )
        roles = job["selection_roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) or not role for role in roles)
            or len(roles) != len(set(roles))
        ):
            raise ConfirmationError(
                f"confirmation job {index} selection_roles are invalid"
            )
        allowed_roles = {"old_incumbent"} | {
            f"screening_rank_{rank}" for rank in range(1, analyze_screen.TOP_K + 1)
        }
        unknown_roles = set(roles) - allowed_roles
        if unknown_roles:
            raise ConfirmationError(
                f"confirmation job {index} has unknown roles {sorted(unknown_roles)}"
            )
        lanes.setdefault(dsl, []).append(job)

    for dsl, lane in lanes.items():
        rank_roles = [
            role
            for job in lane
            for role in job["selection_roles"]
            if role.startswith("screening_rank_")
        ]
        expected_ranks = [
            f"screening_rank_{rank}" for rank in range(1, analyze_screen.TOP_K + 1)
        ]
        _same(rank_roles, expected_ranks, f"{dsl} screening rank roles/order")
        incumbents = [
            job for job in lane if "old_incumbent" in job["selection_roles"]
        ]
        if len(incumbents) != 1:
            raise ConfirmationError(f"{dsl} must contain exactly one old incumbent")
        _same(
            incumbents[0]["grid_id"],
            analyze_screen.OLD_INCUMBENT_GRID_ID,
            f"{dsl} old incumbent grid_id",
        )
        if len(lane) not in (analyze_screen.TOP_K, analyze_screen.TOP_K + 1):
            raise ConfirmationError(
                f"{dsl} must contain top_k jobs plus at most one appended incumbent"
            )
    return doc


def verify_current_screening_sources(receipt: dict[str, Any]) -> None:
    """Require the measured implementations to match the screening receipt."""
    sources = _object(receipt.get("source_sha256"), "screening source_sha256")
    for relative, expected in sources.items():
        if not isinstance(relative, str) or not relative:
            raise ConfirmationError("screening source path must be non-empty")
        _hex64(expected, f"screening source {relative}")
        path = (REPO_ROOT / relative).resolve()
        try:
            path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ConfirmationError(
                f"screening source escapes the repository: {relative!r}"
            ) from exc
        _same(sha256_file(path), expected, f"current screening source {relative}")


def load_bound_confirmation(
    confirmation_path: Path,
    screening_manifest_path: Path,
    screening_launch_receipt_path: Path,
    screening_raw_dir: Path,
    robust_summary_path: Path,
    robust_adapter_manifest_path: Path,
) -> BoundConfirmation:
    """Rebuild the selection and require byte-stable equality to the input."""
    try:
        value, raw = analyze_screen.read_json(confirmation_path)
        document = validate_confirmation_document(value)
        _same(
            raw,
            analyze_screen.stable_json_bytes(document),
            "confirmation stable serialization",
        )
        analysis = analyze_screen.analyze_campaign(
            screening_manifest_path,
            screening_launch_receipt_path,
            screening_raw_dir,
            robust_summary_path,
            robust_adapter_manifest_path,
        )
        expected = analyze_screen.build_confirmation(analysis)
        _same(document, expected, "reconstructed confirmation document")
        verify_current_screening_sources(analysis.receipt)
    except analyze_screen.AnalysisError as exc:
        raise ConfirmationError(str(exc)) from exc
    return BoundConfirmation(document, sha256_bytes(raw), analysis)


def protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dist": args.dist,
        "seed": args.seed,
        "time_only": False,
        "trials": args.trials,
        "warmup_s": args.warmup_s,
    }


def validate_protocol(bound: BoundConfirmation, args: argparse.Namespace) -> str:
    value = protocol(args)
    digest = sha256_bytes(analyze_screen.stable_json_bytes(value))
    _same(
        digest,
        bound.document["provenance"]["screening_protocol_sha256"],
        "confirmation/screening timing protocol hash",
    )
    launch_args = bound.analysis.receipt["launch_args"]
    _same(
        value,
        {name: launch_args[name] for name in value},
        "confirmation/screening timing protocol",
    )
    return digest


def source_hashes() -> dict[str, str]:
    paths = tuple(screen_launch.SOURCE_PATHS) + (
        HERE / "analyze_screen.py",
        Path(__file__).resolve(),
    )
    unique = sorted(set(paths), key=lambda path: str(path))
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in unique
    }


def runtime_provenance(
    bound: BoundConfirmation,
    args: argparse.Namespace,
    protocol_sha256: str,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    sources = source_hashes()
    return {
        "schema_version": 1,
        "record_type": "fused_grid_confirmation_launch",
        "campaign_id": bound.document["campaign_id"],
        "confirmation_sha256": bound.raw_sha256,
        "confirmation_jobs_sha256": bound.document["jobs_sha256"],
        "confirmation_provenance": bound.document["provenance"],
        "protocol": protocol(args),
        "protocol_sha256": protocol_sha256,
        "launch_args": {
            "gpu": args.gpu,
            "order_seed": args.order_seed,
            "reps": args.reps,
            "timeout": args.timeout,
        },
        "artifact_path": {
            name: str(path) for name, path in sorted(artifact_paths.items())
        },
        "artifact_sha256": {
            name: sha256_file(path) for name, path in sorted(artifact_paths.items())
        },
        "source_sha256": sources,
        "source_bundle_sha256": sha256_bytes(
            analyze_screen.stable_json_bytes(sources)
        ),
        "git_commit": screen_launch.git_value("rev-parse", "HEAD"),
        "git_status_porcelain": screen_launch.git_value("status", "--porcelain"),
        "host": platform.node(),
        "python": sys.version,
        "python_executable": sys.executable,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }


def record_binding(
    provenance: dict[str, Any], job: dict[str, Any], rep: int
) -> dict[str, Any]:
    selection = provenance["confirmation_provenance"]
    return {
        "campaign_id": provenance["campaign_id"],
        "confirmation_sha256": provenance["confirmation_sha256"],
        "confirmation_jobs_sha256": provenance["confirmation_jobs_sha256"],
        "screening_launch_receipt_sha256": selection[
            "screening_launch_receipt_sha256"
        ],
        "screening_records_sha256": selection["screening_records_sha256"],
        "robust_summary_sha256": selection["robust_summary_sha256"],
        "robust_adapter_manifest_sha256": selection[
            "robust_adapter_manifest_sha256"
        ],
        "protocol_sha256": provenance["protocol_sha256"],
        "source_bundle_sha256": provenance["source_bundle_sha256"],
        "git_commit": provenance["git_commit"],
        "physical_gpu": provenance["launch_args"]["gpu"],
        "order_seed": provenance["launch_args"]["order_seed"],
        "reps": provenance["launch_args"]["reps"],
        "confirmation_id": job["confirmation_id"],
        "screening_job_id": job["screening_job_id"],
        "dsl": job["dsl"],
        "grid_id": job["grid_id"],
        "rep": rep,
    }


def record_state(
    path: Path, provenance: dict[str, Any], job: dict[str, Any], rep: int
) -> tuple[str, dict[str, Any] | None]:
    if not path.exists():
        return "pending", None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid", None
    if record.get("confirmation_provenance") != record_binding(provenance, job, rep):
        return "foreign", record
    expected_top = {
        "dsl": job["dsl"],
        "variant": job["variant"],
        "rep": rep,
    }
    if any(record.get(name) != value for name, value in expected_top.items()):
        return "invalid", record
    if not isinstance(record.get("ok"), bool):
        return "invalid", record
    if record["ok"]:
        try:
            _finite_positive(
                _object(record.get("timing"), "record timing").get("median_ms"),
                "record timing.median_ms",
            )
        except ConfirmationError:
            return "invalid", record
    error = record.get("error")
    gate_pass = isinstance(error, dict) and error.get("gate_pass") is True
    return ("complete_ok" if record["ok"] and gate_pass else "complete_failed"), record


def output_path(job: dict[str, Any], rep: int, raw_dir: Path) -> Path:
    safe_id = job["confirmation_id"].replace(".", "__")
    return raw_dir / f"{safe_id}__rep{rep}.json"


def scratch_path(job: dict[str, Any], rep: int, scratch_dir: Path) -> Path:
    safe_id = job["confirmation_id"].replace(".", "__")
    return scratch_dir / f"{safe_id}__rep{rep}.runner.json"


def plan_jobs(
    jobs: list[dict[str, Any]], reps: int, order_seed: int
) -> list[tuple[dict[str, Any], int]]:
    plan = [(job, rep) for job in jobs for rep in range(reps)]
    random.Random(order_seed).shuffle(plan)
    return plan


def runner_command(
    job: dict[str, Any], rep: int, args: argparse.Namespace, scratch: Path
) -> list[str]:
    command_args = argparse.Namespace(**vars(args), time_only=False)
    return screen_launch.runner_command(job, rep, command_args, scratch)


def run_fresh_process(
    job: dict[str, Any],
    rep: int,
    args: argparse.Namespace,
    scratch: Path,
) -> dict[str, Any]:
    scratch.parent.mkdir(parents=True, exist_ok=True)
    command = runner_command(job, rep, args, scratch)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(screen_launch.PHASE2) + ":" + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    started = time.time()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            cwd=screen_launch.PHASE2,
            timeout=args.timeout,
        )
        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        )
        stderr = f"TIMEOUT after {args.timeout}s"
        returncode = -9

    record: dict[str, Any] | None = None
    if "###JSON###" in stdout:
        try:
            record = json.loads(stdout.split("###JSON###", 1)[1].strip().splitlines()[0])
        except (json.JSONDecodeError, IndexError):
            record = None
    if record is None:
        record = {
            "ok": False,
            "op": "fused",
            "dsl": job["dsl"],
            "variant": job["variant"],
            "rep": rep,
            "error_msg": "runner produced no parseable JSON",
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    record["wall_s"] = time.time() - started
    record["returncode"] = returncode
    if not record.get("ok"):
        record.setdefault("stderr_tail", stderr[-4000:])
    return record


def print_counts(counts: dict[str, int], total: int) -> None:
    order = ("pending", "complete_ok", "complete_failed", "foreign", "invalid")
    print(
        f"plan: {total} fresh-process records ("
        + ", ".join(f"{name}={counts.get(name, 0)}" for name in order)
        + ")"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument(
        "--screening-manifest", type=Path, default=DEFAULT_SCREENING_MANIFEST
    )
    parser.add_argument("--screening-launch-receipt", type=Path, required=True)
    parser.add_argument("--screening-raw-dir", type=Path, required=True)
    parser.add_argument("--robust-summary", type=Path, required=True)
    parser.add_argument(
        "--robust-adapter-manifest", type=Path, default=DEFAULT_ADAPTER_MANIFEST
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--reps", type=int, default=EXPECTED_REPS)
    parser.add_argument("--dist", choices=("rand", "randn"), default="rand")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    parser.add_argument("--tag", default="fused_gbgs_confirm_robust_v1")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.reps != EXPECTED_REPS:
        parser.error(f"confirmation requires exactly {EXPECTED_REPS} process repetitions")
    if args.trials < 1 or args.warmup_s < 0 or args.timeout < 1:
        parser.error("trials/timeout must be positive and warmup-s non-negative")
    if not TAG_RE.fullmatch(args.tag):
        parser.error("tag must match [A-Za-z0-9][A-Za-z0-9_.-]*")
    return args


def main() -> int:
    args = parse_args()
    artifact_paths = {
        "confirmation": args.confirmation.resolve(),
        "robust_adapter_manifest": args.robust_adapter_manifest.resolve(),
        "robust_summary": args.robust_summary.resolve(),
        "screening_launch_receipt": args.screening_launch_receipt.resolve(),
        "screening_manifest": args.screening_manifest.resolve(),
    }
    try:
        bound = load_bound_confirmation(
            artifact_paths["confirmation"],
            artifact_paths["screening_manifest"],
            artifact_paths["screening_launch_receipt"],
            args.screening_raw_dir.resolve(),
            artifact_paths["robust_summary"],
            artifact_paths["robust_adapter_manifest"],
        )
        protocol_sha256 = validate_protocol(bound, args)
    except ConfirmationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    jobs = bound.document["jobs"]
    print(
        f"validated {bound.document['campaign_id']}: {len(jobs)} jobs x "
        f"{args.reps} fresh processes; confirmation_sha256={bound.raw_sha256}"
    )
    if args.validate_only and not (args.list or args.dry_run):
        return 0

    provenance = runtime_provenance(bound, args, protocol_sha256, artifact_paths)
    plan = plan_jobs(jobs, args.reps, args.order_seed)
    result_root = RESULTS_ROOT / args.tag
    raw_dir = result_root / "raw"
    scratch_dir = result_root / "scratch"
    entries = []
    counts: dict[str, int] = {}
    for job, rep in plan:
        output = output_path(job, rep, raw_dir)
        state, record = record_state(output, provenance, job, rep)
        counts[state] = counts.get(state, 0) + 1
        entries.append((job, rep, output, state, record))
    print_counts(counts, len(entries))

    if args.list or args.dry_run:
        for index, (job, rep, output, state, _record) in enumerate(entries, 1):
            print(
                f"[{index:03d}] {state:<15s} {job['confirmation_id']} "
                f"({job['screening_job_id']}) rep{rep} -> {output.relative_to(HERE)}"
            )
            if args.dry_run and (
                args.force
                or state == "pending"
                or (args.retry_failed and state == "complete_failed")
            ):
                scratch = scratch_path(job, rep, scratch_dir)
                print("      " + shlex.join(runner_command(job, rep, args, scratch)))
        return 0

    unsafe = counts.get("foreign", 0) + counts.get("invalid", 0)
    if unsafe and not args.force:
        print(
            f"ABORT: {unsafe} output record(s) lack matching confirmation provenance; "
            "choose a new --tag or inspect before --force",
            file=sys.stderr,
        )
        return 4

    runnable = [
        entry
        for entry in entries
        if args.force
        or entry[3] == "pending"
        or (args.retry_failed and entry[3] == "complete_failed")
    ]
    if not runnable:
        print("nothing to run; all matching confirmation records are complete")
        return 0

    screen_launch.driver2.preflight(args.gpu, strict=not args.allow_busy)
    raw_dir.mkdir(parents=True, exist_ok=True)
    launch_receipt = result_root / (
        "launch_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{os.getpid()}.json"
    )
    screen_launch.atomic_json(launch_receipt, provenance)
    print(f"running {len(runnable)} records; launch receipt -> {launch_receipt}")

    nok = nfail = 0
    for index, (job, rep, output, state, _old) in enumerate(runnable, 1):
        scratch = scratch_path(job, rep, scratch_dir)
        print(
            f"[{index:03d}/{len(runnable)}] {job['confirmation_id']} "
            f"{job['screening_job_id']} rep{rep}",
            flush=True,
        )
        record = run_fresh_process(job, rep, args, scratch)
        record["confirmation_provenance"] = record_binding(provenance, job, rep)
        screen_launch.atomic_json(output, record)
        final_state, _ = record_state(output, provenance, job, rep)
        if final_state == "complete_ok":
            nok += 1
            print(f"      ok median={record['timing']['median_ms']:.4f} ms")
        else:
            nfail += 1
            if record.get("ok"):
                message = "legacy correctness gate failed or was missing"
            else:
                message = str(record.get("error_msg", "unspecified runner failure"))
            print(f"      FAILED: {message[:200]}")
        screen_launch.atomic_json(
            result_root / "status.json",
            {
                "campaign_id": bound.document["campaign_id"],
                "complete_failed_this_session": nfail,
                "complete_ok_this_session": nok,
                "confirmation_sha256": bound.raw_sha256,
                "last_confirmation_id": job["confirmation_id"],
                "last_rep": rep,
                "planned_this_session": len(runnable),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    print(f"done: {nok} accepted, {nfail} failed -> {raw_dir}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; completed atomic records are resumable", file=sys.stderr)
        raise SystemExit(130)
