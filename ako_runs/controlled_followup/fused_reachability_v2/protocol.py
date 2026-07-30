"""Hash-bound protocol helpers for fused reachability v2."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import random
import statistics
import fcntl
import shutil
import subprocess
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BASE_JOBS = REPO_ROOT / "ako_runs/controlled_followup/fused_grid/jobs/fused_gbgs_grid.json"
CAMPAIGN = HERE / "campaign.json"
JOBS = HERE / "jobs.json"
LOCK = HERE / "launch_lock.json"
ADAPTER = HERE / "adapter_manifest.json"
RESULTS = HERE / "results"
V1_CONFIRMATION_SUMMARY = (
    REPO_ROOT
    / "ako_runs/controlled_followup/fused_grid/results/"
    "fused_gbgs_confirm_robust_v1/summary.json"
)
ORIGINAL_ADAPTER = (
    REPO_ROOT / "ako_runs/controlled_followup/fused_grid/robust_adapter_manifest.json"
)
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

SOURCE_PATHS = (
    "ako_runs/controlled_followup/fused_reachability_v2/campaign.json",
    "ako_runs/controlled_followup/fused_reachability_v2/candidate.py",
    "ako_runs/controlled_followup/fused_reachability_v2/run_one.py",
    "ako_runs/controlled_followup/fused_reachability_v2/protocol.py",
    "ako_runs/controlled_followup/fused_reachability_v2/freeze.py",
    "ako_runs/controlled_followup/fused_reachability_v2/launch.py",
    "ako_runs/controlled_followup/fused_reachability_v2/analyze.py",
    "ako_runs/controlled_followup/fused_reachability_v2/robust_run.py",
    "ako_runs/controlled_followup/fused_reachability_v2/bind_robust.py",
    "ako_runs/controlled_followup/fused_reachability_v2/capture_evidence.py",
    "ako_runs/controlled_followup/fused_reachability_v2/test_protocol.py",
    "ako_runs/phase1_matmul/common.py",
    "ako_runs/phase1_matmul/variants/cuda_noptx_gemm.py",
    "ako_runs/phase1_matmul/variants/cuda_unlimited_gemm.py",
    "ako_runs/phase1_matmul/variants/__init__.py",
    "ako_runs/phase2_fused_sdpa/common2.py",
    "ako_runs/phase2_fused_sdpa/runner2.py",
    "ako_runs/phase2_fused_sdpa/variants2/__init__.py",
    "ako_runs/controlled_followup/fused_grid/robust_adapter.py",
    "ako_runs/controlled_followup/fused_grid/robust_adapter_manifest.json",
    "ako_runs/controlled_followup/robust_gate/__init__.py",
    "ako_runs/controlled_followup/robust_gate/distributions.py",
    "ako_runs/controlled_followup/robust_gate/metrics.py",
    "ako_runs/controlled_followup/robust_gate/oracles.py",
    "ako_runs/controlled_followup/robust_gate/schema.py",
    "ako_runs/controlled_followup/robust_gate/seeds.py",
    "ako_runs/controlled_followup/robust_gate/validate.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def record_filename(job_id: str, rep: int) -> str:
    return f"{job_id.replace('.', '__')}__rep{rep:02d}.json"


def safe_result_root(tag: str) -> Path:
    if tag in (".", "..") or not TAG_RE.fullmatch(tag):
        raise ValueError(f"unsafe result tag: {tag!r}")
    root = (RESULTS / tag).resolve()
    root.relative_to(RESULTS.resolve())
    return root


def acquire_active_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "active.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another launcher is active for {root}") from None
    handle.seek(0)
    handle.truncate()
    handle.write("locked\n")
    handle.flush()
    return handle


def gpu_snapshot(index: int) -> dict[str, str]:
    fields = (
        "index", "uuid", "name", "driver_version", "persistence_mode",
        "compute_cap", "pstate", "memory.total", "memory.used",
        "utilization.gpu", "clocks.sm", "clocks.mem", "clocks.max.sm",
        "clocks.max.memory", "power.limit", "temperature.gpu",
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi snapshot failed")
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != len(fields):
        raise RuntimeError(f"unexpected nvidia-smi field count: {completed.stdout!r}")
    return dict(zip(fields, values))


def validate_gpu_snapshot(snapshot: dict[str, str]) -> None:
    campaign = read_json(CAMPAIGN)
    hardware = campaign["hardware"]
    if snapshot.get("uuid") not in hardware["allowed_gpu_uuids"]:
        raise RuntimeError(f"GPU UUID not in frozen hardware set: {snapshot.get('uuid')}")
    if snapshot.get("name") != hardware["required_name"]:
        raise RuntimeError(f"GPU model mismatch: {snapshot.get('name')}")
    if snapshot.get("compute_cap") != hardware["required_compute_capability"]:
        raise RuntimeError(f"GPU compute capability mismatch: {snapshot.get('compute_cap')}")


def nvcc_fingerprint() -> dict[str, str | None]:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-13.1")
    candidates = [
        str(Path(cuda_home) / "bin/nvcc"),
        "/usr/local/cuda-13.1/bin/nvcc",
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        completed = subprocess.run(
            [candidate, "--version"], capture_output=True, text=True, timeout=20
        )
        return {
            "path": candidate,
            "returncode": str(completed.returncode),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return {"path": None, "returncode": None, "stdout": "", "stderr": "nvcc not found"}


def validate_process_record(
    path: Path,
    *,
    job: dict[str, Any],
    phase: str,
    rep: int,
    physical_gpu: int,
    lock: dict[str, Any],
) -> dict[str, Any]:
    record = read_json(path)
    expected = {
        "schema_version": 1,
        "record_type": "fused_reachability_v2_process",
        "campaign_id": lock["campaign_id"],
        "phase": phase,
        "job": job,
        "job_sha256": lock["job_sha256"][job["job_id"]],
        "rep": rep,
        "seed": 0,
        "dist": "rand",
        "trials": 100,
        "warmup_s": 2.0,
        "physical_gpu": physical_gpu,
        "logical_device": "cuda:0",
        "launch_lock_sha256": file_sha256(LOCK),
        "source_bundle_sha256": lock["source_bundle_sha256"],
    }
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if mismatches:
        raise RuntimeError(f"record contract mismatch in {path}: {mismatches}")
    reps = lock["launch_policy"][phase]["reps"]
    if rep not in range(reps):
        raise RuntimeError(f"rep out of range in {path}: {rep}")
    if not isinstance(record.get("ok"), bool):
        raise RuntimeError(f"record lacks boolean ok: {path}")
    if record["ok"]:
        times = record.get("times_ms")
        if (
            not isinstance(times, list)
            or len(times) != 100
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
                for value in times
            )
        ):
            raise RuntimeError(f"invalid timing samples in {path}")
        median = statistics.median(times)
        mean = statistics.fmean(times)
        if record.get("median_ms") != median or record.get("mean_ms") != mean:
            raise RuntimeError(f"timing summary mismatch in {path}")
        error = record.get("legacy_error")
        if not isinstance(error, dict) or not isinstance(error.get("gate_pass"), bool):
            raise RuntimeError(f"legacy gate missing from successful record: {path}")
    elif not isinstance(record.get("error"), str):
        raise RuntimeError(f"failed record lacks retained error: {path}")
    return record


def validate_launch_receipt(
    result_root: Path, *, phase: str, lane: str, lock: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt_path = result_root / "launch_receipt.json"
    receipt = read_json(receipt_path)
    contract = receipt.get("contract", {})
    required = {
        "campaign_id": lock["campaign_id"],
        "phase": phase,
        "lane": lane,
        "logical_device": "cuda:0",
        "reps": lock["launch_policy"][phase]["reps"],
        "launch_lock_sha256": file_sha256(LOCK),
        "source_bundle_sha256": lock["source_bundle_sha256"],
    }
    if receipt.get("record_type") != "fused_reachability_v2_launch_receipt":
        raise RuntimeError(f"wrong receipt type: {receipt_path}")
    mismatches = [key for key, value in required.items() if contract.get(key) != value]
    if mismatches:
        raise RuntimeError(f"launch receipt mismatch in {receipt_path}: {mismatches}")
    if receipt.get("source_sha256") != lock["source_sha256"]:
        raise RuntimeError(f"launch receipt source map mismatch: {receipt_path}")
    all_jobs = {row["job_id"]: row for row in read_json(JOBS)}
    job_ids = contract.get("jobs")
    if (
        not isinstance(job_ids, list)
        or len(job_ids) != len(set(job_ids))
        or any(job_id not in all_jobs or all_jobs[job_id]["lane"] != lane for job_id in job_ids)
    ):
        raise RuntimeError(f"launch receipt job set is invalid: {receipt_path}")
    jobs = [all_jobs[job_id] for job_id in job_ids]
    reps = required["reps"]
    expected_pairs = {(job_id, rep) for job_id in job_ids for rep in range(reps)}
    observed_pairs = set()
    for item in contract.get("execution_order", []):
        try:
            job_id, rep_text = item.rsplit(":rep", 1)
            observed_pairs.add((job_id, int(rep_text)))
        except (AttributeError, ValueError) as exc:
            raise RuntimeError(f"invalid execution order item {item!r}") from exc
    if observed_pairs != expected_pairs or len(contract.get("execution_order", [])) != len(expected_pairs):
        raise RuntimeError(f"execution order coverage mismatch: {receipt_path}")
    randomizer = random.Random(lock["launch_policy"][phase]["randomization_seed"])
    expected_order = []
    for rep in range(reps):
        block = list(jobs)
        randomizer.shuffle(block)
        expected_order.extend(f"{job['job_id']}:rep{rep}" for job in block)
    if contract.get("execution_order") != expected_order:
        raise RuntimeError(f"execution order does not reproduce frozen blocks: {receipt_path}")
    return receipt, jobs


def source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen source: {relative}")
        result[relative] = file_sha256(path)
    return result


def build_jobs() -> list[dict[str, Any]]:
    wanted_lanes = {"cuda_noptx", "cuda_unlimited"}
    wanted_grid = {f"g{index:02d}" for index in range(5, 13)}
    result = []
    for original in read_json(BASE_JOBS):
        if original["dsl"] not in wanted_lanes or original["grid_id"] not in wanted_grid:
            continue
        set_string = original["set"].replace(
            "x_epilogue=smem", "x_epilogue=streamed_global"
        )
        lane = original["dsl"]
        job = {
            "job_id": f"{lane}_streamed.{original['grid_id']}",
            "lane": lane,
            "grid_id": original["grid_id"],
            "grid_index": original["grid_index"],
            "variant": "GBGS",
            "set": set_string,
            "origin_job_id": original["job_id"],
            "origin_job_sha256": canonical_sha256(original),
        }
        result.append(job)
    result.sort(key=lambda row: (row["lane"], row["grid_index"]))
    if len(result) != 16:
        raise RuntimeError(f"expected 16 prospective jobs, observed {len(result)}")
    return result


def verify_lock() -> dict[str, Any]:
    lock = read_json(LOCK)
    if lock.get("campaign_sha256") != file_sha256(CAMPAIGN):
        raise RuntimeError("campaign.json changed after freeze")
    if lock.get("base_jobs_sha256") != file_sha256(BASE_JOBS):
        raise RuntimeError("base fused-grid jobs changed after freeze")
    jobs = read_json(JOBS)
    if lock.get("jobs_sha256") != file_sha256(JOBS):
        raise RuntimeError("jobs.json changed after freeze")
    if lock.get("jobs_canonical_sha256") != canonical_sha256(jobs):
        raise RuntimeError("jobs canonical hash mismatch")
    if jobs != build_jobs():
        raise RuntimeError("frozen jobs no longer match the bound base-grid projection")
    observed_job_hashes = {row["job_id"]: canonical_sha256(row) for row in jobs}
    if lock.get("job_sha256") != observed_job_hashes:
        raise RuntimeError("per-job canonical hash map mismatch")
    adapter = read_json(ADAPTER)
    if lock.get("adapter_manifest_sha256") != file_sha256(ADAPTER):
        raise RuntimeError("prospective adapter manifest changed after freeze")
    if lock.get("adapter_manifest_canonical_sha256") != canonical_sha256(adapter):
        raise RuntimeError("prospective adapter canonical hash mismatch")
    if adapter.get("source_sha256") != lock.get("source_sha256"):
        raise RuntimeError("prospective adapter source map mismatch")
    if adapter.get("source_bundle_sha256") != lock.get("source_bundle_sha256"):
        raise RuntimeError("prospective adapter source bundle mismatch")
    current = source_hashes()
    if current != lock.get("source_sha256"):
        changed = sorted(
            set(current) | set(lock.get("source_sha256", {}))
        )
        changed = [
            name for name in changed
            if current.get(name) != lock.get("source_sha256", {}).get(name)
        ]
        raise RuntimeError(f"source drift after freeze: {changed}")
    if canonical_sha256(current) != lock.get("source_bundle_sha256"):
        raise RuntimeError("source bundle hash mismatch")
    gate = lock.get("frozen_gate", {})
    for key in ("manifest", "gate_spec"):
        relative = gate.get(f"{key}_path")
        expected = gate.get(f"{key}_sha256")
        if not isinstance(relative, str) or file_sha256(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"frozen {key} changed after launch lock")
    if gate.get("original_adapter_sha256") != file_sha256(ORIGINAL_ADAPTER):
        raise RuntimeError("original fused robust adapter changed after launch lock")
    baseline = lock.get("v1_baseline", {})
    if baseline.get("summary_sha256") != file_sha256(V1_CONFIRMATION_SUMMARY):
        raise RuntimeError("v1 confirmation baseline changed after launch lock")
    return lock
