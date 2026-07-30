"""Collect the frozen 256-seed reachability row-sum stress workload."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

# Pin the toolchain before any transitive torch cpp_extension import snapshots
# CUDA_HOME.  CUDA visibility itself must be set by the launch command.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
_CUDA_BIN = "/usr/local/cuda-13.1/bin"
if _CUDA_BIN not in os.environ.get("PATH", "").split(":"):
    os.environ["PATH"] = _CUDA_BIN + ":" + os.environ.get("PATH", "")

from ako_runs.controlled_followup.robust_gate.distributions import make_fused_inputs
from ako_runs.controlled_followup.robust_gate.metrics import compute_metrics
from ako_runs.controlled_followup.robust_gate.oracles import resolve_output
from ako_runs.controlled_followup.robust_gate.schema import (
    canonical_sha256,
    file_sha256,
    load_json,
)
from ako_runs.controlled_followup.robust_gate.seeds import derive_seed


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
CONTROLLED = HERE.parents[2]
REACH = CONTROLLED / "fused_reachability_v2"
MANIFEST_PATH = HERE / "manifest.json"
FREEZE_PATH = HERE / "receipts" / "freeze_receipt.json"
LAUNCH_PATH = HERE / "receipts" / "launch_receipt.json"
EXECUTION_PATH = HERE / "receipts" / "gpu_execution_receipt.json"
BUILD_PATH = HERE / "receipts" / "build_receipt.json"
COLLECTION_PATH = HERE / "receipts" / "collection_receipt.json"
GATES = ("semantic_mixed", "conformance_mixed")
TENSORS = ("x", "weight", "bias")


def repo_path(relative: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    path.relative_to(REPO_ROOT.resolve())
    return path


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def _reach_protocol() -> ModuleType:
    name = "_frozen_fused_reachability_v2_protocol"
    return sys.modules.get(name) or _module(name, REACH / "protocol.py")


def _reach_candidate() -> ModuleType:
    name = "_frozen_fused_reachability_v2_candidate"
    return sys.modules.get(name) or _module(name, REACH / "candidate.py")


def _seed_map(namespace: str, case_id: str, index: int) -> dict[str, int]:
    return {
        tensor: derive_seed(
            namespace, "fused_softmax", case_id, "validation", tensor, index
        )
        for tensor in TENSORS
    }


def seed_plan(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    split = manifest["stress_split"]
    case_id = manifest["case"]["id"]
    indices = range(split["seed_indices"]["start"], split["seed_indices"]["stop_exclusive"])
    rows = [
        {"seed_index": index, "tensor_seeds": _seed_map(split["namespace"], case_id, index)}
        for index in indices
    ]
    if len(rows) != 256 or split["seeds"] != 256:
        raise ValueError("stress plan must contain exactly 256 seeds")

    new_values = [seed for row in rows for seed in row["tensor_seeds"].values()]
    if len(new_values) != len(set(new_values)):
        raise ValueError("new stress tensor seeds are not internally unique")

    robust_manifest = load_json(repo_path(manifest["registered_gate_binding"]["manifest_path"]))
    original = {
        derive_seed(
            robust_manifest["seed_namespace"],
            "fused_softmax",
            case_id,
            "validation",
            tensor,
            index,
        )
        for index in range(robust_manifest["split_counts"]["validation"])
        for tensor in TENSORS
    }
    prior = manifest["prior_stress_binding"]
    prior_values = {
        derive_seed(
            prior["namespace"],
            "fused_softmax",
            case_id,
            "validation",
            tensor,
            index,
        )
        for index in range(prior["seed_count"])
        for tensor in TENSORS
    }
    overlap_original = set(new_values) & original
    overlap_prior = set(new_values) & prior_values
    if overlap_original or overlap_prior:
        raise ValueError(
            "fresh stress seeds overlap earlier streams: "
            f"original={len(overlap_original)}, prior_stress={len(overlap_prior)}"
        )
    return rows


def _check_file(spec: dict[str, Any], prefix: str) -> Any:
    path = repo_path(spec[f"{prefix}_path"])
    expected = spec[f"{prefix}_sha256"]
    if file_sha256(path) != expected:
        raise ValueError(f"{prefix} raw SHA mismatch: {path}")
    value = load_json(path)
    canonical_key = f"{prefix}_canonical_sha256"
    if canonical_key in spec and canonical_sha256(value) != spec[canonical_key]:
        raise ValueError(f"{prefix} canonical SHA mismatch: {path}")
    return value


def verify_campaign(require_freeze: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(MANIFEST_PATH)
    if manifest.get("append_only") is not True or manifest.get("correctness_only") is not True:
        raise ValueError("audit must remain append-only and correctness-only")
    policy = manifest["performance_selection_policy"]
    for key in (
        "feedback_authorized",
        "frozen_selection_may_change",
        "performance_measurement_authorized",
    ):
        if policy.get(key) is not False:
            raise ValueError(f"performance authority switch changed: {key}")

    reach_spec = manifest["reachability_binding"]
    reach_lock_path = repo_path(reach_spec["launch_lock_path"])
    if file_sha256(reach_lock_path) != reach_spec["launch_lock_sha256"]:
        raise ValueError("reachability launch lock raw SHA mismatch")
    reach_lock = _reach_protocol().verify_lock()
    if (
        reach_lock.get("campaign_id") != reach_spec["campaign_id"]
        or reach_lock.get("source_bundle_sha256") != reach_spec["source_bundle_sha256"]
    ):
        raise ValueError("reachability lock campaign/source binding mismatch")

    selection_path = repo_path(reach_spec["screen_selection_path"])
    if file_sha256(selection_path) != reach_spec["screen_selection_sha256"]:
        raise ValueError("screen selection raw SHA mismatch")
    selection = load_json(selection_path)
    if canonical_sha256(selection) != reach_spec["screen_selection_canonical_sha256"]:
        raise ValueError("screen selection canonical SHA mismatch")
    summary_path = repo_path(reach_spec["screen_summary_path"])
    if file_sha256(summary_path) != reach_spec["screen_summary_sha256"]:
        raise ValueError("screen summary raw SHA mismatch")
    summary = load_json(summary_path)
    if (
        selection.get("launch_lock_sha256") != reach_spec["launch_lock_sha256"]
        or selection.get("source_bundle_sha256") != reach_spec["source_bundle_sha256"]
        or selection.get("screen_summary_sha256") != reach_spec["screen_summary_sha256"]
        or summary.get("record_bundle_sha256") != selection.get("screen_bundle_sha256")
    ):
        raise ValueError("screen selection/summary provenance mismatch")

    selected_projection = [
        {
            "grid_id": row["grid_id"],
            "job_id": row["job_id"],
            "job_sha256": row["job_sha256"],
            "lane": row["lane"],
            "screen_rank": row["screen_rank"],
        }
        for row in selection.get("selected", [])
    ]
    if selected_projection != manifest["selected_candidates"]:
        raise ValueError("manifest candidates differ from frozen screen selection")
    if len(selected_projection) != 6 or len({row["job_id"] for row in selected_projection}) != 6:
        raise ValueError("screen selection must contain exactly six unique candidates")
    for lane in ("cuda_noptx", "cuda_unlimited"):
        if sorted(row["screen_rank"] for row in selected_projection if row["lane"] == lane) != [1, 2, 3]:
            raise ValueError(f"screen ranks changed for {lane}")
    for row in selected_projection:
        if reach_lock["job_sha256"].get(row["job_id"]) != row["job_sha256"]:
            raise ValueError(f"selected job hash mismatch: {row['job_id']}")

    gate_binding = manifest["registered_gate_binding"]
    robust_manifest = _check_file(gate_binding, "manifest")
    gate_spec = _check_file(gate_binding, "gate_spec")
    if gate_spec.get("manifest_sha256") != canonical_sha256(robust_manifest):
        raise ValueError("gate spec no longer binds robust manifest")
    if set(gate_spec.get("gates", {})) != {f"fused_softmax/{gate}" for gate in GATES}:
        raise ValueError("registered gate set changed")
    for gate_id in GATES:
        threshold = gate_spec["gates"][f"fused_softmax/{gate_id}"]["thresholds"]["row_sum_error_max"]
        if threshold.get("value") != gate_binding["registered_row_sum_threshold"]:
            raise ValueError("registered row-sum threshold changed")

    prior = manifest["prior_stress_binding"]
    prior_path = repo_path(prior["manifest_path"])
    if file_sha256(prior_path) != prior["manifest_sha256"]:
        raise ValueError("prior stress manifest raw SHA mismatch")
    prior_manifest = load_json(prior_path)
    if canonical_sha256(prior_manifest) != prior["manifest_canonical_sha256"]:
        raise ValueError("prior stress manifest canonical SHA mismatch")
    if prior_manifest["stress_split"]["namespace"] != prior["namespace"]:
        raise ValueError("prior stress namespace mismatch")

    seeds = seed_plan(manifest)
    if require_freeze:
        if not FREEZE_PATH.is_file():
            raise ValueError("production collection requires the freeze receipt")
        freeze = load_json(FREEZE_PATH)
        if (
            freeze.get("manifest_sha256") != file_sha256(MANIFEST_PATH)
            or freeze.get("manifest_canonical_sha256") != canonical_sha256(manifest)
            or freeze.get("seed_plan_canonical_sha256") != canonical_sha256(seeds)
        ):
            raise ValueError("freeze receipt binding mismatch")
        for relative, expected in freeze.get("source_sha256", {}).items():
            if file_sha256(repo_path(relative)) != expected:
                raise ValueError(f"audit source changed after freeze: {relative}")
        if canonical_sha256(freeze.get("source_sha256", {})) != freeze.get("source_bundle_canonical_sha256"):
            raise ValueError("audit source bundle mismatch")
    return manifest, gate_spec, reach_lock, seeds


def threshold_failures(gate: dict[str, Any], metrics: dict[str, float]) -> list[str]:
    failures = []
    for name, threshold in gate["thresholds"].items():
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{name}=missing/nonfinite")
        elif value > threshold["value"]:
            failures.append(f"{name}={value:.17g}>{threshold['value']:.17g}")
    return failures


class AtomicJsonl:
    def __init__(self, output: Path):
        self.output = output
        self.partial = output.with_name(output.name + ".partial")
        if output.exists() or self.partial.exists():
            raise FileExistsError(f"refusing overwrite of {output} or {self.partial}")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.partial.open("x", encoding="utf-8")
        self.count = 0

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.count += 1

    def finish(self) -> None:
        self.handle.close()
        os.replace(self.partial, self.output)

    def retain(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()


def _exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite of {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def gpu_snapshot(index: int) -> dict[str, str]:
    fields = (
        "index", "uuid", "name", "driver_version", "persistence_mode",
        "compute_cap", "pstate", "memory.total", "memory.used",
        "utilization.gpu", "clocks.sm", "clocks.mem", "clocks.max.sm",
        "clocks.max.memory", "power.limit", "temperature.gpu",
    )
    completed = subprocess.run(
        [
            "nvidia-smi", f"--id={index}", f"--query-gpu={','.join(fields)}",
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
        raise RuntimeError("unexpected nvidia-smi field count")
    return dict(zip(fields, values))


def ensure_idle(index: int) -> None:
    completed = subprocess.run(
        [
            "nvidia-smi", f"--id={index}", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    occupants = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or occupants:
        raise RuntimeError(f"physical GPU {index} is busy or unavailable: {occupants}")


def validate_gpu(snapshot: dict[str, str], manifest: dict[str, Any]) -> None:
    required = manifest["hardware"]
    observed = {
        "uuid": snapshot.get("uuid"),
        "name": snapshot.get("name"),
        "compute_capability": snapshot.get("compute_cap"),
    }
    expected = {
        "uuid": required["required_uuid"],
        "name": required["required_name"],
        "compute_capability": required["required_compute_capability"],
    }
    if observed != expected:
        raise RuntimeError(f"GPU identity mismatch: expected={expected}, observed={observed}")


def nvcc_fingerprint() -> dict[str, Any]:
    candidates = [
        str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.1")) / "bin/nvcc"),
        "/usr/local/cuda-13.1/bin/nvcc",
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            result = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=20)
            return {
                "path": candidate,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
    return {"path": None, "returncode": None, "stdout": "", "stderr": "nvcc not found"}


@dataclass
class Plan:
    row: dict[str, Any]
    execute: Callable[[dict[str, Any], dict[str, Any]], Any] | None
    metadata: dict[str, Any]
    error: str | None = None


def build_plans(manifest: dict[str, Any], reach_lock: dict[str, Any]) -> list[Plan]:
    candidate = _reach_candidate()
    jobs = {row["job_id"]: row for row in load_json(REACH / "jobs.json")}
    plans = []
    for selected in manifest["selected_candidates"]:
        started = time.perf_counter()
        try:
            job = jobs[selected["job_id"]]
            if canonical_sha256(job) != selected["job_sha256"]:
                raise ValueError("job canonical hash mismatch")
            config = candidate.make_config(job)
            built = candidate.build(selected["lane"], config)

            def execute(inputs, prepared, *, _built=built):
                if "x_fp16" not in prepared:
                    prepared["x_fp16"] = inputs["x"].half().contiguous()
                return _built.run(prepared["x_fp16"], inputs["weight"], inputs["bias"])

            plans.append(
                Plan(
                    row=selected,
                    execute=execute,
                    metadata={
                        "build_wall_s": time.perf_counter() - started,
                        "reported_compile_s": built.compile_s,
                        "n_kernels": built.n_kernels,
                        "artifacts": built.artifacts,
                        "config": config.to_dict(),
                    },
                )
            )
        except Exception as exc:  # retained build evidence
            plans.append(
                Plan(
                    row=selected,
                    execute=None,
                    metadata={"build_wall_s": time.perf_counter() - started},
                    error=f"BuildError: {type(exc).__name__}: {exc}",
                )
            )
    if [plan.row for plan in plans] != manifest["selected_candidates"]:
        raise RuntimeError("candidate build order changed")
    return plans


def _base(
    manifest: dict[str, Any], freeze: dict[str, Any], seeds_hash: str,
    plan: Plan, gate_id: str, index: int, seeds: dict[str, int],
    build_sha: str,
) -> dict[str, Any]:
    reach = manifest["reachability_binding"]
    gate = manifest["registered_gate_binding"]
    return {
        "schema_version": "1.0",
        "record_type": "fused_reachability_row_sum_stress_measurement",
        "campaign_id": manifest["campaign_id"],
        "stress_manifest_sha256": file_sha256(MANIFEST_PATH),
        "stress_manifest_canonical_sha256": canonical_sha256(manifest),
        "audit_source_bundle_canonical_sha256": freeze["source_bundle_canonical_sha256"],
        "seed_plan_canonical_sha256": seeds_hash,
        "reachability_launch_lock_sha256": reach["launch_lock_sha256"],
        "reachability_source_bundle_sha256": reach["source_bundle_sha256"],
        "screen_selection_sha256": reach["screen_selection_sha256"],
        "screen_summary_sha256": reach["screen_summary_sha256"],
        "gate_spec_sha256": gate["gate_spec_sha256"],
        "gate_spec_canonical_sha256": gate["gate_spec_canonical_sha256"],
        "build_receipt_sha256": build_sha,
        "candidate": plan.row["job_id"],
        "candidate_job_sha256": plan.row["job_sha256"],
        "lane": plan.row["lane"],
        "grid_id": plan.row["grid_id"],
        "screen_rank": plan.row["screen_rank"],
        "case_id": manifest["case"]["id"],
        "namespace": manifest["stress_split"]["namespace"],
        "seed_index": index,
        "tensor_seeds": seeds,
        "shape": manifest["shape"],
        "gate_id": gate_id,
        "physical_gpu": manifest["hardware"]["physical_gpu"],
        "logical_device": manifest["hardware"]["logical_device"],
        "correctness_only": True,
        "performance_selection_feedback_authorized": False,
    }


def _failure_rows(
    manifest: dict[str, Any], freeze: dict[str, Any], seeds_hash: str,
    plan: Plan, index: int, seeds: dict[str, int], build_sha: str,
    category: str, error: str,
) -> list[dict[str, Any]]:
    rows = []
    for gate_id in GATES:
        row = _base(manifest, freeze, seeds_hash, plan, gate_id, index, seeds, build_sha)
        row.update(
            {
                "ok": False,
                "gate_pass": False,
                "threshold_failures": ["collection_failure"],
                "error_category": category,
                "error": error,
                "raw_safety_exceeded": None,
            }
        )
        rows.append(row)
    return rows


def collect(
    manifest: dict[str, Any], gate_spec: dict[str, Any], reach_lock: dict[str, Any],
    seeds_plan: list[dict[str, Any]], output: Path,
) -> int:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    freeze = load_json(FREEZE_PATH)
    seeds_hash = canonical_sha256(seeds_plan)
    plans = build_plans(manifest, reach_lock)
    build_receipt = {
        "schema_version": "1.0",
        "record_type": "fused_reachability_row_sum_stress_build_receipt",
        "campaign_id": manifest["campaign_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
        "launch_receipt_sha256": file_sha256(LAUNCH_PATH),
        "candidates": [
            {"candidate": plan.row, "metadata": plan.metadata, "error": plan.error}
            for plan in plans
        ],
        "performance_measurement_authorized": False,
    }
    _exclusive(BUILD_PATH, build_receipt)
    build_sha = file_sha256(BUILD_PATH)

    writer = AtomicJsonl(output)
    live_inputs = None
    try:
        for position, seed_row in enumerate(seeds_plan, 1):
            index = seed_row["seed_index"]
            seeds = seed_row["tensor_seeds"]
            runnable = [plan for plan in plans if plan.execute is not None]
            prior_inputs = live_inputs
            try:
                inputs = make_fused_inputs(manifest["shape"], manifest["case"], seeds, "cuda:0")
                live_inputs = inputs
                if prior_inputs is not None:
                    del prior_inputs
            except Exception as exc:
                live_inputs = prior_inputs
                error = f"{type(exc).__name__}: {exc}"
                for plan in plans:
                    category = "build" if plan.error else "setup"
                    reason = plan.error or error
                    for row in _failure_rows(
                        manifest, freeze, seeds_hash, plan, index, seeds,
                        build_sha, category, reason,
                    ):
                        writer.write(row)
                print(f"[setup failure {position}/256] seed={index}: {error}", flush=True)
                continue

            references: dict[str, Any] = {}
            reference_errors: dict[str, str] = {}
            for gate_id in GATES:
                gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
                try:
                    references[gate_id] = resolve_output(
                        gate["reference"]["kind"], "fused_softmax", inputs, gate["contract"]
                    )
                except Exception as exc:
                    reference_errors[gate_id] = f"{type(exc).__name__}: {exc}"

            prepared: dict[str, Any] = {}
            for plan in plans:
                if plan.error or plan.execute is None:
                    for row in _failure_rows(
                        manifest, freeze, seeds_hash, plan, index, seeds,
                        build_sha, "build", plan.error or "candidate is not executable",
                    ):
                        writer.write(row)
                    continue
                started = time.perf_counter()
                try:
                    with torch.no_grad():
                        output_value = plan.execute(inputs, prepared).float()
                    torch.cuda.synchronize()
                    candidate_wall_s = time.perf_counter() - started
                except Exception as exc:
                    for row in _failure_rows(
                        manifest, freeze, seeds_hash, plan, index, seeds,
                        build_sha, "execution", f"{type(exc).__name__}: {exc}",
                    ):
                        writer.write(row)
                    continue

                for gate_id in GATES:
                    row = _base(
                        manifest, freeze, seeds_hash, plan, gate_id, index,
                        seeds, build_sha,
                    )
                    gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
                    row["reference_kind"] = gate["reference"]["kind"]
                    row["contract"] = gate["contract"]
                    row["candidate_wall_s_diagnostic"] = candidate_wall_s
                    if gate_id in reference_errors:
                        row.update(
                            {
                                "ok": False,
                                "gate_pass": False,
                                "threshold_failures": ["collection_failure"],
                                "error_category": "reference",
                                "error": reference_errors[gate_id],
                                "raw_safety_exceeded": None,
                            }
                        )
                    else:
                        try:
                            metrics = compute_metrics(
                                "fused_softmax", references[gate_id], output_value, inputs
                            )
                            failures = threshold_failures(gate, metrics)
                            row_sum = gate["thresholds"]["row_sum_error_max"]
                            raw_cutoff = row_sum["observed_anchor_max"] * row_sum["safety_factor"]
                            row.update(
                                {
                                    "ok": True,
                                    "metrics": metrics,
                                    "threshold_failures": failures,
                                    "gate_pass": not failures,
                                    "registered_row_sum_threshold": row_sum["value"],
                                    "raw_safety_cutoff": raw_cutoff,
                                    "raw_safety_exceeded": metrics["row_sum_error_max"] > raw_cutoff,
                                }
                            )
                        except Exception as exc:
                            row.update(
                                {
                                    "ok": False,
                                    "gate_pass": False,
                                    "threshold_failures": ["collection_failure"],
                                    "error_category": "metric",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "raw_safety_exceeded": None,
                                }
                            )
                    writer.write(row)
                del output_value
            del references
            if position % 8 == 0:
                print(f"reachability row-sum stress: {position}/256 seeds", flush=True)
        writer.finish()
    except BaseException:
        writer.retain()
        print(f"retained {writer.count} rows at {writer.partial}", file=sys.stderr)
        raise
    finally:
        if live_inputs is not None:
            del live_inputs
    return writer.count


def main() -> int:
    manifest, gate_spec, reach_lock, seeds = verify_campaign(require_freeze=True)
    visible = str(manifest["hardware"]["physical_gpu"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
        raise SystemExit(f"runner requires exactly CUDA_VISIBLE_DEVICES={visible}")
    if not LAUNCH_PATH.is_file():
        raise SystemExit("production collection requires launch_receipt.json")
    launch = load_json(LAUNCH_PATH)
    if (
        launch.get("freeze_receipt_sha256") != file_sha256(FREEZE_PATH)
        or launch.get("seed_plan_canonical_sha256") != canonical_sha256(seeds)
        or launch.get("expected_records") != manifest["workload"]["expected_records"]
    ):
        raise SystemExit("launch receipt binding mismatch")

    output = (HERE / manifest["workload"]["output"]).resolve()
    output.relative_to(HERE)
    if EXECUTION_PATH.exists() or BUILD_PATH.exists() or COLLECTION_PATH.exists():
        raise FileExistsError("GPU collection receipts already exist; refusing a second run")
    ensure_idle(manifest["hardware"]["physical_gpu"])
    before = gpu_snapshot(manifest["hardware"]["physical_gpu"])
    validate_gpu(before, manifest)

    active_path = HERE / "active.lock"
    active = active_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        active.close()
        raise RuntimeError("another stress collector is active") from None
    active.seek(0)
    active.truncate()
    active.write("locked\n")
    active.flush()

    os.environ["TORCH_EXTENSIONS_DIR"] = str(HERE / ".torch_extensions")
    os.environ.setdefault("MAX_JOBS", "4")

    execution = {
        "schema_version": "1.0",
        "record_type": "fused_reachability_row_sum_stress_gpu_execution_receipt",
        "campaign_id": manifest["campaign_id"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
        "launch_receipt_sha256": file_sha256(LAUNCH_PATH),
        "seed_plan_canonical_sha256": canonical_sha256(seeds),
        "reachability_launch_lock_sha256": manifest["reachability_binding"]["launch_lock_sha256"],
        "screen_selection_sha256": manifest["reachability_binding"]["screen_selection_sha256"],
        "physical_gpu": manifest["hardware"]["physical_gpu"],
        "logical_device": "cuda:0",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_before": before,
        "nvcc": nvcc_fingerprint(),
        "correctness_only": True,
        "performance_selection_feedback_authorized": False,
    }
    _exclusive(EXECUTION_PATH, execution)
    started = time.perf_counter()
    try:
        count = collect(manifest, gate_spec, reach_lock, seeds, output)
        after = gpu_snapshot(manifest["hardware"]["physical_gpu"])
        validate_gpu(after, manifest)
        collection = {
            "schema_version": "1.0",
            "record_type": "fused_reachability_row_sum_stress_collection_receipt",
            "campaign_id": manifest["campaign_id"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_s_operational": time.perf_counter() - started,
            "gpu_execution_receipt_sha256": file_sha256(EXECUTION_PATH),
            "build_receipt_sha256": file_sha256(BUILD_PATH),
            "raw_path": str(output.relative_to(HERE)),
            "raw_sha256": file_sha256(output),
            "record_count": count,
            "expected_records": manifest["workload"]["expected_records"],
            "gpu_after": after,
            "correctness_only": True,
            "performance_selection_feedback_authorized": False,
        }
        _exclusive(COLLECTION_PATH, collection)
        print(f"wrote {count} records to {output}")
        return 0 if count == manifest["workload"]["expected_records"] else 2
    finally:
        active.close()


if __name__ == "__main__":
    raise SystemExit(main())
