"""Collect the corrected 512-seed, ten-candidate paired robustness audit."""

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

# Pin the toolchain before torch cpp_extension snapshots CUDA_HOME.
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
POLICY_PATH = HERE / "inference_policy.json"
FREEZE_PATH = HERE / "receipts" / "freeze_receipt.json"
SEED_PLAN_PATH = HERE / "receipts" / "seed_plan.json"
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


def _load_bound(spec: dict[str, Any], prefix: str) -> Any:
    path = repo_path(spec[f"{prefix}_path"])
    if file_sha256(path) != spec[f"{prefix}_sha256"]:
        raise ValueError(f"{prefix} raw SHA mismatch: {path}")
    value = load_json(path)
    canonical_key = f"{prefix}_canonical_sha256"
    if canonical_key in spec and canonical_sha256(value) != spec[canonical_key]:
        raise ValueError(f"{prefix} canonical SHA mismatch: {path}")
    return value


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def _reach_protocol() -> ModuleType:
    name = "_same_seed_v2_frozen_reach_protocol"
    return sys.modules.get(name) or _module(name, REACH / "protocol.py")


def _reach_candidate() -> ModuleType:
    name = "_same_seed_v2_frozen_reach_candidate"
    return sys.modules.get(name) or _module(name, REACH / "candidate.py")


def _seed_map(namespace: str, case_id: str, index: int) -> dict[str, int]:
    return {
        tensor: derive_seed(namespace, "fused_softmax", case_id, "validation", tensor, index)
        for tensor in TENSORS
    }


def seed_plan(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    case_id = manifest["case"]["id"]
    rows: list[dict[str, Any]] = []
    for segment in manifest["seed_plan"]["segments"]:
        for index in range(segment["start"], segment["stop_exclusive"]):
            rows.append(
                {
                    "seed_index": index,
                    "segment": segment["label"],
                    "namespace": segment["namespace"],
                    "tensor_seeds": _seed_map(segment["namespace"], case_id, index),
                }
            )
    if [row["seed_index"] for row in rows] != list(range(512)):
        raise ValueError("seed plan must be the exact ordered indices 0..511")
    values = [seed for row in rows for seed in row["tensor_seeds"].values()]
    if len(values) != 1536 or len(set(values)) != 1536:
        raise ValueError("all 1,536 tensor seed integers must be unique")
    return rows


def verify_prior_seed_replay(manifest: dict[str, Any], seeds: list[dict[str, Any]]) -> None:
    prior = manifest["prior_stress_binding"]
    prior_manifest = _load_bound(prior, "manifest")
    if prior_manifest["stress_split"]["namespace"] != prior["namespace"]:
        raise ValueError("prior namespace binding mismatch")
    raw_path = repo_path(prior["raw_path"])
    if file_sha256(raw_path) != prior["raw_sha256"]:
        raise ValueError("prior raw SHA mismatch")
    by_index: dict[int, set[str]] = {}
    count = 0
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        index = row.get("seed_index")
        if not isinstance(index, int) or not 0 <= index < 256:
            raise ValueError("unexpected seed index in prior raw")
        if row.get("namespace") != prior["namespace"]:
            raise ValueError("prior raw namespace mismatch")
        by_index.setdefault(index, set()).add(canonical_sha256(row.get("tensor_seeds")))
        count += 1
    if count != 2048 or set(by_index) != set(range(256)):
        raise ValueError("prior raw does not have the expected 2,048-record census")
    for index in range(256):
        expected = seeds[index]["tensor_seeds"]
        if by_index[index] != {canonical_sha256(expected)}:
            raise ValueError(f"seed tuple does not exactly replay prior raw index {index}")


def verify_campaign(
    require_freeze: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(MANIFEST_PATH)
    if manifest.get("append_only") is not True or manifest.get("correctness_only") is not True:
        raise ValueError("campaign must remain append-only and correctness-only")
    if manifest.get("performance_selection_feedback_authorized") is not False:
        raise ValueError("performance feedback is forbidden")
    candidates = manifest["candidates"]
    if len(candidates) != 10 or [row["candidate_id"] for row in candidates] != manifest["candidate_order"]:
        raise ValueError("candidate roster/order must contain exactly the ten preregistered candidates")
    if len({row["candidate_id"] for row in candidates}) != 10:
        raise ValueError("candidate roster contains duplicates")

    policy_spec = manifest["inference_policy_binding"]
    policy = load_json(repo_path(policy_spec["path"]))
    if (
        file_sha256(repo_path(policy_spec["path"])) != policy_spec["sha256"]
        or canonical_sha256(policy) != policy_spec["canonical_sha256"]
        or policy["paired_contrasts"] != manifest["paired_contrasts"]
    ):
        raise ValueError("inference policy binding mismatch")
    legacy_spec = manifest["legacy_v1_disposition_binding"]
    legacy = load_json(repo_path(legacy_spec["path"]))
    if (
        file_sha256(repo_path(legacy_spec["path"])) != legacy_spec["sha256"]
        or canonical_sha256(legacy) != legacy_spec["canonical_sha256"]
        or legacy.get("controlling") is not False
    ):
        raise ValueError("legacy v1 disposition binding mismatch")

    old = manifest["old_candidate_binding"]
    adapter = load_json(repo_path(old["adapter_manifest_path"]))
    if (
        file_sha256(repo_path(old["adapter_manifest_path"])) != old["adapter_manifest_sha256"]
        or canonical_sha256(adapter) != old["adapter_manifest_canonical_sha256"]
        or adapter["source_bundle_sha256"] != old["source_bundle_sha256"]
    ):
        raise ValueError("old candidate adapter binding mismatch")
    for row in candidates[:4]:
        if adapter["grid"]["job_sha256"].get(row["candidate_id"]) != row["job_sha256"]:
            raise ValueError(f"old candidate job hash mismatch: {row['candidate_id']}")

    streamed = manifest["streamed_candidate_binding"]
    reach_lock_path = repo_path(streamed["launch_lock_path"])
    if (
        file_sha256(reach_lock_path) != streamed["launch_lock_sha256"]
        or canonical_sha256(load_json(reach_lock_path)) != streamed["launch_lock_canonical_sha256"]
    ):
        raise ValueError("reachability launch lock binding mismatch")
    reach_lock = _reach_protocol().verify_lock()
    jobs = _load_bound(streamed, "jobs")
    if reach_lock["source_bundle_sha256"] != streamed["source_bundle_sha256"]:
        raise ValueError("reachability source bundle mismatch")
    job_by_id = {row["job_id"]: row for row in jobs}
    for row in candidates[4:]:
        if (
            reach_lock["job_sha256"].get(row["candidate_id"]) != row["job_sha256"]
            or canonical_sha256(job_by_id[row["candidate_id"]]) != row["job_sha256"]
        ):
            raise ValueError(f"streamed candidate job hash mismatch: {row['candidate_id']}")

    gate_binding = manifest["registered_gate_binding"]
    robust_manifest = _load_bound(gate_binding, "manifest")
    gate_spec = _load_bound(gate_binding, "gate_spec")
    if gate_spec.get("manifest_sha256") != canonical_sha256(robust_manifest):
        raise ValueError("gate spec no longer binds the robust manifest")
    if tuple(manifest["gates"]) != GATES or set(gate_spec["gates"]) != {
        f"fused_softmax/{gate}" for gate in GATES
    }:
        raise ValueError("mixed gate roster changed")
    for gate_id in GATES:
        if gate_spec["gates"][f"fused_softmax/{gate_id}"]["thresholds"]["row_sum_error_max"]["value"] != gate_binding["registered_row_sum_threshold"]:
            raise ValueError("registered row-sum threshold changed")

    seeds = seed_plan(manifest)
    verify_prior_seed_replay(manifest, seeds)
    if require_freeze:
        if not FREEZE_PATH.is_file() or not SEED_PLAN_PATH.is_file():
            raise ValueError("production work requires frozen source and seed receipts")
        freeze = load_json(FREEZE_PATH)
        frozen_seeds = load_json(SEED_PLAN_PATH)
        if (
            freeze.get("manifest_sha256") != file_sha256(MANIFEST_PATH)
            or freeze.get("manifest_canonical_sha256") != canonical_sha256(manifest)
            or freeze.get("inference_policy_sha256") != file_sha256(POLICY_PATH)
            or freeze.get("seed_plan_sha256") != file_sha256(SEED_PLAN_PATH)
            or freeze.get("seed_plan_canonical_sha256") != canonical_sha256(seeds)
            or frozen_seeds != seeds
        ):
            raise ValueError("freeze/seed binding mismatch")
        for relative, expected in freeze.get("source_sha256", {}).items():
            if file_sha256(repo_path(relative)) != expected:
                raise ValueError(f"campaign source changed after freeze: {relative}")
        if canonical_sha256(freeze.get("source_sha256", {})) != freeze.get("source_bundle_canonical_sha256"):
            raise ValueError("frozen source bundle mismatch")
    return manifest, gate_spec, reach_lock, seeds


def threshold_failures(gate: dict[str, Any], metrics: dict[str, float]) -> list[str]:
    failures = []
    for name, rule in gate["thresholds"].items():
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{name}=missing/nonfinite")
        elif value > rule["value"]:
            failures.append(f"{name}={value:.17g}>{rule['value']:.17g}")
    return failures


class AppendOnlyJsonl:
    """Append to a partial stream and atomically seal it when census is complete."""

    def __init__(self, output: Path):
        self.output = output
        self.partial = output.with_name(output.name + ".partial")
        if output.exists():
            raise FileExistsError(f"refusing overwrite of sealed stream: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.keys: set[tuple[str, str, int]] = set()
        if self.partial.exists():
            for number, line in enumerate(self.partial.read_text(encoding="utf-8").splitlines(), 1):
                if not line:
                    continue
                row = json.loads(line)
                key = (row["candidate_id"], row["gate_id"], row["seed_index"])
                if key in self.keys:
                    raise ValueError(f"duplicate key in partial stream at line {number}: {key}")
                self.keys.add(key)
        self.handle = self.partial.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        key = (row["candidate_id"], row["gate_id"], row["seed_index"])
        if key in self.keys:
            raise ValueError(f"refusing duplicate append: {key}")
        self.handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.keys.add(key)

    def finish(self, expected: int) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if len(self.keys) != expected:
            raise ValueError(f"cannot seal incomplete stream: {len(self.keys)} != {expected}")
        os.replace(self.partial, self.output)

    def retain(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()


def _exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite of {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def gpu_snapshot(index: int) -> dict[str, str]:
    fields = (
        "index", "uuid", "name", "driver_version", "compute_cap", "pstate",
        "memory.total", "memory.used", "utilization.gpu", "clocks.sm", "clocks.mem",
        "power.limit", "temperature.gpu",
    )
    result = subprocess.run(
        ["nvidia-smi", f"--id={index}", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi snapshot failed")
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != len(fields):
        raise RuntimeError("unexpected nvidia-smi field count")
    return dict(zip(fields, values))


def ensure_idle(index: int) -> None:
    result = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    occupants = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or occupants:
        raise RuntimeError(f"physical GPU {index} busy or unavailable: {occupants}")


def validate_gpu(snapshot: dict[str, str], manifest: dict[str, Any]) -> None:
    required = manifest["hardware"]
    observed = (snapshot.get("uuid"), snapshot.get("name"), snapshot.get("compute_cap"))
    expected = (required["required_uuid"], required["required_name"], required["required_compute_capability"])
    if observed != expected:
        raise RuntimeError(f"GPU identity mismatch: expected={expected}, observed={observed}")


def nvcc_fingerprint() -> dict[str, Any]:
    options = ["/usr/local/cuda-13.1/bin/nvcc", shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"]
    for option in options:
        if option and Path(option).is_file():
            result = subprocess.run([option, "--version"], capture_output=True, text=True, timeout=20)
            return {"path": option, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    return {"path": None, "returncode": None, "stdout": "", "stderr": "nvcc not found"}


@dataclass
class Plan:
    row: dict[str, Any]
    execute: Callable[[dict[str, Any], dict[str, Any]], Any] | None
    metadata: dict[str, Any]
    error: str | None = None


def build_plans(manifest: dict[str, Any], reach_lock: dict[str, Any]) -> list[Plan]:
    from ako_runs.controlled_followup.fused_grid.robust_adapter import (
        build_phase2_candidates,
        load_repository,
        select_jobs,
    )

    old_rows = manifest["candidates"][:4]
    context = load_repository(repo_path(manifest["old_candidate_binding"]["adapter_manifest_path"]))
    phase2 = REPO_ROOT / "ako_runs" / "phase2_fused_sdpa"
    if str(phase2) not in sys.path:
        sys.path.insert(0, str(phase2))
    import common2  # type: ignore

    common2.ARTIFACTS_DIR = str(HERE / "build_artifacts")
    old_jobs = select_jobs(context, [row["candidate_id"] for row in old_rows])
    old_built = build_phase2_candidates(context, old_jobs)
    plans: list[Plan] = []
    for row, built in zip(old_rows, old_built, strict=True):
        if built.job_sha256 != row["job_sha256"]:
            raise ValueError(f"built old job hash mismatch: {row['candidate_id']}")
        plans.append(
            Plan(
                row=row,
                execute=built.execute,
                metadata={"config": built.config, "build_metadata": built.build_metadata},
                error=built.build_error,
            )
        )

    candidate_module = _reach_candidate()
    jobs = {row["job_id"]: row for row in load_json(REACH / "jobs.json")}
    for row in manifest["candidates"][4:]:
        started = time.perf_counter()
        try:
            job = jobs[row["candidate_id"]]
            if canonical_sha256(job) != row["job_sha256"] or reach_lock["job_sha256"][row["candidate_id"]] != row["job_sha256"]:
                raise ValueError("streamed job hash mismatch")
            config = candidate_module.make_config(job)
            built = candidate_module.build(row["lane"], config)

            def execute(inputs, prepared, *, _built=built):
                if "x_fp16" not in prepared:
                    prepared["x_fp16"] = inputs["x"].half().contiguous()
                return _built.run(prepared["x_fp16"], inputs["weight"], inputs["bias"])

            plans.append(
                Plan(
                    row=row,
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
        except Exception as exc:
            plans.append(
                Plan(
                    row=row,
                    execute=None,
                    metadata={"build_wall_s": time.perf_counter() - started, "traceback": traceback.format_exc()},
                    error=f"BuildError: {type(exc).__name__}: {exc}",
                )
            )
    if [plan.row for plan in plans] != manifest["candidates"]:
        raise RuntimeError("candidate build order changed")
    return plans


def _base(
    manifest: dict[str, Any], freeze: dict[str, Any], seed_hash: str, plan: Plan,
    gate_id: str, seed: dict[str, Any], build_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "record_type": "fused_same_seed_stress_measurement",
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "manifest_canonical_sha256": canonical_sha256(manifest),
        "source_bundle_canonical_sha256": freeze["source_bundle_canonical_sha256"],
        "seed_plan_canonical_sha256": seed_hash,
        "inference_policy_sha256": file_sha256(POLICY_PATH),
        "build_receipt_sha256": build_sha,
        "gate_spec_sha256": manifest["registered_gate_binding"]["gate_spec_sha256"],
        "old_source_bundle_sha256": manifest["old_candidate_binding"]["source_bundle_sha256"],
        "streamed_source_bundle_sha256": manifest["streamed_candidate_binding"]["source_bundle_sha256"],
        "candidate_id": plan.row["candidate_id"],
        "candidate_job_sha256": plan.row["job_sha256"],
        "generation": plan.row["generation"],
        "candidate_source": plan.row["source"],
        "lane": plan.row["lane"],
        "grid_id": plan.row["grid_id"],
        "case_id": manifest["case"]["id"],
        "seed_index": seed["seed_index"],
        "seed_segment": seed["segment"],
        "seed_namespace": seed["namespace"],
        "tensor_seeds": seed["tensor_seeds"],
        "shape": manifest["shape"],
        "gate_id": gate_id,
        "physical_gpu": manifest["hardware"]["physical_gpu"],
        "logical_device": manifest["hardware"]["logical_device"],
        "correctness_only": True,
        "performance_selection_feedback_authorized": False,
    }


def _failure_row(
    manifest: dict[str, Any], freeze: dict[str, Any], seed_hash: str, plan: Plan,
    gate_id: str, seed: dict[str, Any], build_sha: str, category: str, error: str,
) -> dict[str, Any]:
    row = _base(manifest, freeze, seed_hash, plan, gate_id, seed, build_sha)
    row.update(
        {
            "ok": False,
            "gate_pass": False,
            "threshold_failures": ["collection_failure"],
            "error_category": category,
            "error": error,
            "raw_safety_exceeded": None,
            "threshold_ratios": None,
        }
    )
    return row


def _load_or_create_build(plans: list[Plan], manifest: dict[str, Any]) -> str:
    value = {
        "schema_version": "2.0",
        "record_type": "fused_same_seed_stress_build_receipt",
        "campaign_id": manifest["campaign_id"],
        "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
        "launch_receipt_sha256": file_sha256(LAUNCH_PATH),
        "candidates": [
            {"candidate": plan.row, "metadata": plan.metadata, "error": plan.error}
            for plan in plans
        ],
        "performance_measurement_authorized": False,
    }
    if not BUILD_PATH.exists():
        value["created_utc"] = datetime.now(timezone.utc).isoformat()
        _exclusive(BUILD_PATH, value)
    else:
        observed = load_json(BUILD_PATH)
        expected_signature = [(p.row["candidate_id"], bool(p.error)) for p in plans]
        observed_signature = [(p["candidate"]["candidate_id"], bool(p["error"])) for p in observed.get("candidates", [])]
        if observed_signature != expected_signature:
            raise ValueError("resumed build status differs from frozen partial stream build")
    return file_sha256(BUILD_PATH)


def collect(
    manifest: dict[str, Any], gate_spec: dict[str, Any], reach_lock: dict[str, Any],
    seeds: list[dict[str, Any]], output: Path,
) -> int:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    plans = build_plans(manifest, reach_lock)
    build_sha = _load_or_create_build(plans, manifest)
    freeze = load_json(FREEZE_PATH)
    seed_hash = canonical_sha256(seeds)
    writer = AppendOnlyJsonl(output)
    live_inputs = None
    try:
        for position, seed in enumerate(seeds, 1):
            missing_by_candidate = {
                plan.row["candidate_id"]: [
                    gate for gate in GATES
                    if (plan.row["candidate_id"], gate, seed["seed_index"]) not in writer.keys
                ]
                for plan in plans
            }
            if not any(missing_by_candidate.values()):
                continue
            prior_inputs = live_inputs
            try:
                inputs = make_fused_inputs(manifest["shape"], manifest["case"], seed["tensor_seeds"], "cuda:0")
                live_inputs = inputs
                if prior_inputs is not None:
                    del prior_inputs
            except Exception as exc:
                live_inputs = prior_inputs
                for plan in plans:
                    for gate_id in missing_by_candidate[plan.row["candidate_id"]]:
                        writer.write(_failure_row(manifest, freeze, seed_hash, plan, gate_id, seed, build_sha, "setup", f"{type(exc).__name__}: {exc}"))
                continue

            reference: dict[str, Any] = {}
            reference_error: dict[str, str] = {}
            needed_gates = {gate for values in missing_by_candidate.values() for gate in values}
            for gate_id in needed_gates:
                gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
                try:
                    reference[gate_id] = resolve_output(gate["reference"]["kind"], "fused_softmax", inputs, gate["contract"])
                except Exception as exc:
                    reference_error[gate_id] = f"{type(exc).__name__}: {exc}"

            prepared: dict[str, Any] = {}
            for plan in plans:
                missing_gates = missing_by_candidate[plan.row["candidate_id"]]
                if not missing_gates:
                    continue
                if plan.error or plan.execute is None:
                    for gate_id in missing_gates:
                        writer.write(_failure_row(manifest, freeze, seed_hash, plan, gate_id, seed, build_sha, "build", plan.error or "candidate not executable"))
                    continue
                started = time.perf_counter()
                try:
                    with torch.no_grad():
                        output_value = plan.execute(inputs, prepared).float()
                    torch.cuda.synchronize()
                    wall_s = time.perf_counter() - started
                except Exception as exc:
                    for gate_id in missing_gates:
                        writer.write(_failure_row(manifest, freeze, seed_hash, plan, gate_id, seed, build_sha, "execution", f"{type(exc).__name__}: {exc}"))
                    continue

                for gate_id in missing_gates:
                    if gate_id in reference_error:
                        writer.write(_failure_row(manifest, freeze, seed_hash, plan, gate_id, seed, build_sha, "reference", reference_error[gate_id]))
                        continue
                    row = _base(manifest, freeze, seed_hash, plan, gate_id, seed, build_sha)
                    gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
                    try:
                        metrics = compute_metrics("fused_softmax", reference[gate_id], output_value, inputs)
                        failures = threshold_failures(gate, metrics)
                        ratios = {
                            name: metrics[name] / rule["value"]
                            for name, rule in gate["thresholds"].items()
                            if name in metrics and rule["value"] > 0
                        }
                        row_sum = gate["thresholds"]["row_sum_error_max"]
                        raw_cutoff = row_sum["observed_anchor_max"] * row_sum["safety_factor"]
                        row.update(
                            {
                                "ok": True,
                                "reference_kind": gate["reference"]["kind"],
                                "contract": gate["contract"],
                                "candidate_wall_s_diagnostic": wall_s,
                                "metrics": metrics,
                                "threshold_ratios": ratios,
                                "threshold_failures": failures,
                                "gate_pass": not failures,
                                "registered_row_sum_threshold": row_sum["value"],
                                "raw_safety_cutoff": raw_cutoff,
                                "raw_safety_exceeded": metrics["row_sum_error_max"] > raw_cutoff,
                            }
                        )
                    except Exception as exc:
                        row = _failure_row(manifest, freeze, seed_hash, plan, gate_id, seed, build_sha, "metric", f"{type(exc).__name__}: {exc}")
                    writer.write(row)
                del output_value
            del reference
            if position % 8 == 0:
                print(f"same-seed v2: {position}/512 seeds; rows={len(writer.keys)}/10240", flush=True)
        writer.finish(manifest["workload"]["expected_records"])
        return len(writer.keys)
    except BaseException:
        writer.retain()
        print(f"retained {len(writer.keys)} unique rows at {writer.partial}", file=sys.stderr)
        raise
    finally:
        if live_inputs is not None:
            del live_inputs


def main() -> int:
    manifest, gate_spec, reach_lock, seeds = verify_campaign(require_freeze=True)
    physical = manifest["hardware"]["physical_gpu"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise SystemExit(f"runner requires exactly CUDA_VISIBLE_DEVICES={physical}")
    if not LAUNCH_PATH.is_file():
        raise SystemExit("production collection requires launch_receipt.json")
    launch = load_json(LAUNCH_PATH)
    if (
        launch.get("freeze_receipt_sha256") != file_sha256(FREEZE_PATH)
        or launch.get("seed_plan_sha256") != file_sha256(SEED_PLAN_PATH)
        or launch.get("expected_records") != manifest["workload"]["expected_records"]
    ):
        raise SystemExit("launch receipt binding mismatch")
    output = (HERE / manifest["workload"]["output"]).resolve()
    output.relative_to(HERE)
    if output.exists() or COLLECTION_PATH.exists():
        raise FileExistsError("sealed collection already exists")

    ensure_idle(physical)
    before = gpu_snapshot(physical)
    validate_gpu(before, manifest)
    active = (HERE / "active.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        active.close()
        raise RuntimeError("another collector is active") from None
    os.environ["TORCH_EXTENSIONS_DIR"] = str(HERE / ".torch_extensions")
    os.environ.setdefault("MAX_JOBS", "4")

    if not EXECUTION_PATH.exists():
        _exclusive(
            EXECUTION_PATH,
            {
                "schema_version": "2.0",
                "record_type": "fused_same_seed_stress_gpu_execution_receipt",
                "campaign_id": manifest["campaign_id"],
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "host": platform.node(),
                "python": sys.version,
                "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
                "launch_receipt_sha256": file_sha256(LAUNCH_PATH),
                "seed_plan_sha256": file_sha256(SEED_PLAN_PATH),
                "gate_spec_sha256": manifest["registered_gate_binding"]["gate_spec_sha256"],
                "old_source_bundle_sha256": manifest["old_candidate_binding"]["source_bundle_sha256"],
                "streamed_source_bundle_sha256": manifest["streamed_candidate_binding"]["source_bundle_sha256"],
                "physical_gpu": physical,
                "logical_device": "cuda:0",
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "gpu_before": before,
                "nvcc": nvcc_fingerprint(),
                "correctness_only": True,
            },
        )
    else:
        execution = load_json(EXECUTION_PATH)
        if execution.get("launch_receipt_sha256") != file_sha256(LAUNCH_PATH):
            raise ValueError("existing execution receipt does not bind launch")

    started = time.perf_counter()
    try:
        count = collect(manifest, gate_spec, reach_lock, seeds, output)
        after = gpu_snapshot(physical)
        validate_gpu(after, manifest)
        _exclusive(
            COLLECTION_PATH,
            {
                "schema_version": "2.0",
                "record_type": "fused_same_seed_stress_collection_receipt",
                "campaign_id": manifest["campaign_id"],
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "wall_s_operational_this_invocation": time.perf_counter() - started,
                "gpu_execution_receipt_sha256": file_sha256(EXECUTION_PATH),
                "build_receipt_sha256": file_sha256(BUILD_PATH),
                "raw_path": str(output.relative_to(HERE)),
                "raw_sha256": file_sha256(output),
                "record_count": count,
                "expected_records": manifest["workload"]["expected_records"],
                "gpu_after": after,
                "correctness_only": True,
            },
        )
        print(f"wrote and sealed {count} records at {output}")
        return 0
    finally:
        active.close()


if __name__ == "__main__":
    raise SystemExit(main())
