#!/usr/bin/env python3
"""Core adapter from fused-grid Phase-2 jobs to the frozen robust gate.

Candidate imports are deferred until :func:`build_phase2_candidates`, keeping
manifest checks, seed planning, threshold logic, and CPU smoke tests runnable
on hosts without CUDA, TileLang, or Triton.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


HERE = Path(__file__).resolve().parent
CONTROLLED = HERE.parent
REPO_ROOT = HERE.parents[2]
ROBUST = CONTROLLED / "robust_gate"
PHASE1 = REPO_ROOT / "ako_runs/phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs/phase2_fused_sdpa"
DEFAULT_ADAPTER_MANIFEST = HERE / "robust_adapter_manifest.json"

if str(CONTROLLED) not in sys.path:
    sys.path.insert(0, str(CONTROLLED))

from robust_gate import SCHEMA_VERSION  # noqa: E402
from robust_gate.distributions import make_inputs  # noqa: E402
from robust_gate.metrics import compute_metrics  # noqa: E402
from robust_gate.oracles import native_mixed_reference, resolve_output  # noqa: E402
from robust_gate.schema import (  # noqa: E402
    SchemaError,
    canonical_sha256,
    file_sha256,
    load_json,
    validate_gate_spec,
    validate_manifest,
)
from robust_gate.seeds import tensor_seeds  # noqa: E402
from robust_gate.validate import validate_records  # noqa: E402


GATE_IDS = ("semantic_mixed", "conformance_mixed")
ARTIFACT_KEYS = (
    "n_regs",
    "n_spills",
    "shared_bytes",
    "grid",
    "block",
    "backend_detail",
    "wcache",
    "n_kernels",
    "tile",
    "b_global_dtype",
    "transpose_b",
    "num_warps",
    "num_stages",
)


def _repo_file(relative: str) -> Path:
    candidate = (REPO_ROOT / relative).resolve()
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {relative!r}") from exc
    return candidate


def _load_exact(path: Path, expected_hash: str, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = file_sha256(path)
    if actual != expected_hash:
        raise ValueError(
            f"{label} hash mismatch: expected {expected_hash}, observed {actual}"
        )
    return load_json(path)


@dataclass(frozen=True)
class RepositoryContext:
    adapter_path: Path
    adapter: dict[str, Any]
    adapter_sha256: str
    grid_manifest: dict[str, Any]
    jobs: tuple[dict[str, Any], ...]
    robust_manifest: dict[str, Any]
    gate_spec: dict[str, Any]

    @property
    def manifest_sha256(self) -> str:
        return self.adapter["robust_gate"]["manifest_canonical_sha256"]

    @property
    def source_bundle_sha256(self) -> str:
        return self.adapter["source_bundle_sha256"]

    @property
    def cases(self) -> tuple[str, ...]:
        return tuple(self.adapter["robust_gate"]["case_ids"])

    def job_by_id(self) -> dict[str, dict[str, Any]]:
        return {job["job_id"]: job for job in self.jobs}


def load_repository(
    adapter_path: str | os.PathLike[str] = DEFAULT_ADAPTER_MANIFEST,
) -> RepositoryContext:
    """Load and re-hash all frozen inputs and execution sources."""
    adapter_path = Path(adapter_path).resolve()
    adapter = load_json(adapter_path)
    if adapter.get("schema_version") != 1:
        raise ValueError("unsupported robust adapter manifest schema")
    if adapter.get("operation") != "fused_softmax":
        raise ValueError("adapter manifest is not for fused_softmax")

    source_hashes = adapter.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("adapter manifest has no source hash map")
    for relative, expected in source_hashes.items():
        path = _repo_file(relative)
        if file_sha256(path) != expected:
            raise ValueError(f"execution source changed after freeze: {relative}")
    if canonical_sha256(source_hashes) != adapter.get("source_bundle_sha256"):
        raise ValueError("adapter source bundle hash is inconsistent")

    grid_info = adapter["grid"]
    grid_path = _repo_file(grid_info["manifest_path"])
    grid_manifest = _load_exact(
        grid_path, grid_info["manifest_sha256"], "fused-grid manifest"
    )
    if canonical_sha256(grid_manifest) != grid_info["manifest_canonical_sha256"]:
        raise ValueError("fused-grid canonical manifest hash mismatch")
    jobs_path = _repo_file(grid_info["jobs_path"])
    jobs_value = _load_exact(jobs_path, grid_info["jobs_sha256"], "fused-grid jobs")
    if not isinstance(jobs_value, list):
        raise ValueError("fused-grid jobs must be a JSON array")
    if len(jobs_value) != grid_info["job_count"]:
        raise ValueError("fused-grid job count changed")
    observed_ids = set()
    for job in jobs_value:
        job_id = job.get("job_id")
        expected = grid_info["job_sha256"].get(job_id)
        if not expected or canonical_sha256(job) != expected:
            raise ValueError(f"grid job hash mismatch: {job_id!r}")
        if job_id in observed_ids:
            raise ValueError(f"duplicate grid job {job_id!r}")
        observed_ids.add(job_id)
        if job.get("variant") != "GBGS" or job.get("geom") != "fused":
            raise ValueError(f"job is not fused GBGS: {job!r}")
    if observed_ids != set(grid_info["job_sha256"]):
        raise ValueError("adapter job hash map and job list differ")

    robust_info = adapter["robust_gate"]
    robust_manifest = _load_exact(
        _repo_file(robust_info["manifest_path"]),
        robust_info["manifest_sha256"],
        "robust-gate manifest",
    )
    gate_spec = _load_exact(
        _repo_file(robust_info["gate_spec_path"]),
        robust_info["gate_spec_sha256"],
        "frozen fused gate spec",
    )
    validate_manifest(robust_manifest)
    validate_gate_spec(gate_spec)
    if canonical_sha256(robust_manifest) != robust_info["manifest_canonical_sha256"]:
        raise ValueError("robust manifest canonical hash mismatch")
    if canonical_sha256(gate_spec) != robust_info["gate_spec_canonical_sha256"]:
        raise ValueError("gate spec canonical hash mismatch")
    if gate_spec["manifest_sha256"] != canonical_sha256(robust_manifest):
        raise ValueError("gate spec does not bind the robust manifest")
    expected_keys = [f"fused_softmax/{gate_id}" for gate_id in GATE_IDS]
    if robust_info["gate_keys"] != expected_keys:
        raise ValueError("adapter gate order/content changed")
    if set(gate_spec["gates"]) != set(expected_keys):
        raise ValueError("frozen gate spec does not contain exactly both mixed gates")
    if robust_info["split_counts"] != {"tuning": 8, "validation": 64}:
        raise ValueError("adapter does not preserve the 8/64 split")

    return RepositoryContext(
        adapter_path=adapter_path,
        adapter=adapter,
        adapter_sha256=file_sha256(adapter_path),
        grid_manifest=grid_manifest,
        jobs=tuple(jobs_value),
        robust_manifest=robust_manifest,
        gate_spec=gate_spec,
    )


def parse_phase2_set(set_string: str) -> dict[str, Any]:
    """Parse the checked-in runner2 set syntax without importing GPU code."""
    parsed: dict[str, Any] = {}
    for part in (set_string or "").split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        key, value = key.strip(), value.strip()
        if key in ("cast", "arith", "algo"):
            parsed[key] = value
        elif key.startswith("x_"):
            parsed.setdefault("extra", {})[key[2:]] = value
        else:
            parsed[key] = int(value)
    return parsed


def select_jobs(
    context: RepositoryContext,
    job_ids: Iterable[str] | None = None,
    *,
    all_jobs: bool = False,
) -> list[dict[str, Any]]:
    if all_jobs and job_ids:
        raise ValueError("choose either explicit jobs or all jobs, not both")
    by_id = context.job_by_id()
    if all_jobs:
        return list(context.jobs)
    requested = list(job_ids or [])
    if not requested:
        raise ValueError("select at least one --job, or pass --all-jobs")
    if len(requested) != len(set(requested)):
        raise ValueError("job selection contains duplicates")
    unknown = [job_id for job_id in requested if job_id not in by_id]
    if unknown:
        raise ValueError(f"unknown grid job(s): {', '.join(unknown)}")
    return [by_id[job_id] for job_id in requested]


def case_by_id(context: RepositoryContext, case_id: str) -> dict[str, Any]:
    for case in context.robust_manifest["operations"]["fused_softmax"]["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def seed_indices(
    context: RepositoryContext,
    split: str,
    seed_start: int = 0,
    max_seeds: int | None = None,
) -> tuple[int, ...]:
    if split not in ("tuning", "validation"):
        raise ValueError("only tuning and validation splits are supported")
    count = context.adapter["robust_gate"]["split_counts"][split]
    if seed_start < 0 or seed_start >= count:
        raise ValueError(f"seed_start must be in [0, {count})")
    if max_seeds is not None and max_seeds <= 0:
        raise ValueError("max_seeds must be positive")
    stop = count if max_seeds is None else min(count, seed_start + max_seeds)
    return tuple(range(seed_start, stop))


def threshold_failures(
    gate: dict[str, Any], metrics: dict[str, float]
) -> list[str]:
    failures = []
    for name, threshold in gate["thresholds"].items():
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{name}=missing/nonfinite")
        elif value > threshold["value"]:
            failures.append(f"{name}={value:.9g} > {threshold['value']:.9g}")
    return failures


@dataclass
class CandidatePlan:
    candidate: str
    job: dict[str, Any] | None
    job_sha256: str | None
    config: dict[str, Any] | None
    build_metadata: dict[str, Any]
    execute: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None
    build_error: str | None = None
    build_traceback: str | None = None

    @property
    def job_id(self) -> str | None:
        return self.job.get("job_id") if self.job else None


def candidate_name(context: RepositoryContext, job: dict[str, Any]) -> str:
    digest = context.adapter["grid"]["job_sha256"][job["job_id"]]
    return f"fused-grid:{job['job_id']}:{digest[:12]}"


def _failed_build_plans(
    context: RepositoryContext,
    jobs: list[dict[str, Any]],
    message: str,
    trace: str,
) -> list[CandidatePlan]:
    return [
        CandidatePlan(
            candidate=candidate_name(context, job),
            job=job,
            job_sha256=context.adapter["grid"]["job_sha256"][job["job_id"]],
            config=None,
            build_metadata={},
            build_error=message,
            build_traceback=trace,
        )
        for job in jobs
    ]


def build_phase2_candidates(
    context: RepositoryContext,
    jobs: list[dict[str, Any]],
    on_result: Callable[[CandidatePlan, int, int], None] | None = None,
) -> list[CandidatePlan]:
    """Build every selected job once; convert every build error into evidence."""
    try:
        if str(PHASE2) not in sys.path:
            sys.path.insert(0, str(PHASE2))
        if str(PHASE1) not in sys.path:
            sys.path.insert(0, str(PHASE1))
        import torch
        import common2
        import variants2

        common2.setup_cuda_env()
    except Exception as exc:  # noqa: BLE001 - retained for every selected job
        return _failed_build_plans(
            context,
            jobs,
            f"Phase2ImportError: {type(exc).__name__}: {exc}",
            traceback.format_exc(),
        )

    expected_shape = context.adapter["robust_gate"]["shape"]
    plans: list[CandidatePlan] = []
    for job in jobs:
        job_hash = context.adapter["grid"]["job_sha256"][job["job_id"]]
        config_dict = None
        started = time.perf_counter()
        try:
            overrides = parse_phase2_set(job["set"])
            cfg = common2.make_fused_config(job["dsl"], job["variant"], **overrides)
            config_dict = cfg.to_dict()
            if (cfg.M, cfg.K, cfg.N) != (
                expected_shape["M"],
                expected_shape["K"],
                expected_shape["N"],
            ):
                raise ValueError("Phase-2 config shape differs from frozen gate shape")
            if cfg.variant != "GBGS" or cfg.arith != "fp16" or cfg.cast != "precast":
                raise ValueError("Phase-2 config left the GBGS/fp16/precast contract")
            if cfg.extra.get("wcache") != "cached" or cfg.extra.get("epilogue") != "smem":
                raise ValueError("Phase-2 config left cached/smem fixed factors")
            built = variants2.build("fused", cfg)
            if built.x_dtype != torch.float16:
                raise ValueError(
                    f"precast grid candidate unexpectedly requires {built.x_dtype}"
                )
            build_wall_s = time.perf_counter() - started
            artifacts = {
                key: value for key, value in built.artifacts.items() if key in ARTIFACT_KEYS
            }
            metadata = {
                "adapter_build_wall_s": build_wall_s,
                "reported_compile_s": built.compile_s,
                "n_kernels": built.n_kernels,
                "notes": built.notes,
                "artifacts": artifacts,
                "x_dtype": str(built.x_dtype),
            }

            def execute(inputs, prepared, *, _built=built):
                # Every selected grid job has the same precast activation.  The
                # half copy is therefore shared safely within this case/seed;
                # each candidate still owns its declared cached weight path.
                if "x_fp16" not in prepared:
                    prepared["x_fp16"] = inputs["x"].half().contiguous()
                return _built.run(
                    prepared["x_fp16"], inputs["weight"], inputs["bias"]
                )

            plan = CandidatePlan(
                    candidate=candidate_name(context, job),
                    job=job,
                    job_sha256=job_hash,
                    config=config_dict,
                    build_metadata=metadata,
                    execute=execute,
                )
            plans.append(plan)
        except Exception as exc:  # noqa: BLE001 - build failure is evidence
            plan = CandidatePlan(
                    candidate=candidate_name(context, job),
                    job=job,
                    job_sha256=job_hash,
                    config=config_dict,
                    build_metadata={
                        "adapter_build_wall_s": time.perf_counter() - started
                    },
                    build_error=f"BuildError: {type(exc).__name__}: {exc}",
                    build_traceback=traceback.format_exc(),
                )
            plans.append(plan)
        if on_result is not None:
            on_result(plan, len(plans), len(jobs))
    return plans


def _sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _record_base(
    context: RepositoryContext,
    plan: CandidatePlan,
    *,
    gate_id: str,
    case_id: str,
    split: str,
    seed_index: int,
    seed_map: dict[str, int],
    shape: dict[str, int],
    device: str,
    shared: dict[str, Any],
) -> dict[str, Any]:
    gate = context.gate_spec["gates"][f"fused_softmax/{gate_id}"]
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "robust_gate_measurement",
        "campaign_id": context.robust_manifest["campaign_id"],
        "manifest_sha256": context.manifest_sha256,
        "op": "fused_softmax",
        "gate_id": gate_id,
        "case_id": case_id,
        "split": split,
        "seed_index": seed_index,
        "tensor_seeds": seed_map,
        "candidate": plan.candidate,
        "role": "candidate",
        "device": device,
        "shape": dict(shape),
        "reference_kind": gate["reference"]["kind"],
        "contract": dict(gate["contract"]),
        "source_sha256": context.source_bundle_sha256,
        "source_bundle_sha256": context.source_bundle_sha256,
        "adapter_manifest_sha256": context.adapter_sha256,
        "grid_manifest_sha256": context.adapter["grid"]["manifest_sha256"],
        "grid_jobs_sha256": context.adapter["grid"]["jobs_sha256"],
        "gate_spec_sha256": context.adapter["robust_gate"]["gate_spec_sha256"],
        "gate_spec_canonical_sha256": context.adapter["robust_gate"][
            "gate_spec_canonical_sha256"
        ],
        "phase2_config": plan.config,
        "build_metadata": plan.build_metadata,
        **shared,
    }
    try:
        import torch

        record["torch_version"] = torch.__version__
    except Exception:  # pragma: no cover - robust modules already require torch
        record["torch_version"] = "unavailable"
    if plan.job is not None:
        record.update(
            {
                "grid_job_id": plan.job["job_id"],
                "grid_job_sha256": plan.job_sha256,
                "grid_job": plan.job,
            }
        )
    else:
        record["adapter_mode"] = "cpu-surrogate"
    return record


def _failure_records(
    context: RepositoryContext,
    plan: CandidatePlan,
    *,
    case_id: str,
    split: str,
    seed_index: int,
    seed_map: dict[str, int],
    shape: dict[str, int],
    device: str,
    error: str,
    trace: str,
    shared: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for gate_id in GATE_IDS:
        record = _record_base(
            context,
            plan,
            gate_id=gate_id,
            case_id=case_id,
            split=split,
            seed_index=seed_index,
            seed_map=seed_map,
            shape=shape,
            device=device,
            shared=shared or {},
        )
        record.update(
            {
                "ok": False,
                "gate_pass": False,
                "error": error,
                "traceback": trace,
                "threshold_failures": [error],
            }
        )
        records.append(record)
    return records


def evaluate_case_seed(
    context: RepositoryContext,
    plans: list[CandidatePlan],
    *,
    case_id: str,
    split: str,
    seed_index: int,
    device: str,
    shape: dict[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any] | None]:
    """Evaluate one case/seed, reusing inputs, references, and candidate output.

    Returns records keyed by candidate name and the live input mapping.  The
    caller deliberately retains that mapping until the next seed is allocated;
    this prevents allocator pointer reuse from fooling Phase-2's address-keyed
    weight cache.
    """
    op = "fused_softmax"
    op_spec = context.robust_manifest["operations"][op]
    shape = dict(shape or op_spec["shape"])
    case = case_by_id(context, case_id)
    seed_map = tensor_seeds(context.robust_manifest, op, case_id, split, seed_index)
    result: dict[str, list[dict[str, Any]]] = {}

    runnable = []
    for plan in plans:
        if plan.build_error:
            result[plan.candidate] = _failure_records(
                context,
                plan,
                case_id=case_id,
                split=split,
                seed_index=seed_index,
                seed_map=seed_map,
                shape=shape,
                device=device,
                error=plan.build_error,
                trace=plan.build_traceback or "",
            )
        else:
            runnable.append(plan)
    if not runnable:
        return result, None

    setup_started = time.perf_counter()
    try:
        inputs = make_inputs(op, shape, case, seed_map, device=device)
        _sync_if_cuda(device)
    except Exception as exc:  # noqa: BLE001 - repeated as retained evidence
        error = f"InputSetupError: {type(exc).__name__}: {exc}"
        trace = traceback.format_exc()
        shared = {"shared_input_wall_s": time.perf_counter() - setup_started}
        for plan in runnable:
            result[plan.candidate] = _failure_records(
                context,
                plan,
                case_id=case_id,
                split=split,
                seed_index=seed_index,
                seed_map=seed_map,
                shape=shape,
                device=device,
                error=error,
                trace=trace,
                shared=shared,
            )
        return result, None
    input_wall_s = time.perf_counter() - setup_started

    reference_cache: dict[tuple[str, str], Any] = {}
    reference_errors: dict[tuple[str, str], tuple[str, str]] = {}
    reference_wall: dict[str, float] = {}
    for gate_id in GATE_IDS:
        gate = context.gate_spec["gates"][f"{op}/{gate_id}"]
        kind = gate["reference"]["kind"]
        cache_key = (kind, canonical_sha256(gate["contract"]))
        if cache_key in reference_cache or cache_key in reference_errors:
            continue
        started = time.perf_counter()
        try:
            reference_cache[cache_key] = resolve_output(
                kind, op, inputs, gate["contract"]
            )
            _sync_if_cuda(device)
        except Exception as exc:  # noqa: BLE001 - per-gate failure evidence
            reference_errors[cache_key] = (
                f"ReferenceError: {type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )
        reference_wall[kind] = time.perf_counter() - started

    shared = {
        "shared_input_wall_s": input_wall_s,
        "shared_reference_wall_s": reference_wall,
    }
    prepared: dict[str, Any] = {}
    for plan in runnable:
        started = time.perf_counter()
        try:
            assert plan.execute is not None
            output = plan.execute(inputs, prepared)
            _sync_if_cuda(device)
            run_wall_s = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001 - candidate failure is evidence
            result[plan.candidate] = _failure_records(
                context,
                plan,
                case_id=case_id,
                split=split,
                seed_index=seed_index,
                seed_map=seed_map,
                shape=shape,
                device=device,
                error=f"CandidateError: {type(exc).__name__}: {exc}",
                trace=traceback.format_exc(),
                shared={**shared, "candidate_wall_s": time.perf_counter() - started},
            )
            continue

        records = []
        for gate_id in GATE_IDS:
            gate = context.gate_spec["gates"][f"{op}/{gate_id}"]
            kind = gate["reference"]["kind"]
            cache_key = (kind, canonical_sha256(gate["contract"]))
            base = _record_base(
                context,
                plan,
                gate_id=gate_id,
                case_id=case_id,
                split=split,
                seed_index=seed_index,
                seed_map=seed_map,
                shape=shape,
                device=device,
                shared={**shared, "candidate_wall_s": run_wall_s},
            )
            base["output_dtype"] = str(getattr(output, "dtype", type(output).__name__))
            base["output_shape"] = list(getattr(output, "shape", ()))
            if cache_key in reference_errors:
                error, trace = reference_errors[cache_key]
                base.update(
                    {
                        "ok": False,
                        "gate_pass": False,
                        "error": error,
                        "traceback": trace,
                        "threshold_failures": [error],
                    }
                )
            else:
                metric_started = time.perf_counter()
                try:
                    metrics = compute_metrics(op, reference_cache[cache_key], output, inputs)
                    failures = threshold_failures(gate, metrics)
                    base.update(
                        {
                            "ok": True,
                            "metrics": metrics,
                            "gate_pass": not failures,
                            "threshold_failures": failures,
                            "metric_wall_s": time.perf_counter() - metric_started,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - metric failure is evidence
                    error = f"MetricError: {type(exc).__name__}: {exc}"
                    base.update(
                        {
                            "ok": False,
                            "gate_pass": False,
                            "error": error,
                            "traceback": traceback.format_exc(),
                            "threshold_failures": [error],
                            "metric_wall_s": time.perf_counter() - metric_started,
                        }
                    )
            records.append(base)
        result[plan.candidate] = records
        del output
    return result, inputs


def cpu_smoke_records(
    context: RepositoryContext,
    *,
    case_ids: Iterable[str] | None = None,
    seed_index: int = 0,
) -> list[dict[str, Any]]:
    """Exercise both frozen gates with a CPU mixed-arithmetic surrogate."""
    contract = context.gate_spec["gates"][
        "fused_softmax/conformance_mixed"
    ]["contract"]

    def execute(inputs, _prepared):
        return native_mixed_reference("fused_softmax", inputs, contract)

    plan = CandidatePlan(
        candidate="cpu-smoke:native_mixed",
        job=None,
        job_sha256=None,
        config=None,
        build_metadata={"surrogate": "native_mixed_reference"},
        execute=execute,
    )
    shape = context.robust_manifest["operations"]["fused_softmax"]["cpu_test_shape"]
    records = []
    for case_id in tuple(case_ids or context.cases):
        rows, _inputs = evaluate_case_seed(
            context,
            [plan],
            case_id=case_id,
            split="tuning",
            seed_index=seed_index,
            device="cpu",
            shape=shape,
        )
        records.extend(rows[plan.candidate])
    return records


def _launch_coverage(
    records: list[dict[str, Any]],
    candidates: Iterable[str],
    case_ids: Iterable[str],
    indices: Iterable[int],
) -> dict[str, Any]:
    candidates, case_ids, indices = tuple(candidates), tuple(case_ids), tuple(indices)
    expected = {
        (candidate, gate_id, case_id, index)
        for candidate in candidates
        for gate_id in GATE_IDS
        for case_id in case_ids
        for index in indices
    }
    observed = {
        (row.get("candidate"), row.get("gate_id"), row.get("case_id"), row.get("seed_index"))
        for row in records
    }
    duplicates = len(records) - len(observed)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    return {
        "expected_records": len(expected),
        "observed_records": len(records),
        "missing_records": len(missing),
        "unexpected_records": len(unexpected),
        "duplicate_records": duplicates,
        "complete": not missing and not unexpected and duplicates == 0,
        "missing_examples": [list(value) for value in missing[:20]],
        "unexpected_examples": [list(value) for value in unexpected[:20]],
    }


def _summary_provenance(context: RepositoryContext) -> dict[str, str]:
    return {
        "adapter_manifest_sha256": context.adapter_sha256,
        "source_bundle_sha256": context.source_bundle_sha256,
        "grid_manifest_sha256": context.adapter["grid"]["manifest_sha256"],
        "grid_jobs_sha256": context.adapter["grid"]["jobs_sha256"],
        "robust_manifest_sha256": context.manifest_sha256,
        "gate_spec_sha256": context.adapter["robust_gate"]["gate_spec_sha256"],
        "gate_spec_canonical_sha256": context.adapter["robust_gate"][
            "gate_spec_canonical_sha256"
        ],
    }


def _candidate_job_binding(
    context: RepositoryContext, candidate: str
) -> dict[str, str]:
    for job in context.jobs:
        if candidate_name(context, job) == candidate:
            return {
                "grid_job_id": job["job_id"],
                "grid_job_sha256": context.adapter["grid"]["job_sha256"][
                    job["job_id"]
                ],
            }
    return {}


def summarize_tuning(
    context: RepositoryContext,
    records: list[dict[str, Any]],
    *,
    candidates: Iterable[str],
    case_ids: Iterable[str],
    indices: Iterable[int],
) -> dict[str, Any]:
    candidates, case_ids, indices = tuple(candidates), tuple(case_ids), tuple(indices)
    coverage = _launch_coverage(records, candidates, case_ids, indices)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("split") != "tuning":
            raise SchemaError("tuning summary received a non-tuning record")
        grouped[(row["candidate"], row["gate_id"])].append(row)
    groups = []
    for candidate in candidates:
        for gate_id in GATE_IDS:
            rows = grouped[(candidate, gate_id)]
            gate = context.gate_spec["gates"][f"fused_softmax/{gate_id}"]
            failures = [
                row
                for row in rows
                if (
                    not row.get("ok", False)
                    or threshold_failures(gate, row.get("metrics", {}))
                    or not row.get("gate_pass", False)
                )
            ]
            groups.append(
                {
                    "candidate": candidate,
                    "gate_id": gate_id,
                    **_candidate_job_binding(context, candidate),
                    "n_records": len(rows),
                    "n_failed_records": len(failures),
                    "selected_coverage_complete": len(rows) == len(case_ids) * len(indices),
                    "selected_success": len(rows) == len(case_ids) * len(indices) and not failures,
                }
            )
    full_coverage = set(case_ids) == set(context.cases) and set(indices) == set(range(8))
    selected_success = coverage["complete"] and all(g["selected_success"] for g in groups)
    if full_coverage:
        status = "PASS" if selected_success else "FAIL"
    else:
        status = "SCREEN_PASS" if selected_success else "SCREEN_FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": context.robust_manifest["campaign_id"],
        "split": "tuning",
        "status": status,
        "success": full_coverage and selected_success,
        "selected_success": selected_success,
        "full_tuning_coverage": full_coverage,
        "launch_coverage": coverage,
        "groups": groups,
        **_summary_provenance(context),
    }


def summarize_validation(
    context: RepositoryContext,
    records: list[dict[str, Any]],
    *,
    candidates: Iterable[str],
    case_ids: Iterable[str],
    indices: Iterable[int],
) -> dict[str, Any]:
    candidates = tuple(candidates)
    coverage = _launch_coverage(records, candidates, case_ids, indices)
    if records:
        summary = validate_records(
            context.robust_manifest,
            context.gate_spec,
            records,
            allow_incomplete=False,
        )
    else:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": context.robust_manifest["campaign_id"],
            "manifest_sha256": context.manifest_sha256,
            "gate_spec_sha256": canonical_sha256(context.gate_spec),
            "success_rule": "all_metrics_all_cases_all_seeds_and_complete_coverage",
            "success": False,
            "groups": [],
            "failures": [],
        }
    observed_candidates = {row.get("candidate") for row in records}
    absent_candidates = sorted(set(candidates) - observed_candidates)
    for group in summary["groups"]:
        group.update(_candidate_job_binding(context, group["candidate"]))
    summary["launch_coverage"] = coverage
    summary["absent_candidates"] = absent_candidates
    summary.update(_summary_provenance(context))
    summary["success"] = bool(
        summary.get("success") and coverage["complete"] and not absent_candidates
    )
    summary["status"] = "PASS" if summary["success"] else "FAIL"
    return summary
