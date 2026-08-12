#!/usr/bin/env python3
"""Tag-scoped, performance-blind admission for transfer artifacts."""
from __future__ import annotations

import hashlib
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import protocol


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".llir", ".ptx", ".source", ".ttgir", ".ttir"}
_ENTRY_ID = re.compile(r"tt2_[0-9a-f]{24}")


def admission_plan(manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    manifest = manifest or protocol.make_admission_manifest()
    rows = manifest.get("rows") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or len(rows) != 160:
        raise protocol.ProtocolError("artifact admission requires the frozen 160-row manifest")
    if len({row.get("entry_id") for row in rows if isinstance(row, dict)}) != len(rows):
        raise protocol.ProtocolError("artifact admission entry IDs are not unique")
    return rows


def entry_paths(entry_id: str, root: Path) -> dict[str, Path]:
    if _ENTRY_ID.fullmatch(entry_id) is None:
        raise protocol.ProtocolError(f"unsafe artifact entry ID: {entry_id!r}")
    root = root.resolve()
    entry = root / "artifacts" / entry_id
    return {
        "root": entry,
        "cache": entry / "cache",
        "tmp": entry / "runtime_tmp",
        "gate": entry / "gate.jsonl",
        "failure": entry / "failure.json",
        "build": entry / "build_record.json",
        "verify": entry / "verify_record.json",
        "entry": entry / "entry.json",
    }


def cache_environment(entry_id: str, mode: str, root: Path) -> dict[str, str]:
    if mode not in {"admit", "load_only"}:
        raise protocol.ProtocolError(f"unknown artifact mode: {mode}")
    paths = entry_paths(entry_id, root)
    cache, runtime_tmp = paths["cache"].resolve(), paths["tmp"].resolve()
    value = {
        "TRAJECTORY_TRANSFER_ARTIFACT_MODE": mode,
        "TRAJECTORY_TRANSFER_ARTIFACT_ENTRY": entry_id,
        "PHASE1_TL_CACHE": "1",
        "PHASE2_TL_CACHE": "1",
        "TILELANG_CACHE_DIR": str(cache / "tilelang"),
        "TILELANG_EXECUTION_BACKEND": "tvm_ffi",
        "TILELANG_TMP_DIR": str(runtime_tmp / "tilelang"),
        "TILELANG_TARGET": "cuda",
        "TILELANG_DISABLE_CACHE": "0",
        "TILELANG_CLEAR_CACHE": "0",
        "TRITON_ALWAYS_COMPILE": "0",
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TRITON_KERNEL_DUMP": "0",
        "TRITON_KERNEL_OVERRIDE": "0",
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
    }
    if mode == "admit":
        value.update({name: str(runtime_tmp) for name in ("TMPDIR", "TMP", "TEMP")})
    if mode == "load_only":
        value["TRITON_CACHE_MANAGER"] = (
            "ako_runs.controlled_followup.native_trajectory_replication_ada_v3.artifacts:"
            "ReadOnlyTritonCacheManager"
        )
    return value


def prepare_cache_environment(entry_id: str, mode: str, root: Path) -> dict[str, str]:
    paths = entry_paths(entry_id, root)
    if mode == "admit":
        if paths["root"].exists():
            raise FileExistsError(f"refusing existing artifact root: {paths['root']}")
        paths["cache"].mkdir(parents=True)
        paths["tmp"].mkdir()
        (paths["tmp"] / "tilelang").mkdir()
    elif mode == "load_only":
        if not all(path.is_dir() for path in (paths["cache"], paths["tmp"], paths["tmp"] / "tilelang")):
            raise protocol.ProtocolError("load-only mode requires the admitted cache/temp tree")
    else:
        raise protocol.ProtocolError(f"unknown artifact mode: {mode}")
    if paths["cache"].is_symlink() or paths["tmp"].is_symlink():
        raise protocol.ProtocolError("artifact cache/temp root may not be a symlink")
    if paths["cache"].stat().st_dev != paths["tmp"].stat().st_dev:
        raise protocol.ProtocolError("artifact cache and atomic temp root are on different filesystems")
    value = cache_environment(entry_id, mode, root)
    os.environ.update(value)
    return value


def validate_cache_environment(entry_id: str, mode: str, root: Path) -> None:
    mismatch = {
        key: (os.environ.get(key), expected)
        for key, expected in cache_environment(entry_id, mode, root).items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise protocol.ProtocolError(f"artifact cache environment changed: {mismatch}")


def _is_code_object(path: Path) -> bool:
    return path.suffix == ".cubin" or path.name.endswith(".so")


def cache_snapshot(entry_id: str, root: Path) -> dict[str, Any]:
    cache = entry_paths(entry_id, root)["cache"]
    if not cache.is_dir() or cache.is_symlink():
        raise protocol.ProtocolError("admitted cache is missing or unsafe")
    files = []
    for path in sorted(cache.rglob("*"), key=lambda item: item.relative_to(cache).as_posix()):
        if path.is_symlink():
            raise protocol.ProtocolError(f"admitted cache contains a symlink: {path}")
        if not path.is_file():
            continue
        if path.stat().st_size <= 0:
            raise protocol.ProtocolError(f"admitted cache contains an empty file: {path}")
        if path.suffix in SOURCE_SUFFIXES:
            source = path.read_text(encoding="utf-8", errors="replace")
            if "<<<" in source and '#include "checked_cuda_launch.h"' not in source:
                raise protocol.ProtocolError(f"generated CUDA wrapper lacks checked launches: {path}")
        files.append({
            "path": path.relative_to(cache).as_posix(),
            "sha256": protocol.file_sha256(path),
            "size": path.stat().st_size,
        })
    sources = [row for row in files if Path(row["path"]).suffix in SOURCE_SUFFIXES]
    objects = [row for row in files if _is_code_object(Path(row["path"]))]
    if not sources or not objects:
        raise protocol.ProtocolError("admission must retain generated source and a loadable object")
    return {
        "cache_root": str(cache.resolve().relative_to(protocol.REPO_ROOT.resolve())),
        "file_count": len(files),
        "files": files,
        "files_sha256": protocol.canonical_sha256(files),
        "generated_sources": sources,
        "loadable_code_objects": objects,
    }


def runtime_tmp_snapshot(entry_id: str, root: Path) -> dict[str, Any]:
    runtime_tmp = entry_paths(entry_id, root)["tmp"]
    if not runtime_tmp.is_dir() or runtime_tmp.is_symlink():
        raise protocol.ProtocolError("artifact runtime temp root is missing or unsafe")
    rows = []
    for path in sorted(runtime_tmp.rglob("*"), key=lambda item: item.relative_to(runtime_tmp).as_posix()):
        if path.is_symlink():
            raise protocol.ProtocolError(f"artifact runtime temp contains a symlink: {path}")
        rows.append({
            "kind": "directory" if path.is_dir() else "file",
            "path": path.relative_to(runtime_tmp).as_posix(),
            **({
                "sha256": protocol.file_sha256(path),
                "size": path.stat().st_size,
            } if path.is_file() else {}),
        })
    return {
        "root": str(runtime_tmp.resolve().relative_to(protocol.REPO_ROOT.resolve())),
        "rows": rows,
        "rows_sha256": protocol.canonical_sha256(rows),
    }


def _runtime_tmp_from_build(build: dict[str, Any]) -> dict[str, Any]:
    expected = build.get("runtime_tmp")
    if not isinstance(expected, dict) or not isinstance(expected.get("root"), str):
        raise protocol.ProtocolError("artifact build record lacks runtime-temp binding")
    root = (protocol.REPO_ROOT / expected["root"]).resolve()
    try:
        root.relative_to(protocol.REPO_ROOT.resolve())
    except ValueError as exc:
        raise protocol.ProtocolError("artifact runtime temp escapes the repository") from exc
    if not root.is_dir() or root.is_symlink():
        raise protocol.ProtocolError("artifact runtime temp root is missing or unsafe")
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise protocol.ProtocolError(f"artifact runtime temp contains a symlink: {path}")
        rows.append({
            "kind": "directory" if path.is_dir() else "file",
            "path": path.relative_to(root).as_posix(),
            **({"sha256": protocol.file_sha256(path), "size": path.stat().st_size} if path.is_file() else {}),
        })
    observed = {"root": expected["root"], "rows": rows, "rows_sha256": protocol.canonical_sha256(rows)}
    if observed != expected:
        raise protocol.ProtocolError("artifact runtime temp changed after admission")
    return observed


def generated_artifact_identity(cache: dict[str, Any]) -> str:
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3 import artifacts as native

    return native.generated_artifact_identity(cache)


@contextmanager
def capture_torch_extensions() -> Iterator[list[dict[str, Any]]]:
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3 import artifacts as native

    with native.capture_torch_extensions() as requests:
        yield requests


def _source_cell(row: dict[str, Any]) -> dict[str, Any]:
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core

    matches = [cell for cell in core.load_cells(require_resolved=True) if cell["cell_id"] == row.get("coordinate_cell_id")]
    if len(matches) != 1:
        raise protocol.ProtocolError("admission row has no unique resolved source cell")
    return matches[0]


def gate_built(built: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Run all 512 frozen validation records for one already-built artifact."""
    from ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.audit import fixed_gate_summary
    from ako_runs.controlled_followup.fused_grid import robust_adapter

    cell = _source_cell(row)
    context = robust_adapter.load_repository()

    def execute(inputs, prepared):
        if "x_fp16" not in prepared:
            prepared["x_fp16"] = inputs["x"].half().contiguous()
        return built.run(prepared["x_fp16"], inputs["weight"], inputs["bias"])

    plan = robust_adapter.CandidatePlan(
        candidate=f"trajectory-transfer:{row['entry_id']}",
        job=cell["origin_job"],
        job_sha256=cell["origin_job_sha256"],
        config=built.config,
        build_metadata={"n_kernels": built.metadata.get("n_kernels")},
        execute=execute,
    )
    records: list[dict[str, Any]] = []
    live_inputs = None
    for case_id in context.adapter["robust_gate"]["case_ids"]:
        for seed_index in range(64):
            prior = live_inputs
            evaluated, live_inputs = robust_adapter.evaluate_case_seed(
                context, [plan], case_id=case_id, split="validation",
                seed_index=seed_index, device="cuda:0",
            )
            if prior is not None:
                del prior
            records.extend(evaluated[plan.candidate])
    summary = fixed_gate_summary(context, records)
    return {"records": records, "summary": summary}


def quick_gate_built(built: Any) -> dict[str, Any]:
    """Fresh load-only launch check; the build receipt retains the full gate."""
    import torch

    from ako_runs.phase2_fused_sdpa import common2
    from ako_runs.phase1_matmul import common

    x, weight, bias = common2.fused_inputs(seed=0, dist="rand")
    with torch.no_grad():
        reference = common2.fused_reference(x, weight, bias, arm="GBGS", dtype=torch.float32)
        observed = built.run(x.half().contiguous(), weight, bias)
        torch.cuda.synchronize()
    result = common.gate_stats(reference, observed.float())
    if result.get("gate_pass") is not True:
        raise protocol.ProtocolError("load-only artifact failed its fresh launch check")
    return result


def dynamic_work_audit(built: Any) -> dict[str, Any]:
    """Observe every CUDA device event from exactly one fresh call."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from ako_runs.phase2_fused_sdpa import common2

    x, weight, bias = common2.fused_inputs(seed=0, dist="rand")
    x16 = x.half().contiguous()
    with torch.no_grad():
        warmup = built.run(x16, weight, bias)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(warmup).all().item()):
            raise protocol.ProtocolError("dynamic-work audit warmup produced nonfinite output")
        torch.cuda.synchronize()
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as observed:
        output = built.run(x16, weight, bias)
        torch.cuda.synchronize()
    names = []
    for event in observed.events():
        device = str(getattr(event, "device_type", "")).upper()
        name = str(getattr(event, "name", ""))
        if device.endswith("CUDA") and name:
            names.append(name)
    value = {
        "schema_version": 1,
        "method": "torch_profiler_cuda_activity_v1",
        "profiled_calls": 1,
        "input": {"seed": 0, "distribution": "positive"},
        "expected_cuda_device_event_count": 2,
        "observed_cuda_device_event_count": len(names),
        "cuda_device_event_names": names,
        "cuda_device_event_names_sha256": protocol.canonical_sha256(names),
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "all_finite": bool(torch.isfinite(output).all().item()),
        },
        "passed": len(names) == 2,
        "performance_observations": [],
    }
    validate_dynamic_work_audit(value)
    return value


def validate_dynamic_work_audit(value: Any) -> dict[str, Any]:
    names = value.get("cuda_device_event_names") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != {
            "schema_version", "method", "profiled_calls", "input",
            "expected_cuda_device_event_count", "observed_cuda_device_event_count",
            "cuda_device_event_names", "cuda_device_event_names_sha256", "output",
            "passed", "performance_observations",
        }
        or value.get("schema_version") != 1
        or value.get("method") != "torch_profiler_cuda_activity_v1"
        or value.get("profiled_calls") != 1
        or value.get("input") != {"seed": 0, "distribution": "positive"}
        or value.get("expected_cuda_device_event_count") != 2
        or value.get("observed_cuda_device_event_count") != 2
        or not isinstance(names, list)
        or len(names) != 2
        or any(not isinstance(name, str) or not name for name in names)
        or value.get("cuda_device_event_names_sha256") != protocol.canonical_sha256(names)
        or value.get("output")
        != {"shape": [1024, 8192], "dtype": "torch.float32", "all_finite": True}
        or value.get("passed") is not True
        or value.get("performance_observations") != []
    ):
        raise protocol.ProtocolError("fresh dynamic-work audit failed or is malformed")
    return value


def _clean_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_metadata(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def build_record(
    row: dict[str, Any], built: Any, gate_summary: dict[str, Any],
    gate_path: str, gate_sha256: str, provenance: dict[str, Any],
    torch_requests: list[dict[str, Any]], dynamic_audit: dict[str, Any], root: Path,
) -> dict[str, Any]:
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3 import artifacts as native

    snapshot = cache_snapshot(row["entry_id"], root)
    return {
        "schema_version": 1,
        "record_type": "trajectory_transfer_ada_v2_artifact_build_record",
        "campaign_id": protocol.CAMPAIGN_ID,
        "entry_id": row["entry_id"],
        "admission_row": row,
        "admission_row_sha256": protocol.canonical_sha256(row),
        "artifact_identity_sha256": generated_artifact_identity(snapshot),
        "cache": snapshot,
        "config": built.config,
        "dynamic_work_audit": validate_dynamic_work_audit(dynamic_audit),
        "gate_jsonl_path": gate_path,
        "gate_jsonl_sha256": gate_sha256,
        "gate_summary": gate_summary,
        "n_kernels": built.metadata.get("n_kernels"),
        "performance_observations": [],
        "provenance": provenance,
        "runtime_tmp": runtime_tmp_snapshot(row["entry_id"], root),
        "source_build_metadata": _clean_metadata(built.metadata),
        "torch_inline_requests": native.bind_torch_requests(torch_requests, snapshot),
    }


def validate_build_record(value: Any, row: dict[str, Any], root: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise protocol.ProtocolError(f"invalid performance-blind build record: {row['entry_id']}")
    validate_dynamic_work_audit(value.get("dynamic_work_audit"))
    if (
        value.get("record_type") != "trajectory_transfer_ada_v2_artifact_build_record"
        or value.get("campaign_id") != protocol.CAMPAIGN_ID
        or value.get("entry_id") != row["entry_id"]
        or value.get("admission_row") != row
        or value.get("admission_row_sha256") != protocol.canonical_sha256(row)
        or value.get("n_kernels") != 2
        or value.get("performance_observations") != []
        or value.get("gate_summary", {}).get("full_gate_pass") is not True
        or value.get("gate_summary", {}).get("observed_records") != 512
        or value.get("cache") != cache_snapshot(row["entry_id"], root)
        or value.get("runtime_tmp") != runtime_tmp_snapshot(row["entry_id"], root)
        or value.get("artifact_identity_sha256") != generated_artifact_identity(value.get("cache", {}))
    ):
        raise protocol.ProtocolError(f"invalid performance-blind build record: {row['entry_id']}")
    gate_path = protocol.REPO_ROOT / str(value.get("gate_jsonl_path", ""))
    if not gate_path.is_file() or protocol.file_sha256(gate_path) != value.get("gate_jsonl_sha256"):
        raise protocol.ProtocolError("artifact gate evidence changed")
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3 import artifacts as native
    native.validate_torch_requests(value.get("torch_inline_requests"), value["cache"])
    return value


def _validate_gate_record_bindings(
    context: Any, records: list[dict[str, Any]], build: dict[str, Any], row: dict[str, Any],
) -> None:
    expected_shape = context.adapter["robust_gate"]["shape"]
    expected_job_sha = context.adapter["grid"]["job_sha256"][row["origin_job"]["job_id"]]
    candidate = f"trajectory-transfer:{row['entry_id']}"
    for record in records:
        if (
            record.get("candidate") != candidate
            or record.get("split") != "validation"
            or record.get("device") != "cuda:0"
            or record.get("shape") != expected_shape
            or record.get("phase2_config") != build.get("config")
            or record.get("build_metadata") != {"n_kernels": 2}
            or record.get("grid_job") != row["origin_job"]
            or record.get("grid_job_sha256") != expected_job_sha
            or record.get("grid_job_id") != row["origin_job"]["job_id"]
            or record.get("campaign_id") != context.robust_manifest["campaign_id"]
            or record.get("manifest_sha256") != context.manifest_sha256
            or record.get("source_sha256") != context.source_bundle_sha256
            or record.get("source_bundle_sha256") != context.source_bundle_sha256
            or record.get("adapter_manifest_sha256") != context.adapter_sha256
            or record.get("gate_spec_sha256")
            != context.adapter["robust_gate"]["gate_spec_sha256"]
        ):
            raise protocol.ProtocolError("artifact gate record is transplanted or foreign")


def validate_full_gate(
    build: dict[str, Any], row: dict[str, Any], context: Any | None = None,
) -> None:
    """Parse and independently summarize the retained 512-row JSONL gate."""
    import json

    from ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1.audit import (
        fixed_gate_summary,
    )
    from ako_runs.controlled_followup.fused_grid import robust_adapter

    path = protocol.REPO_ROOT / str(build.get("gate_jsonl_path", ""))
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise protocol.ProtocolError("artifact gate JSONL is unreadable") from exc
    if len(records) != 512 or not all(isinstance(record, dict) for record in records):
        raise protocol.ProtocolError("artifact gate JSONL does not contain exactly 512 records")
    try:
        context = context or robust_adapter.load_repository()
        summary = fixed_gate_summary(context, records)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise protocol.ProtocolError("artifact gate JSONL cannot be independently summarized") from exc
    if summary != build.get("gate_summary") or summary.get("full_gate_pass") is not True:
        raise protocol.ProtocolError("artifact gate summary failed independent rederivation")
    _validate_gate_record_bindings(context, records, build, row)


@contextmanager
def load_only_guards(build: dict[str, Any]) -> Iterator[dict[str, Any]]:
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3 import artifacts as native

    compatible = {**build, "cell_id": build["admission_row"]["coordinate_cell_id"]}
    _runtime_tmp_from_build(build)
    with native.load_only_guards(compatible) as evidence:
        yield evidence
    _runtime_tmp_from_build(build)


def validate_load_evidence(value: Any, row: dict[str, Any], build: dict[str, Any]) -> dict[str, Any]:
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3 import artifacts as native

    compatible = {**build, "cell_id": row["coordinate_cell_id"]}
    return native.validate_load_evidence(value, _source_cell(row), compatible)


def verify_record(
    row: dict[str, Any], build: dict[str, Any], built: Any,
    live_gate: dict[str, Any], load_evidence: dict[str, Any],
    provenance: dict[str, Any], root: Path,
) -> dict[str, Any]:
    snapshot = cache_snapshot(row["entry_id"], root)
    identity = generated_artifact_identity(snapshot)
    if identity != build["artifact_identity_sha256"] or built.config != build["config"]:
        raise protocol.ProtocolError("load-only verification resolved to another artifact")
    validate_load_evidence(load_evidence, row, build)
    return {
        "schema_version": 1,
        "record_type": "trajectory_transfer_ada_v2_artifact_verify_record",
        "campaign_id": protocol.CAMPAIGN_ID,
        "entry_id": row["entry_id"],
        "admission_row_sha256": protocol.canonical_sha256(row),
        "artifact_identity_sha256": identity,
        "build_record_sha256": protocol.file_sha256(entry_paths(row["entry_id"], root)["build"]),
        "cache": snapshot,
        "config": built.config,
        "build_gate_jsonl_sha256": build["gate_jsonl_sha256"],
        "dynamic_work_audit": build["dynamic_work_audit"],
        "live_gate": live_gate,
        "load_evidence": load_evidence,
        "n_kernels": built.metadata.get("n_kernels"),
        "performance_observations": [],
        "provenance": provenance,
        "runtime_tmp": _runtime_tmp_from_build(build),
    }


def validate_verify_record(
    value: Any, row: dict[str, Any], build: dict[str, Any], root: Path,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("record_type") != "trajectory_transfer_ada_v2_artifact_verify_record"
        or value.get("campaign_id") != protocol.CAMPAIGN_ID
        or value.get("entry_id") != row["entry_id"]
        or value.get("admission_row_sha256") != protocol.canonical_sha256(row)
        or value.get("build_record_sha256") != protocol.file_sha256(entry_paths(row["entry_id"], root)["build"])
        or value.get("cache") != build["cache"]
        or value.get("artifact_identity_sha256") != build["artifact_identity_sha256"]
        or value.get("config") != build["config"]
        or value.get("n_kernels") != 2
        or value.get("performance_observations") != []
        or value.get("build_gate_jsonl_sha256") != build.get("gate_jsonl_sha256")
        or value.get("dynamic_work_audit") != build.get("dynamic_work_audit")
        or value.get("live_gate", {}).get("gate_pass") is not True
        or value.get("cache") != cache_snapshot(row["entry_id"], root)
        or value.get("runtime_tmp") != build.get("runtime_tmp")
    ):
        raise protocol.ProtocolError(f"invalid load-only verification record: {row['entry_id']}")
    validate_load_evidence(value.get("load_evidence"), row, build)
    return value
