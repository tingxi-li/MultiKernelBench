#!/usr/bin/env python3
"""Performance-blind admission and load-only artifact helpers."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    from . import protocol
except ImportError:
    import protocol  # type: ignore


ADMISSION_ROOT = protocol.RESULTS_ROOT / "artifact_admission"
ARTIFACTS_ROOT = ADMISSION_ROOT / "artifacts"
MANIFEST_PATH = ADMISSION_ROOT / "manifest.json"
INCIDENT_PATH = protocol.PREDECESSOR_INCIDENT
INCIDENT_SHA256 = protocol.PREDECESSOR_INCIDENT_SHA256
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".llir", ".ptx", ".source", ".ttgir", ".ttir"}
SIDE_COMPILE_DIAGNOSTIC_KEYS = (
    "ptxas_log", "n_regs", "n_spills", "stack_frame_bytes", "static_smem_bytes",
    "sass_ffma_count", "sass_hmma_count", "sass_imma_count", "sass_ldgsts_count",
    "kernel_resources",
)


class ReadOnlyTritonCacheManager:
    """Resolved lazily by Triton; replaced with a real subclass below."""


def _readonly_triton_cache_manager_class():
    from triton import knobs
    from triton.runtime.cache import FileCacheManager

    class _ReadOnlyTritonCacheManager(FileCacheManager):
        def __init__(self, key, override=False, dump=False):
            if override or dump:
                raise protocol.ProtocolError("Triton override/dump caches are forbidden")
            self.key = key
            self.cache_dir = os.path.join(knobs.cache.dir, key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            if not os.path.isdir(self.cache_dir):
                raise protocol.ProtocolError(f"Triton admitted cache miss: {key}")

        def put(self, *_args, **_kwargs):
            raise protocol.ProtocolError("Triton cache writes are forbidden")

        def put_group(self, *_args, **_kwargs):
            raise protocol.ProtocolError("Triton cache-group writes are forbidden")

    _ReadOnlyTritonCacheManager.__name__ = "ReadOnlyTritonCacheManager"
    return _ReadOnlyTritonCacheManager


# Triton's environment loader requires the configured class itself to inherit
# CacheManager. The class reads the cache root only when Triton constructs it,
# after the child process has installed its entry-specific environment.
ReadOnlyTritonCacheManager = _readonly_triton_cache_manager_class()


def admission_plan(contract: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    contract = contract or protocol.load_contract()
    rows = [
        {
            "cell_id": material["cell_id"],
            "entry_id": material["cell_id"],
            "role": "shared_sham_base" if material["cell_id"] == protocol.SHAM_CELL_ID else "native_cell",
        }
        for material in contract["materials"]["selected_prefixes"]
    ]
    for position, row in enumerate(rows):
        row["position"] = position
    if (
        len(rows) != 12
        or len({row["cell_id"] for row in rows}) != 12
        or sum(row["role"] == "shared_sham_base" for row in rows) != 1
    ):
        raise protocol.ProtocolError("artifact admission must contain the exact 12 native g01 cells")
    return rows


def entry_slug(cell_id: str) -> str:
    if re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+\.g[0-9]{2}", cell_id) is None:
        raise protocol.ProtocolError(f"unsafe admitted cell ID: {cell_id!r}")
    return cell_id.replace(".", "__")


def entry_root(cell_id: str) -> Path:
    return ARTIFACTS_ROOT / entry_slug(cell_id)


def entry_paths(cell_id: str) -> dict[str, Path]:
    root = entry_root(cell_id)
    return {
        "root": root,
        "cache": root / "cache",
        "tmp": root / "runtime_tmp",
        "build": root / "build_record.json",
        "verify": root / "verify_record.json",
        "entry": root / "entry.json",
    }


def cache_environment(cell_id: str, mode: str) -> dict[str, str]:
    if mode not in {"admit", "load_only"}:
        raise protocol.ProtocolError(f"unknown artifact mode: {mode}")
    cache = entry_paths(cell_id)["cache"].resolve()
    runtime_tmp = entry_paths(cell_id)["tmp"].resolve()
    tilelang = cache / "tilelang"
    value = {
        "NATIVE_TRAJECTORY_ARTIFACT_MODE": mode,
        "NATIVE_TRAJECTORY_ARTIFACT_CELL": cell_id,
        "PHASE1_TL_CACHE": "1",
        "PHASE2_TL_CACHE": "1",
        "TILELANG_CACHE_DIR": str(tilelang),
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
        "TMPDIR": str(runtime_tmp),
        "TMP": str(runtime_tmp),
        "TEMP": str(runtime_tmp),
    }
    if mode == "load_only":
        value["TRITON_CACHE_MANAGER"] = (
            "ako_runs.controlled_followup.native_trajectory_replication_ada_v3.artifacts:"
            "ReadOnlyTritonCacheManager"
        )
    return value


def prepare_cache_environment(cell_id: str, mode: str) -> dict[str, str]:
    value = cache_environment(cell_id, mode)
    paths = entry_paths(cell_id)
    paths["cache"].mkdir(parents=True, exist_ok=True)
    paths["tmp"].mkdir(parents=True, exist_ok=True)
    Path(value["TILELANG_TMP_DIR"]).mkdir(parents=True, exist_ok=True)
    atomic_temp_device_snapshot(cell_id)
    return value


def atomic_temp_device_snapshot(cell_id: str) -> dict[str, Any]:
    paths = entry_paths(cell_id)
    cache, runtime_tmp = paths["cache"].resolve(), paths["tmp"].resolve()
    expected_parent = entry_root(cell_id).resolve()
    if cache.parent != expected_parent or runtime_tmp.parent != expected_parent:
        raise protocol.ProtocolError("artifact cache/temp roots are not entry-local siblings")
    if not cache.is_dir() or cache.is_symlink() or not runtime_tmp.is_dir() or runtime_tmp.is_symlink():
        raise protocol.ProtocolError("artifact cache/temp roots are missing or unsafe")
    cache_device, temp_device = cache.stat().st_dev, runtime_tmp.stat().st_dev
    if cache_device != temp_device:
        raise protocol.ProtocolError("TileLang atomic temp root is on another filesystem")
    return {
        "cache_root": protocol.repo_path(cache),
        "cache_st_dev": cache_device,
        "runtime_tmp_root": protocol.repo_path(runtime_tmp),
        "runtime_tmp_st_dev": temp_device,
        "same_filesystem": True,
    }


def validate_cache_environment(cell_id: str, mode: str) -> None:
    mismatch = {
        key: (os.environ.get(key), expected)
        for key, expected in cache_environment(cell_id, mode).items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise protocol.ProtocolError(f"artifact cache environment changed: {mismatch}")


def _is_code_object(path: Path) -> bool:
    return path.suffix == ".cubin" or path.name.endswith(".so")


def cache_snapshot(cell_id: str) -> dict[str, Any]:
    cache = entry_paths(cell_id)["cache"]
    if not cache.is_dir() or cache.is_symlink():
        raise protocol.ProtocolError(f"admitted cache is missing or unsafe: {cache}")
    files = []
    for path in sorted(cache.rglob("*"), key=lambda item: item.relative_to(cache).as_posix()):
        if path.is_symlink():
            raise protocol.ProtocolError(f"admitted cache contains a symlink: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(cache).as_posix(),
                    "sha256": protocol.file_sha256(path),
                    "size": path.stat().st_size,
                }
            )
    if not files:
        raise protocol.ProtocolError("admitted cache contains no files")
    source_files = [row for row in files if Path(row["path"]).suffix in SOURCE_SUFFIXES]
    code_objects = [row for row in files if _is_code_object(Path(row["path"]))]
    if not source_files or not code_objects:
        raise protocol.ProtocolError(
            "each admitted two-kernel cell must retain generated source and a loadable code object"
        )
    return {
        "cache_root": protocol.repo_path(cache),
        "file_count": len(files),
        "files": files,
        "files_sha256": protocol.canonical_sha256(files),
        "generated_sources": source_files,
        "loadable_code_objects": code_objects,
    }


def generated_artifact_identity(cache: dict[str, Any]) -> str:
    """Hash executable source/object bytes; diagnostic metadata is out of scope."""
    if not isinstance(cache, dict) or not isinstance(cache.get("generated_sources"), list) or not isinstance(
        cache.get("loadable_code_objects"), list
    ):
        raise protocol.ProtocolError("artifact identity lacks generated source/code-object rows")
    def bytes_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if any(
            not isinstance(row, dict)
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None
            or isinstance(row.get("size"), bool)
            or not isinstance(row.get("size"), int)
            or row["size"] < 0
            for row in rows
        ):
            raise protocol.ProtocolError("artifact identity contains a malformed byte binding")
        return sorted(
            ({"sha256": row["sha256"], "size": row["size"]} for row in rows),
            key=lambda row: (row["sha256"], row["size"]),
        )

    return protocol.canonical_sha256(
        {
            "generated_sources": bytes_only(cache["generated_sources"]),
            "loadable_code_objects": bytes_only(cache["loadable_code_objects"]),
        }
    )


def _source_fingerprints(value: Any) -> list[dict[str, Any]]:
    if value is None:
        value = []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or any(not isinstance(item, str) for item in values):
        raise protocol.ProtocolError("torch extension sources are not strings")
    return [
        {"sha256": hashlib.sha256(item.encode("utf-8")).hexdigest(), "size": len(item.encode("utf-8"))}
        for item in values
    ]


def _torch_inline_request(function: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    bound = inspect.signature(function).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    name = values.get("name")
    if not isinstance(name, str) or not name:
        raise protocol.ProtocolError("torch extension request has no module name")
    functions = values.get("functions")
    if isinstance(functions, dict):
        normalized_functions: Any = {str(key): str(value) for key, value in sorted(functions.items())}
    elif isinstance(functions, (list, tuple)):
        normalized_functions = [str(value) for value in functions]
    elif functions is None:
        normalized_functions = None
    else:
        normalized_functions = str(functions)
    return {
        "compile_options": {
            key: _strip_performance(values.get(key))
            for key in (
                "extra_cflags",
                "extra_cuda_cflags",
                "extra_include_paths",
                "extra_ldflags",
                "extra_sycl_cflags",
                "is_python_module",
                "keep_intermediates",
                "no_implicit_headers",
                "use_pch",
                "with_cuda",
                "with_pytorch_error_handling",
                "with_sycl",
            )
        },
        "cpp_sources": _source_fingerprints(values.get("cpp_sources", [])),
        "cuda_sources": _source_fingerprints(values.get("cuda_sources", [])),
        "functions": normalized_functions,
        "name": name,
        "sycl_sources": _source_fingerprints(values.get("sycl_sources", [])),
    }


@contextmanager
def capture_torch_extensions() -> Iterator[list[dict[str, Any]]]:
    import torch.utils.cpp_extension as cpp_extension

    original = cpp_extension.load_inline
    requests: list[dict[str, Any]] = []

    def capture(*args, **kwargs):
        request = _torch_inline_request(original, args, kwargs)
        module = original(*args, **kwargs)
        module_path = Path(str(getattr(module, "__file__", ""))).resolve()
        requests.append({"module_path": str(module_path), "request": request})
        return module

    cpp_extension.load_inline = capture
    try:
        yield requests
    finally:
        cpp_extension.load_inline = original


def bind_torch_requests(
    requests: list[dict[str, Any]], cache: dict[str, Any]
) -> list[dict[str, Any]]:
    allowed = _allowed_files({"cache": cache})
    result = []
    for value in requests:
        row = _verify_bound_file(Path(value["module_path"]), allowed)
        if not _is_code_object(Path(row["path"])):
            raise protocol.ProtocolError("torch extension did not resolve to an admitted code object")
        result.append({"module": row, "request": value["request"]})
    if len({row["request"]["name"] for row in result}) != len(result):
        raise protocol.ProtocolError("torch extension admission contains duplicate module names")
    return result


def validate_torch_requests(value: Any, cache: dict[str, Any]) -> list[dict[str, Any]]:
    admitted = {row["path"]: row for row in cache["files"]}
    if not isinstance(value, list):
        raise protocol.ProtocolError("torch extension request census is invalid")
    names = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"module", "request"}:
            raise protocol.ProtocolError("torch extension request is malformed")
        module, request = row["module"], row["request"]
        if (
            not isinstance(module, dict)
            or admitted.get(module.get("path")) != module
            or not _is_code_object(Path(str(module.get("path", ""))))
            or not isinstance(request, dict)
            or set(request) != {
                "compile_options", "cpp_sources", "cuda_sources", "functions", "name", "sycl_sources"
            }
            or not isinstance(request["name"], str)
            or not request["name"]
        ):
            raise protocol.ProtocolError("torch extension request is not artifact-bound")
        if not isinstance(request["compile_options"], dict):
            raise protocol.ProtocolError("torch extension compile options are malformed")
        for sources in (request["cpp_sources"], request["cuda_sources"], request["sycl_sources"]):
            if not isinstance(sources, list) or any(
                not isinstance(source, dict)
                or set(source) != {"sha256", "size"}
                or re.fullmatch(r"[0-9a-f]{64}", str(source["sha256"])) is None
                or not isinstance(source["size"], int)
                or source["size"] < 0
                for source in sources
            ):
                raise protocol.ProtocolError("torch extension source fingerprint is malformed")
        names.append(request["name"])
    if len(names) != len(set(names)):
        raise protocol.ProtocolError("torch extension request names are not unique")
    return value


def _strip_performance(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_performance(item)
            for key, item in value.items()
            if str(key) not in {"build_wall_s", "compile_s", "reported_compile_s"}
        }
    if isinstance(value, (list, tuple)):
        return [_strip_performance(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def build_record(
    row: dict[str, Any], built: Any, correctness: dict[str, Any],
    provenance: dict[str, Any], torch_requests: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot = cache_snapshot(row["cell_id"])
    metadata = _strip_performance(built.metadata)
    requests = bind_torch_requests(torch_requests, snapshot)
    return {
        "schema_version": 1,
        "record_type": "native_trajectory_replication_ada_artifact_build_record",
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": row["cell_id"],
        "admission_row": row,
        "admission_row_sha256": protocol.canonical_sha256(row),
        "admission_provenance": provenance,
        "atomic_temp_device": atomic_temp_device_snapshot(row["cell_id"]),
        "cache": snapshot,
        "config": built.config,
        "correctness": correctness,
        "artifact_identity_sha256": generated_artifact_identity(snapshot),
        "n_kernels": built.metadata["n_kernels"],
        "performance_observations": [],
        "source_build_metadata": metadata,
        "torch_inline_requests": requests,
    }


def validate_build_record(value: Any, row: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "record_type",
        "campaign_id",
        "cell_id",
        "admission_row",
        "admission_row_sha256",
        "admission_provenance",
        "artifact_identity_sha256",
        "atomic_temp_device",
        "cache",
        "config",
        "correctness",
        "n_kernels",
        "performance_observations",
        "source_build_metadata",
        "torch_inline_requests",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != 1
        or value.get("record_type") != "native_trajectory_replication_ada_artifact_build_record"
        or value.get("campaign_id") != protocol.CAMPAIGN_ID
        or value.get("cell_id") != row["cell_id"]
        or value.get("admission_row") != row
        or value.get("admission_row_sha256") != protocol.canonical_sha256(row)
        or value.get("n_kernels") != 2
        or value.get("performance_observations") != []
        or value.get("correctness", {}).get("gate_pass") is not True
        or value.get("artifact_identity_sha256") != generated_artifact_identity(value.get("cache", {}))
    ):
        raise protocol.ProtocolError(f"invalid performance-blind build record: {row['cell_id']}")
    _validate_provenance(value["admission_provenance"], row["cell_id"], "admit")
    if value["atomic_temp_device"] != atomic_temp_device_snapshot(row["cell_id"]):
        raise protocol.ProtocolError(f"artifact atomic-temp binding changed: {row['cell_id']}")
    validate_torch_requests(value["torch_inline_requests"], value["cache"])
    if value["cache"] != cache_snapshot(row["cell_id"]):
        raise protocol.ProtocolError(f"admitted cache changed after build: {row['cell_id']}")
    return value


def _allowed_files(build: dict[str, Any]) -> dict[Path, dict[str, Any]]:
    cache = (protocol.REPO_ROOT / build["cache"]["cache_root"]).resolve()
    return {(cache / row["path"]).resolve(): row for row in build["cache"]["files"]}


def _verify_bound_file(path: Path, allowed: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    resolved = path.resolve()
    row = allowed.get(resolved)
    if row is None or not resolved.is_file() or protocol.file_sha256(resolved) != row["sha256"]:
        raise protocol.ProtocolError(f"runtime attempted to load an unadmitted artifact: {path}")
    return row


def _negative_out_idx_candidates(value: Any) -> list[Any]:
    if isinstance(value, int):
        indexes = [value]
        scalar = True
    elif isinstance(value, list) and value and all(isinstance(index, int) for index in value):
        indexes = value
        scalar = False
    else:
        return []
    if not any(index < 0 for index in indexes):
        return []
    candidates = []
    # The frozen fused kernels have two to four parameters. Replaying these
    # legalizations reads existing frontend keys only; compilation stays blocked.
    for parameter_count in range(1, 5):
        normalized = [parameter_count + index if index < 0 else index for index in indexes]
        if all(0 <= index < parameter_count for index in normalized):
            candidate: Any = normalized[0] if scalar else normalized
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _admitted_cuda_noptx_side_compile(build: dict[str, Any]) -> dict[str, Any]:
    """Replay frozen diagnostics; the load-only child must not run nvcc."""
    cell_id = build.get("cell_id")
    metadata = build.get("source_build_metadata")
    values = metadata.get("artifacts") if isinstance(metadata, dict) else None
    if not isinstance(cell_id, str) or ".cuda_noptx." not in cell_id or not isinstance(values, dict):
        raise protocol.ProtocolError("CUDA-no-PTX admission lacks its frozen diagnostics")
    strategy = cell_id.split(".", 1)[0]
    if strategy == "global_intermediate":
        values = values.get("gemm")
    elif strategy == "register_common_postprocess":
        values = values.get("lane_gbg")
    elif strategy != "register_fused":
        values = None
    if not isinstance(values, dict):
        raise protocol.ProtocolError("CUDA-no-PTX admission has no strategy diagnostics")
    replay = {key: values[key] for key in SIDE_COMPILE_DIAGNOSTIC_KEYS if key in values}
    log = replay.get("ptxas_log")
    if not isinstance(log, str) or not log:
        raise protocol.ProtocolError("CUDA-no-PTX admission has no ptxas log to replay")
    if strategy != "global_intermediate":
        resources = protocol.core.ptxas_kernel_resources(log, "fused_kernel")
        if replay.get("kernel_resources") != resources:
            raise protocol.ProtocolError("CUDA-no-PTX admitted resource diagnostics changed")
    return replay


def _load_tilelang_admitted(
    original: Any, args: tuple[Any, ...], kwargs: dict[str, Any],
    allowed: dict[Path, dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    raw_requested = kwargs.get("out_idx")
    requested = raw_requested.copy() if isinstance(raw_requested, list) else raw_requested

    def bound(kernel: Any, resolved: Any) -> tuple[Any, dict[str, Any]]:
        cache_path = Path(str(getattr(kernel, "_tilelang_cache_path", ""))).resolve()
        rows = [
            _verify_bound_file(path, allowed)
            for path in allowed
            if path.parent == cache_path and _is_code_object(path)
        ]
        if len(rows) != 1:
            raise protocol.ProtocolError("TileLang cache hit lacks one admitted loadable object")
        return kernel, {
            "cache_path": str(cache_path),
            "loadable_code_objects": rows,
            "requested_out_idx": requested,
            "resolved_out_idx": resolved,
        }

    kernel = original(*args, **kwargs)
    if kernel is not None:
        return bound(kernel, requested)
    candidates = _negative_out_idx_candidates(requested)
    if not candidates:
        raise protocol.ProtocolError("TileLang admitted frontend-cache miss")
    hits = []
    for candidate in candidates:
        retry = {**kwargs, "out_idx": candidate.copy() if isinstance(candidate, list) else candidate}
        kernel = original(*args, **retry)
        if kernel is not None:
            hits.append(bound(kernel, candidate))
    if len(hits) != 1:
        raise protocol.ProtocolError(
            f"TileLang negative-out_idx fallback found {len(hits)} admitted cache hits"
        )
    return hits[0]


@contextmanager
def load_only_guards(build: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Reject every compiler miss and record the exact admitted objects loaded."""
    tilelang_cache = importlib.import_module("tilelang.cache")
    tilelang_kernel_cache = importlib.import_module("tilelang.cache.kernel_cache")
    tilelang_jit = importlib.import_module("tilelang.jit")
    import torch.utils.cpp_extension as cpp_extension
    import triton
    import triton.runtime.build as triton_build

    allowed = _allowed_files(build)
    evidence: dict[str, Any] = {
        "mode": "load_only",
        "diagnostic_side_compile_skips": [],
        "tilelang_cache_hits": [],
        "torch_extension_loads": [],
        "triton_cache_hits": [],
        "triton_kernel_loads": [],
    }
    original_tl_load = tilelang_cache.load_frontend_cached
    original_tl_compile = tilelang_jit.JITImpl.compile
    original_tl_safe_write = tilelang_kernel_cache.KernelCache._safe_write_file
    original_triton_build = triton_build._build
    original_torch_inline = cpp_extension.load_inline
    original_torch_ninja = cpp_extension._run_ninja_build
    original_listener = triton.knobs.compilation.listener
    expected_torch = validate_torch_requests(build["torch_inline_requests"], build["cache"])
    used_torch: set[int] = set()
    if original_listener is not None:
        raise protocol.ProtocolError("foreign Triton compilation listener is installed")

    def tl_load(*args, **kwargs):
        kernel, hit = _load_tilelang_admitted(original_tl_load, args, kwargs, allowed)
        evidence["tilelang_cache_hits"].append(hit)
        return kernel

    def reject_tilelang_compile(*_args, **_kwargs):
        raise protocol.ProtocolError("TileLang compilation is forbidden during artifact loading")

    def reject_tilelang_cache_write(*_args, **_kwargs):
        raise protocol.ProtocolError("TileLang cache writes are forbidden during artifact loading")

    def reject_native_build(*_args, **_kwargs):
        raise protocol.ProtocolError("native launcher compilation is forbidden during artifact loading")

    def load_torch_extension(*args, **kwargs):
        request = _torch_inline_request(original_torch_inline, args, kwargs)
        matches = [
            index
            for index, row in enumerate(expected_torch)
            if index not in used_torch and row["request"] == request
        ]
        if len(matches) != 1:
            raise protocol.ProtocolError(f"torch extension request is not uniquely admitted: {request['name']}")
        index = matches[0]
        used_torch.add(index)
        expected = expected_torch[index]
        paths = [path for path, row in allowed.items() if row == expected["module"]]
        if len(paths) != 1:
            raise protocol.ProtocolError("admitted torch extension module is not unique")
        path = paths[0]
        module_row = _verify_bound_file(path, allowed)
        if request["name"] in sys.modules:
            raise protocol.ProtocolError("torch extension module was loaded before the admitted request")
        spec = importlib.util.spec_from_file_location(request["name"], path)
        if spec is None or spec.loader is None:
            raise protocol.ProtocolError(f"cannot load admitted torch extension: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[request["name"]] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(request["name"], None)
            raise
        evidence["torch_extension_loads"].append(
            {"module": module_row, "request": request}
        )
        return module

    def skip_side_compile(kernel_source: str, name: str):
        diagnostics = _admitted_cuda_noptx_side_compile(build)
        evidence["diagnostic_side_compile_skips"].append(
            {
                "admitted_diagnostics_sha256": protocol.canonical_sha256(diagnostics),
                "name": str(name),
                "source_sha256": hashlib.sha256(kernel_source.encode("utf-8")).hexdigest(),
            }
        )
        return diagnostics

    def triton_listener(*, metadata_group, cache_hit, **_kwargs):
        if cache_hit is not True:
            raise protocol.ProtocolError("Triton admitted cache miss")
        rows = [_verify_bound_file(Path(path), allowed) for path in metadata_group.values()]
        evidence["triton_cache_hits"].append(rows)

    def triton_loaded(_module, _function, _name, metadata_group, _hash):
        rows = [
            _verify_bound_file(Path(path), allowed)
            for path in metadata_group.values()
            if _is_code_object(Path(path))
        ]
        if not rows:
            raise protocol.ProtocolError("Triton loaded kernel lacks an admitted code object")
        evidence["triton_kernel_loads"].append(rows)

    side_compile_module = None
    original_side_compile = None
    original_side_load_inline = None
    if ".cuda_noptx." in build["cell_id"]:
        side_compile_module = importlib.import_module("variants.cuda_noptx_gemm")
        original_side_compile = side_compile_module._side_compile
        original_side_load_inline = side_compile_module.load_inline
        side_compile_module._side_compile = skip_side_compile
        side_compile_module.load_inline = load_torch_extension
    tilelang_cache.load_frontend_cached = tl_load
    tilelang_jit.JITImpl.compile = reject_tilelang_compile
    tilelang_kernel_cache.KernelCache._safe_write_file = staticmethod(reject_tilelang_cache_write)
    triton_build._build = reject_native_build
    cpp_extension.load_inline = load_torch_extension
    cpp_extension._run_ninja_build = reject_native_build
    triton.knobs.compilation.listener = triton_listener
    triton.knobs.runtime.kernel_load_end_hook.add(triton_loaded)
    try:
        yield evidence
    finally:
        if side_compile_module is not None:
            side_compile_module._side_compile = original_side_compile
            side_compile_module.load_inline = original_side_load_inline
        triton.knobs.runtime.kernel_load_end_hook.remove(triton_loaded)
        triton.knobs.compilation.listener = original_listener
        triton_build._build = original_triton_build
        cpp_extension._run_ninja_build = original_torch_ninja
        cpp_extension.load_inline = original_torch_inline
        tilelang_kernel_cache.KernelCache._safe_write_file = staticmethod(original_tl_safe_write)
        tilelang_jit.JITImpl.compile = original_tl_compile
        tilelang_cache.load_frontend_cached = original_tl_load


def validate_load_evidence(
    value: Any, cell: dict[str, Any], build: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "diagnostic_side_compile_skips",
        "mode",
        "tilelang_cache_hits",
        "torch_extension_loads",
        "triton_cache_hits",
        "triton_kernel_loads",
    } or value.get("mode") != "load_only":
        raise protocol.ProtocolError("invalid artifact load evidence")
    cache = build.get("cache") if isinstance(build, dict) else None
    if not isinstance(cache, dict) or not isinstance(cache.get("files"), list):
        raise protocol.ProtocolError("artifact load evidence lacks its admitted cache")
    admitted = {
        row.get("path"): row
        for row in cache["files"]
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    if len(admitted) != len(cache["files"]):
        raise protocol.ProtocolError("admitted cache file census is invalid")

    def checked_row(row: Any, *, code_object: bool = False) -> dict[str, Any]:
        if (
            not isinstance(row, dict)
            or admitted.get(row.get("path")) != row
            or (code_object and not _is_code_object(Path(str(row.get("path", "")))))
        ):
            raise protocol.ProtocolError("artifact load evidence names an unadmitted object")
        return row

    loaded_code_objects: list[str] = []
    if value["torch_extension_loads"] != build.get("torch_inline_requests"):
        raise protocol.ProtocolError("torch extension loads differ from admitted requests")
    for extension in value["torch_extension_loads"]:
        loaded_code_objects.append(checked_row(extension["module"], code_object=True)["path"])

    if not isinstance(value["tilelang_cache_hits"], list):
        raise protocol.ProtocolError("invalid TileLang load evidence")
    for hit in value["tilelang_cache_hits"]:
        if (
            not isinstance(hit, dict)
            or set(hit) != {
                "cache_path", "loadable_code_objects", "requested_out_idx", "resolved_out_idx"
            }
            or not isinstance(hit["cache_path"], str)
            or not isinstance(hit["loadable_code_objects"], list)
            or not hit["loadable_code_objects"]
            or (
                hit["resolved_out_idx"] != hit["requested_out_idx"]
                and hit["resolved_out_idx"]
                not in _negative_out_idx_candidates(hit["requested_out_idx"])
            )
        ):
            raise protocol.ProtocolError("invalid TileLang cache-hit evidence")
        if len(hit["loadable_code_objects"]) != 1:
            raise protocol.ProtocolError("one TileLang cache hit must bind one code object")
        cache_path = Path(hit["cache_path"]).resolve()
        for row in hit["loadable_code_objects"]:
            checked = checked_row(row, code_object=True)
            object_path = (protocol.REPO_ROOT / cache["cache_root"] / checked["path"]).resolve()
            if object_path.parent != cache_path:
                raise protocol.ProtocolError("TileLang load evidence names another cache entry")
            loaded_code_objects.append(checked["path"])

    if not isinstance(value["triton_cache_hits"], list):
        raise protocol.ProtocolError("invalid Triton cache-hit evidence")
    for group in value["triton_cache_hits"]:
        if not isinstance(group, list) or not group:
            raise protocol.ProtocolError("invalid Triton cache-hit group")
        checked = [checked_row(row) for row in group]
        if len({row["path"] for row in checked}) != len(checked) or not any(
            _is_code_object(Path(row["path"])) for row in checked
        ):
            raise protocol.ProtocolError("invalid Triton cache-hit object census")

    if not isinstance(value["triton_kernel_loads"], list):
        raise protocol.ProtocolError("invalid Triton kernel-load evidence")
    for group in value["triton_kernel_loads"]:
        if not isinstance(group, list) or not group:
            raise protocol.ProtocolError("invalid Triton kernel-load group")
        checked = [checked_row(row, code_object=True) for row in group]
        if len(checked) != 1:
            raise protocol.ProtocolError("one Triton kernel load must bind one code object")
        loaded_code_objects.append(checked[0]["path"])

    fused = cell["strategy"] == "register_fused"
    expected_lane_kernels = 2 if fused else 1
    expected_common = int(cell["strategy"] in {"global_intermediate", "register_common_postprocess"})
    expected_torch = expected_common
    expected_side_skips = 0
    if cell["lane"] == "tilelang":
        counts = (len(value["tilelang_cache_hits"]), len(value["triton_cache_hits"]), len(value["triton_kernel_loads"]))
        expected = (expected_lane_kernels, 0, 0)
    elif cell["lane"] == "triton":
        counts = (len(value["tilelang_cache_hits"]), len(value["triton_cache_hits"]), len(value["triton_kernel_loads"]))
        expected = (0, expected_lane_kernels, expected_lane_kernels)
    elif cell["lane"] in {"cuda_noptx", "cuda_unlimited"}:
        counts = (len(value["tilelang_cache_hits"]), len(value["triton_cache_hits"]), len(value["triton_kernel_loads"]))
        expected = (0, 0, 0)
        expected_torch = 1 + expected_common
        expected_side_skips = int(cell["lane"] == "cuda_noptx")
    else:
        raise protocol.ProtocolError("artifact admission names an unknown lane")
    skips = value["diagnostic_side_compile_skips"]
    expected_diagnostics_sha256 = (
        protocol.canonical_sha256(_admitted_cuda_noptx_side_compile(build))
        if expected_side_skips else None
    )
    if (
        counts != expected
        or len(value["torch_extension_loads"]) != expected_torch
        or not isinstance(skips, list)
        or len(skips) != expected_side_skips
        or any(
            not isinstance(skip, dict)
            or set(skip) != {"admitted_diagnostics_sha256", "name", "source_sha256"}
            or skip["admitted_diagnostics_sha256"] != expected_diagnostics_sha256
            or not isinstance(skip["name"], str)
            or re.fullmatch(r"[0-9a-f]{64}", str(skip["source_sha256"])) is None
            for skip in skips
        )
    ):
        raise protocol.ProtocolError(
            f"artifact load census changed for {cell['cell_id']}"
        )
    expected_objects = expected_lane_kernels + expected_common if cell["lane"] in {"tilelang", "triton"} else expected_torch
    if len(loaded_code_objects) != expected_objects or len(set(loaded_code_objects)) != expected_objects:
        raise protocol.ProtocolError(
            f"artifact did not load its distinct admitted code objects: {cell['cell_id']}"
        )
    return value


def verify_record(
    row: dict[str, Any], build: dict[str, Any], built: Any,
    correctness: dict[str, Any], load_evidence: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    snapshot = cache_snapshot(row["cell_id"])
    identity = generated_artifact_identity(snapshot)
    if identity != build["artifact_identity_sha256"] or built.config != build["config"]:
        raise protocol.ProtocolError("load-only verification resolved to another admitted artifact")
    return {
        "schema_version": 1,
        "record_type": "native_trajectory_replication_ada_artifact_verify_record",
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": row["cell_id"],
        "admission_row_sha256": protocol.canonical_sha256(row),
        "admission_provenance": provenance,
        "artifact_identity_sha256": identity,
        "atomic_temp_device": atomic_temp_device_snapshot(row["cell_id"]),
        "build_record_sha256": protocol.file_sha256(entry_paths(row["cell_id"])["build"]),
        "cache": snapshot,
        "correctness": correctness,
        "load_evidence": load_evidence,
        "n_kernels": built.metadata["n_kernels"],
        "performance_observations": [],
    }


def validate_verify_record(
    value: Any, row: dict[str, Any], build: dict[str, Any], cell: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "record_type",
        "campaign_id",
        "cell_id",
        "admission_row_sha256",
        "admission_provenance",
        "artifact_identity_sha256",
        "atomic_temp_device",
        "build_record_sha256",
        "cache",
        "correctness",
        "load_evidence",
        "n_kernels",
        "performance_observations",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != 1
        or value.get("record_type") != "native_trajectory_replication_ada_artifact_verify_record"
        or value.get("campaign_id") != protocol.CAMPAIGN_ID
        or value.get("cell_id") != row["cell_id"]
        or value.get("admission_row_sha256") != protocol.canonical_sha256(row)
        or value.get("build_record_sha256") != protocol.file_sha256(entry_paths(row["cell_id"])["build"])
        or value.get("cache") != build["cache"]
        or value.get("artifact_identity_sha256") != build["artifact_identity_sha256"]
        or value.get("n_kernels") != 2
        or value.get("performance_observations") != []
        or value.get("correctness", {}).get("gate_pass") is not True
    ):
        raise protocol.ProtocolError(f"invalid cache-hit verification record: {row['cell_id']}")
    _validate_provenance(value["admission_provenance"], row["cell_id"], "load_only")
    if value["atomic_temp_device"] != atomic_temp_device_snapshot(row["cell_id"]):
        raise protocol.ProtocolError(f"artifact atomic-temp binding changed: {row['cell_id']}")
    if value["admission_provenance"]["process_pid"] == build["admission_provenance"]["process_pid"]:
        raise protocol.ProtocolError("artifact build and cache-hit verification reused one process")
    validate_load_evidence(value["load_evidence"], cell, build)
    if value["cache"] != cache_snapshot(row["cell_id"]):
        raise protocol.ProtocolError(f"admitted cache changed during verification: {row['cell_id']}")
    return value


def _validate_provenance(value: Any, cell_id: str, mode: str) -> dict[str, Any]:
    required = {
        "cache_environment",
        "compute_pids_postflight",
        "compute_pids_preflight",
        "execution_lock_sha256",
        "gpu_postflight",
        "gpu_preflight",
        "launch_receipt_path",
        "launch_receipt_sha256",
        "parent_pid",
        "process_pid",
        "toolchain",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("execution_lock_sha256") != protocol.file_sha256(protocol.EXECUTION_LOCK_PATH)
        or value.get("cache_environment") != cache_environment(cell_id, mode)
        or value.get("launch_receipt_path")
        != protocol.repo_path(ADMISSION_ROOT / "launch_receipt.json")
        or value.get("launch_receipt_sha256")
        != protocol.file_sha256(ADMISSION_ROOT / "launch_receipt.json")
        or not isinstance(value.get("process_pid"), int)
        or value["process_pid"] <= 0
        or not isinstance(value.get("parent_pid"), int)
        or value["parent_pid"] <= 0
        or not isinstance(value.get("gpu_preflight"), dict)
        or value.get("gpu_postflight") != value.get("gpu_preflight")
        or value.get("compute_pids_preflight") != []
        or not isinstance(value.get("compute_pids_postflight"), list)
        or any(
            isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
            for pid in value.get("compute_pids_postflight", [])
        )
        or not set(value.get("compute_pids_postflight", [])) <= {value.get("process_pid")}
        or not isinstance(value.get("toolchain"), dict)
    ):
        raise protocol.ProtocolError("invalid artifact-admission process provenance")
    runner = protocol.local_runner_module()
    contract = protocol.load_contract()
    runner.validate_gpu_snapshot(value["gpu_preflight"], contract)
    if value["toolchain"] != runner.validate_execution_frozen(rehash_materials=False)[2]["toolchain"]:
        raise protocol.ProtocolError("artifact-admission toolchain changed")
    return value


def make_entry(row: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    paths = entry_paths(row["cell_id"])
    build = validate_build_record(protocol.read_json(paths["build"]), row)
    verify = validate_verify_record(protocol.read_json(paths["verify"]), row, build, cell)
    artifact = {
        "artifact_identity_sha256": build["artifact_identity_sha256"],
        "cell_id": row["cell_id"],
        "config": build["config"],
        "cache_files_sha256": build["cache"]["files_sha256"],
        "generated_sources": build["cache"]["generated_sources"],
        "loadable_code_objects": build["cache"]["loadable_code_objects"],
    }
    return {
        "schema_version": 1,
        "record_type": "native_trajectory_replication_ada_admitted_artifact",
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": row["cell_id"],
        "role": row["role"],
        "admission_row_sha256": protocol.canonical_sha256(row),
        "artifact_identity_sha256": build["artifact_identity_sha256"],
        "artifact_sha256": protocol.canonical_sha256(artifact),
        "build_record_path": protocol.repo_path(paths["build"]),
        "build_record_sha256": protocol.file_sha256(paths["build"]),
        "cache": build["cache"],
        "config": build["config"],
        "performance_observations": [],
        "verify_record_path": protocol.repo_path(paths["verify"]),
        "verify_record_sha256": protocol.file_sha256(paths["verify"]),
    }


def validate_entry(value: Any, row: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    expected = make_entry(row, cell)
    if value != expected:
        raise protocol.ProtocolError(f"admitted artifact entry changed: {row['cell_id']}")
    return value


def make_manifest(launch_receipt_path: Path, cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if protocol.file_sha256(INCIDENT_PATH) != INCIDENT_SHA256:
        raise protocol.ProtocolError("predecessor incident receipt changed")
    plan = admission_plan()
    entries = []
    for row in plan:
        path = entry_paths(row["cell_id"])["entry"]
        value = validate_entry(protocol.read_json(path), row, cells[row["cell_id"]])
        entries.append(
            {
                "artifact_sha256": value["artifact_sha256"],
                "artifact_identity_sha256": value["artifact_identity_sha256"],
                "cell_id": row["cell_id"],
                "entry_path": protocol.repo_path(path),
                "entry_sha256": protocol.file_sha256(path),
                "role": row["role"],
            }
        )
    return {
        "schema_version": 1,
        "record_type": "native_trajectory_replication_ada_artifact_admission_manifest",
        "campaign_id": protocol.CAMPAIGN_ID,
        "complete": True,
        "admission_plan": plan,
        "admission_plan_sha256": protocol.canonical_sha256(plan),
        "entries": entries,
        "entry_count": 12,
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "launch_receipt_path": protocol.repo_path(launch_receipt_path),
        "launch_receipt_sha256": protocol.file_sha256(launch_receipt_path),
        "performance_blind": True,
        "predecessor_incident_path": protocol.repo_path(INCIDENT_PATH),
        "predecessor_incident_sha256": INCIDENT_SHA256,
        "shared_sham_artifact_count": 1,
    }


def validate_manifest(value: Any, cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    launch_path = protocol.REPO_ROOT / str(value.get("launch_receipt_path", ""))
    expected = make_manifest(launch_path, cells)
    if value != expected:
        raise protocol.ProtocolError("artifact admission manifest changed")
    return value


def manifest_paths(value: dict[str, Any]) -> list[str]:
    paths = [
        protocol.repo_path(MANIFEST_PATH),
        value["launch_receipt_path"],
        protocol.repo_path(ADMISSION_ROOT / "run_status.json"),
    ]
    for row in value["entries"]:
        entry = protocol.read_json(protocol.REPO_ROOT / row["entry_path"])
        paths.extend([row["entry_path"], entry["build_record_path"], entry["verify_record_path"]])
        paths.extend(
            protocol.repo_path(protocol.REPO_ROOT / entry["cache"]["cache_root"] / file["path"])
            for file in entry["cache"]["files"]
        )
    if len(paths) != len(set(paths)):
        raise protocol.ProtocolError("artifact admission path closure contains duplicates")
    return paths


def binding(cell_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in manifest["entries"] if row["cell_id"] == cell_id]
    if len(rows) != 1:
        raise protocol.ProtocolError(f"cell has no unique admitted artifact: {cell_id}")
    row = rows[0]
    return {
        "artifact_admission_manifest_path": protocol.repo_path(MANIFEST_PATH),
        "artifact_admission_manifest_sha256": protocol.file_sha256(MANIFEST_PATH),
        "admitted_artifact_sha256": row["artifact_sha256"],
        "admitted_artifact_identity_sha256": row["artifact_identity_sha256"],
        "admitted_entry_path": row["entry_path"],
        "admitted_entry_sha256": row["entry_sha256"],
    }
