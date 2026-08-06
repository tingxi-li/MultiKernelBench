#!/usr/bin/env python3
"""Performance-blind admission and load-only artifact helpers."""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import textwrap
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
INCIDENT_PATH = (
    protocol.REPO_ROOT
    / "ako_runs/controlled_followup/finite_frontier_ada_v1/INCIDENT_SELECTION_20260806.json"
)
INCIDENT_SHA256 = "bef0a1e0eba76849420c918734d7837050e8b5a99a51d70b3307836cd081d9d2"
ADMISSION_INCIDENT_PATH = (
    protocol.REPO_ROOT
    / "ako_runs/controlled_followup/finite_frontier_ada_v2/INCIDENT_ADMISSION_20260806.json"
)
ADMISSION_INCIDENT_SHA256 = "89571304d78dcb6fa0e09e997218c66c6d6564af074af50d00e578571c1d4a6e"
FRONTEND_INCIDENT_PATH = protocol.REPO_ROOT / protocol.PREDECESSOR_FRONTEND_INCIDENT_PATH
FRONTEND_INCIDENT_SHA256 = protocol.PREDECESSOR_FRONTEND_INCIDENT_SHA256
READINESS_INCIDENT_PATH = protocol.REPO_ROOT / protocol.PREDECESSOR_READINESS_INCIDENT_PATH
READINESS_INCIDENT_SHA256 = protocol.PREDECESSOR_READINESS_INCIDENT_SHA256
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".llir", ".ptx", ".source", ".ttgir", ".ttir"}


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
    selection = contract["manifest"]["selection_confirm"]
    rows = [
        {"cell_id": cell_id, "entry_id": cell_id, "role": "candidate"}
        for dsl in protocol.DSLS
        for cell_id in selection["candidate_ids"][dsl]
    ]
    sham = selection["sham_base_cell"]
    rows.append({"cell_id": sham, "entry_id": sham, "role": "shared_sham_base"})
    for position, row in enumerate(rows):
        row["position"] = position
    if len(rows) != 7 or len({row["cell_id"] for row in rows}) != 7:
        raise protocol.ProtocolError("artifact admission must contain six candidates and one shared sham")
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
        "FINITE_FRONTIER_ARTIFACT_MODE": mode,
        "FINITE_FRONTIER_ARTIFACT_CELL": cell_id,
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
            "ako_runs.controlled_followup.finite_frontier_ada_v5.artifacts:"
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
    if not source_files or len(code_objects) < 2:
        raise protocol.ProtocolError(
            "each admitted two-kernel cell must retain generated source and at least two loadable code objects"
        )
    return {
        "cache_root": protocol.repo_path(cache),
        "file_count": len(files),
        "files": files,
        "files_sha256": protocol.canonical_sha256(files),
        "generated_sources": source_files,
        "loadable_code_objects": code_objects,
    }


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
    provenance: dict[str, Any],
) -> dict[str, Any]:
    snapshot = cache_snapshot(row["cell_id"])
    metadata = _strip_performance(built.metadata)
    return {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_artifact_build_record",
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": row["cell_id"],
        "admission_row": row,
        "admission_row_sha256": protocol.canonical_sha256(row),
        "admission_provenance": provenance,
        "atomic_temp_device": atomic_temp_device_snapshot(row["cell_id"]),
        "cache": snapshot,
        "config": built.config,
        "correctness": correctness,
        "implementation_sha256": built.metadata["implementation_sha256"],
        "n_kernels": built.metadata["n_kernels"],
        "performance_observations": [],
        "source_build_metadata": metadata,
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
        "atomic_temp_device",
        "cache",
        "config",
        "correctness",
        "implementation_sha256",
        "n_kernels",
        "performance_observations",
        "source_build_metadata",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != 1
        or value.get("record_type") != "finite_frontier_ada_artifact_build_record"
        or value.get("campaign_id") != protocol.CAMPAIGN_ID
        or value.get("cell_id") != row["cell_id"]
        or value.get("admission_row") != row
        or value.get("admission_row_sha256") != protocol.canonical_sha256(row)
        or value.get("n_kernels") != 2
        or value.get("performance_observations") != []
        or value.get("correctness", {}).get("gate_pass") is not True
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("implementation_sha256"))) is None
    ):
        raise protocol.ProtocolError(f"invalid performance-blind build record: {row['cell_id']}")
    _validate_provenance(value["admission_provenance"], row["cell_id"], "admit")
    if value["atomic_temp_device"] != atomic_temp_device_snapshot(row["cell_id"]):
        raise protocol.ProtocolError(f"artifact atomic-temp binding changed: {row['cell_id']}")
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


def _normalized_out_idx_candidates(
    frontend_key_data: Any, out_idx: Any,
) -> list[list[int]]:
    """Derive Python-style nonnegative indices from the cached frontend source."""
    if (
        not isinstance(frontend_key_data, dict)
        or not isinstance(out_idx, list)
        or not out_idx
        or any(isinstance(index, bool) or not isinstance(index, int) for index in out_idx)
        or not any(index < 0 for index in out_idx)
        or not isinstance(frontend_key_data.get("source"), str)
    ):
        return []
    try:
        tree = ast.parse(textwrap.dedent(frontend_key_data["source"]))
    except (IndentationError, SyntaxError):
        return []

    def is_prim_func(decorator: ast.expr) -> bool:
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        return (
            isinstance(decorator, ast.Name) and decorator.id == "prim_func"
        ) or (
            isinstance(decorator, ast.Attribute) and decorator.attr == "prim_func"
        )

    arities = sorted({
        len(node.args.posonlyargs) + len(node.args.args)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(is_prim_func(decorator) for decorator in node.decorator_list)
        and node.args.vararg is None
        and node.args.kwarg is None
        and not node.args.kwonlyargs
    })
    candidates = []
    for arity in arities:
        normalized = [arity + index if index < 0 else index for index in out_idx]
        if (
            normalized != out_idx
            and all(0 <= index < arity for index in normalized)
            and normalized not in candidates
        ):
            candidates.append(normalized)
    return candidates


def _postprocess_loader(build: dict[str, Any], evidence: dict[str, Any]):
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import postprocess

    allowed = _allowed_files(build)

    def load(*, has_bias: bool, has_gelu: bool):
        metadata = postprocess.source_metadata(has_bias=has_bias, has_gelu=has_gelu)
        module_name = f"fused_crossed_v2_pp_{metadata['cuda_source_sha256'][:12]}"
        matches = [
            path for path in allowed
            if path.name == f"{module_name}.so"
        ]
        if len(matches) != 1:
            raise protocol.ProtocolError(f"admitted postprocess module is not unique: {module_name}")
        path = matches[0]
        row = _verify_bound_file(path, allowed)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise protocol.ProtocolError(f"cannot load admitted postprocess module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        evidence["postprocess_loads"].append(row)
        return postprocess.BuiltPostprocess(
            run=module.common_postprocess,
            compile_s=0.0,
            artifacts={
                "backend_detail": "common CUDA compile-time bias/GELU plus row-softmax",
                "block": [256, 1, 1],
                "cuda_source_sha256": metadata["cuda_source_sha256"],
                "grid": [1024, 1, 1],
                "has_bias": metadata["has_bias"],
                "has_gelu": metadata["has_gelu"],
                "n_kernels": 1,
                "operand_smem_bytes": 0,
                "epilogue_tile_bytes": 0,
                "shared_bytes": 0,
                "softmax_source_sha256": metadata["softmax_source_sha256"],
                "total_dynamic_shared_bytes": 0,
            },
        )

    return postprocess, load


@contextmanager
def load_only_guards(build: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Reject every compiler miss and record the exact admitted objects loaded."""
    tilelang_cache = importlib.import_module("tilelang.cache")
    tilelang_kernel_cache = importlib.import_module("tilelang.cache.kernel_cache")
    tilelang_jit = importlib.import_module("tilelang.jit")
    import triton
    import triton.runtime.build as triton_build

    allowed = _allowed_files(build)
    evidence: dict[str, Any] = {
        "mode": "load_only",
        "postprocess_loads": [],
        "tilelang_cache_hits": [],
        "triton_cache_hits": [],
        "triton_kernel_loads": [],
    }
    original_tl_load = tilelang_cache.load_frontend_cached
    original_tl_compile = tilelang_jit.JITImpl.compile
    original_tl_safe_write = tilelang_kernel_cache.KernelCache._safe_write_file
    original_triton_build = triton_build._build
    original_listener = triton.knobs.compilation.listener
    if original_listener is not None:
        raise protocol.ProtocolError("foreign Triton compilation listener is installed")

    def tl_load(*args, **kwargs):
        kernel = original_tl_load(*args, **kwargs)
        if kernel is None:
            frontend_key_data = args[0] if args else kwargs.get("frontend_key_data")
            hits = []
            for candidate in _normalized_out_idx_candidates(
                frontend_key_data, kwargs.get("out_idx")
            ):
                retry_kwargs = {**kwargs, "out_idx": candidate}
                retry = original_tl_load(*args, **retry_kwargs)
                if retry is not None:
                    hits.append((retry, candidate))
            if len(hits) != 1:
                raise protocol.ProtocolError(
                    "TileLang admitted frontend-cache miss or ambiguous normalized out_idx"
                )
            kernel, _resolved_out_idx = hits[0]
        cache_path = Path(str(getattr(kernel, "_tilelang_cache_path", ""))).resolve()
        rows = [
            _verify_bound_file(path, allowed)
            for path in allowed
            if path.parent == cache_path and _is_code_object(path)
        ]
        if not rows:
            raise protocol.ProtocolError("TileLang cache hit lacks an admitted loadable object")
        evidence["tilelang_cache_hits"].append(
            {"cache_path": str(cache_path), "loadable_code_objects": rows}
        )
        return kernel

    def reject_tilelang_compile(*_args, **_kwargs):
        raise protocol.ProtocolError("TileLang compilation is forbidden during artifact loading")

    def reject_tilelang_cache_write(*_args, **_kwargs):
        raise protocol.ProtocolError("TileLang cache writes are forbidden during artifact loading")

    def reject_native_build(*_args, **_kwargs):
        raise protocol.ProtocolError("native launcher compilation is forbidden during artifact loading")

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

    postprocess, admitted_postprocess = _postprocess_loader(build, evidence)
    original_postprocess = postprocess.build
    tilelang_cache.load_frontend_cached = tl_load
    tilelang_jit.JITImpl.compile = reject_tilelang_compile
    tilelang_kernel_cache.KernelCache._safe_write_file = staticmethod(reject_tilelang_cache_write)
    triton_build._build = reject_native_build
    triton.knobs.compilation.listener = triton_listener
    triton.knobs.runtime.kernel_load_end_hook.add(triton_loaded)
    postprocess.build = admitted_postprocess
    try:
        yield evidence
    finally:
        postprocess.build = original_postprocess
        triton.knobs.runtime.kernel_load_end_hook.remove(triton_loaded)
        triton.knobs.compilation.listener = original_listener
        triton_build._build = original_triton_build
        tilelang_kernel_cache.KernelCache._safe_write_file = staticmethod(original_tl_safe_write)
        tilelang_jit.JITImpl.compile = original_tl_compile
        tilelang_cache.load_frontend_cached = original_tl_load


def validate_load_evidence(
    value: Any, cell: dict[str, Any], cache: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "mode",
        "postprocess_loads",
        "tilelang_cache_hits",
        "triton_cache_hits",
        "triton_kernel_loads",
    } or value.get("mode") != "load_only":
        raise protocol.ProtocolError("invalid artifact load evidence")
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
    if not isinstance(value["postprocess_loads"], list):
        raise protocol.ProtocolError("invalid postprocess load evidence")
    for row in value["postprocess_loads"]:
        loaded_code_objects.append(checked_row(row, code_object=True)["path"])

    if not isinstance(value["tilelang_cache_hits"], list):
        raise protocol.ProtocolError("invalid TileLang load evidence")
    for hit in value["tilelang_cache_hits"]:
        if (
            not isinstance(hit, dict)
            or set(hit) != {"cache_path", "loadable_code_objects"}
            or not isinstance(hit["cache_path"], str)
            or not isinstance(hit["loadable_code_objects"], list)
            or not hit["loadable_code_objects"]
        ):
            raise protocol.ProtocolError("invalid TileLang cache-hit evidence")
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
    expected_postprocess = int(cell["strategy"] in {"global_intermediate", "register_common_postprocess"})
    if cell["lane"] == "tilelang":
        counts = (len(value["tilelang_cache_hits"]), len(value["triton_cache_hits"]), len(value["triton_kernel_loads"]))
        expected = (expected_lane_kernels, 0, 0)
    elif cell["lane"] == "triton":
        counts = (len(value["tilelang_cache_hits"]), len(value["triton_cache_hits"]), len(value["triton_kernel_loads"]))
        expected = (0, expected_lane_kernels, expected_lane_kernels)
    else:
        raise protocol.ProtocolError("admission is restricted to TileLang and Triton")
    if counts != expected or len(value["postprocess_loads"]) != expected_postprocess:
        raise protocol.ProtocolError(
            f"artifact load census changed for {cell['cell_id']}: {counts}/{len(value['postprocess_loads'])}"
        )
    if len(loaded_code_objects) != 2 or len(set(loaded_code_objects)) != 2:
        raise protocol.ProtocolError(
            f"two-kernel artifact did not load two distinct admitted code objects: {cell['cell_id']}"
        )
    return value


def verify_record(
    row: dict[str, Any], build: dict[str, Any], built: Any,
    correctness: dict[str, Any], load_evidence: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if built.metadata["implementation_sha256"] != build["implementation_sha256"]:
        raise protocol.ProtocolError("cache-hit verification resolved to another implementation")
    return {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_artifact_verify_record",
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": row["cell_id"],
        "admission_row_sha256": protocol.canonical_sha256(row),
        "admission_provenance": provenance,
        "atomic_temp_device": atomic_temp_device_snapshot(row["cell_id"]),
        "build_record_sha256": protocol.file_sha256(entry_paths(row["cell_id"])["build"]),
        "cache": cache_snapshot(row["cell_id"]),
        "correctness": correctness,
        "implementation_sha256": built.metadata["implementation_sha256"],
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
        "atomic_temp_device",
        "build_record_sha256",
        "cache",
        "correctness",
        "implementation_sha256",
        "load_evidence",
        "n_kernels",
        "performance_observations",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != 1
        or value.get("record_type") != "finite_frontier_ada_artifact_verify_record"
        or value.get("campaign_id") != protocol.CAMPAIGN_ID
        or value.get("cell_id") != row["cell_id"]
        or value.get("admission_row_sha256") != protocol.canonical_sha256(row)
        or value.get("build_record_sha256") != protocol.file_sha256(entry_paths(row["cell_id"])["build"])
        or value.get("cache") != build["cache"]
        or value.get("implementation_sha256") != build["implementation_sha256"]
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
    validate_load_evidence(value["load_evidence"], cell, build["cache"])
    if value["cache"] != cache_snapshot(row["cell_id"]):
        raise protocol.ProtocolError(f"admitted cache changed during verification: {row['cell_id']}")
    return value


def _validate_provenance(value: Any, cell_id: str, mode: str) -> dict[str, Any]:
    required = {
        "execution_lock_sha256",
        "cache_environment",
        "gpu_idle_postflight",
        "gpu_idle_preflight",
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
        or not isinstance(value.get("gpu_idle_preflight"), dict)
        or not isinstance(value.get("gpu_idle_postflight"), dict)
        or not isinstance(value.get("toolchain"), dict)
    ):
        raise protocol.ProtocolError("invalid artifact-admission process provenance")
    launch = protocol.local_launch_module()
    contract = protocol.load_contract()
    launch.validate_gpu_snapshot(value["gpu_preflight"], contract["manifest"]["hardware"])
    launch.validate_idle_snapshot(value["gpu_idle_preflight"], "record_pre")
    launch.validate_idle_snapshot(value["gpu_idle_postflight"], "record_post")
    if value["toolchain"] != contract["manifest"]["toolchain"]:
        raise protocol.ProtocolError("artifact-admission toolchain changed")
    return value


def make_entry(row: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    paths = entry_paths(row["cell_id"])
    build = validate_build_record(protocol.read_json(paths["build"]), row)
    verify = validate_verify_record(protocol.read_json(paths["verify"]), row, build, cell)
    artifact = {
        "cell_id": row["cell_id"],
        "config": build["config"],
        "implementation_sha256": build["implementation_sha256"],
        "cache_files_sha256": build["cache"]["files_sha256"],
        "generated_sources": build["cache"]["generated_sources"],
        "loadable_code_objects": build["cache"]["loadable_code_objects"],
    }
    return {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_admitted_artifact",
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": row["cell_id"],
        "role": row["role"],
        "admission_row_sha256": protocol.canonical_sha256(row),
        "artifact_sha256": protocol.canonical_sha256(artifact),
        "build_record_path": protocol.repo_path(paths["build"]),
        "build_record_sha256": protocol.file_sha256(paths["build"]),
        "cache": build["cache"],
        "config": build["config"],
        "implementation_sha256": build["implementation_sha256"],
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
        raise protocol.ProtocolError("v1 incident receipt changed")
    if protocol.file_sha256(ADMISSION_INCIDENT_PATH) != ADMISSION_INCIDENT_SHA256:
        raise protocol.ProtocolError("v2 admission incident receipt changed")
    if protocol.file_sha256(FRONTEND_INCIDENT_PATH) != FRONTEND_INCIDENT_SHA256:
        raise protocol.ProtocolError("v3 frontend incident receipt changed")
    if protocol.file_sha256(READINESS_INCIDENT_PATH) != READINESS_INCIDENT_SHA256:
        raise protocol.ProtocolError("v4 readiness incident receipt changed")
    plan = admission_plan()
    entries = []
    for row in plan:
        path = entry_paths(row["cell_id"])["entry"]
        value = validate_entry(protocol.read_json(path), row, cells[row["cell_id"]])
        entries.append(
            {
                "artifact_sha256": value["artifact_sha256"],
                "cell_id": row["cell_id"],
                "entry_path": protocol.repo_path(path),
                "entry_sha256": protocol.file_sha256(path),
                "role": row["role"],
            }
        )
    return {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_artifact_admission_manifest",
        "campaign_id": protocol.CAMPAIGN_ID,
        "complete": True,
        "admission_plan": plan,
        "admission_plan_sha256": protocol.canonical_sha256(plan),
        "entries": entries,
        "entry_count": 7,
        "execution_lock_sha256": protocol.file_sha256(protocol.EXECUTION_LOCK_PATH),
        "launch_receipt_path": protocol.repo_path(launch_receipt_path),
        "launch_receipt_sha256": protocol.file_sha256(launch_receipt_path),
        "performance_blind": True,
        "predecessor_admission_incident_path": protocol.repo_path(ADMISSION_INCIDENT_PATH),
        "predecessor_admission_incident_sha256": ADMISSION_INCIDENT_SHA256,
        "predecessor_frontend_incident_path": protocol.repo_path(FRONTEND_INCIDENT_PATH),
        "predecessor_frontend_incident_sha256": FRONTEND_INCIDENT_SHA256,
        "predecessor_incident_path": protocol.repo_path(INCIDENT_PATH),
        "predecessor_incident_sha256": INCIDENT_SHA256,
        "predecessor_readiness_incident_path": protocol.repo_path(READINESS_INCIDENT_PATH),
        "predecessor_readiness_incident_sha256": READINESS_INCIDENT_SHA256,
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
        "admitted_entry_path": row["entry_path"],
        "admitted_entry_sha256": row["entry_sha256"],
    }
