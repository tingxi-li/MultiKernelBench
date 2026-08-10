#!/usr/bin/env python3
"""Hash and load only the exact TileLang objects admitted by this campaign."""
from __future__ import annotations

import ast
import importlib
import os
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import protocol


CUDA_ROOT = Path("/usr/local/cuda-13.1")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh"}


def configure_compiler_root(gpu: int | None = None) -> None:
    """Pin CUDA before any import can initialize TileLang's compiler state."""
    os.environ["CUDA_HOME"] = str(CUDA_ROOT)
    os.environ["CUDA_PATH"] = str(CUDA_ROOT)
    parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    cuda_bin = str(CUDA_ROOT / "bin")
    os.environ["PATH"] = os.pathsep.join([cuda_bin, *[part for part in parts if part != cuda_bin]])
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def exact_import(name: str, relative: str):
    """Import a legacy top-level module only from its hash-bound source path."""
    expected = protocol.repo_path(relative).resolve()
    loaded = sys.modules.get(name)
    if loaded is not None:
        actual = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if actual != expected:
            raise protocol.ProtocolError(f"generic module shadowed: {name}: {actual} != {expected}")
        return loaded
    search_root = expected.parent.parent if expected.name == "__init__.py" else expected.parent
    sys.path.insert(0, str(search_root))
    try:
        module = importlib.import_module(name)
    finally:
        try:
            sys.path.remove(str(search_root))
        except ValueError:
            pass
    actual = Path(str(getattr(module, "__file__", ""))).resolve()
    if actual != expected:
        raise protocol.ProtocolError(f"generic module shadowed: {name}: {actual} != {expected}")
    return module


def exact_robust_adapter():
    """Load the gate adapter only after pinning every generic robust_gate import."""
    modules = {
        "robust_gate": "ako_runs/controlled_followup/robust_gate/__init__.py",
        "robust_gate.distributions": "ako_runs/controlled_followup/robust_gate/distributions.py",
        "robust_gate.metrics": "ako_runs/controlled_followup/robust_gate/metrics.py",
        "robust_gate.oracles": "ako_runs/controlled_followup/robust_gate/oracles.py",
        "robust_gate.schema": "ako_runs/controlled_followup/robust_gate/schema.py",
        "robust_gate.seeds": "ako_runs/controlled_followup/robust_gate/seeds.py",
        "robust_gate.validate": "ako_runs/controlled_followup/robust_gate/validate.py",
    }
    for name, relative in modules.items():
        exact_import(name, relative)
    return exact_import(
        "robust_adapter", "ako_runs/controlled_followup/fused_grid/robust_adapter.py"
    )


def exact_recovery_audit():
    """Load the gate-summary recovery with its complete executed module closure."""
    prefix = "ako_runs.controlled_followup.fused_epilogue_crossed_v1.recovery_v1r1"
    modules = {
        prefix: "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/__init__.py",
        f"{prefix}.common": "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/common.py",
        f"{prefix}.validate": "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/validate.py",
        f"{prefix}.audit": "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/audit.py",
    }
    loaded = None
    for name, relative in modules.items():
        loaded = exact_import(name, relative)
    return loaded


def artifact_paths(root: Path) -> dict[str, Path]:
    root = root.resolve()
    return {"root": root, "cache": root / "cache", "tmp": root / "runtime_tmp"}


def cache_environment(root: Path, mode: str) -> dict[str, str]:
    if mode not in {"admit", "load_only"}:
        raise protocol.ProtocolError(f"unknown artifact mode: {mode}")
    paths = artifact_paths(root)
    environment = {
        "TILELANG_ABSTRACTION_ARTIFACT_MODE": mode,
        "PHASE2_TL_CACHE": "1",
        "TILELANG_CACHE_DIR": str(paths["cache"] / "tilelang"),
        "TILELANG_EXECUTION_BACKEND": "tvm_ffi",
        "TILELANG_TMP_DIR": str(paths["tmp"] / "tilelang"),
        "TILELANG_TARGET": "cuda",
        "TILELANG_DISABLE_CACHE": "0",
        "TILELANG_CLEAR_CACHE": "0",
        "TORCH_EXTENSIONS_DIR": str(paths["cache"] / "torch_extensions"),
    }
    if mode == "admit":
        environment.update({name: str(paths["tmp"]) for name in ("TMPDIR", "TMP", "TEMP")})
    return environment


def prepare_environment(root: Path, mode: str, gpu: int) -> None:
    configure_compiler_root(gpu)
    if "tilelang" in sys.modules or "tilelang.cache" in sys.modules:
        raise protocol.ProtocolError("artifact environment must be installed before importing TileLang")
    if os.environ.get("TILELANG_ABSTRACTION_CAPTURE_DIR") or os.environ.get("PHASE1_TL_ASM"):
        raise protocol.ProtocolError("legacy raw PTX/SASS capture is forbidden")
    paths = artifact_paths(root)
    if mode == "admit":
        if paths["root"].exists():
            raise FileExistsError(f"refusing existing admitted artifact root: {paths['root']}")
        paths["cache"].mkdir(parents=True)
        paths["tmp"].mkdir()
        (paths["tmp"] / "tilelang").mkdir()
    elif not all(paths[key].is_dir() and not paths[key].is_symlink() for key in ("cache", "tmp")):
        raise protocol.ProtocolError("admitted artifact cache/temp root is missing or unsafe")
    if paths["cache"].stat().st_dev != paths["tmp"].stat().st_dev:
        raise protocol.ProtocolError("TileLang cache and atomic temp directory are on different filesystems")
    os.environ.update(cache_environment(root, mode))


def _is_code_object(path: Path) -> bool:
    return path.suffix == ".cubin" or path.name.endswith(".so")


def cache_snapshot(root: Path) -> dict[str, Any]:
    cache = artifact_paths(root)["cache"]
    if not cache.is_dir() or cache.is_symlink():
        raise protocol.ProtocolError("admitted cache is missing or unsafe")
    files = []
    for path in sorted(cache.rglob("*"), key=lambda item: item.relative_to(cache).as_posix()):
        if path.is_symlink():
            raise protocol.ProtocolError(f"admitted cache contains a symlink: {path}")
        if path.is_file():
            if path.stat().st_size <= 0:
                raise protocol.ProtocolError(f"admitted cache contains an empty file: {path}")
            files.append({
                "path": path.relative_to(cache).as_posix(),
                "sha256": protocol.file_sha256(path),
                "bytes": path.stat().st_size,
            })
    sources = [row for row in files if Path(row["path"]).suffix in SOURCE_SUFFIXES]
    code = [row for row in files if _is_code_object(Path(row["path"]))]
    if not sources or len(code) != 2 or len({row["sha256"] for row in code}) != 2:
        raise protocol.ProtocolError("admission must retain source and two distinct TileLang code objects")
    return {
        "root": protocol.repo_relative(cache),
        "file_count": len(files),
        "files": files,
        "files_sha256": protocol.canonical_sha256(files),
        "generated_sources": sources,
        "loadable_code_objects": code,
    }


def validate_cache_receipt(value: Any, expected_root: Path) -> str:
    if not isinstance(value, dict) or value != cache_snapshot(expected_root):
        raise protocol.ProtocolError("admitted cache receipt differs from retained bytes")
    return value["files_sha256"]


def _allowed_files(cache: dict[str, Any]) -> dict[Path, dict[str, Any]]:
    root = (protocol.REPO / cache["root"]).resolve()
    return {(root / row["path"]).resolve(): row for row in cache["files"]}


def _verify_bound_file(path: Path, allowed: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    resolved = path.resolve()
    row = allowed.get(resolved)
    if row is None or not resolved.is_file() or protocol.file_sha256(resolved) != row["sha256"]:
        raise protocol.ProtocolError(f"runtime attempted to load an unadmitted object: {path}")
    return row


def _normalized_out_idx_candidates(frontend_key_data: Any, out_idx: Any) -> list[list[int]]:
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

    def prim_func(decorator: ast.expr) -> bool:
        decorator = decorator.func if isinstance(decorator, ast.Call) else decorator
        return (isinstance(decorator, ast.Name) and decorator.id == "prim_func") or (
            isinstance(decorator, ast.Attribute) and decorator.attr == "prim_func"
        )

    arities = sorted({
        len(node.args.posonlyargs) + len(node.args.args)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(prim_func(decorator) for decorator in node.decorator_list)
        and node.args.vararg is None and node.args.kwarg is None and not node.args.kwonlyargs
    })
    return [
        normalized for arity in arities
        if (normalized := [arity + index if index < 0 else index for index in out_idx]) != out_idx
        and all(0 <= index < arity for index in normalized)
    ]


@contextmanager
def load_only_guards(cache: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Reject compilation/writes and record both admitted executable loads."""
    tilelang_cache = importlib.import_module("tilelang.cache")
    kernel_cache = importlib.import_module("tilelang.cache.kernel_cache")
    tilelang_jit = importlib.import_module("tilelang.jit")
    allowed = _allowed_files(cache)
    evidence: dict[str, Any] = {"mode": "load_only", "tilelang_cache_hits": []}
    original_load = tilelang_cache.load_frontend_cached
    original_compile = tilelang_jit.JITImpl.compile
    original_write = kernel_cache.KernelCache._safe_write_file

    def load(*args, **kwargs):
        kernel = original_load(*args, **kwargs)
        if kernel is None:
            hits = []
            frontend = args[0] if args else kwargs.get("frontend_key_data")
            for candidate in _normalized_out_idx_candidates(frontend, kwargs.get("out_idx")):
                retry = original_load(*args, **{**kwargs, "out_idx": candidate})
                if retry is not None:
                    hits.append(retry)
            if len(hits) != 1:
                raise protocol.ProtocolError("TileLang admitted frontend-cache miss")
            kernel = hits[0]
        cache_path = Path(str(getattr(kernel, "_tilelang_cache_path", ""))).resolve()
        rows = [
            _verify_bound_file(path, allowed)
            for path in allowed
            if path.parent == cache_path and _is_code_object(path)
        ]
        if len(rows) != 1:
            raise protocol.ProtocolError("TileLang cache hit does not bind one admitted executable")
        evidence["tilelang_cache_hits"].append({"cache_path": str(cache_path), "code_object": rows[0]})
        return kernel

    def reject_compile(*_args, **_kwargs):
        raise protocol.ProtocolError("TileLang compilation is forbidden in load-only mode")

    def reject_write(*_args, **_kwargs):
        raise protocol.ProtocolError("TileLang cache writes are forbidden in load-only mode")

    tilelang_cache.load_frontend_cached = load
    tilelang_jit.JITImpl.compile = reject_compile
    kernel_cache.KernelCache._safe_write_file = staticmethod(reject_write)
    try:
        yield evidence
    finally:
        kernel_cache.KernelCache._safe_write_file = staticmethod(original_write)
        tilelang_jit.JITImpl.compile = original_compile
        tilelang_cache.load_frontend_cached = original_load


def validate_load_evidence(value: Any, cache: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"mode", "tilelang_cache_hits"} or value.get("mode") != "load_only":
        raise protocol.ProtocolError("invalid TileLang load evidence")
    admitted = {row["path"]: row for row in cache.get("files", [])}
    hits = value["tilelang_cache_hits"]
    if not isinstance(hits, list) or len(hits) != 2:
        raise protocol.ProtocolError("load-only build must hit exactly two TileLang cache entries")
    loaded = []
    for hit in hits:
        if not isinstance(hit, dict) or set(hit) != {"cache_path", "code_object"}:
            raise protocol.ProtocolError("malformed TileLang cache-hit evidence")
        row = hit["code_object"]
        if not isinstance(row, dict) or admitted.get(row.get("path")) != row or not _is_code_object(Path(row["path"])):
            raise protocol.ProtocolError("load evidence names an unadmitted code object")
        if ((protocol.REPO / cache["root"]) / row["path"]).resolve().parent != Path(hit["cache_path"]).resolve():
            raise protocol.ProtocolError("load evidence names another cache entry")
        loaded.append(row["path"])
    if len(set(loaded)) != 2:
        raise protocol.ProtocolError("load-only build did not load two distinct admitted executables")
    return value
