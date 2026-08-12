#!/usr/bin/env python3
"""Tag-scoped, performance-blind admission for transfer artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import protocol


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".llir", ".ptx", ".source", ".ttgir", ".ttir"}
_ENTRY_ID = re.compile(r"tt4_[0-9a-f]{24}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
TILELANG_OFF_FRONTEND_KEY = "b99b4b29b6aac24d9a91bfb41567cad2e43e368d3164e753aa5a54ceb327b6a7"
_TILELANG_KERNEL_FILES = {
    "device_kernel.cu", "executable.so", "host_kernel.cu", "params.pkl", "prim_func.pkl",
}
_TILELANG_EXACT_LOAD_STATE: dict[str, Any] | None = None


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
        "seed": entry / "pair_seed.json",
    }


def _file_rows(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise protocol.ProtocolError(f"unsafe cache directory: {root}")
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise protocol.ProtocolError(f"cache directory contains a symlink or special file: {path}")
        if path.is_file():
            if path.stat().st_size <= 0:
                raise protocol.ProtocolError(f"cache directory contains an empty file: {path}")
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": protocol.file_sha256(path),
                "size": path.stat().st_size,
            })
    return rows


def _tilelang_layout(entry_id: str, root: Path) -> tuple[Path, Path, Path]:
    tilelang = entry_paths(entry_id, root)["cache"] / "tilelang"
    namespaces = [path for path in tilelang.iterdir()] if tilelang.is_dir() else []
    if len(namespaces) != 1 or not namespaces[0].is_dir() or namespaces[0].is_symlink():
        raise protocol.ProtocolError("TileLang cache namespace census is not exactly one")
    namespace = namespaces[0]
    frontend, kernels, staging = namespace / "frontend", namespace / "kernels", namespace / ".staging"
    if (
        {path.name for path in namespace.iterdir()} != {".staging", "frontend", "kernels"}
        or not frontend.is_dir() or frontend.is_symlink()
        or not kernels.is_dir() or kernels.is_symlink()
        or not staging.is_dir() or staging.is_symlink()
        or any(staging.iterdir())
    ):
        raise protocol.ProtocolError("TileLang frontend/kernel cache roots are missing or unsafe")
    return namespace, frontend, kernels


def _frontend_entries(frontend: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in frontend.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix != ".json" \
                or _SHA256.fullmatch(path.stem) is None:
            raise protocol.ProtocolError("TileLang frontend cache contains a foreign entry")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise protocol.ProtocolError("TileLang frontend cache entry is unreadable") from exc
        if not isinstance(value, dict) or set(value) != {"kernel_key"} \
                or _SHA256.fullmatch(str(value.get("kernel_key", ""))) is None:
            raise protocol.ProtocolError("TileLang frontend cache entry is malformed")
        entries[path.stem] = {
            "kernel_key": value["kernel_key"],
            "path": path,
            "sha256": protocol.file_sha256(path),
            "size": path.stat().st_size,
        }
    return entries


def _kernel_binding(kernel: Path) -> dict[str, Any]:
    if kernel.is_symlink() or not kernel.is_dir() or _SHA256.fullmatch(kernel.name) is None:
        raise protocol.ProtocolError("TileLang kernel cache entry is unsafe")
    if {path.name for path in kernel.iterdir()} != _TILELANG_KERNEL_FILES:
        raise protocol.ProtocolError("TileLang kernel cache entry is not the exact five-file closure")
    files = _file_rows(kernel)
    if {item["path"] for item in files} != _TILELANG_KERNEL_FILES:
        raise protocol.ProtocolError("TileLang kernel cache entry is not the exact five-file closure")
    selected = {item["path"]: item for item in files}
    return {
        "schema_version": 1,
        "kernel_key": kernel.name,
        "files": files,
        "files_sha256": protocol.canonical_sha256(files),
        "device_source_sha256": selected["device_kernel.cu"]["sha256"],
        "host_source_sha256": selected["host_kernel.cu"]["sha256"],
        "executable_sha256": selected["executable.so"]["sha256"],
    }


def tilelang_held_gemm_cache_binding(
    row: dict[str, Any], built_or_record: Any, root: Path,
) -> dict[str, Any] | None:
    """Bind the exact generated GEMM cache entry, not merely its schedule/key."""
    if row.get("destination") != "tilelang":
        return None
    metadata = (
        built_or_record.metadata
        if hasattr(built_or_record, "metadata")
        else built_or_record.get("source_build_metadata", {})
    )
    source = metadata.get("held_gemm_source_binding", {}).get("source_sha256")
    if not isinstance(source, str) or _SHA256.fullmatch(source) is None:
        raise protocol.ProtocolError("TileLang build lacks its held GEMM device-source digest")
    namespace, frontend, kernels = _tilelang_layout(row["entry_id"], root)
    matches = []
    for kernel in kernels.iterdir():
        device = kernel / "device_kernel.cu"
        if kernel.is_dir() and not kernel.is_symlink() and device.is_file() \
                and not device.is_symlink() and protocol.file_sha256(device) == source:
            matches.append(kernel)
    if len(matches) != 1 or _SHA256.fullmatch(matches[0].name) is None:
        raise protocol.ProtocolError("TileLang held GEMM cache entry is absent or ambiguous")
    kernel = matches[0]
    binding = _kernel_binding(kernel)
    aliases = _frontend_entries(frontend)
    if row.get("mechanism_enabled"):
        if any(alias["kernel_key"] == kernel.name for alias in aliases.values()):
            raise protocol.ProtocolError("TileLang on arm may not reach the held GEMM through a frontend alias")
    elif aliases.get(TILELANG_OFF_FRONTEND_KEY, {}).get("kernel_key") != kernel.name:
        raise protocol.ProtocolError("TileLang off held GEMM frontend alias missed its frozen key")
    return {**binding, "cache_namespace": namespace.name}


def _paired_off_row(on_row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row for row in manifest["rows"]
        if row.get("destination") == "tilelang"
        and row.get("mechanism_state") == "off"
        and row.get("adaptation") == on_row.get("adaptation")
        and row.get("grid_id") == on_row.get("grid_id")
    ]
    if len(matches) != 1:
        raise protocol.ProtocolError("TileLang on row has no unique paired off row")
    return matches[0]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise protocol.ProtocolError("TileLang pair cache path escapes its admission root") from exc


def _write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True))


def _validated_off_pair(
    on_row: dict[str, Any], manifest: dict[str, Any], root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], dict[str, Any]]:
    off_row = _paired_off_row(on_row, manifest)
    off_paths = entry_paths(off_row["entry_id"], root)
    if not off_paths["build"].is_file() or not off_paths["verify"].is_file():
        raise protocol.ProtocolError("TileLang pair seed requires a completed fresh-v4 off artifact")
    off_build = validate_build_record(protocol.read_json(off_paths["build"]), off_row, root)
    validate_verify_record(protocol.read_json(off_paths["verify"]), off_row, off_build, root)
    binding = tilelang_held_gemm_cache_binding(off_row, off_build, root)
    assert binding is not None
    _namespace, frontend, kernels = _tilelang_layout(off_row["entry_id"], root)
    aliases = _frontend_entries(frontend)
    if set(aliases) != {TILELANG_OFF_FRONTEND_KEY} \
            or {path.name for path in kernels.iterdir()} != {binding["kernel_key"]}:
        raise protocol.ProtocolError("TileLang off artifact is not the exact one-GEMM cache closure")
    return off_row, off_build, off_paths, binding


def _seed_rows(on_row: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    tilelang = entry_paths(on_row["entry_id"], root)["cache"] / "tilelang"
    return _file_rows(tilelang)


def _seed_receipt(
    on_row: dict[str, Any], off_row: dict[str, Any], off_paths: dict[str, Path],
    binding: dict[str, Any], root: Path, seeded_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    off_namespace, off_frontend, off_kernels = _tilelang_layout(off_row["entry_id"], root)
    on_paths = entry_paths(on_row["entry_id"], root)
    on_namespace, on_frontend, on_kernels = _tilelang_layout(on_row["entry_id"], root)
    source_alias = off_frontend / f"{TILELANG_OFF_FRONTEND_KEY}.json"
    if source_alias.is_symlink() or not source_alias.is_file():
        raise protocol.ProtocolError("TileLang pair source alias is missing or unsafe")
    rows = _seed_rows(on_row, root) if seeded_rows is None else seeded_rows
    load_contract = {
        "method": "tilelang_tvm_ffi_exact_kernel_key_v1",
        "kernel_key": binding["kernel_key"],
        "cache_namespace": binding["cache_namespace"],
        "device_source_sha256": binding["device_source_sha256"],
        "frontend_lookup_used": False,
        "fallback_lowering_allowed": False,
    }
    return {
        "schema_version": 1,
        "record_type": "trajectory_transfer_ada_v4_tilelang_pair_seed",
        "campaign_id": protocol.CAMPAIGN_ID,
        "off_entry_id": off_row["entry_id"],
        "off_build_record_sha256": protocol.file_sha256(off_paths["build"]),
        "off_verify_record_sha256": protocol.file_sha256(off_paths["verify"]),
        "on_entry_id": on_row["entry_id"],
        "on_admission_row_sha256": protocol.canonical_sha256(on_row),
        "off_frontend_key": TILELANG_OFF_FRONTEND_KEY,
        "source_paths": {
            "build_record": _relative(off_paths["build"], root),
            "verify_record": _relative(off_paths["verify"], root),
            "frontend_alias": _relative(source_alias, root),
            "kernel_directory": _relative(off_kernels / binding["kernel_key"], root),
        },
        "destination_paths": {
            "kernel_directory": _relative(on_kernels / binding["kernel_key"], root),
        },
        "source_frontend_alias_sha256": protocol.file_sha256(source_alias),
        "held_gemm_cache": binding,
        "exact_load_contract": load_contract,
        "seeded_cache_files": rows,
        "seeded_cache_files_sha256": protocol.canonical_sha256(rows),
        "post_build_contract": {
            "fresh_frontend_entries": 1,
            "fresh_kernel_entries": 1,
            "held_gemm_recompile_or_mutation_allowed": False,
        },
        "performance_observations": [],
    }


def _require_distinct_copies(
    source: Path, destination: Path, binding: dict[str, Any],
) -> None:
    for row in binding["files"]:
        before, after = source / row["path"], destination / row["path"]
        if (before.stat().st_dev, before.stat().st_ino) == (after.stat().st_dev, after.stat().st_ino):
            raise protocol.ProtocolError("TileLang pair seed may not hard-link held GEMM files")


def _validate_seeded_cache(
    on_row: dict[str, Any], receipt: dict[str, Any], root: Path, *, post_build: bool,
) -> None:
    on_paths = entry_paths(on_row["entry_id"], root)
    namespace, frontend, kernels = _tilelang_layout(on_row["entry_id"], root)
    aliases = _frontend_entries(frontend)
    held = receipt["held_gemm_cache"]
    if any(alias["kernel_key"] == held["kernel_key"] for alias in aliases.values()):
        raise protocol.ProtocolError("TileLang on held GEMM may not have a frontend alias")
    current = {row["path"]: row for row in _seed_rows(on_row, root)}
    seeded = {row["path"]: row for row in receipt["seeded_cache_files"]}
    expected_seeded_paths = {
        *(f"{namespace.name}/kernels/{held['kernel_key']}/{name}"
          for name in _TILELANG_KERNEL_FILES),
    }
    if set(seeded) != expected_seeded_paths:
        raise protocol.ProtocolError("TileLang seed receipt does not bind the exact five-file closure")
    if any(current.get(path) != row for path, row in seeded.items()):
        raise protocol.ProtocolError("TileLang seeded held GEMM cache was mutated")
    if not post_build:
        if aliases \
                or {path.name for path in kernels.iterdir()} != {held["kernel_key"]} \
                or list(current.values()) != receipt["seeded_cache_files"]:
            raise protocol.ProtocolError("TileLang pre-build seed cache contains foreign entries")
        if any(path.name != "tilelang" or not path.is_dir() or path.is_symlink()
               for path in on_paths["tmp"].iterdir()):
            raise protocol.ProtocolError("TileLang pre-build runtime temp is not fresh")
        if any((on_paths["tmp"] / "tilelang").iterdir()):
            raise protocol.ProtocolError("TileLang pre-build runtime temp is not empty")
        return
    if len(aliases) != 1:
        raise protocol.ProtocolError("TileLang post-build frontend census is not exactly fresh F1")
    f1_frontend_key, f1_alias = next(iter(aliases.items()))
    f1_key = f1_alias["kernel_key"]
    if f1_key == held["kernel_key"] or {path.name for path in kernels.iterdir()} != {held["kernel_key"], f1_key}:
        raise protocol.ProtocolError("TileLang post-build kernel census is not held GEMM plus fresh F1")
    f1 = _kernel_binding(kernels / f1_key)
    expected_new = {
        f"{namespace.name}/frontend/{f1_frontend_key}.json",
        *(f"{namespace.name}/kernels/{f1_key}/{row['path']}" for row in f1["files"]),
    }
    if set(current) - set(seeded) != expected_new or set(seeded) - set(current):
        raise protocol.ProtocolError("TileLang post-build cache delta contains foreign files")


def _post_build_seed_binding(
    on_row: dict[str, Any], built_or_record: Any, receipt: dict[str, Any], root: Path,
) -> dict[str, Any]:
    _validate_seeded_cache(on_row, receipt, root, post_build=True)
    held = tilelang_held_gemm_cache_binding(on_row, built_or_record, root)
    if held != receipt["held_gemm_cache"]:
        raise protocol.ProtocolError("TileLang on build did not retain the exact seeded GEMM cache")
    metadata = (
        built_or_record.metadata
        if hasattr(built_or_record, "metadata")
        else built_or_record.get("source_build_metadata", {})
    )
    reported = metadata.get("artifacts", {})
    if reported.get("pair_seed") != receipt \
            or reported.get("seeded_gemm_cache_key") != held["kernel_key"] \
            or reported.get("seeded_gemm_exact_load") != receipt["exact_load_contract"] \
            or reported.get("seeded_gemm_fallback_compile_forbidden") is not True \
            or reported.get("tilelang_compile_mode") != "admit" \
            or not isinstance(reported.get("tilelang_fresh_compile_qualnames"), list) \
            or len(reported["tilelang_fresh_compile_qualnames"]) != 1 \
            or "_f1.<locals>._k" not in reported["tilelang_fresh_compile_qualnames"][0]:
        raise protocol.ProtocolError("TileLang on build lacks its no-fallback pair-seed binding")
    namespace, frontend, kernels = _tilelang_layout(on_row["entry_id"], root)
    aliases = _frontend_entries(frontend)
    f1_frontend_key, f1_alias = next(iter(aliases.items()))
    f1_key = f1_alias["kernel_key"]
    f1 = _kernel_binding(kernels / f1_key)
    delta = [
        row for row in _seed_rows(on_row, root)
        if row["path"] not in {item["path"] for item in receipt["seeded_cache_files"]}
    ]
    return {
        "schema_version": 1,
        "cache_namespace": namespace.name,
        "fresh_f1_frontend_key": f1_frontend_key,
        "fresh_f1_frontend_sha256": f1_alias["sha256"],
        "fresh_f1_kernel": f1,
        "post_build_delta_files": delta,
        "post_build_delta_sha256": protocol.canonical_sha256(delta),
        "seed_receipt_sha256": protocol.file_sha256(entry_paths(on_row["entry_id"], root)["seed"]),
    }


def seed_tilelang_pair(
    on_row: dict[str, Any], manifest: dict[str, Any], root: Path,
) -> dict[str, Any]:
    """Seed an on-arm cache only from its freshly admitted v4 off-arm GEMM."""
    if on_row.get("destination") != "tilelang" or on_row.get("mechanism_state") != "on":
        raise protocol.ProtocolError("only a TileLang on arm may receive a pair seed")
    if any(part in {"trajectory_transfer_ada_v2", "trajectory_transfer_ada_v3"}
           for part in root.resolve().parts):
        raise protocol.ProtocolError("v4 pair seeding may not read a predecessor result path")
    off_row, _off_build, off_paths, binding = _validated_off_pair(on_row, manifest, root)
    on_paths = entry_paths(on_row["entry_id"], root)
    if on_paths["root"].exists() or on_paths["root"].is_symlink():
        raise protocol.ProtocolError("TileLang on seed target already exists")
    off_namespace, _off_frontend, off_kernels = _tilelang_layout(off_row["entry_id"], root)
    on_paths["cache"].mkdir(parents=True)
    on_paths["tmp"].mkdir()
    (on_paths["tmp"] / "tilelang").mkdir()
    on_namespace = on_paths["cache"] / "tilelang" / binding["cache_namespace"]
    (on_namespace / "frontend").mkdir(parents=True)
    (on_namespace / "kernels").mkdir()
    (on_namespace / ".staging").mkdir()
    source_kernel = off_kernels / binding["kernel_key"]
    target_kernel = on_namespace / "kernels" / binding["kernel_key"]
    shutil.copytree(source_kernel, target_kernel, copy_function=shutil.copy2)
    _require_distinct_copies(source_kernel, target_kernel, binding)
    seeded = tilelang_held_gemm_cache_binding(
        on_row, {"source_build_metadata": {"held_gemm_source_binding": {
            "source_sha256": binding["device_source_sha256"],
        }}}, root,
    )
    if seeded != binding:
        raise protocol.ProtocolError("TileLang pair seed changed the held GEMM cache bytes")
    receipt = _seed_receipt(on_row, off_row, off_paths, binding, root)
    _write_new_json(on_paths["seed"], receipt)
    _validate_seeded_cache(on_row, receipt, root, post_build=False)
    return receipt


def validate_tilelang_pair_seed(
    on_row: dict[str, Any], manifest: dict[str, Any], root: Path,
    built_or_record: Any | None = None,
) -> dict[str, Any]:
    if on_row.get("destination") != "tilelang" or on_row.get("mechanism_state") != "on":
        raise protocol.ProtocolError("only a TileLang on arm may carry a pair seed")
    off_row, _off_build, off_paths, binding = _validated_off_pair(on_row, manifest, root)
    on_paths = entry_paths(on_row["entry_id"], root)
    value = protocol.read_json(on_paths["seed"])
    retained_rows = value.get("seeded_cache_files") if isinstance(value, dict) else None
    if not isinstance(retained_rows, list) \
            or value.get("seeded_cache_files_sha256") != protocol.canonical_sha256(retained_rows):
        raise protocol.ProtocolError("TileLang pair seed receipt cache census is malformed")
    expected = _seed_receipt(on_row, off_row, off_paths, binding, root, retained_rows)
    if value != expected:
        raise protocol.ProtocolError("TileLang pair seed receipt/cache changed")
    if built_or_record is None and on_paths["build"].is_file():
        built_or_record = protocol.read_json(on_paths["build"])
    if built_or_record is None:
        _validate_seeded_cache(on_row, value, root, post_build=False)
    else:
        _post_build_seed_binding(on_row, built_or_record, value, root)
    return value


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
    metadata = _clean_metadata(built.metadata)
    held_cache = tilelang_held_gemm_cache_binding(row, built, root)
    if held_cache is not None:
        metadata["held_gemm_cache_binding"] = held_cache
    pair_seed = None
    pair_post = None
    if row.get("destination") == "tilelang" and row.get("mechanism_state") == "on":
        manifest = protocol.read_json(protocol.ADMISSION_MANIFEST_PATH)
        pair_seed = validate_tilelang_pair_seed(row, manifest, root, built)
        pair_post = _post_build_seed_binding(row, built, pair_seed, root)
    return {
        "schema_version": 1,
        "record_type": "trajectory_transfer_ada_v4_artifact_build_record",
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
        "source_build_metadata": metadata,
        "tilelang_pair_seed": pair_seed,
        "tilelang_pair_postcondition": pair_post,
        "torch_inline_requests": native.bind_torch_requests(torch_requests, snapshot),
    }


def validate_build_record(value: Any, row: dict[str, Any], root: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise protocol.ProtocolError(f"invalid performance-blind build record: {row['entry_id']}")
    validate_dynamic_work_audit(value.get("dynamic_work_audit"))
    if (
        value.get("record_type") != "trajectory_transfer_ada_v4_artifact_build_record"
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
    held_cache = tilelang_held_gemm_cache_binding(row, value, root)
    metadata = value.get("source_build_metadata", {})
    if metadata.get("held_gemm_cache_binding") != held_cache:
        raise protocol.ProtocolError("TileLang held GEMM cache binding changed")
    seeded = row.get("destination") == "tilelang" and row.get("mechanism_state") == "on"
    if seeded:
        manifest = protocol.read_json(protocol.ADMISSION_MANIFEST_PATH)
        receipt = validate_tilelang_pair_seed(row, manifest, root, value)
        post = _post_build_seed_binding(row, value, receipt, root)
        if value.get("tilelang_pair_seed") != receipt \
                or value.get("tilelang_pair_postcondition") != post:
            raise protocol.ProtocolError("TileLang pair seed build binding changed")
    elif value.get("tilelang_pair_seed") is not None \
            or value.get("tilelang_pair_postcondition") is not None:
        raise protocol.ProtocolError("unseeded artifact carries TileLang pair evidence")
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


def record_exact_tilelang_gemm_load(kernel: Any, binding: dict[str, Any]) -> None:
    """Add the alias-free held-GEMM hit to the active native load-evidence census."""
    state = _TILELANG_EXACT_LOAD_STATE
    mode = os.environ.get("TRAJECTORY_TRANSFER_ARTIFACT_MODE")
    if mode == "admit":
        return
    if mode != "load_only" or not isinstance(state, dict) or state.get("count") != 0:
        raise protocol.ProtocolError("exact TileLang GEMM load occurred outside its one-hit guard")
    build = state["build"]
    if binding != build.get("tilelang_pair_seed", {}).get("exact_load_contract"):
        raise protocol.ProtocolError("exact TileLang GEMM load differs from its admitted seed")
    cache_path = Path(str(getattr(kernel, "_tilelang_cache_path", ""))).resolve()
    cache_root = (protocol.REPO_ROOT / build["cache"]["cache_root"]).resolve()
    try:
        relative = (cache_path / "executable.so").relative_to(cache_root).as_posix()
    except ValueError as exc:
        raise protocol.ProtocolError("exact TileLang GEMM load escaped its admitted cache") from exc
    rows = [row for row in build["cache"]["files"] if row.get("path") == relative]
    if len(rows) != 1 or protocol.file_sha256(cache_root / relative) != rows[0].get("sha256"):
        raise protocol.ProtocolError("exact TileLang GEMM load lacks its admitted executable")
    state["evidence"]["tilelang_cache_hits"].append({
        "cache_path": str(cache_path),
        "loadable_code_objects": rows,
        "requested_out_idx": [-1],
        "resolved_out_idx": [-1],
    })
    state["count"] = 1


@contextmanager
def load_only_guards(build: dict[str, Any]) -> Iterator[dict[str, Any]]:
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3 import artifacts as native

    compatible = {**build, "cell_id": build["admission_row"]["coordinate_cell_id"]}
    _runtime_tmp_from_build(build)
    global _TILELANG_EXACT_LOAD_STATE
    seeded = (
        build.get("admission_row", {}).get("destination") == "tilelang"
        and build.get("admission_row", {}).get("mechanism_state") == "on"
    )
    if _TILELANG_EXACT_LOAD_STATE is not None:
        raise protocol.ProtocolError("nested exact TileLang load guard")
    with native.load_only_guards(compatible) as evidence:
        _TILELANG_EXACT_LOAD_STATE = {"build": build, "evidence": evidence, "count": 0}
        try:
            yield evidence
            if _TILELANG_EXACT_LOAD_STATE["count"] != int(seeded):
                raise protocol.ProtocolError("exact TileLang GEMM load census changed")
        finally:
            _TILELANG_EXACT_LOAD_STATE = None
    _runtime_tmp_from_build(build)


def validate_load_evidence(value: Any, row: dict[str, Any], build: dict[str, Any]) -> dict[str, Any]:
    from ako_runs.controlled_followup.native_trajectory_replication_ada_v3 import artifacts as native

    compatible = {**build, "cell_id": row["coordinate_cell_id"]}
    native.validate_load_evidence(value, _source_cell(row), compatible)
    if row.get("destination") == "tilelang" and row.get("mechanism_state") == "on":
        seed = build.get("tilelang_pair_seed", {})
        expected = (
            (protocol.REPO_ROOT / build["cache"]["cache_root"]).resolve().parents[2]
            / seed.get("destination_paths", {}).get("kernel_directory", "")
        ).resolve()
        direct = [
            hit for hit in value["tilelang_cache_hits"]
            if Path(str(hit.get("cache_path", ""))).resolve() == expected
        ]
        if len(direct) != 1 or direct[0].get("requested_out_idx") != [-1] \
                or direct[0].get("resolved_out_idx") != [-1]:
            raise protocol.ProtocolError("alias-free held-GEMM load evidence is absent or ambiguous")
    return value


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
        "record_type": "trajectory_transfer_ada_v4_artifact_verify_record",
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
        or value.get("record_type") != "trajectory_transfer_ada_v4_artifact_verify_record"
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
