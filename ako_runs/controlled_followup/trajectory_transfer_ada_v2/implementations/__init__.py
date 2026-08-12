"""Hash-bound adapters for the selected accumulator-to-softmax transition."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


DESTINATIONS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
OFF_STRATEGY = "register_common_postprocess"
ON_STRATEGY = "register_fused"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HERE = Path(__file__).resolve().parent.parent
_REPO = _HERE.parents[2]
_PRIMITIVE_MAP = _HERE / "primitive_map.json"


class AdapterError(RuntimeError):
    pass


class UnsupportedRoute(AdapterError):
    pass


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _resolved_cells() -> tuple[dict[str, Any], ...]:
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import core

    return tuple(core.load_cells(require_resolved=True))


def _generated_source_bindings(value: Any, path: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(item, str) and str(key).endswith("source_sha256") and _SHA256.fullmatch(item):
                found.append({"path": child, "sha256": item})
            elif isinstance(item, str) and "source" in str(key):
                found.append({"path": child, "sha256": hashlib.sha256(item.encode()).hexdigest()})
            else:
                found.extend(_generated_source_bindings(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_generated_source_bindings(item, f"{path}[{index}]"))
    return found


def _paired_config(config: dict[str, Any]) -> dict[str, Any]:
    """Project away language/arm labels while retaining every execution control."""
    expected = {
        "dsl", "variant", "M", "N", "K", "BM", "BN", "BK", "threads",
        "kc", "stages", "arith", "cast", "extra", "input_dtype",
    }
    if (
        not isinstance(config, dict)
        or set(config) != expected
        or config.get("variant") not in {"GBG", "GBGS", "F1"}
        or not isinstance(config.get("extra"), dict)
    ):
        raise AdapterError("destination builder returned an unexpected arm configuration")
    return {key: value for key, value in config.items() if key not in {"dsl", "variant"}}


def _primitive_recipe(destination: str, route: str, spec: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(_PRIMITIVE_MAP.read_text(encoding="utf-8"))
        recipe = value["destinations"][destination]
    except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
        raise AdapterError("cannot resolve the frozen destination primitive recipe") from exc
    selected, review = recipe.get(route), recipe.get("review")
    off_coordinate = f"{OFF_STRATEGY}.{destination}.g01"
    on_coordinate = f"{ON_STRATEGY}.{destination}.g01"
    on_implementation = (
        "tilelang_abstraction.F1.g01" if destination == "tilelang" else on_coordinate
    )
    if (
        recipe.get("route") != route
        or not isinstance(selected, dict)
        or selected.get("status") != "ELIGIBLE"
        or selected.get("off_coordinate_cell_id") != off_coordinate
        or selected.get("off_implementation_id") != off_coordinate
        or selected.get("on_coordinate_cell_id") != on_coordinate
        or selected.get("on_implementation_id") != on_implementation
        or not isinstance(review, dict)
        or selected.get("recipe", {}).get("source_review_sha256") != _canonical_sha256(review)
    ):
        raise AdapterError("destination primitive recipe is not selected and pre-reviewed")
    source = _REPO / str(review.get("source_path", ""))
    if not source.is_file() or _file_sha256(source) != review.get("source_sha256"):
        raise AdapterError("destination primitive recipe source is missing")
    absence = spec.get("primitive_absence_receipt")
    absence_sha256 = spec.get("primitive_absence_receipt_sha256")
    if route == "manual_reconstruction":
        shared = value.get("shared_receipts", {}).get("cuda_cpp_row_reduction_primitive_absence")
        if (
            destination not in {"cuda_noptx", "cuda_unlimited"}
            or not isinstance(shared, dict)
            or absence != shared
            or absence_sha256 != _canonical_sha256(shared)
            or selected.get("primitive_absence_receipt_sha256") != absence_sha256
        ):
            raise UnsupportedRoute("manual reconstruction lacks its exact primitive-absence receipt")
        manual_source = _REPO / str(selected.get("recipe", {}).get("manual_source_path", ""))
        if (
            not manual_source.is_file()
            or _file_sha256(manual_source) != selected.get("recipe", {}).get("manual_source_sha256")
        ):
            raise AdapterError("manual reconstruction source is missing or changed")
    elif route == "direct_primitive_mapping":
        if absence is not None or absence_sha256 is not None:
            raise UnsupportedRoute("direct mapping may not carry a primitive-absence receipt")
    else:
        raise UnsupportedRoute(f"unknown transfer route: {route}")
    return {
        "destination_recipe": recipe,
        "destination_recipe_sha256": _canonical_sha256(recipe),
        "primitive_map_sha256": _file_sha256(_PRIMITIVE_MAP),
        "reviewed_source_path": str(source.resolve().relative_to(_REPO.resolve())),
        "reviewed_source_sha256": _file_sha256(source),
    }


def _validated_spec(
    spec: dict[str, Any], destination: str, route: str, mechanism_enabled: bool
) -> tuple[str, str]:
    if not isinstance(spec, dict):
        raise AdapterError("adapter spec must be an object")
    required = {
        "origin_id", "donor_step_id", "destination", "origin_job",
        "primitive_graph_sha256",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise AdapterError(f"adapter spec is missing: {missing}")
    if destination not in DESTINATIONS or spec["destination"] != destination:
        raise AdapterError("adapter destination changed")
    if not isinstance(spec["origin_id"], str) or not spec["origin_id"]:
        raise AdapterError("adapter origin_id is empty")
    if not isinstance(spec["donor_step_id"], str) or not spec["donor_step_id"]:
        raise AdapterError("adapter donor_step_id is empty")
    if not isinstance(spec["origin_job"], dict):
        raise AdapterError("adapter origin_job must be an object")
    graph = spec["primitive_graph_sha256"]
    if not isinstance(graph, str) or _SHA256.fullmatch(graph) is None:
        raise AdapterError("adapter primitive graph is not SHA-256-bound")
    from .. import protocol
    expected_graph = protocol._primitive_graph_sha256(
        destination, "on" if mechanism_enabled else "off", route
    )
    if graph != expected_graph:
        raise AdapterError("adapter primitive graph is cross-wired to another mechanism state")
    declared_route = spec.get("route")
    if route != declared_route:
        raise AdapterError("requested route differs from its frozen spec")
    from .. import protocol
    if route != protocol.ROUTE_BY_DESTINATION.get(destination):
        raise UnsupportedRoute("route is not preregistered for this destination")
    grid_id = spec.get("grid_id", spec.get("config_id"))
    config_id = spec.get("config_id", grid_id)
    if not isinstance(grid_id, str) or re.fullmatch(r"g[0-9]{2}", grid_id) is None:
        raise AdapterError("adapter grid_id is invalid")
    if config_id != grid_id:
        raise AdapterError("config_id and grid_id disagree")
    strategy = ON_STRATEGY if mechanism_enabled else OFF_STRATEGY
    coordinate = f"{strategy}.{destination}.{grid_id}"
    implementation = (
        f"tilelang_abstraction.F1.{grid_id}"
        if destination == "tilelang" and mechanism_enabled else coordinate
    )
    if spec.get("coordinate_cell_id", coordinate) != coordinate:
        raise AdapterError("adapter coordinate cell identity changed")
    if spec.get("implementation_id", implementation) != implementation:
        raise AdapterError("adapter implementation identity changed")
    return grid_id, graph


def _build_tilelang_f1(cell: dict[str, Any]):
    """Reuse the existing F1 builder while retaining its generated GEMM source."""
    import torch
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import candidates
    from ako_runs.phase2_fused_sdpa.variants2 import fused_tilelang_abstraction as abstraction

    cfg = candidates._phase2_config(cell, "regs", "GBGS")
    cfg.dsl, cfg.variant = "tilelang_abs", "F1"
    captured: dict[str, str] = {}
    original = abstraction._kernel_gemm

    def capture(*args, **kwargs):
        kernel = original(*args, **kwargs)
        captured["cuda_source_gemm"] = kernel.get_kernel_source()
        return kernel

    started = time.perf_counter()
    abstraction._kernel_gemm = capture
    try:
        with candidates._artifact_scope(cell["cell_id"]) as root:
            built = abstraction.build(cfg)
    finally:
        abstraction._kernel_gemm = original
    source = captured.get("cuda_source_gemm")
    if built.n_kernels != 2 or built.x_dtype != torch.float16 or not source:
        raise AdapterError("TileLang F1 violated the full two-kernel/source contract")
    built.artifacts["cuda_source_gemm"] = source
    built.artifacts["cuda_source_gemm_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return candidates._finish(
        cell, built, cfg, builder="trajectory-transfer:tilelang:F1:full",
        wall_s=time.perf_counter() - started, artifact_root=root,
    )


def _cuda_gemm_body(source: str) -> str:
    """Remove the second kernel/wrapper and the arm-only softmax switch."""
    if not isinstance(source, str) or not source:
        raise AdapterError("CUDA builder did not retain generated source")
    cuts = [
        index for marker in ("/* ---- Phase-2 row softmax", '#include "checked_cuda_launch.h"')
        if (index := source.find(marker)) >= 0
    ]
    if not cuts:
        raise AdapterError("CUDA generated source has no kernel boundary")
    return re.sub(
        r"^#define HAS_SOFTMAX [01]$", "#define HAS_SOFTMAX <paired>",
        source[:min(cuts)], flags=re.MULTILINE,
    )


def _held_gemm_binding(
    destination: str, built: Any, paired_config: dict[str, Any]
) -> dict[str, Any]:
    artifacts = built.metadata.get("artifacts", {})
    lane = artifacts.get("lane_gbg", artifacts) if isinstance(artifacts, dict) else {}
    if destination == "tilelang":
        source = lane.get("cuda_source_gemm", lane.get("cuda_source"))
        kind = "generated_tilelang_cuda_gemm"
    elif destination.startswith("cuda_"):
        source = _cuda_gemm_body(lane.get("cuda_source"))
        kind = "generated_cuda_first_kernel_body"
    else:
        path = _REPO / "ako_runs/phase2_fused_sdpa/variants2/fused_triton.py"
        text = path.read_text(encoding="utf-8")
        try:
            source = "@triton.jit\ndef _fused_gemm_kernel" + text.split(
                "@triton.jit\ndef _fused_gemm_kernel", 1
            )[1].split("\n\n@triton.jit", 1)[0]
        except (IndexError, ValueError) as exc:
            raise AdapterError("cannot bind the Triton GEMM source body") from exc
        kind = "triton_jit_gemm_source_body"
    if not isinstance(source, str) or not source:
        raise AdapterError("builder did not expose its first GEMM source/body")
    return {
        "kind": kind,
        "schedule": paired_config,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _build_destination(
    destination: str,
    spec: dict[str, Any],
    mechanism_enabled: bool,
    route: str | None = None,
):
    if not isinstance(mechanism_enabled, bool):
        raise AdapterError("mechanism_enabled must be boolean")
    route = spec.get("route") if route is None and isinstance(spec, dict) else route
    if not isinstance(route, str):
        raise UnsupportedRoute("transfer route is missing")
    grid_id, graph_sha256 = _validated_spec(spec, destination, route, mechanism_enabled)
    recipe = _primitive_recipe(destination, route, spec)
    strategy = ON_STRATEGY if mechanism_enabled else OFF_STRATEGY
    matches = [
        cell for cell in _resolved_cells()
        if cell.get("strategy") == strategy
        and cell.get("lane") == destination
        and cell.get("grid_id") == grid_id
    ]
    if len(matches) != 1:
        raise AdapterError(
            f"expected one resolved crossed-v2 cell for {strategy}.{destination}.{grid_id}"
        )
    cell = matches[0]
    coordinate_cell_id = f"{strategy}.{destination}.{grid_id}"
    implementation_id = (
        f"tilelang_abstraction.F1.{grid_id}"
        if destination == "tilelang" and mechanism_enabled else coordinate_cell_id
    )
    if cell.get("cell_id") != coordinate_cell_id:
        raise AdapterError("resolved coordinate cell identity changed")
    if spec["origin_job"] != cell.get("origin_job"):
        raise AdapterError("frozen destination origin_job differs from the resolved builder input")

    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import candidates

    built = (
        _build_tilelang_f1(cell)
        if destination == "tilelang" and mechanism_enabled
        else candidates.build(cell)
    )
    if built.metadata.get("n_kernels") != 2:
        raise AdapterError("the selected common-to-native-softmax pair must retain two kernels")
    paired_config = _paired_config(built.config)
    held_gemm = _held_gemm_binding(destination, built, paired_config)
    module_path = Path(__file__).with_name(f"{destination}.py")
    sources = sorted(
        _generated_source_bindings(built.metadata.get("artifacts", {})),
        key=lambda row: (row["path"], row["sha256"]),
    )
    metadata = dict(built.metadata)
    metadata.update(
        {
            "adapter_dispatch_sha256": _file_sha256(Path(__file__)),
            "adapter_module_sha256": _file_sha256(module_path),
            "config_sha256": _canonical_sha256(built.config),
            "coordinate_cell_id": coordinate_cell_id,
            "coordinate_cell_sha256": _canonical_sha256(cell),
            "destination": destination,
            "donor_step_id": spec["donor_step_id"],
            "generated_source_bindings": sources,
            "generated_source_sha256": _canonical_sha256(sources),
            "held_gemm_source_binding": held_gemm,
            "held_gemm_source_sha256": _canonical_sha256(held_gemm),
            "implementation_id": implementation_id,
            "mechanism_enabled": mechanism_enabled,
            "origin_id": spec["origin_id"],
            "primitive_graph_sha256": graph_sha256,
            "primitive_mapping": recipe,
            "paired_config_sha256": _canonical_sha256(paired_config),
            "protocol_route": route,
            "route": route,
        }
    )
    built.metadata = metadata
    return built


def build(
    spec: dict[str, Any], mechanism_enabled: bool, route: str | None = None
):
    """Build one frozen destination pair member through its lane module."""
    destination = spec.get("destination") if isinstance(spec, dict) else None
    if destination not in DESTINATIONS:
        raise AdapterError(f"unknown destination: {destination!r}")
    module = importlib.import_module(f"{__name__}.{destination}")
    return module.build(spec, mechanism_enabled, route)
