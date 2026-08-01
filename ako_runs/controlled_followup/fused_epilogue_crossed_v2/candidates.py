#!/usr/bin/env python3
"""Build one v2 cell without mutating receipt-bound historical sources."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from .core import HERE, REPO_ROOT, ProtocolError, canonical_sha256, parse_set
except ImportError:  # direct script execution
    from core import HERE, REPO_ROOT, ProtocolError, canonical_sha256, parse_set


PHASE1 = REPO_ROOT / "ako_runs/phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs/phase2_fused_sdpa"
V1 = REPO_ROOT / "ako_runs/controlled_followup/fused_epilogue_crossed_v1"
for path in (str(REPO_ROOT), str(PHASE2), str(PHASE1), str(V1), str(HERE)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

class UnsupportedStrategy(ProtocolError):
    """A measured support probe retained this path as unsupported."""


@dataclass
class BuiltCell:
    cell_id: str
    run: Callable[[Any, Any, Any], Any]
    compile_s: float
    config: dict[str, Any]
    metadata: dict[str, Any]


@contextmanager
def _artifact_scope(cell_id: str):
    import common
    import common2

    root = Path(tempfile.gettempdir()) / "fused_epilogue_crossed_v2" / f"pid{os.getpid()}" / cell_id
    root.mkdir(parents=True, exist_ok=True)
    old_phase1, old_phase2 = common.ARTIFACTS_DIR, common2.ARTIFACTS_DIR
    common.ARTIFACTS_DIR, common2.ARTIFACTS_DIR = str(root / "phase1"), str(root / "phase2")
    try:
        yield root
    finally:
        common.ARTIFACTS_DIR, common2.ARTIFACTS_DIR = old_phase1, old_phase2


def _phase2_config(cell: dict[str, Any], epilogue: str, arm: str):
    import common2

    overrides = parse_set(cell["origin_job"]["set"])
    overrides.setdefault("extra", {})["epilogue"] = epilogue
    overrides["extra"]["wcache"] = "cached"
    cfg = common2.make_fused_config(cell["lane"], arm, **overrides)
    if (cfg.M, cfg.K, cfg.N) != (1024, 8192, 8192):
        raise ProtocolError("candidate left the frozen shape")
    if cfg.arith != "fp16" or cfg.cast != "precast" or cfg.extra.get("wcache") != "cached":
        raise ProtocolError("candidate left the fp16/precast/cached contract")
    if cfg.extra.get("epilogue") != epilogue:
        raise ProtocolError("candidate did not consume the requested epilogue")
    return cfg


def _normalize_artifacts(artifacts: dict[str, Any], cfg=None) -> dict[str, Any]:
    value = dict(artifacts)
    dynamic = next(
        (
            value[key]
            for key in (
                "total_dynamic_shared_bytes",
                "total_dynamic_smem_bytes",
                "dynamic_smem_bytes",
                "shared_bytes",
            )
            if value.get(key) is not None
        ),
        None,
    )
    operand = value.get(
        "operand_smem_bytes",
        value.get("operand_shared_bytes", value.get("geom_info", {}).get("smem_bytes")),
    )
    epilogue = value.get(
        "epilogue_tile_bytes",
        value.get("epilogue_tile_shared_bytes", value.get("shared_epilogue_bytes")),
    )
    if epilogue is None and cfg is not None:
        epilogue = cfg.BM * (cfg.BN + 4) * 4 if cfg.extra.get("epilogue") == "smem" else 0
    if dynamic is None and value.get("shared_epilogue_bytes") is not None and cfg is not None:
        operand = (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * 2 * cfg.stages
        dynamic = max(int(operand), int(epilogue))
        value["dynamic_allocation_accounting"] = (
            "TileLang source allocations: stages*(As+Bs), lifetime-overlapped with Cs"
        )
    if dynamic is not None:
        dynamic = int(dynamic)
    value.update(
        {
            "dynamic_smem_bytes": dynamic,
            "epilogue_tile_bytes": int(epilogue) if epilogue is not None else None,
            "operand_smem_bytes": int(operand) if operand is not None else None,
            "shared_bytes": dynamic,
            "total_dynamic_shared_bytes": dynamic,
            "total_dynamic_smem_bytes": dynamic,
        }
    )
    return value


def _source_digests(value: Any, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(item, str) and "source" in str(key):
                found.append((child, hashlib.sha256(item.encode("utf-8")).hexdigest()))
            elif str(key).endswith("source_sha256") and isinstance(item, str):
                found.append((child, item))
            else:
                found.extend(_source_digests(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_source_digests(item, f"{path}[{index}]"))
    return found


def _finish(cell: dict[str, Any], built, cfg, *, builder: str, wall_s: float, artifact_root: Path) -> BuiltCell:
    artifacts = _normalize_artifacts(built.artifacts, cfg)
    metadata = {
        "artifacts": artifacts,
        "build_wall_s": wall_s,
        "builder": builder,
        "epilogue_request": cfg.extra.get("epilogue"),
        "isolated_artifact_root": str(artifact_root),
        "n_kernels": built.n_kernels,
        "notes": built.notes,
    }
    metadata["implementation_sha256"] = canonical_sha256(
        {
            "cell": cell["cell_id"],
            "config": cfg.to_dict(),
            "sources": sorted(_source_digests(artifacts)),
        }
    )
    return BuiltCell(cell["cell_id"], built.run, built.compile_s, cfg.to_dict(), metadata)


def _build_phase2(cell: dict[str, Any], epilogue: str, arm: str):
    import torch
    import variants2
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import (
        cuda_unlimited as v2_cuda_unlimited,
        support_probes as v2_support_probes,
    )

    cfg = _phase2_config(cell, epilogue, arm)
    started = time.perf_counter()
    with _artifact_scope(cell["cell_id"]) as root:
        if cell["lane"] == "cuda_unlimited":
            built = v2_cuda_unlimited.build(cfg)
        elif cell.get("support_probe_key") and cell["lane"] in {"cuda_noptx", "triton"}:
            built = v2_support_probes.build_probe_candidate(cell["support_probe_key"], cfg)
        else:
            built = variants2.build("fused", cfg)
    expected_kernels = 2 if arm == "GBGS" else 1
    if built.x_dtype != torch.float16 or built.n_kernels != expected_kernels:
        raise ProtocolError(
            f"{arm} builder violated the precast/{expected_kernels}-kernel contract"
        )
    return _finish(
        cell,
        built,
        cfg,
        builder=f"crossed-v2:{cell['lane']}:{arm}:{epilogue}",
        wall_s=time.perf_counter() - started,
        artifact_root=root,
    )


def _build_tilelang_smem(cell: dict[str, Any]) -> BuiltCell:
    import torch
    from ako_runs.controlled_followup.fused_epilogue_crossed_v1 import (
        tilelang_smem as v1_tilelang_smem,
    )

    cfg = _phase2_config(cell, "smem", "GBGS")
    epilogue_bytes = cfg.BM * (cfg.BN + 4) * 4
    if epilogue_bytes > 101_376:
        raise ProtocolError(
            f"smem epilogue tile {epilogue_bytes} B exceeds the sm_89 cap at setup"
        )
    started = time.perf_counter()
    with _artifact_scope(cell["cell_id"]) as root:
        built = v1_tilelang_smem.build(cfg)
    if built.x_dtype != torch.float16 or built.n_kernels != 2:
        raise ProtocolError("TileLang smem builder violated the precast two-kernel contract")
    return _finish(cell, built, cfg, builder="crossed-v2:tilelang-smem", wall_s=time.perf_counter() - started, artifact_root=root)


def _build_global(cell: dict[str, Any]) -> BuiltCell:
    import common
    import common2
    import torch
    import variants
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import (
        postprocess as v2_postprocess,
    )

    parsed = parse_set(cell["origin_job"]["set"])
    cfg = common.Config(
        dsl=cell["lane"], variant="D", M=1024, N=8192, K=8192,
        BM=parsed["BM"], BN=parsed["BN"], BK=parsed["BK"],
        threads=parsed["threads"], kc=parsed["kc"], stages=parsed["stages"],
        arith="fp16", cast="precast",
    )
    started = time.perf_counter()
    with _artifact_scope(cell["cell_id"]) as root:
        saved = (common.M, common.N, common.K)
        if cell["lane"] == "cuda_noptx":
            common.M, common.N, common.K = cfg.M, cfg.N, cfg.K
        try:
            gemm = variants.build(cfg)
        finally:
            common.M, common.N, common.K = saved
        if gemm.input_dtype != torch.float16:
            raise ProtocolError("global GEMM violated the precast fp16 contract")
        combined = v2_postprocess.build(has_bias=True, has_gelu=True)
    cached_weight = common2.weight_fn("cached")

    def run(x, weight, bias):
        return combined.run(gemm.run(x, cached_weight(weight)), bias)

    artifacts = {
        "gemm": _normalize_artifacts(gemm.artifacts),
        "postprocess": combined.artifacts,
        "strategy": "global_intermediate",
    }
    metadata = {
        "artifacts": artifacts,
        "build_wall_s": time.perf_counter() - started,
        "builder": f"phase1:{cell['lane']}+crossed-v2:common-postprocess",
        "implementation_sha256": canonical_sha256(
            {"cell": cell["cell_id"], "config": cfg.to_dict(), "sources": sorted(_source_digests(artifacts))}
        ),
        "isolated_artifact_root": str(root),
        "n_kernels": 2,
        "notes": "lane-native GEMM plus common bias+exact-GELU+softmax",
        "common_softmax_source_sha256": combined.artifacts["softmax_source_sha256"],
    }
    return BuiltCell(cell["cell_id"], run, gemm.compile_s + combined.compile_s, cfg.to_dict(), metadata)


def _build_register_common(cell: dict[str, Any]) -> BuiltCell:
    from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import (
        postprocess as v2_postprocess,
    )

    lane = _build_phase2(cell, "regs", "GBG")
    started = time.perf_counter()
    common = v2_postprocess.build(has_bias=False, has_gelu=False)

    def run(x, weight, bias):
        return common.run(lane.run(x, weight, bias), bias)

    artifacts = {
        "lane_gbg": lane.metadata["artifacts"],
        "postprocess": common.artifacts,
        "strategy": "register_common_postprocess",
    }
    metadata = {
        "artifacts": artifacts,
        "build_wall_s": lane.metadata["build_wall_s"] + (time.perf_counter() - started),
        "builder": f"crossed-v2:{cell['lane']}:GBG-regs+common-softmax",
        "epilogue_request": "regs",
        "implementation_sha256": canonical_sha256(
            {"cell": cell["cell_id"], "config": lane.config, "sources": sorted(_source_digests(artifacts))}
        ),
        "isolated_artifact_root": lane.metadata["isolated_artifact_root"],
        "n_kernels": 2,
        "notes": "lane-native register bias+exact-GELU followed by byte-identical common softmax",
        "common_softmax_source_sha256": common.artifacts["softmax_source_sha256"],
    }
    return BuiltCell(cell["cell_id"], run, lane.compile_s + common.compile_s, lane.config, metadata)


def build(cell: dict[str, Any]) -> BuiltCell:
    if cell.get("support_declared") is None:
        raise ProtocolError(f"support unresolved for {cell['cell_id']}")
    if cell["support_declared"] is False:
        raise UnsupportedStrategy(cell["support_detail"])
    strategy, lane = cell["strategy"], cell["lane"]
    if strategy == "register_fused":
        return _build_phase2(cell, "regs", "GBGS")
    if strategy == "smem_staged":
        return _build_tilelang_smem(cell) if lane == "tilelang" else _build_phase2(cell, "smem", "GBGS")
    if strategy == "global_intermediate":
        return _build_global(cell)
    if strategy == "register_common_postprocess":
        return _build_register_common(cell)
    raise ProtocolError(f"unknown strategy: {strategy}")
