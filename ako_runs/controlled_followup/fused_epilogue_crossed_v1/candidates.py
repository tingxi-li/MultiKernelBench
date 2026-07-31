#!/usr/bin/env python3
"""Fail-closed builders for one crossed epilogue cell."""
from __future__ import annotations

import sys
import time
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core import HERE, REPO_ROOT, SUPPORT, ProtocolError, parse_set


PHASE1 = REPO_ROOT / "ako_runs/phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs/phase2_fused_sdpa"
for path in (str(REPO_ROOT), str(PHASE2), str(PHASE1), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)


class UnsupportedStrategy(ProtocolError):
    """The requested path cannot be expressed by the checked builder."""


@dataclass
class BuiltCell:
    cell_id: str
    run: Callable[[Any, Any, Any], Any]
    compile_s: float
    config: dict[str, Any]
    metadata: dict[str, Any]


@contextmanager
def _artifact_scope(cell_id: str):
    """Keep imported historical builders from touching receipt-bound outputs."""
    import common
    import common2

    root = Path(tempfile.gettempdir()) / "fused_epilogue_crossed_v1" / f"pid{os.getpid()}" / cell_id
    root.mkdir(parents=True, exist_ok=True)
    old_phase1 = common.ARTIFACTS_DIR
    old_phase2 = common2.ARTIFACTS_DIR
    common.ARTIFACTS_DIR = str(root / "phase1")
    common2.ARTIFACTS_DIR = str(root / "phase2")
    try:
        yield root
    finally:
        common.ARTIFACTS_DIR = old_phase1
        common2.ARTIFACTS_DIR = old_phase2


def _phase2_config(cell: dict[str, Any], epilogue: str):
    import common2

    overrides = parse_set(cell["origin_job"]["set"])
    overrides.setdefault("extra", {})["epilogue"] = epilogue
    overrides["extra"]["wcache"] = "cached"
    cfg = common2.make_fused_config(cell["lane"], "GBGS", **overrides)
    if (cfg.M, cfg.K, cfg.N) != (1024, 8192, 8192):
        raise ProtocolError("candidate left the frozen shape")
    if cfg.arith != "fp16" or cfg.cast != "precast" or cfg.extra.get("wcache") != "cached":
        raise ProtocolError("candidate left the fp16/precast/cached contract")
    return cfg


def _build_phase2(cell: dict[str, Any], epilogue: str) -> BuiltCell:
    import torch
    import variants2

    cfg = _phase2_config(cell, epilogue)
    started = time.perf_counter()
    with _artifact_scope(cell["cell_id"]) as artifact_root:
        built = variants2.build("fused", cfg)
    if built.x_dtype != torch.float16 or built.n_kernels != 2:
        raise ProtocolError("phase2 builder violated the precast two-kernel contract")
    return BuiltCell(
        cell_id=cell["cell_id"],
        run=built.run,
        compile_s=built.compile_s,
        config=cfg.to_dict(),
        metadata={
            "builder": f"phase2:{cell['lane']}",
            "build_wall_s": time.perf_counter() - started,
            "n_kernels": built.n_kernels,
            "notes": built.notes,
            "artifacts": built.artifacts,
            "isolated_artifact_root": str(artifact_root),
        },
    )


def _build_tilelang_smem(cell: dict[str, Any]) -> BuiltCell:
    import torch
    import tilelang_smem

    cfg = _phase2_config(cell, "smem")
    started = time.perf_counter()
    with _artifact_scope(cell["cell_id"]) as artifact_root:
        built = tilelang_smem.build(cfg)
    if built.x_dtype != torch.float16 or built.n_kernels != 2:
        raise ProtocolError("TileLang smem builder violated its declared contract")
    return BuiltCell(
        cell_id=cell["cell_id"],
        run=built.run,
        compile_s=built.compile_s,
        config=cfg.to_dict(),
        metadata={
            "builder": "crossed-v1:tilelang_smem",
            "build_wall_s": time.perf_counter() - started,
            "n_kernels": built.n_kernels,
            "notes": built.notes,
            "artifacts": built.artifacts,
            "isolated_artifact_root": str(artifact_root),
        },
    )


def _build_global(cell: dict[str, Any]) -> BuiltCell:
    import torch
    import common
    import common2
    import variants
    import postprocess

    parsed = parse_set(cell["origin_job"]["set"])
    cfg = common.Config(
        dsl=cell["lane"],
        variant="D",
        M=1024,
        N=8192,
        K=8192,
        BM=parsed["BM"],
        BN=parsed["BN"],
        BK=parsed["BK"],
        threads=parsed["threads"],
        kc=parsed["kc"],
        stages=parsed["stages"],
        arith="fp16",
        cast="precast",
    )
    started = time.perf_counter()
    # The receipt-bound no-PTX generator still reads Phase-1's module-level
    # shape while producing its source and warm-up allocation. Bind that shape
    # for the complete build call, then restore it; no source is edited.
    with _artifact_scope(cell["cell_id"]) as artifact_root:
        saved_shape = (common.M, common.N, common.K)
        if cell["lane"] == "cuda_noptx":
            common.M, common.N, common.K = cfg.M, cfg.N, cfg.K
        try:
            gemm = variants.build(cfg)
        finally:
            common.M, common.N, common.K = saved_shape
        if gemm.input_dtype != torch.float16:
            raise ProtocolError("global-intermediate GEMM does not accept precast fp16")
        combined = postprocess.build()
    cached_weight = common2.weight_fn("cached")

    def run(x, weight, bias):
        intermediate = gemm.run(x, cached_weight(weight))
        return combined.run(intermediate, bias)

    metadata = {
        "builder": f"phase1:{cell['lane']}+crossed-v1:combined_postprocess",
        "build_wall_s": time.perf_counter() - started,
        "n_kernels": 2,
        "notes": "lane-native GEMM writes fp32 global intermediate; one common CUDA kernel performs bias+exact-GELU+softmax",
        "artifacts": {
            "gemm": gemm.artifacts,
            "postprocess": combined.artifacts,
            "strategy": "global_intermediate",
        },
        "reported_compile_s_components": {
            "gemm": gemm.compile_s,
            "postprocess": combined.compile_s,
        },
        "isolated_artifact_root": str(artifact_root),
    }
    return BuiltCell(
        cell_id=cell["cell_id"],
        run=run,
        compile_s=gemm.compile_s + combined.compile_s,
        config=cfg.to_dict(),
        metadata=metadata,
    )


def build(cell: dict[str, Any]) -> BuiltCell:
    supported, reason = SUPPORT[(cell["strategy"], cell["lane"])]
    if cell.get("support_declared") is not supported or cell.get("support_detail") != reason:
        raise ProtocolError("cell support declaration differs from checked support matrix")
    if not supported:
        raise UnsupportedStrategy(reason)
    strategy, lane = cell["strategy"], cell["lane"]
    if strategy == "register_fused":
        return _build_phase2(cell, "regs")
    if strategy == "smem_staged":
        if lane == "tilelang":
            return _build_tilelang_smem(cell)
        return _build_phase2(cell, "smem")
    if strategy == "global_intermediate":
        return _build_global(cell)
    raise ProtocolError(f"unknown strategy: {strategy}")
