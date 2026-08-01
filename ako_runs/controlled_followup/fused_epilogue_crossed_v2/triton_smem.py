#!/usr/bin/env python3
"""Pinned-Triton explicit shared-accumulator falsification candidate."""
from __future__ import annotations

import json


class ExplicitSharedMemoryUnavailable(RuntimeError):
    """The pinned public Triton language has no explicit shared allocator."""

    capability_limitation = True


class ExplicitSharedMemoryReviewRequired(RuntimeError):
    """A new API exists but has not been preregistered or validated."""


def make_source(cfg) -> str:
    """Retain the exact capability request, including the measured geometry."""
    request = {
        "BM": cfg.BM,
        "BN": cfg.BN,
        "BK": cfg.BK,
        "operation": "allocate fp32 BMxBN user-managed shared accumulator tile",
        "required_semantics": ["store tl.dot accumulator", "barrier", "reload", "bias", "exact GELU"],
        "triton_api": "public triton.language",
    }
    return (
        "# crossed-v2 measured Triton capability request\n"
        f"REQUEST = {json.dumps(request, sort_keys=True)}\n"
        "import triton.language as tl\n"
        "shared = tl.alloc_shared((REQUEST['BM'], REQUEST['BN']), tl.float32)\n"
        "# The probe succeeds only if the pinned public API resolves this explicit allocation.\n"
    )


def build(cfg):
    if cfg.arith != "fp16" or cfg.cast != "precast":
        raise ValueError("Triton smem probe requires fp16/precast")
    if cfg.extra.get("wcache", "cached") != "cached":
        raise ValueError("Triton smem probe requires cached weight")
    if cfg.extra.get("epilogue") != "smem":
        raise ValueError("Triton smem probe requires epilogue=smem")

    import triton
    import triton.language as tl

    candidates = sorted(
        name for name in dir(tl) if "shared" in name.lower() or "alloc" in name.lower()
    )
    allocator = getattr(tl, "alloc_shared", None)
    if allocator is None:
        raise ExplicitSharedMemoryUnavailable(
            f"Triton {triton.__version__} public triton.language has no alloc_shared; "
            f"shared/allocation-like symbols={candidates!r}"
        )
    raise ExplicitSharedMemoryReviewRequired(
        f"Triton {triton.__version__} unexpectedly exposes alloc_shared={allocator!r}; "
        "the pinned campaign must be reviewed before this unregistered API can define a strategy"
    )
